import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import os
import tqdm
import wandb
import copy
from torch.optim.lr_scheduler import _LRScheduler

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.1, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
    
class InverseSquareRootScheduler(_LRScheduler):    
    def __init__(self, optimizer, warmup_steps, init_lr=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        super(InverseSquareRootScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1

        if step <= self.warmup_steps:
            lr_scale = step / max(1, self.warmup_steps)
            return [self.init_lr + lr_scale * (base_lr - self.init_lr) for base_lr in self.base_lrs]

        else:
            return [base_lr * (self.warmup_steps ** 0.5) * (step ** -0.5) for base_lr in self.base_lrs]


def compute_class_weights_json(json_path, device):

    print(f"Loading annotations from {json_path}")
    
    class_names = ["advertisement", "cover", "story", "textstory", "first-page"]
    
    class_counts = {name: 0 for name in class_names}
    unknown_pages = 0
    total_pages = 0
    
    with open(json_path, 'r') as f:
        comics_data = json.load(f)
    
    category_mapping = {
        "advertisements": "advertisement",
        "covers": "cover",
        "stories": "story",
        "textstories": "textstory",
        "first-pages": "first-page" 
    }

    print("Counting pages per class...")
    for book in comics_data:
        book_pages = 0  

        for category, singular in category_mapping.items():
            if category in book:
                for item in book[category]:
        
                    start = item.get("page_start", 1) - 1 
                    end = item.get("page_end", start + 1) - 1
                    
                    if start < 0: 
                        start = 0
                    
                    pages_in_item = (end - start + 1)
                    class_counts[singular] += pages_in_item
                    book_pages += pages_in_item
        
        max_page = 0
        for category in category_mapping:
            if category in book:
                for item in book[category]:
                    end = item.get("page_end", 1)
                    if end > max_page:
                        max_page = end
        
        total_pages += max_page
        unknown_pages += max(0, max_page - book_pages)
    
    count_tensor = torch.tensor([class_counts[name] for name in class_names], dtype=torch.float)
    
    total_labeled = count_tensor.sum()
    weights = total_labeled / (count_tensor * len(class_names) + 1e-5)
    
    weights = weights / weights.sum() * len(class_names)
    
    print("\nClass distribution:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {int(count_tensor[i])} pages ({count_tensor[i]/total_labeled*100:.2f}%), weight={weights[i]:.4f}")
    print(f"  unknown/unlabeled: {unknown_pages} pages")
    print(f"  Total pages: {total_pages}")
    
    return weights.to(device)

def compute_class_weights(dataset, device):

    print(f"Computing class weights from dataset with {len(dataset.books)} books...")

    class_names = dataset.class_names 
    class_counts = {name: 0 for name in class_names}
    unknown_pages = 0 
    total_pages = 0

    for book in dataset.books:
        for label_idx in book['page_labels']:
            total_pages += 1
            if label_idx == -1 or label_idx is None:
                if "unknown" in class_counts: 
                    unknown_pages += 1
            elif 0 <= label_idx < len(class_names):
                label_name = class_names[label_idx]
                class_counts[label_name] += 1
            else:
                print(f"Warning: Found unexpected label index {label_idx} in dataset.")
                if "unknown" in class_counts:
                    unknown_pages += 1

    count_tensor = torch.tensor([class_counts[name] for name in class_names], dtype=torch.float)
    
    total_labeled = count_tensor.sum()
    if total_labeled == 0:
         print("Warning: No labeled pages found in the dataset. Returning uniform weights.")
         return torch.ones(len(class_names), device=device) / len(class_names)

    weights = total_labeled / (count_tensor * len(class_names) + 1e-7) 
    weights = weights / weights.sum() * len(class_names) 
    
    print("\nClass distribution from dataset:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {int(count_tensor[i])} pages ({count_tensor[i]/total_labeled*100:.2f}%), weight={weights[i]:.4f}")
    if "unknown" in class_counts:
         print(f"  unknown/unlabeled: {unknown_pages} pages")
    print(f"  Total pages processed: {total_pages}")
    
    return weights.to(device)

def train_model(run, model, train_loader, val_loader, test_loader, num_epochs=10, lr=1e-3, checkpoints='./checkpoints', 
                device='cuda', class_weights=None, name = 'BertBook', warmup = 44, initial_lr = 1e-7):
    
    os.makedirs(checkpoints, exist_ok=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    if class_weights is not None:
        print(f"Using weighted CrossEntropy with weights: {class_weights.cpu().numpy()}")
    else:
        print("Using standard CrossEntropyLoss.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    early_stopper = EarlyStopping(patience=7, min_delta=0.0001, restore_best_weights=True)
    
    scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=warmup, init_lr=initial_lr)
    
    best_val_f1 = -1.0 
    best_model_pth = os.path.join(checkpoints, f'best_{name}.pt')
    
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc="Epochs")
    
    for epoch in epoch_pbar:
        model.train()
        
        total_train_loss = 0.0
        total_train_samples = 0
        train_preds, train_labels = [], []
        
        train_pbar = tqdm.tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False)
        
        for idx, batch in enumerate(train_pbar):
            features = batch['features'].to(device) 
            attention_mask = batch['attention_mask'].to(device)
            page_labels = batch['page_labels'].to(device)  
            
            logits = model(features, attention_mask)
            batch_size, seq_length, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = page_labels.view(-1)

            valid_mask = (labels_flat != -1)
            valid_count = valid_mask.sum().item()

            loss = criterion(logits_flat, labels_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            if valid_count > 0:
                total_train_loss += loss.item() * valid_count
                total_train_samples += valid_count
            
            with torch.no_grad():
                predictions = logits_flat.argmax(dim=1)
                
            train_preds.extend(predictions[valid_mask].cpu().numpy())
            train_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            if idx % 10 == 0:
                run.log({"batch_train_loss": loss.item(), "epoch": epoch})
        
        train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
        train_acc = accuracy_score(train_labels, train_preds) if total_train_samples > 0 else 0
        train_f1 = f1_score(train_labels, train_preds, average='macro')  if total_train_samples > 0 else 0
        
        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0
        val_preds, val_labels = [], []
        
        val_pbar = tqdm.tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}", leave=False)
        
        with torch.no_grad():
            for idx, batch in enumerate(val_pbar):
                features = batch['features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                page_labels = batch['page_labels'].to(device)
                
                logits = model(features, attention_mask)
                batch_size, seq_length, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = page_labels.view(-1)
                
                valid_mask = (labels_flat != -1)
                valid_count = valid_mask.sum().item()
                
                loss = criterion(logits_flat, labels_flat)
       
                if valid_count > 0:
                    total_val_loss += loss.item() * valid_count
                    total_val_samples += valid_count
                
                predictions = logits_flat.argmax(dim=1)
                val_preds.extend(predictions[valid_mask].cpu().numpy())
                val_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
                val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                if idx % 10 == 0:
                    run.log({"batch_val_loss": loss.item(), "epoch": epoch})
        
        val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
        val_acc = accuracy_score(val_labels, val_preds) if total_val_samples > 0 else 0
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0) if total_val_samples > 0 else 0
        
        total_test_loss = 0.0
        total_test_samples = 0
        test_preds, test_labels = [], []
        
        test_pbar = tqdm.tqdm(test_loader, desc=f"Test {epoch+1}/{num_epochs}", leave=False)
        
        with torch.no_grad():
            for idx, batch in enumerate(test_pbar):
                features = batch['features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                page_labels = batch['page_labels'].to(device)
                
                logits = model(features, attention_mask)
                batch_size, seq_length, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = page_labels.view(-1)
                
                valid_mask = (labels_flat != -1)
                valid_count = valid_mask.sum().item()
                
                loss = criterion(logits_flat, labels_flat)
       
                if valid_count > 0:
                    total_test_loss += loss.item() * valid_count
                    total_test_samples += valid_count
                
                predictions = logits_flat.argmax(dim=1)
                test_preds.extend(predictions[valid_mask].cpu().numpy())
                test_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
                test_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                if idx % 10 == 0:
                    run.log({"batch_test_loss": loss.item(), "epoch": epoch})
        
        test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0
        test_acc = accuracy_score(test_labels, test_preds) if total_test_samples > 0 else 0
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0) if total_test_samples > 0 else 0
        
        
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_f1": f"{train_f1:.4f}",
            "val_loss": f"{val_loss:.4f}", 
            "val_f1": f"{val_f1:.4f}"
        })
        
        print(f'Epoch {epoch+1} / {num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_pth)
            print(f'Model saved in: {best_model_pth} (val F1: {val_f1:.4f})')
        
        run.log({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1,
            'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1 
        })
        
        if early_stopper(model, val_loss):
            print(early_stopper.status)
            break 
        
        print(early_stopper.status)
    
    print("Training finished.")
    
    if os.path.exists(best_model_pth):
        print(f"Loading best model (based on val F1) from {best_model_pth}")
        model.load_state_dict(torch.load(best_model_pth, map_location=device, weights_only=False))
    else:
        print("Warning: No best F1 model file found. Returning model from last epoch or early stopping point.")            
        
    return model

def train_model_detection(run, model, train_loader, val_loader, test_loader, num_epochs=10, lr=1e-3, checkpoints='./checkpoints', 
                device='cuda', class_weights=None, name = 'BertBook', warmup = 44, initial_lr = 1e-7):
    
    os.makedirs(checkpoints, exist_ok=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    if class_weights is not None:
        print(f"Using weighted CrossEntropy with weights: {class_weights.cpu().numpy()}")
    else:
        print("Using standard CrossEntropyLoss.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    early_stopper = EarlyStopping(patience=7, min_delta=0.0001, restore_best_weights=True)
    
    scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=warmup, init_lr=initial_lr)
    
    best_val_f1 = -1.0 
    best_model_pth = os.path.join(checkpoints, f'best_{name}.pt')
    
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc="Epochs")
    
    for epoch in epoch_pbar:
        model.train()
        
        total_train_loss = 0.0
        total_train_samples = 0
        train_preds, train_labels = [], []
        
        train_pbar = tqdm.tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False)
        
        for idx, batch in enumerate(train_pbar):
            features = batch['features'].to(device) 
            detection_features = batch['detection_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            page_labels = batch['page_labels'].to(device)  
            
            logits = model(features, attention_mask, detection_features)
            batch_size, seq_length, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = page_labels.view(-1)

            valid_mask = (labels_flat != -1)
            valid_count = valid_mask.sum().item()

            loss = criterion(logits_flat, labels_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            if valid_count > 0:
                total_train_loss += loss.item() * valid_count
                total_train_samples += valid_count
            
            with torch.no_grad():
                predictions = logits_flat.argmax(dim=1)
                
            train_preds.extend(predictions[valid_mask].cpu().numpy())
            train_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            if idx % 10 == 0:
                run.log({"batch_train_loss": loss.item(), "epoch": epoch})
        
        train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
        train_acc = accuracy_score(train_labels, train_preds) if total_train_samples > 0 else 0
        train_f1 = f1_score(train_labels, train_preds, average='macro')  if total_train_samples > 0 else 0
        
        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0
        val_preds, val_labels = [], []
        
        val_pbar = tqdm.tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}", leave=False)
        
        with torch.no_grad():
            for idx, batch in enumerate(val_pbar):
                features = batch['features'].to(device)
                detection_features = batch['detection_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                page_labels = batch['page_labels'].to(device)
                
                logits = model(features, attention_mask, detection_features)
                batch_size, seq_length, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = page_labels.view(-1)
                
                valid_mask = (labels_flat != -1)
                valid_count = valid_mask.sum().item()
                
                loss = criterion(logits_flat, labels_flat)
       
                if valid_count > 0:
                    total_val_loss += loss.item() * valid_count
                    total_val_samples += valid_count
                
                predictions = logits_flat.argmax(dim=1)
                val_preds.extend(predictions[valid_mask].cpu().numpy())
                val_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
                val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                if idx % 10 == 0:
                    run.log({"batch_val_loss": loss.item(), "epoch": epoch})
        
        val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
        val_acc = accuracy_score(val_labels, val_preds) if total_val_samples > 0 else 0
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0) if total_val_samples > 0 else 0
        
        total_test_loss = 0.0
        total_test_samples = 0
        test_preds, test_labels = [], []
        
        test_pbar = tqdm.tqdm(test_loader, desc=f"Test {epoch+1}/{num_epochs}", leave=False)
        
        with torch.no_grad():
            for idx, batch in enumerate(test_pbar):
                features = batch['features'].to(device)
                detection_features = batch['detection_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                page_labels = batch['page_labels'].to(device)
                
                logits = model(features, attention_mask, detection_features)
                batch_size, seq_length, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = page_labels.view(-1)
                
                valid_mask = (labels_flat != -1)
                valid_count = valid_mask.sum().item()
                
                loss = criterion(logits_flat, labels_flat)
       
                if valid_count > 0:
                    total_test_loss += loss.item() * valid_count
                    total_test_samples += valid_count
                
                predictions = logits_flat.argmax(dim=1)
                test_preds.extend(predictions[valid_mask].cpu().numpy())
                test_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
                test_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                if idx % 10 == 0:
                    run.log({"batch_test_loss": loss.item(), "epoch": epoch})
        
        test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0
        test_acc = accuracy_score(test_labels, test_preds) if total_test_samples > 0 else 0
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0) if total_test_samples > 0 else 0
        
        
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_f1": f"{train_f1:.4f}",
            "val_loss": f"{val_loss:.4f}", 
            "val_f1": f"{val_f1:.4f}"
        })
        
        print(f'Epoch {epoch+1} / {num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_pth)
            print(f'Model saved in: {best_model_pth} (val F1: {val_f1:.4f})')
        
        run.log({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1,
            'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1 
        })
        
        if early_stopper(model, val_loss):
            print(early_stopper.status)
            break 
        
        print(early_stopper.status)
        
    print("Training finished.")
    
    if os.path.exists(best_model_pth):
        print(f"Loading best model (based on val F1) from {best_model_pth}")
        model.load_state_dict(torch.load(best_model_pth, map_location=device, weights_only=False))
    else:
        print("Warning: No best F1 model file found. Returning model from last epoch or early stopping point.")            
    
    return model

def train_model_fusion(run, model, train_loader, val_loader, test_loader, num_epochs=10, lr=1e-3, checkpoints='./checkpoints', 
                device='cuda', class_weights=None, name = 'BertBook', warmup = 44, initial_lr = 1e-7):
    
    os.makedirs(checkpoints, exist_ok=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    if class_weights is not None:
        print(f"Using weighted CrossEntropy with weights: {class_weights.cpu().numpy()}")
    else:
        print("Using standard CrossEntropyLoss.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    early_stopper = EarlyStopping(patience=7, min_delta=0.0001, restore_best_weights=True)
    
    scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=warmup, init_lr=initial_lr)
    
    best_val_f1 = -1.0 
    best_model_pth = os.path.join(checkpoints, f'best_{name}.pt')
    
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc="Epochs")
    
    for epoch in epoch_pbar:
        model.train()
        
        total_train_loss = 0.0
        total_train_samples = 0
        train_preds, train_labels = [], []
        
        train_pbar = tqdm.tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False)
        
        for idx, batch in enumerate(train_pbar):
            fusion_features_batch = batch['features']
            features_on_device = {
                bb_name: tensor.to(device) for bb_name, tensor in fusion_features_batch.items()
            }
            attention_mask = batch['attention_mask'].to(device)
            page_labels = batch['page_labels'].to(device)  
            
            logits = model(features_on_device, attention_mask)
            batch_size, seq_length, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = page_labels.view(-1)

            valid_mask = (labels_flat != -1)
            valid_count = valid_mask.sum().item()

            loss = criterion(logits_flat, labels_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            if valid_count > 0:
                total_train_loss += loss.item() * valid_count
                total_train_samples += valid_count
            
            with torch.no_grad():
                predictions = logits_flat.argmax(dim=1)
                
            train_preds.extend(predictions[valid_mask].cpu().numpy())
            train_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            if idx % 10 == 0:
                run.log({"batch_train_loss": loss.item(), "epoch": epoch})
        
        train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
        train_acc = accuracy_score(train_labels, train_preds) if total_train_samples > 0 else 0
        train_f1 = f1_score(train_labels, train_preds, average='macro')  if total_train_samples > 0 else 0
        
        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0
        val_preds, val_labels = [], []
        
        val_pbar = tqdm.tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}", leave=False)
        
        with torch.no_grad():
            for idx, batch in enumerate(val_pbar):
                fusion_features_batch = batch['features']
                features_on_device = {
                    bb_name: tensor.to(device) for bb_name, tensor in fusion_features_batch.items()
                }
                attention_mask = batch['attention_mask'].to(device)
                page_labels = batch['page_labels'].to(device)
                
                logits = model(features_on_device, attention_mask)
                batch_size, seq_length, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = page_labels.view(-1)
                
                valid_mask = (labels_flat != -1)
                valid_count = valid_mask.sum().item()
                
                loss = criterion(logits_flat, labels_flat)
       
                if valid_count > 0:
                    total_val_loss += loss.item() * valid_count
                    total_val_samples += valid_count
                
                predictions = logits_flat.argmax(dim=1)
                val_preds.extend(predictions[valid_mask].cpu().numpy())
                val_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
                val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                if idx % 10 == 0:
                    run.log({"batch_val_loss": loss.item(), "epoch": epoch})
        
        val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
        val_acc = accuracy_score(val_labels, val_preds) if total_val_samples > 0 else 0
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0) if total_val_samples > 0 else 0
        
        total_test_loss = 0.0
        total_test_samples = 0
        test_preds, test_labels = [], []
        
        test_pbar = tqdm.tqdm(test_loader, desc=f"Test {epoch+1}/{num_epochs}", leave=False)
        
        with torch.no_grad():
            for idx, batch in enumerate(test_pbar):
                fusion_features_batch = batch['features']
                features_on_device = {
                    bb_name: tensor.to(device) for bb_name, tensor in fusion_features_batch.items()
                }
                attention_mask = batch['attention_mask'].to(device)
                page_labels = batch['page_labels'].to(device)
                
                logits = model(features_on_device, attention_mask)
                batch_size, seq_length, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = page_labels.view(-1)
                
                valid_mask = (labels_flat != -1)
                valid_count = valid_mask.sum().item()
                
                loss = criterion(logits_flat, labels_flat)
       
                if valid_count > 0:
                    total_test_loss += loss.item() * valid_count
                    total_test_samples += valid_count
                
                predictions = logits_flat.argmax(dim=1)
                test_preds.extend(predictions[valid_mask].cpu().numpy())
                test_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
                test_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                if idx % 10 == 0:
                    run.log({"batch_test_loss": loss.item(), "epoch": epoch})
        
        test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0
        test_acc = accuracy_score(test_labels, test_preds) if total_test_samples > 0 else 0
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0) if total_test_samples > 0 else 0
        
        
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_f1": f"{train_f1:.4f}",
            "val_loss": f"{val_loss:.4f}", 
            "val_f1": f"{val_f1:.4f}"
        })
        
        print(f'Epoch {epoch+1} / {num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_pth)
            print(f'Model saved in: {best_model_pth} (val F1: {val_f1:.4f})')
        
        run.log({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1,
            'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1 
        })
        
        if early_stopper(model, val_loss):
            print(early_stopper.status)
            break 
        
        print(early_stopper.status)
        
    print("Training finished.")
    
    if os.path.exists(best_model_pth):
        print(f"Loading best model (based on val F1) from {best_model_pth}")
        model.load_state_dict(torch.load(best_model_pth, map_location=device, weights_only=False))
    else:
        print("Warning: No best F1 model file found. Returning model from last epoch or early stopping point.")            
        
    return model

def train_multimodal(run, model, train_loader, val_loader, test_loader, num_epochs=10, lr=1e-3, checkpoints='./checkpoints', 
                device='cuda', class_weights=None, name = 'BertBook', warmup = 44, initial_lr = 1e-7):
    
    os.makedirs(checkpoints, exist_ok=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    if class_weights is not None:
        print(f"Using weighted CrossEntropy with weights: {class_weights.cpu().numpy()}")
    else:
        print("Using standard CrossEntropyLoss.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    early_stopper = EarlyStopping(patience=7, min_delta=0.0001, restore_best_weights=True)
    
    scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=warmup, init_lr=initial_lr)
    
    best_val_f1 = -1.0 
    best_model_pth = os.path.join(checkpoints, f'best_{name}.pt')
    
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc="Epochs")
    
    for epoch in epoch_pbar:
        model.train()
        
        total_train_loss = 0.0
        total_train_samples = 0
        train_preds, train_labels = [], []
        
        train_pbar = tqdm.tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False)
        
        for idx, batch in enumerate(train_pbar):
            textual_features = batch['textual_features'].to(device) 
            visual_features = batch['visual_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            page_labels = batch['page_labels'].to(device)  
            
            logits = model(textual_features, visual_features, attention_mask)
            batch_size, seq_length, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = page_labels.view(-1)

            valid_mask = (labels_flat != -1)
            valid_count = valid_mask.sum().item()

            loss = criterion(logits_flat, labels_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            if valid_count > 0:
                total_train_loss += loss.item() * valid_count
                total_train_samples += valid_count
            
            with torch.no_grad():
                predictions = logits_flat.argmax(dim=1)
                
            train_preds.extend(predictions[valid_mask].cpu().numpy())
            train_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            if idx % 10 == 0:
                run.log({"batch_train_loss": loss.item(), "epoch": epoch})
        
        train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
        train_acc = accuracy_score(train_labels, train_preds) if total_train_samples > 0 else 0
        train_f1 = f1_score(train_labels, train_preds, average='macro')  if total_train_samples > 0 else 0
        
        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0
        val_preds, val_labels = [], []
        
        val_pbar = tqdm.tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}", leave=False)
        
        with torch.no_grad():
            for idx, batch in enumerate(val_pbar):
                textual_features = batch['textual_features'].to(device) 
                visual_features = batch['visual_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                page_labels = batch['page_labels'].to(device)  
                
                logits = model(textual_features, visual_features, attention_mask)
                batch_size, seq_length, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = page_labels.view(-1)
                
                valid_mask = (labels_flat != -1)
                valid_count = valid_mask.sum().item()
                
                loss = criterion(logits_flat, labels_flat)
       
                if valid_count > 0:
                    total_val_loss += loss.item() * valid_count
                    total_val_samples += valid_count
                
                predictions = logits_flat.argmax(dim=1)
                val_preds.extend(predictions[valid_mask].cpu().numpy())
                val_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
                val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                if idx % 10 == 0:
                    run.log({"batch_val_loss": loss.item(), "epoch": epoch})
        
        val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
        val_acc = accuracy_score(val_labels, val_preds) if total_val_samples > 0 else 0
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0) if total_val_samples > 0 else 0
        
        total_test_loss = 0.0
        total_test_samples = 0
        test_preds, test_labels = [], []
        
        test_pbar = tqdm.tqdm(test_loader, desc=f"Test {epoch+1}/{num_epochs}", leave=False)
        
        with torch.no_grad():
            for idx, batch in enumerate(test_pbar):
                textual_features = batch['textual_features'].to(device) 
                visual_features = batch['visual_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                page_labels = batch['page_labels'].to(device)  
                
                logits = model(textual_features, visual_features, attention_mask)
                batch_size, seq_length, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = page_labels.view(-1)
                
                valid_mask = (labels_flat != -1)
                valid_count = valid_mask.sum().item()
                
                loss = criterion(logits_flat, labels_flat)
       
                if valid_count > 0:
                    total_test_loss += loss.item() * valid_count
                    total_test_samples += valid_count
                
                predictions = logits_flat.argmax(dim=1)
                test_preds.extend(predictions[valid_mask].cpu().numpy())
                test_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
                test_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                if idx % 10 == 0:
                    run.log({"batch_test_loss": loss.item(), "epoch": epoch})
        
        test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0
        test_acc = accuracy_score(test_labels, test_preds) if total_test_samples > 0 else 0
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0) if total_test_samples > 0 else 0
        
        
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_f1": f"{train_f1:.4f}",
            "val_loss": f"{val_loss:.4f}", 
            "val_f1": f"{val_f1:.4f}"
        })
        
        print(f'Epoch {epoch+1} / {num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_pth)
            print(f'Model saved in: {best_model_pth} (val F1: {val_f1:.4f})')
        
        run.log({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1,
            'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1 
        })
        
        if early_stopper(model, val_loss):
            print(early_stopper.status)
            break 
        
        print(early_stopper.status)
    
    print("Training finished.")
    
    if os.path.exists(best_model_pth):
        print(f"Loading best model (based on val F1) from {best_model_pth}")
        model.load_state_dict(torch.load(best_model_pth, map_location=device, weights_only=False))
    else:
        print("Warning: No best F1 model file found. Returning model from last epoch or early stopping point.")            
        
    return model