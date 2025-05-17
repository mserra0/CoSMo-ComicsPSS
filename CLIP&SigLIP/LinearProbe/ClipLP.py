import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image
import os
import numpy as np
import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ComicDataset(Dataset):    
    def __init__(self, root_dir, json_path, processor, cache_dir=None):

        self.root_dir = root_dir
        self.processor = processor
        self.image_paths = []
        self.labels = []
        self.label_to_id = {}

        cache_file = f"{cache_dir}/metadata.pt" if cache_dir else None
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached metadata from {cache_file}")
            cached_data = torch.load(cache_file)
            self.image_paths = cached_data['paths']
            self.labels = cached_data['labels']
            self.label_to_id = cached_data['label_map']
            self.class_names = list(self.label_to_id.keys())
        else:
            
            self._process_json(json_path)
     
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save({
                    'paths': self.image_paths,
                    'labels': self.labels,
                    'label_map': self.label_to_id
                }, f"{cache_dir}/metadata.pt")
    
    def _process_json(self, json_path):
        
        with open(json_path, 'r') as f:
            ground_truth = json.load(f)
        
        category_mapping = {
            "stories": "story",
            "textstories": "textstory",
            "advertisements": "advertisement",
            "covers": "cover"
        }
        
        valid_paths = []
        valid_labels = []
        skipped_count = 0
        
        for book in ground_truth:
            book_hash = book["hash_code"]
            for category_key, label in category_mapping.items():
                for item in book.get(category_key, []):
                    for page in range(item["page_start"], item["page_end"] + 1):
                        image_path = os.path.join(self.root_dir, book_hash, f"{page:03d}.jpg")
                        if os.path.exists(image_path):
                            try:
                                with Image.open(image_path) as img:
                                    img.verify()  
                                valid_paths.append(image_path)
                                valid_labels.append(label)
                            except Exception as e:
                                print(f"Warning: Skipping corrupted image - {image_path}: {e}")
                                skipped_count += 1
                        else:
                            print(f"Warning: Image not found - {image_path}")
                            skipped_count += 1
        
        print(f"Total images skipped: {skipped_count}")
        print(f"Total valid images: {len(valid_paths)}")
        
        unique_labels = sorted(list(set(valid_labels)))
        self.class_names = unique_labels
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        print(f"Found {len(unique_labels)} classes: {unique_labels}")
        
        self.image_paths = valid_paths
        self.labels = [self.label_to_id[label] for label in valid_labels]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
            processed = self.processor(images=image, return_tensors="pt")
            pixel_values = processed['pixel_values'].squeeze(0)
            return pixel_values, label
        except Exception as e:
            print(f"Error loading image {image_path} during training: {e}")
            dummy_image = torch.zeros((3, 224, 224)) 
            return dummy_image, label
    
    def get_num_classes(self):
        return len(self.label_to_id)
    
    def get_class_names(self):
        return self.class_names
    
    def get_sample_weights(self):
        class_counts = {}
        for label in self.labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        num_samples = len(self.labels)
        class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
        
        sample_weights = [class_weights[label] for label in self.labels]
        
        print("Class distribution:")
        for cls, count in class_counts.items():
            class_name = self.class_names[cls]
            print(f"  Class {cls} ({class_name}): {count} samples, weight={class_weights[cls]:.4f}")
        
        return sample_weights


class CLIPLinearProbe(nn.Module):
    def __init__(self, clip_model, num_classes, hidden_dim = 512, dropout_p = 0.3):
        super(CLIPLinearProbe, self).__init__()
        self.clip_model = clip_model
        self.feature_dim = clip_model.visual_projection.out_features
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_classes)
        )
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, pixel_values):
        with torch.no_grad():
            features = self.clip_model.get_image_features(pixel_values=pixel_values)
            
        logits = self.classifier(features)
        return logits
    
def train_model(run, model, train_loader, val_loader, device, num_epochs=10, lr=1e-4, out_dir=''):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    best_val_f1 = 0
    
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc="Epochs")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        train_pbar = tqdm.tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False)
        
        for idx, (images, labels) in enumerate(train_pbar):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if idx % 10 == 0:
                run.log({"batch_train_loss": loss.item(), "epoch": epoch})
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        val_pbar = tqdm.tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}", leave=False)
        
        with torch.no_grad():
            for idx, (images, labels) in enumerate(val_pbar):
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if idx % 10 == 0:
                    run.log({"batch_val_loss": loss.item(), "epoch": epoch})
                
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
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
            torch.save(model.state_dict(), f'{out_dir}/bestClipLPbase.pth')
            print(f'Model saved in: {out_dir}')
            
        run.log({'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1})
        run.log({'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1})
        
    return model  

def main(run, train=True, lr = 1e-4, dropout_p=0.3, epochs = 10, batch_size=32):
        
    root_dir = "/home/mserrao/PSSComics/Comics/datasets.unify/DCM/images"
    train_json = "/home/mserrao/PSSComics/Comics/DatasetDCM/comics100_all/comics_dev_books.json"
    val_json = "/home/mserrao/PSSComics/Comics/DatasetDCM/comics100_all/comics_val_books.json"
    test_json = "/home/mserrao/PSSComics/Comics/DatasetDCM/comics100_all/comics_test_books.json"
    cache_dir = '/home/mserrao/PSSComics/Comics/CLIP&SigLIP/ComicsCLIP'
    out_dir = '/home/mserrao/PSSComics/Comics/CLIP&SigLIP/LinearProbe/out'
    
    os.makedirs(out_dir, exist_ok=True)
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_id = 'openai/clip-vit-large-patch14'
    print(f"Loading model: {model_id}")
    processor = CLIPProcessor.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id)
    clip_model.to(device)
    
    print("Creating train dataset...")
    train_dataset = ComicDataset(root_dir, train_json, processor, cache_dir=f"{cache_dir}/dev")
    print("Creating validation dataset...")
    val_dataset = ComicDataset(root_dir, val_json, processor, cache_dir=f"{cache_dir}/val")
    print("Creating validation dataset...")
    test_dataset = ComicDataset(root_dir, test_json, processor, cache_dir=f"{cache_dir}/test")
    
    print("Calculating sample weights for balancing training...")
    train_weights = train_dataset.get_sample_weights()
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,  
        num_workers=2
    )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    num_classes = train_dataset.get_num_classes()
    model = CLIPLinearProbe(clip_model, num_classes, dropout_p=dropout_p).to(device)
    if train:
        model = train_model(run, model, train_loader, val_loader, device, num_epochs=epochs, lr=lr, out_dir=out_dir)
    else:
        try:
            model.load_state_dict(torch.load(f'{out_dir}/bestClipLPbase.pth', weights_only=True))
            print(f"Loaded pre-trained model from {out_dir}/bestClipLPbase.pth")
        except Exception as e:
            print(f'No model to load from {out_dir}/bestClipLPbase.pth', e)
            
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader, desc="Evaluation"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nFinal Model Performance:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.get_class_names()))
    
    # report_dict = classification_report(all_labels, all_preds, target_names=test_dataset.get_class_names(), output_dict=True)
    # report_df = pd.DataFrame(report_dict).T
    # wandb.log({"Classification report": wandb.Table(dataframe=report_df)})
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    print("Confusion Matrix Normalized:")
    print(cm_normalized)
    
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("Per-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        class_name = test_dataset.get_class_names()[i]
        print(f"  Class {i} ({class_name}): {acc:.4f}")
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=test_dataset.get_class_names(), yticklabels=test_dataset.get_class_names())
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    image_path = f"{out_dir}/normCM_CLIP.png"
    plt.savefig(image_path)

    wandb.log({"norm_confusion_matrix_CLIP": wandb.Image(image_path)})

    plt.close()
        
    print("Training completed!")
    
if __name__ == '__main__':
    
    run = wandb.init(
        project="ClipLP",
        name="clipSimpleMLP",
        config={
            "lr": 1e-4,
            "architecture": "ClipLPbase",
            "dataset": "DCM small",
            "epochs": 20,
            "dropout": 0.3,
            "batch_size": 128, 
            "model_id": 'openai/clip-vit-large-patch14'
        },
        )
    
    main(run,
        train=False, 
        lr = run.config.lr, 
        dropout_p=run.config.dropout, 
        epochs = run.config.epochs,
        batch_size=run.config.batch_size)