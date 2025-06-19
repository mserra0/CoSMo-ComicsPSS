import torch
from transformers import AutoModel
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import os
import numpy as np
import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data import ComicTransform
from utils.training import compute_class_weights, train_model_fusion
from utils.metrics import calculate_mndd, panoptic_quality_metrics
from pss_datasets.pss_fusion_dataset import PSSFusionDataset
from models.book_bert import BookBERTfusion, BookBERTMultiVision
import json
import random
import pandas as pd
import yaml
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def set_all_seeds(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_class_distribution(dataset_name, json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    categories = {"stories": 0, "textstories": 0, "advertisements": 0, "covers": 0}
    
    for book in data:
        for category in categories:
            if category in book:
                categories[category] += len(book[category])
    
    print(f"\n{dataset_name} distribution:")
    for category, count in categories.items():
        print(f"  - {category}: {count/sum(categories.values()):.2f}")

def main(run, gpu_id = 0,train=True, lr = 1e-4, dropout_p=0.4, epochs = 10, batch_size=32,
         seed=10, augmentations = False, num_aug_copies = 5, num_synthetic_books=200, 
         num_attention_heads = 4, num_hidden_layers = 4, positional_embeddings = 'absolute', hidden_dim = 256,
         model_name = 'BookBERT', warmup = 44, initial_lr = 1e-6, projection_dim = 1024, bert_input_dim = 1024):
    
    root_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images'
    precompute_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
    checkpoint_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/checkpoints'
    data_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
    out_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/out'
    
    backbone_ids = [
        "openai/clip-vit-large-patch14-336",
        # "facebook/dinov2-base",
        "google/siglip-so400m-patch14-384",
        "google/siglip2-large-patch16-512"
    ]
    
    set_all_seeds(seed)
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Process using device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # print(f"Loading features from models: \n {'\n '.join(backbone_ids)}")
    
    backbones = {}
    for model_id in backbone_ids:
        parts = model_id.split('/')[1].split('-')
        backbone_name = f'{parts[0]}_{parts[-1]}'

        backbone = AutoModel.from_pretrained(model_id).eval()
        
        if 'dinov2' in backbone_name:
            feature_dim = backbone.config.hidden_size
        elif 'clip' in backbone_name:
            feature_dim = backbone.config.vision_config.projection_dim
        elif 'siglip' in backbone_name:
            feature_dim = backbone.config.vision_config.hidden_size
        else:
            raise ValueError(f"Warning: Unknown backbone '{backbone_name}'")
        
        backbones[backbone_name] = feature_dim
        
        print(f'Loaded {backbone_name} with feature dim {feature_dim} to the configuration.')
    
    transformations = ComicTransform()
    
    train_dataset = PSSFusionDataset(
                root_dir = root_dir, 
                backbones = backbones, 
                annotations_path=f'{data_dir}/comics_train.json', 
                precompute_dir=f'{precompute_dir}/features_train.pt',
                max_seq_len=512, 
                device = device,
                augment_data=augmentations,
                num_augmented_copies = num_aug_copies,
                removal_p=0.05,
                num_synthetic_books=num_synthetic_books,
                min_stories=1,
                max_stories=3,
                synthetic_remove_p=0.15
                )
    
    val_dataset = PSSFusionDataset(
                root_dir = root_dir, 
                backbones = backbones, 
                annotations_path=f'{data_dir}/comics_val.json', 
                precompute_dir=f'{precompute_dir}/features_val.pt',
                max_seq_len=512, 
                device = device,
                augment_data=False
                )
    
    test_dataset = PSSFusionDataset(
                root_dir = root_dir, 
                backbones = backbones, 
                annotations_path=f'{data_dir}/comics_test.json', 
                precompute_dir=f'{precompute_dir}/features_test.pt',
                max_seq_len=512, 
                device = device,
                augment_data=False
                )

    # visualize_book(train_dataset, book_id='synthetic_1', book_idx=None, output_path=f"{out_dir}/book_sample.png", dpi=300, transforms=None)
    # visualize_book(train_dataset, book_id='synthetic_2', book_idx=None, output_path=f"{out_dir}/book_sample1.png", dpi=300, transforms=None)
    # visualize_book(train_dataset, book_id='synthetic_3', book_idx=None, output_path=f"{out_dir}/book_sample2.png", dpi=300, transforms=None)
    # return 0 
    
    print('Computing class weights...')
    class_weights = compute_class_weights(train_dataset, device) 
 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True 
    )
    
    val_loader = DataLoader( val_dataset, batch_size=batch_size)
    
    test_loader = DataLoader( test_dataset, batch_size=batch_size)
    
    num_classes = train_dataset.get_num_classes()
    
    model = BookBERTMultiVision(backbones=backbones, num_classes=num_classes, hidden_dim=hidden_dim, num_attention_heads=num_attention_heads,
                     num_hidden_layers=num_hidden_layers, dropout_p=dropout_p, positional_embeddings=positional_embeddings,
                     projection_dim=projection_dim, bert_input_dim=bert_input_dim)
    
    if train:
        model = train_model_fusion(run, model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, lr=lr, 
                            device=device, num_epochs=epochs, class_weights=class_weights, checkpoints=checkpoint_dir,
                            name = model_name, warmup=warmup, initial_lr=initial_lr)
    else:
        try:
            model.load_state_dict(torch.load(f'{checkpoint_dir}/best_{model_name}.pt', weights_only=True))
            model.to(device)
            print(f"Loaded pre-trained model from {checkpoint_dir}/best_model.pt")
        except Exception as e:
            print(f'No model to load from {checkpoint_dir}/best_model.pt', e)
            
                
    model.eval()
    all_preds = []
    all_labels = []
    book_mndd_scores = []
    docs_prec = []
    docs_recall = []
    docs_f1 = []
    docs_sq = []  
    docs_pq = []  

    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Evaluation"):
            fusion_features_batch = batch['features']
            features_on_device = {
                bb_name: tensor.to(device) for bb_name, tensor in fusion_features_batch.items()
            }
            attention_mask = batch['attention_mask'].to(device)
            page_labels = batch['page_labels'].to(device)

            logits = model(features_on_device, attention_mask)
            batch_size, seq_length, num_classes = logits.shape
        
            predictions = logits.argmax(dim=2)
        
            for i in range(batch_size):
                mask = attention_mask[i].bool()
                book_preds = predictions[i][mask].cpu().numpy()
                book_labels = page_labels[i][mask].cpu().numpy()
            
                if len(book_preds) == 0:
                    continue
                book_score = calculate_mndd(book_preds, book_labels)
                book_mndd_scores.append(book_score)
                
                metrics = panoptic_quality_metrics(book_preds, book_labels)
                docs_prec.append(metrics["precision"])
                docs_recall.append(metrics["recall"]) 
                docs_f1.append(metrics["f1"])
                docs_sq.append(metrics["sq"])  
                docs_pq.append(metrics["pq"])  
       
            logits_flat = logits.view(-1, num_classes)
            labels_flat = page_labels.view(-1)
            predictions_flat = logits_flat.argmax(dim=1)
            
            mask = (labels_flat != -1)
            all_preds.extend(predictions_flat[mask].cpu().numpy())
            all_labels.extend(labels_flat[mask].cpu().numpy())
                
    class_names = test_dataset.get_class_names()
    
    print("\nFinal Model Performance:")
    print(f"\nAverage Book MNDD: {np.mean(book_mndd_scores):.4f}")
    print(f'\nDocument-level Metrics:')
    print(f'  Precision: {np.mean(docs_prec):.4f}')
    print(f'  Recall: {np.mean(docs_recall):.4f}')
    print(f'  F1: {np.mean(docs_f1):.4f}')
    print(f'  Segmentation Quality (SQ): {np.mean(docs_sq):.4f}')
    print(f'  Panoptic Quality (PQ): {np.mean(docs_pq):.4f}')
    
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    
    run.log({'MnDD' : np.mean(book_mndd_scores),
        'DocF1': np.mean(docs_f1),
        'SQ': np.mean(docs_sq),
        'PQ': np.mean(docs_pq)
            })
    
    
    # print("\nFinding and visualizing poorly segmented books...")
    # worst_books, _ = save_artifacts(
    #     model=model,
    #     test_dataset=test_dataset,
    #     test_loader=test_loader,
    #     device=device,
    #     top_n=10, 
    #     output_dir=f"{out_dir}/poorly_segmented"
    # )
    
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
        class_name = class_names[i]
        print(f"  Class {i} ({class_name}): {acc:.4f}")
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    image_path = f"{out_dir}/normCM_BookBERT.png"
    plt.savefig(image_path)

    wandb.log({"norm_confusion_matrix_BookBERT": wandb.Image(image_path)})

    plt.close()
        
    print("Training completed!")
    

def run_sweep():
    run = wandb.init(
        project="BookBERT",
        name="BigAugmentations",
        )
        
    try: 
        main(run,
            model_name=run.name,
            train=False, 
            lr = run.config.lr, 
            dropout_p=run.config.dropout_p, 
            epochs = run.config.epochs,
            batch_size=run.config.batch_size, 
            seed = run.config.seed,
            augmentations=run.config.augmentations,
            num_aug_copies = run.config.num_aug_copies, 
            num_synthetic_books = run.config.num_synth_books, 
            num_attention_heads = run.config.num_attention_heads,
            num_hidden_layers = run.config.num_hidden_layers,
            positional_embeddings = run.config.positional_embeddings,
            hidden_dim = run.config.hidden_dim,
            warmup=run.config.warmup,
            initial_lr=run.config.initial_lr)
        
    except Exception as e:
        print(f'Error during sweep run {run.id}: {e}')
        wandb.log({"error": str(e)})
        run.finish(exit_code=1) 
        
    else: 
        run.finish() 
    return 0

def find_best_params(out_file='./configs/best_params', project_name='my-first-sweep', sweep_id=None):
    print(f"\nFinding best parameters for sweep '{sweep_id}' in project '{project_name}'...")
    try:
        
        api = wandb.Api()
        sweep_path = f"{api.default_entity}/{project_name}/{sweep_id}"
        sweep = api.sweep(sweep_path)

        runs = [run for run in sweep.runs if run.state == "finished"]
        
        if not runs:
            print("No finished runs found in the sweep.")
            return None
        
        default_value = -float('inf') 
        sorted_runs = sorted(runs, key=lambda run: run.summary.get('val_f1', default_value), reverse=True)

        best_run = sorted_runs[0]
        best_params = best_run.config
        best_metric_value = best_run.summary.get('val_f1', 'N/A')

        print(f"Best run found: {best_run.name} (ID: {best_run.id})")
        print(f"  Best {'val_f1'}: {best_metric_value}")
        print(f"  Best parameters: {best_params}")
        
        output_dir = os.path.dirname(out_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(out_file, 'w') as f:
            yaml.dump(best_params, f, indent=4, default_flow_style=False)
        print(f"Best parameters saved to '{out_file}'")

        return best_params

    except Exception as e:
        print(f"Error finding/saving best parameters: {e}")
        return None
    
if __name__ == "__main__":
    
    # sweep_config = '/home/mserrao/PSSComics/Comics/EncoderClassifier/configs/sweep_conf2.yaml'
    
    # with open(sweep_config, 'r') as file:
    #     config = yaml.safe_load(file)
    
    # sweep_id = wandb.sweep(sweep=config, project="my-first-sweep")

    # wandb.agent(sweep_id, function=run_sweep, count=30)
    
    # out_file = f'/home/mserrao/PSSComics/Comics/EncoderClassifier/configs/best_config{sweep_config}.yaml'
    
    # best_params = find_best_params(out_file=out_file)
    
    parser = argparse.ArgumentParser(description='Train BookBERT model')
    parser.add_argument('--config', type=str, default='configs/siglip_config.yaml', 
                        required=True,
                        help='Path to the configuration file')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train the model')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for this process (0-indexed)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    run = wandb.init(
        project=config_data["project"],
        name=config_data["name"],
        config=config_data["config"]
    )
    
    main(run,
        model_name=run.name,
        gpu_id = args.gpu_id,
        train=args.train, 
        lr = float(run.config.lr), 
        dropout_p=run.config.dropout, 
        epochs = run.config.epochs,
        batch_size=run.config.batch_size,
        seed = run.config.seed,
        augmentations=run.config.augmentations,
        num_aug_copies = run.config.num_aug_copies, 
        num_synthetic_books = run.config.num_synth_books, 
        num_attention_heads = run.config.num_attention_heads,
        num_hidden_layers = run.config.num_hidden_layers,
        positional_embeddings = run.config.positional_embeddings,
        hidden_dim = run.config.hidden_dim,
        warmup=run.config.warmup,
        initial_lr=float(run.config.initial_lr),
        projection_dim=run.config.projection_dim,
        bert_input_dim=run.config.bert_input_dim
    )