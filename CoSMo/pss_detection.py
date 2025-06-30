import torch
from transformers import SiglipImageProcessor, AutoProcessor, AutoModel
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader
import os
import numpy as np
import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data import combine_json_files, split_data, ComicTransform
from utils.visualitzation import visualize_book, save_artifacts
from utils.training import compute_class_weights, train_model_detection
from utils.metrics import calculate_mndd, panoptic_quality_metrics
from pss_datasets.pss_dataset_with_stats import PSSDatasetWithStats
from models.book_bert import BookBERT2
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

def main(run, gpu_id = 0,train=True, precompute = False, lr = 1e-4, dropout_p=0.4, epochs = 10, batch_size=32, model_id='openai/clip-vit-large-patch14', 
         seed=10, augmentations = False, num_aug_copies = 5, num_synthetic_books=200, 
         num_attention_heads = 4, num_hidden_layers = 4, positional_embeddings = 'absolute', hidden_dim = 256,
         model_name = 'BookBERT', warmup = 44, initial_lr = 1e-6 , precompute_detection_features = False):
    
    root_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images'
    annotations_dir = '/home/mserrao/PSSComics/Comics/DatasetDCM/comics_all_430.json'
    precompute_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
    checkpoint_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/checkpoints'
    data_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
    out_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/out'
    detections_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/data/DCM/magi'
    set_all_seeds(seed)
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Process using device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # split_data(annotations_dir, data_dir, train=0.7, val=0.1, test=0.2, seed=seed)
    
    print(f"Loading model: {model_id}")
    
    parts = model_id.split('/')[1].split('-')
    backbone_name = f'{parts[0]}_{parts[-1]}'
    
    backbone = AutoModel.from_pretrained(model_id).eval()
    
    if 'siglip2' in backbone_name:
        processor = SiglipImageProcessor.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        
    backbone.to(device)
    
    if 'dinov2' in backbone_name:
        feature_dim = backbone.config.hidden_size
    elif 'clip' in backbone_name:
        feature_dim = backbone.config.vision_config.projection_dim
    elif 'siglip' in backbone_name:
        feature_dim = backbone.config.vision_config.hidden_size
    else:
        raise ValueError(f"Warning: Unknown backbone '{backbone_name}'")
    
    print(f'Loaded {backbone_name} with feature dim {feature_dim}')
    
    transformations = ComicTransform()
    
    train_dataset = PSSDatasetWithStats(root_dir=root_dir, 
                               model_id = model_id,
                               backbone=backbone, 
                               backbone_name = backbone_name,
                               feature_dim = feature_dim,
                               processor=processor, 
                               device=device, 
                               annotations_path=f'{data_dir}/comics_train.json', 
                               precompute_features=precompute,
                               precompute_dir=f'{precompute_dir}/features_train.pt', 
                               augment_data=augmentations,
                               num_augmented_copies = num_aug_copies,
                               transform=transformations, 
                               removal_p=0.05,
                               num_synthetic_books=num_synthetic_books,
                               min_stories=1,
                               max_stories=3,
                               synthetic_remove_p=0.15,
                               detections_json_path=f'{detections_dir}/train.json',
                               stats_cache_path=f'{precompute_dir}/detection_features_train.pt',
                               precompute_detection_features=precompute_detection_features)
    
    val_dataset = PSSDatasetWithStats(root_dir=root_dir, 
                            model_id = model_id,
                            backbone=backbone,
                            backbone_name = backbone_name, 
                            feature_dim = feature_dim,
                            processor=processor, 
                            device=device, 
                            annotations_path=f'{data_dir}/comics_val.json', 
                            precompute_features=precompute,
                            precompute_dir=f'{precompute_dir}/features_val.pt', 
                            augment_data=False,
                            detections_json_path=f'{detections_dir}/val.json',
                            stats_cache_path=f'{precompute_dir}/detection_features_val.pt',
                            precompute_detection_features=precompute_detection_features)
    
    test_dataset = PSSDatasetWithStats(root_dir=root_dir, 
                            model_id = model_id,
                            backbone=backbone, 
                            backbone_name = backbone_name,
                            feature_dim = feature_dim,
                            processor=processor, 
                            device=device, 
                            annotations_path=f'{data_dir}/comics_test.json', 
                            precompute_features=precompute,
                            precompute_dir=f'{precompute_dir}/features_test.pt', 
                            augment_data=False,
                            detections_json_path=f'{detections_dir}/test.json',
                            stats_cache_path=f'{precompute_dir}/detection_features_test.pt',
                            precompute_detection_features=True)

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
    detection_feature_dim = train_dataset.detection_feature_dim
    
    model = BookBERT2(feature_dim = feature_dim, num_classes=num_classes, hidden_dim=hidden_dim, num_attention_heads=num_attention_heads,
                     num_hidden_layers=num_hidden_layers, dropout_p=dropout_p, positional_embeddings=positional_embeddings,
                     detections_feature_dim=detection_feature_dim)
    
    if train:
        model = train_model_detection(run, model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, lr=lr, 
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
            features = batch['features'].to(device)
            detection_features = batch['detection_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            page_labels = batch['page_labels'].to(device)

            logits = model(features, attention_mask, detection_features)
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
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    print(f'F1 Macro Score: {f1_macro}')
    acc = accuracy_score(all_labels, all_preds)
    print(f'Accuracy Score: {acc}')
    
    run.log({'MnDD' : np.mean(book_mndd_scores),
            'DocF1': np.mean(docs_f1),
            'SQ': np.mean(docs_sq),
            'PQ': np.mean(docs_pq),
            'F1macro': f1_macro,
            'Acc' : acc
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
    parser.add_argument('--precompute_backbone', action='store_true', default=False,
                    help='Whether to precompute the backbone features')
    parser.add_argument('--precompute_detection_features', action='store_true', default=False,
                        help='Recmpute the handcrafted features')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for this process (0-indexed)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        
    if config_data['config']['augmentations']:
        run_name = config_data['name']+'_aug'
    else:
        run_name = config_data['name']
    
    run = wandb.init(
        project=config_data["project"],
        name=run_name,
        config=config_data["config"]
    )
    
    main(run,
        model_name=run.name,
        gpu_id = args.gpu_id,
        train=args.train, 
        precompute=args.precompute_backbone,
        lr = float(run.config.lr), 
        dropout_p=run.config.dropout, 
        epochs = run.config.epochs,
        batch_size=run.config.batch_size,
        model_id = run.config.model_id, 
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
        precompute_detection_features=args.precompute_detection_features
    )