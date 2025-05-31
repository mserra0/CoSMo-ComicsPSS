import torch
from .metrics import calculate_mndd, get_breaking_points
import os
import math
import numpy as np
import wandb
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from collections import Counter

def visualize_book(dataset, book_id=None, book_idx=None, output_path=None, max_cols=5, figsize=(15, 20), dpi=300, transforms=None):
    print('Creating visualization...')
    
    if book_id is None and book_idx is None:
        raise ValueError("Either book_id or book_idx must be provided")
    
    if book_id is not None:
        book = None
        for b in dataset.books:
            if b['book_id'] == book_id:
                book = b
                break
        if book is None:
            raise ValueError(f"Book with ID '{book_id}' not found in the dataset")
    else:
        book = dataset.books[book_idx]
        book_id = book['book_id']  
    
    image_paths = book['image_paths']
    page_labels = book['page_labels']
    class_names = dataset.get_class_names()
    
    num_pages = len(image_paths)
    num_cols = min(max_cols, num_pages)
    num_rows = math.ceil(num_pages / num_cols)
    
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FF99FF'] 
    class_colors = {i: colors[i] for i in range(len(class_names))}
    class_colors[-1] = '#CCCCCC'  

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.suptitle(f"Book: {book_id} ({num_pages} pages)", fontsize=18)
    
    for i, (img_path, label) in enumerate(zip(image_paths, page_labels)):
        img = Image.open(img_path).convert('RGB')
        if transforms:
            img = transforms(img)
        
        ax = plt.subplot(num_rows, num_cols, i + 1)
        
        if label == -1 or label is None:
            label_name = "unknown"
        else:
            label_name = class_names[label]
        
        ax.imshow(img)
        ax.set_title(f"Page {i+1}: {label_name}", fontsize=10)
        
        border_color = class_colors.get(label, '#CCCCCC')
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3)
            
        ax.set_xticks([])
        ax.set_yticks([])
    
    legend_elements = []

    for i, class_name in enumerate(class_names):
        legend_elements.append(Patch(facecolor=class_colors[i], label=class_name))
    if -1 in class_colors:
        legend_elements.append(Patch(facecolor=class_colors[-1], label="unknown"))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), 
               bbox_to_anchor=(0.5, 0.02), frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(f'{output_path}/{book_id}')
        print(f"Saved visualization to {output_path}/{book_id}")
    
    return fig

def visualize_book_predictions(model, dataset, book_id=None, book_idx=None, output_path=None, 
                               device='cuda', max_cols=5, figsize=(15, 25), dpi=200):
    """
    Creates a visualization comparing ground truth and model predictions for a book.
    """
    print(f'Creating prediction visualization for book {book_id or book_idx}...')
    
    if book_id is None and book_idx is None:
        raise ValueError("Either book_id or book_idx must be provided")
    
    if book_id is not None:
        book = None
        for i, b in enumerate(dataset.books):
            if b['book_id'] == book_id:
                book = b
                book_idx = i
                break
        if book is None:
            raise ValueError(f"Book with ID '{book_id}' not found in the dataset")
    else:
        book = dataset.books[book_idx]
        book_id = book['book_id']
    
    image_paths = book['image_paths']
    true_labels = book['page_labels']
    class_names = dataset.get_class_names()
  
    model.eval()
    with torch.no_grad():
        features = []
    
        for img_path in image_paths:
            if dataset.precompute_features:
                feature = dataset.get_precomputed_features(book_idx)[image_paths.index(img_path)]
            else:
                feature = dataset._extract_clip_features(img_path)
            features.append(feature)
        
        if not features:
            return None
            
        features_tensor = torch.stack(features).unsqueeze(0).to(device)  
        attention_mask = torch.ones(1, len(features), dtype=torch.long).to(device)
        
        logits = model(features_tensor, attention_mask)
        predictions = logits.squeeze(0).argmax(dim=1).cpu().numpy()
    
    num_pages = len(image_paths)
    num_cols = min(max_cols, num_pages)
    num_rows = 2 * math.ceil(num_pages / num_cols)  
    
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FF99FF'] 
    class_colors = {i: colors[i] for i in range(len(class_names))}
    class_colors[-1] = '#CCCCCC'
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.suptitle(f"Book: {book_id} - Ground Truth vs Predictions", fontsize=18)
    
    true_bp = get_breaking_points(true_labels)
    pred_bp = get_breaking_points(predictions)
    
    mndd_score = calculate_mndd(predictions, true_labels)
 
    
    for i, (img_path, label, is_bp) in enumerate(zip(image_paths, true_labels, true_bp)):
        img = Image.open(img_path).convert('RGB')
        
        ax = plt.subplot(num_rows, num_cols, i + 1)
        
        if label == -1 or label is None:
            label_name = "unknown"
        else:
            label_name = class_names[label]
        
        ax.imshow(img)
        title = f"Page {i+1}: {label_name}"
        if is_bp:
            title = f"↓ {title} ↓"  
        ax.set_title(title, fontsize=10)
        
        border_color = class_colors.get(label, '#CCCCCC')
        border_width = 5 if is_bp else 3
        
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(border_width)
            
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.figtext(0.5, 0.55, "GROUND TRUTH", fontsize=14, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    for i, (img_path, pred, is_bp) in enumerate(zip(image_paths, predictions, pred_bp)):
        img = Image.open(img_path).convert('RGB')
        
        ax = plt.subplot(num_rows, num_cols, i + 1 + num_cols * math.ceil(num_pages / num_cols))
        
        pred_name = class_names[pred]
        
        ax.imshow(img)
        title = f"Page {i+1}: {pred_name}"
        if is_bp:
            title = f"↑ {title} ↑"  
        ax.set_title(title, fontsize=10)
        
        border_color = class_colors.get(pred, '#CCCCCC')
        border_width = 5 if is_bp else 3
        
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(border_width)
            spine.set_linestyle('-')
            
        ax.set_xticks([])
        ax.set_yticks([])

    plt.figtext(0.5, 0.45, "PREDICTIONS", fontsize=14, ha='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    plt.figtext(0.5, 0.04, f"MNDD Score: {mndd_score:.2f}", fontsize=14, ha='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    legend_elements = []
    for i, class_name in enumerate(class_names):
        legend_elements.append(Patch(facecolor=class_colors[i], label=class_name))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), 
               bbox_to_anchor=(0.5, 0.01), frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.4)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    
    return fig, mndd_score

def save_artifacts(model, test_dataset, test_loader, device, top_n=5, output_dir=None):
    """
    Saves separate visualizations of ground truth and predictions for poorly segmented books.
    Creates a folder structure: artifacts/[book_id]/[book_id]_pred.png and [book_id]_gt.png
    """
    print(f"Finding {top_n} books with worst segmentation...")
  
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
 
    book_scores = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(test_loader, desc="Evaluating segmentation")):
            fusion_features_batch = batch['features']
            features_on_device = {
                bb_name: tensor.to(device) for bb_name, tensor in fusion_features_batch.items()
            }
            attention_mask = batch['attention_mask'].to(device)
            page_labels = batch['page_labels'].to(device)
        
            book_ids = batch['book_id'] if 'book_id' in batch else None
            
            logits = model(features_on_device, attention_mask)
            batch_size = page_labels.shape[0]
            
            predictions = logits.argmax(dim=2)
            
            for i in range(batch_size):
                mask = attention_mask[i].bool()
                book_preds = predictions[i][mask].cpu().numpy()
                book_labels = page_labels[i][mask].cpu().numpy()
                
                if len(book_preds) == 0:
                    continue
                
                book_score = calculate_mndd(book_preds, book_labels)
                
                if book_ids is not None:
                    if isinstance(book_ids, (list, tuple)) or (hasattr(book_ids, 'ndim') and book_ids.ndim > 0):
                        book_id = book_ids[i] if i < len(book_ids) else f"unknown_{batch_idx}_{i}"
                    else:
                        book_id = book_ids
                else:
                    dataset_idx = batch_idx * test_loader.batch_size + i
                    if dataset_idx < len(test_dataset):
                        book_id = test_dataset.books[dataset_idx]['book_id']
                    else:
                        book_id = f"unknown_{batch_idx}_{i}"
                
                book_scores.append((book_id, book_score, book_preds))
    
    book_scores.sort(key=lambda x: x[1], reverse=True)
    
    worst_books = book_scores[:top_n]
    
    print(f"Top {len(worst_books)} books with worst segmentation:")
    for book_id, score, _ in worst_books:
        print(f"Book {book_id}: MNDD = {score:.4f}")
    
    all_figs = []
    
    for book_id, score, _ in worst_books:
        book_dir = os.path.join(output_dir, book_id)
        os.makedirs(book_dir, exist_ok=True)

        book_found = False
        for i, book in enumerate(test_dataset.books):
            if book['book_id'] == book_id:
                book_idx = i
                book_found = True
                break
        
        if not book_found:
            print(f"Warning: Couldn't find book {book_id} in dataset")
            continue
  
        gt_path = os.path.join(book_dir, f"{book_id}_gt.png")
        try:
            gt_fig = visualize_book(
                test_dataset,
                book_id=book_id,
                output_path=gt_path,
                dpi=200
            )
            
            wandb.log({
                f"poorly_segmented/{book_id}/ground_truth": wandb.Image(gt_path)
            })

            plt.close(gt_fig)
            
        except Exception as e:
            print(f"Error visualizing ground truth for book {book_id}: {e}")

        pred_path = os.path.join(book_dir, f"{book_id}_pred.png")
        try:
            book = test_dataset.books[book_idx]
            image_paths = book['image_paths']

            with torch.no_grad():
                features = []
                
                for img_path in image_paths:
                    if test_dataset.precompute_features:
                        feature = test_dataset.get_precomputed_features(book_idx)[image_paths.index(img_path)]
                    else:
                        feature = test_dataset._extract_clip_features(img_path)
                    features.append(feature)
                
                if not features:
                    continue
       
                features_tensor = torch.stack(features).unsqueeze(0).to(device)
                attention_mask = torch.ones(1, len(features), dtype=torch.long).to(device)
                logits = model(features_tensor, attention_mask)
                predictions = logits.squeeze(0).argmax(dim=1).cpu().numpy()
                
            pred_book = book.copy()
            pred_book['page_labels'] = predictions
            
            # Temporarily swap the book in dataset for visualization
            original_book = test_dataset.books[book_idx]
            test_dataset.books[book_idx] = pred_book
            
            # Visualize predictions
            pred_fig = visualize_book(
                test_dataset,
                book_id=book_id,
                output_path=pred_path,
                dpi=200
            )
            
            # Restore original book
            test_dataset.books[book_idx] = original_book
            
            # Log to wandb
            wandb.log({
                f"poorly_segmented/{book_id}/predictions": wandb.Image(pred_path)
            })
            
            # Save MNDD score to a text file
            with open(os.path.join(book_dir, f"{book_id}_mndd_score.txt"), 'w') as f:
                f.write(f"MNDD Score: {score:.4f}\n")
            
            # Close figure to free memory
            plt.close(pred_fig)
            
        except Exception as e:
            print(f"Error visualizing predictions for book {book_id}: {e}")
    
    return worst_books, all_figs

def analyze_book_types(dataset):
    """
    Analyzes and visualizes distributions and statistics for original, augmented, and synthetic books in a PSSDataset.
    
    Args:
        dataset (PSSDataset): Your PSSDataset object.
    """
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract data for analysis
    data = []
    class_names = dataset.get_class_names()
    
    for book in dataset.books:
        # Determine book type
        source = book['metadata']['source']
        if source == 'original':
            book_type = 'original'
        elif '_aug_' in book['book_id']:
            book_type = 'augmented'
        else:
            book_type = 'synthetic'
        
        # Calculate metrics
        page_labels = book['page_labels']
        total_pages = len(page_labels)
        
        # Count stories (including first-page)
        story_pages = sum(1 for label in page_labels 
                         if label in [dataset.class_to_idx['story'], dataset.class_to_idx['first-page']])
        
        # Count each page type
        label_counts = Counter(page_labels)
        
        # Count number of story sequences
        num_stories = 0
        in_story = False
        for label in page_labels:
            if label == dataset.class_to_idx['first-page']:
                num_stories += 1
                in_story = True
            elif label == dataset.class_to_idx['story'] and not in_story:
                num_stories += 1
                in_story = True
            elif label != dataset.class_to_idx['story'] and label != dataset.class_to_idx['first-page']:
                in_story = False
        
        data.append({
            'book_id': book['book_id'],
            'type': book_type,
            'total_pages': total_pages,
            'story_pages': story_pages,
            'num_stories': num_stories,
            'avg_story_length': story_pages / max(num_stories, 1),
            'cover_pages': label_counts.get(dataset.class_to_idx['cover'], 0),
            'ad_pages': label_counts.get(dataset.class_to_idx['advertisement'], 0),
            'textstory_pages': label_counts.get(dataset.class_to_idx['textstory'], 0),
            'first_pages': label_counts.get(dataset.class_to_idx['first-page'], 0)
        })
    
    df = pd.DataFrame(data)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Book length distribution
    plt.subplot(3, 3, 1)
    sns.histplot(data=df, x='total_pages', hue='type', kde=True, alpha=0.7)
    plt.title('Book Length Distribution (Total Pages)')
    plt.xlabel('Number of Pages')
    plt.ylabel('Count')
    
    # 2. Story pages distribution
    plt.subplot(3, 3, 2)
    sns.histplot(data=df, x='story_pages', hue='type', kde=True, alpha=0.7)
    plt.title('Story Content Distribution (Story Pages)')
    plt.xlabel('Number of Story Pages')
    plt.ylabel('Count')
    
    # 3. Number of stories per book
    plt.subplot(3, 3, 3)
    sns.histplot(data=df, x='num_stories', hue='type', kde=True, alpha=0.7)
    plt.title('Number of Stories per Book')
    plt.xlabel('Number of Stories')
    plt.ylabel('Count')
    
    # 4. Average story length
    plt.subplot(3, 3, 4)
    sns.histplot(data=df, x='avg_story_length', hue='type', kde=True, alpha=0.7)
    plt.title('Average Story Length')
    plt.xlabel('Average Pages per Story')
    plt.ylabel('Count')
    
    # 5. Box plots for comparison
    plt.subplot(3, 3, 5)
    sns.boxplot(data=df, x='type', y='total_pages')
    plt.title('Book Length by Type (Box Plot)')
    plt.ylabel('Total Pages')
    
    # 6. Advertisement pages distribution
    plt.subplot(3, 3, 6)
    sns.histplot(data=df, x='ad_pages', hue='type', kde=True, alpha=0.7)
    plt.title('Advertisement Pages Distribution')
    plt.xlabel('Number of Ad Pages')
    plt.ylabel('Count')
    
    # 7. Text story pages distribution
    plt.subplot(3, 3, 7)
    sns.histplot(data=df, x='textstory_pages', hue='type', kde=True, alpha=0.7)
    plt.title('Text Story Pages Distribution')
    plt.xlabel('Number of Text Story Pages')
    plt.ylabel('Count')
    
    # 8. Cover pages distribution
    plt.subplot(3, 3, 8)
    sns.histplot(data=df, x='cover_pages', hue='type', kde=True, alpha=0.7)
    plt.title('Cover Pages Distribution')
    plt.xlabel('Number of Cover Pages')
    plt.ylabel('Count')
    
    # 9. Story vs Non-story ratio
    plt.subplot(3, 3, 9)
    df['story_ratio'] = df['story_pages'] / df['total_pages']
    sns.histplot(data=df, x='story_ratio', hue='type', kde=True, alpha=0.7)
    plt.title('Story Content Ratio')
    plt.xlabel('Story Pages / Total Pages')
    plt.ylabel('Count')
    
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS BY BOOK TYPE")
    print("="*80)
    
    for book_type in ['original', 'augmented', 'synthetic']:
        if book_type in df['type'].values:
            type_data = df[df['type'] == book_type]
            print(f"\n{book_type.upper()} BOOKS ({len(type_data)} books):")
            print("-" * 50)
            
            metrics = ['total_pages', 'story_pages', 'num_stories', 'avg_story_length', 
                      'ad_pages', 'textstory_pages', 'cover_pages']
            
            for metric in metrics:
                values = type_data[metric]
                print(f"{metric:20s}: mean={values.mean():.2f}, std={values.std():.2f}, "
                      f"min={values.min():.2f}, max={values.max():.2f}")
    
    # Statistical comparison
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS")
    print("="*80)
    
    from scipy import stats
    
    # Compare means between book types
    metrics_to_compare = ['total_pages', 'story_pages', 'num_stories', 'avg_story_length']
    
    book_types = df['type'].unique()
    for metric in metrics_to_compare:
        print(f"\n{metric.upper()}:")
        for i, type1 in enumerate(book_types):
            for type2 in book_types[i+1:]:
                data1 = df[df['type'] == type1][metric]
                data2 = df[df['type'] == type2][metric]
                
                if len(data1) > 0 and len(data2) > 0:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    print(f"  {type1} vs {type2}: t={t_stat:.3f}, p={p_value:.4f}")
    
    # Page type distribution comparison
    print("\n" + "="*80)
    print("PAGE TYPE DISTRIBUTION BY BOOK TYPE")
    print("="*80)
    
    page_type_cols = ['cover_pages', 'story_pages', 'ad_pages', 'textstory_pages']
    page_dist = df.groupby('type')[page_type_cols].mean()
    print(page_dist.round(2))
    
    return df

# Usage example:
# dataset = PSSDataset(...)  # your dataset
# analysis_df = analyze_book_types(dataset)