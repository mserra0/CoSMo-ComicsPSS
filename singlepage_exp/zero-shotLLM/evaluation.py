import json
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from pss_datasets.pss_dataset import PSSDataset
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

root_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images'
annotations_dir = '/home/mserrao/PSSComics/Comics/DatasetDCM/comics_all_430.json'
precompute_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
checkpoint_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/checkpoints'
data_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
out_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/out'

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON {filepath}: {e}")
        return None
    
def extract_predictions(dataset, root_dir):
    label_map = {
        "Cover" : 'cover', 
        "Story-Start" : 'first-page', 
        "Story-Page" : 'story', 
        "Advertisement" : 'advertisement', 
        "Text-Story" : 'textstory'
    }
    predictions = {}
    for book in dataset.books:
        book_id = book['book_id']
        book_img_pths = book['image_paths']
        book_path = f'{root_dir}/{book_id}'
        book_preds = {}
        for path in book_img_pths:
            base = os.path.basename(path)
            img_num = os.path.splitext(base)[0]
            pred_file = load_json(f'{book_path}/{img_num}.json')
            book_preds[img_num] = {'chosen_label' : label_map[pred_file["PageType"]] }
            
        predictions[book_id] = book_preds
    return predictions
    
def extract_book_data(dataset, predictions, verbose=True):
    class_to_idx = dataset.class_to_idx
    books_data = {}
    for book in dataset.books:
        book_id = book['book_id']
        book_img_pths = book['image_paths']
        book_labels = book['page_labels']
        book_preds =predictions[book_id]
        y_true, y_pred, img_paths = [], [], []
        for idx, path in enumerate(book_img_pths):
            base = os.path.basename(path)
            img_num = os.path.splitext(base)[0]
            
            if img_num in book_preds.keys():
                label = book_labels[idx]
                chosen_label = book_preds[img_num]['chosen_label']
                if chosen_label == 'none': chosen_label = 'story'
                pred = class_to_idx['textstory' if chosen_label == 'text-story' else chosen_label]
                y_true.append(label)
                y_pred.append(pred)
                img_paths.append(path)
            else:
                if verbose:
                    print(f'No data added for image {path}')
                    
        books_data[book_id] = {"y_true" : np.array(y_true), "y_pred" : np.array(y_pred), "img_paths" : img_paths}
        
    return books_data

def evaluate(preds_json):
    model_id = "openai/clip-vit-large-patch14-336"

    parts = model_id.split('/')[1].split('-')
    backbone_name = f'{parts[0]}_{parts[-1]}'

    backbone = AutoModel.from_pretrained(model_id).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    feature_dim = backbone.config.vision_config.projection_dim

    test_dataset = PSSDataset(root_dir=root_dir, 
                            model_id = model_id,
                            backbone=backbone, 
                            backbone_name = backbone_name,
                            feature_dim = feature_dim,
                            processor=processor, 
                            batch_size=8,
                            device='cuda', 
                            annotations_path=f'{data_dir}/comics_test.json', 
                            precompute_features=False,
                            precompute_dir=f'{precompute_dir}/features_test.pt', 
                            augment_data=False)

    test_preds = load_json(preds_json)
    # test_preds = extract_predictions(test_dataset, root_dir)
    test_data = extract_book_data(test_dataset, test_preds)

    class_names = test_dataset.class_names
    all_gt, all_pred = [], []
    errors = {}
    for book_id, data in test_data.items():
        y_true = data['y_true']
        y_pred = data['y_pred']
        image_paths = data['img_paths']
        
        # error_mask = (y_true != y_pred)
        # error_positions = np.where(error_mask)[0]
        # errors[book_id] = image_paths[error_positions].tolist()
        
        all_gt.extend(y_true)
        all_pred.extend(y_pred)
        
    f1_macro = f1_score(all_gt, all_pred, average='macro')
    
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    
    print("Classification Report:")
    print(classification_report(all_gt, all_pred, target_names=class_names, digits=4))
    
    cm_name = 'norm_CM_zero-shot(CLS_prompt).png'
    # cm_name = 'norm_CM_zero-shot(OCR_prompt).png'
    
    cm_normalized = confusion_matrix(all_gt, all_pred, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(f'{out_dir}/{cm_name}', dpi=300, bbox_inches='tight')
    plt.close()  
    print(f"Normalized confusion matrix saved to {out_dir}/{cm_name}")
    
if __name__ == '__main__':
    pred_json = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data/zero-shot_qwen.json'
    evaluate(pred_json)