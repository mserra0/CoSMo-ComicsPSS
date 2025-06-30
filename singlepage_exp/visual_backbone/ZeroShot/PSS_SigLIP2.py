import torch
import os
import numpy as np
from tqdm.auto import tqdm
from transformers import pipeline
from datasets import load_from_disk
from PIL import Image
from PSS_CLIP import create_comic_dataset
import gc

dataset_path = '/home/mserrao/PSSComics/ComicsCLIP'
out_folder = 'Comics/out'
os.makedirs(out_folder, exist_ok=True)

if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
    comic_dataset = load_from_disk(dataset_path)
else:
    root_dir = "/home/mserrao/PSSComics/Comics/datasets.unify/DCM/images"
    split_json = "/home/mserrao/PSSComics/Comics/comics100_all/comics_val_books.json"
    comic_dataset = create_comic_dataset(root_dir, split_json)
    comic_dataset.save_to_disk("ComicsCLIP")
print(f"Created dataset with {len(comic_dataset)} examples")

labels = comic_dataset.info.features['label'].names
print(f"Labels: {labels}")

candidate_labels = [
    "a comic book cover page",
    "a comic book advertisement page",
    "a comic book story page with panels",
    "a text-heavy comic book page"
]

print("Loading zero-shot classification pipeline...")
image_classifier = pipeline(
    task="zero-shot-image-classification",
    model="google/siglip2-so400m-patch14-384", 
    device="cuda"
)

print("Processing images...")
preds = []
scores = []

for i in tqdm(range(len(comic_dataset))):
    try:
    
        img = comic_dataset[i]['image']
        
        result = image_classifier(
            images=img,  
            candidate_labels=candidate_labels  
        )
 
        if result[0]['label'] == "a comic book cover page":
            pred_idx = labels.index("cover")
        elif result[0]['label'] == "a comic book advertisement page":
            pred_idx = labels.index("advertisement")
        elif result[0]['label'] == "a comic book story page with panels":
            pred_idx = labels.index("story")
        elif result[0]['label'] == "a text-heavy comic book page":
            pred_idx = labels.index("textstory")
        else:
            pred_idx = labels.index("story")
            
        preds.append(pred_idx)
        scores.append(result[0]['score'])

        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"Error processing image {i}: {e}")
        continue

if len(preds) == 0:
    print("No predictions were made. Please check the pipeline.")
    exit(1)

print(f"Finished processing {len(preds)} images")

correct = sum(preds[i] == comic_dataset[i]['label'] for i in range(len(preds)))
accuracy = correct / len(preds)
print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{len(preds)})")

class_correct = {}
class_total = {}

for i in range(len(preds)):
    true_label = comic_dataset[i]['label']
    pred_label = preds[i]
    
    true_label_name = labels[true_label]
    
    if true_label_name not in class_total:
        class_total[true_label_name] = 0
        class_correct[true_label_name] = 0

    class_total[true_label_name] += 1
    if pred_label == true_label:
        class_correct[true_label_name] += 1
        
print("\nPer-Class Accuracy:")
print("-" * 40)
print(f"{'Class':<15} {'Accuracy':<10} {'Correct/Total':<15}")
print("-" * 40)

for label_name in sorted(class_total.keys()):
    class_acc = class_correct[label_name] / class_total[label_name]
    print(f"{label_name:<15} {class_acc:.4f}     {class_correct[label_name]}/{class_total[label_name]}")

print("\nConfusion Matrix:")
confusion = np.zeros((len(labels), len(labels)), dtype=int)
for i in range(len(preds)):
    true_label = comic_dataset[i]['label']
    pred_label = preds[i]
    confusion[true_label][pred_label] += 1

print("True \\ Pred", end="\t")
for label_name in labels:
    print(f"{label_name[:5]}", end="\t")
print()

for i, true_label in enumerate(labels):
    print(f"{true_label[:10]}", end="\t")
    for j in range(len(labels)):
        print(f"{confusion[i][j]}", end="\t")
    print()