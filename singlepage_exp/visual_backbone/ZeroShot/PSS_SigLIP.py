from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
from PSS_CLIP import create_comic_dataset
from datasets import load_from_disk
import torch
import numpy as np
from tqdm.auto import tqdm
import gc
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


root_dir = "/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images"
split_json = "/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data/comics_test.json"
dataset_path = '/home-local/mserrao/PSSComics/multimodal-comic-pss/CLIP&SigLIP/ZeroShot/data/test'
out_folder = '/home-local/mserrao/PSSComics/multimodal-comic-pss/CLIP&SigLIP/ZeroShot/data/out'


if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
    comic_dataset = load_from_disk(dataset_path)
else:
    comic_dataset = create_comic_dataset(root_dir, split_json, dataset_path)
    comic_dataset.save_to_disk("ComicsSiglip")
print(f"Created dataset with {len(comic_dataset)} examples")
print(f"Sample: {comic_dataset[0]}")
    
model_id = "google/siglip-so400m-patch14-384"
print(f"Loading model: {model_id}")
model = AutoModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
model.to(device)

labels = comic_dataset.info.features['label'].names
print(comic_dataset.info.features['label'])
print(f"Labels: {labels}")


clip_prompts = {
    "cover": [
        'a colorful, full-page comic book front with a prominent title, logos, and main characters. No speech bubbles.',
        'the front cover of a comic book with the main title, issue number, and dynamic artwork.',
        'a vibrant, illustrated comic book cover featuring a central character or scene, designed to attract readers.',
        'artwork for a comic book cover, including the title and publisher logos.'
    ],
    "advertisement": [
        'a full-page product advertisement from a comic book.',
        'a colorful ad page from a vintage comic book, often with coupons or mail-in offers.',
        'a page from a comic book dedicated to advertising products, with promotional text and images.',
        'an advertisement within a comic book, distinct from the story pages.'
    ],
    "story": [
        'a standard comic book page with multiple panels showing sequential action and dialogue in speech bubbles.',
        'a page from a comic book narrative, with panels, characters, and a clear story progression.',
        'a comic book story page with artwork depicting scenes, characters in action, and dialogue.',
        'a sequential art page from a comic book, telling a story through illustrations and text.'
    ],
    "textstory": [
        'a page from a comic book consisting almost entirely of dense paragraphs of text, with minimal or no illustrations.',
        'a text-only story page from a vintage comic book, formatted like a short story or letter column.',
        'a page of prose narrative within a comic book, without comic panels or speech bubbles.',
        'a dense block of text on a page from a comic book, telling a story without sequential art.'
    ],
    "First-Page": [
        'the opening page of a comic book story, featuring a large title for the story and an establishing scene in panels.',
        'the first page of a comic story, with a title card, credits, and the initial panels of the narrative.',
        'an introductory page of a comic book story, combining a story title with sequential art to begin the plot.',
        'the beginning of a comic book chapter, with a title and panels that introduce characters or setting.'
    ]
}


prompt_to_class_idx = {}
all_prompts = []

for i, class_name in enumerate(labels):
    if class_name in clip_prompts:
        for prompt in clip_prompts[class_name]:
            all_prompts.append(prompt)
            prompt_to_class_idx[len(all_prompts) - 1] = i

print(f"Created {len(all_prompts)} prompts for {len(labels)} classes")

print("Processing text labels...")
with torch.no_grad():
    label_tokens = processor(
        text=all_prompts,
        padding="max_length", 
        return_tensors='pt'
    )
    label_tokens = {key: val.to(device) for key, val in label_tokens.items()}
    
    label_emb = model.get_text_features(**label_tokens)
    label_emb = label_emb.detach().cpu().numpy()
    label_emb = label_emb / np.linalg.norm(label_emb, axis=1, keepdims=True)
    
    torch.cuda.empty_cache()
    del label_tokens

print("Processing images...")
preds = []
batch_size = 16 

for i in tqdm(range(0, len(comic_dataset), batch_size)):
    try:
        i_end = min(i + batch_size, len(comic_dataset))
        
        with torch.no_grad(): 
            images = processor(
                text=None,
                images=comic_dataset[i:i_end]['image'],
                return_tensors='pt'
            )['pixel_values'].to(device)

            img_emb = model.get_image_features(images)
            img_emb = img_emb.detach().cpu().numpy()
            img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
            
            del images
            torch.cuda.empty_cache()
        
        scores = np.dot(img_emb, label_emb.T)
        
        class_scores = np.zeros((scores.shape[0], len(labels)))
        for prompt_idx, class_idx in prompt_to_class_idx.items():
            class_scores[:, class_idx] += scores[:, prompt_idx]
            
        for class_idx in range(len(labels)):
            prompt_count = sum(1 for idx in prompt_to_class_idx.values() if idx == class_idx)
            if prompt_count > 0:  
                class_scores[:, class_idx] /= prompt_count
        
        batch_preds = np.argmax(class_scores, axis=1)
        preds.extend(batch_preds)
        
        del img_emb, scores, class_scores, batch_preds
        gc.collect()
        
    except Exception as e:
        print(f"Error processing batch {i}-{i_end}: {e}")
        if batch_size > 1:
            print("Reducing batch size and retrying...")
            batch_size = max(1, batch_size // 2)
            i = max(0, i - batch_size)
            
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

try:
    np.save('siglip_predictions.npy', np.array(preds))
    print("Predictions saved to siglip_predictions.npy")
except Exception as e:
    print(f"Failed to save predictions: {e}")


