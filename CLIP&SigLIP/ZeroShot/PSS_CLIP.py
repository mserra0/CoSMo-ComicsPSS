from datasets import Dataset, Image as HFImage, ClassLabel, load_from_disk
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from tqdm.auto import tqdm
import gc
import os
import json
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_comic_dataset(root_dir, json_annot, dataset_path):
    image_paths = []
    labels = []
    
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        comic_dataset = load_from_disk(dataset_path)
        print(f"Loaded dataset with {len(comic_dataset)} examples.")
        return comic_dataset
    
    with open(json_annot, 'r') as f:
        ground_truth = json.load(f)
        
    category_mapping = {
        "stories": "story",
        "textstories": "textstory",
        "advertisements": "advertisement",
        "covers": "cover"
    }
    
    for book in ground_truth:
        book_hash = book["hash_code"]
        for category_key, label in category_mapping.items():
            for item in book.get(category_key, []):
                for page in range(item["page_start"], item["page_end"] + 1):
                    image_path = os.path.join(root_dir, book_hash, f"{page:03d}.jpg")
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        labels.append(label)
                    else:
                        print(f"Warning: Image not found - {image_path}")
    
    dataset_dict = {
        'image': image_paths,
        'label': labels
    }
    
    comic_dataset = Dataset.from_dict(dataset_dict)
    
    def load_image(examples, indices):
        images = []
        for i, path in enumerate(examples["image"]):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                images.append(Image.new('RGB', (224, 224), color='black'))
        examples["image"] = images
        return examples
    
    comic_dataset = comic_dataset.map(
        load_image,
        batched=True,
        batch_size=64, 
        with_indices=True,
        desc="Loading images",
        num_proc=4  
    )
    
    print('Mapping Done!')
    label_names = list(set(category_mapping.values()))
    comic_dataset = comic_dataset.cast_column('image', HFImage())
    comic_dataset = comic_dataset.cast_column('label', ClassLabel(names=label_names))
            
    comic_dataset.save_to_disk(dataset_path)
    print(f"Created dataset with {len(comic_dataset)} examples")
    
    return comic_dataset

if __name__ == '__main__':

    root_dir = "/home/mserrao/PSSComics/Comics/datasets.unify/DCM/images"
    split_json = "/home/mserrao/PSSComics/Comics/comics100_all/comics_test_books.json"
    dataset_path = '/home/mserrao/PSSComics/ComicsCLIP/test'
    out_folder = 'Comics/out'

    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        comic_dataset = load_from_disk(dataset_path)
    else:
        comic_dataset = create_comic_dataset(root_dir, split_json)
        comic_dataset.save_to_disk(dataset_path)
    print(f"Created dataset with {len(comic_dataset)} examples")
    print(f"Sample: {comic_dataset[0]}")
        
    model_id = 'openai/clip-vit-large-patch14'
    print(f"Loading model: {model_id}")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)

    labels = comic_dataset.info.features['label'].names
    print(comic_dataset.info.features['label'])
    print(f"Labels: {labels}")


    clip_prompts = {
        "cover": [
            'a colorful, full-page comic book front with title, logo, and main characters',
            'a comic book cover page with title and artwork',
            'the front cover of a comic book showing the main character',
            'a vibrant illustrated comic book cover with superhero characters'
        ],
        "advertisement": [
            'a photo of a comic book advertisement',
            'a colorful ad page from a comic book',
            'a vintage comic book page selling products to readers',
            'an advertisement with products and promotional text in a comic book'
        ],
        "story": [
            'a comic page with panels, characters, speech bubbles, and action scenes',
            'a page from a comic book with sequential art panels',
            'a comic book page showing a visual narrative with characters',
            'a comic book story page with characters in action and dialog bubbles'
        ],
        "textstory": [
            'a picture of a vintage comic book dense storytelling page full of text',
            'a vintage letter',
            'a page with dense paragraphs of text from a comic book',
            'a text-heavy narrative page from a vintage comic book',
        ]
    }

    model_id = 'openai/clip-vit-large-patch14'
    print(f"Loading model: {model_id}")

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)


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
            padding=True, 
            images=None, 
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
        np.save('clip_predictions.npy', np.array(preds))
        print("Predictions saved to clip_predictions.npy")
    except Exception as e:
        print(f"Failed to save predictions: {e}")