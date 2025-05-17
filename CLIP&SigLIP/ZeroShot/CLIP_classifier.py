from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from tqdm.auto import tqdm
import gc
from CLIP_dataset import PageStreamDataset

root_dir = "/home/mserrao/PSSComics/Comics/datasets.unify/DCM/images"
split_json = "/home/mserrao/PSSComics/Comics/comics100_all/comics_val_books.json"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

imagenette = load_dataset(
    'frgfm/imagenette',
    '320px',
    split='validation',
    revision='4d512db',
    trust_remote_code=True
)

print(imagenette[0:4])

print(f"Dataset loaded with {len(imagenette)} images")

labels = imagenette.info.features['label'].names
print(f"Labels: {labels}")

clip_labels = [f'a photo of a {label}' for label in labels]

model_id = 'openai/clip-vit-large-patch14'
print(f"Loading model: {model_id}")

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
model.to(device)

print("Processing text labels...")
with torch.no_grad():
    label_tokens = processor(
        text=clip_labels,
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

for i in tqdm(range(0, len(imagenette), batch_size)):
    try:
        i_end = min(i + batch_size, len(imagenette))
        
        with torch.no_grad(): 
            images = processor(
                text=None,
                images=imagenette[i:i_end]['image'],
                return_tensors='pt'
            )['pixel_values'].to(device)

            img_emb = model.get_image_features(images)
            img_emb = img_emb.detach().cpu().numpy()
            
            img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
            
            del images
            torch.cuda.empty_cache()
        
        scores = np.dot(img_emb, label_emb.T)
        batch_preds = np.argmax(scores, axis=1)
        preds.extend(batch_preds)
        
        del img_emb, scores, batch_preds
        gc.collect()
        
    except Exception as e:
        print(f"Error processing batch {i}-{i_end}: {e}")
        if batch_size > 1:
            print("Reducing batch size and retrying...")
            batch_size = max(1, batch_size // 2)
            i = max(0, i - batch_size)  

print(f"Finished processing {len(preds)} images")

correct = sum(preds[i] == imagenette[i]['label'] for i in range(len(preds)))
accuracy = correct / len(preds)
print(f"Accuracy: {accuracy:.4f} ({correct}/{len(preds)})")

try:
    np.save('clip_predictions.npy', np.array(preds))
    print("Predictions saved to clip_predictions.npy")
except Exception as e:
    print(f"Failed to save predictions: {e}")