from EncoderClassifier.pss_datasets.pss_dataset import PSSDataset
from EncoderClassifier.utils.visualitzation import analyze_book_types, visualize_book
from EncoderClassifier.utils.data import ComicTransform
import matplotlib.pyplot as plt
import os
from transformers import SiglipImageProcessor, AutoProcessor, AutoModel
import torch
import random 
import numpy as np
import tqdm
root_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images'
annotations_dir = '/home/mserrao/PSSComics/Comics/DatasetDCM/comics_all_430.json'
precompute_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
checkpoint_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/checkpoints'
data_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
out_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/out'
    
model_id = 'openai/clip-vit-large-patch14-336'
gpu_id = 3
seed = 10
num_aug_copies = 5
num_synthetic_books = 1000

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

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

transformations = ComicTransform()

test_dataset = PSSDataset(root_dir=root_dir, 
                            model_id = model_id,
                            backbone=backbone, 
                            backbone_name = backbone_name,
                            feature_dim = feature_dim,
                            processor=processor, 
                            device=device, 
                            annotations_path=f'{data_dir}/comics_test.json', 
                            precompute_features=False,
                            precompute_dir=f'{precompute_dir}/features_test.pt', 
                            augment_data=False,
                            num_augmented_copies = num_aug_copies,
                            transform=None, 
                            removal_p=0.05,
                            num_synthetic_books=num_synthetic_books,
                            min_stories=2,
                            max_stories=3,
                            synthetic_remove_p=0.15)
# analyze_book_types(test_dataset)
visualize_book(test_dataset, book_idx=2, dpi=200, transforms=None)
plt.show()
def batch_visualize_books(dataset, output_dir, dpi=150):
    """
    Generate and save visualizations for all books in the dataset
    
    Args:
        dataset: The PSSDataset instance
        output_dir: Directory to save visualizations
        dpi: Resolution for saved images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(dataset.books)} unique books to visualize")
    
    for book in dataset.books:
        try:
            book_id = book['book_id']
            fig = visualize_book(dataset, book_id=book_id, dpi=dpi, transforms=dataset.transform)

            output_path = os.path.join(output_dir, f"{book_id}.png")
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
        except Exception as e:
            print(f"Error visualizing book {book_id}: {e}")
    
    print(f"Visualizations saved to {output_dir}")
visualization_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/visualizations/test_dataset'

batch_visualize_books(test_dataset, visualization_dir)