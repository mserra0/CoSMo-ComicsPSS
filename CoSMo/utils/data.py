from torch.utils.data import Subset
import json
import os
import random
import math
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from matplotlib.patches import Patch

class ComicTransform:
    def __init__(self):
        self.paper_colors = [(255, 255, 245), (250, 248, 239), (251, 247, 240), (252, 252, 250)]
        self.params={}
    def __call__(self, img):
    
        orig_width, orig_height = img.size
    
        scale = random.uniform(0.98, 1.15)
        self.params['scale'] = scale
        
        fill_1 = random.choice(self.paper_colors)
        self.params['fill_1'] = fill_1
        
        fill_2 = random.choice(self.paper_colors)
        self.params['fill_2'] = fill_2
        
        fill_3 = random.choice(self.paper_colors)
        self.params['fill_3'] = fill_3
        
        transform_pipe = transforms.Compose([
            
            transforms.RandomRotation(degrees=2, fill=fill_1), 
    
            transforms.Resize((int(orig_height * scale), int(orig_width * scale)), 
                             interpolation=transforms.InterpolationMode.BICUBIC),
            
            transforms.RandomRotation(degrees=5, fill=fill_2), 
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.05)
            ], p=0.5),
            
            transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 3.0)),
            
            transforms.Lambda(lambda x: F.adjust_sharpness(x, 0.9)),
    
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3, fill=fill_3),
            
            transforms.CenterCrop((orig_height, orig_width))
        ])
        
        return transform_pipe(img), self._get_description()
    
    def _get_description(self):
        description = f"Image scaled {self.params['scale']:.3f}x, wiht the following fills: {self.params['fill_1'], self.params['fill_2'], self.params['fill_3']}"
        return description
    

def combine_json_files(output_path=None):
    json_files = [
        "/home/mserrao/PSSComics/Comics/DatasetDCM/comics100_all/comics_dev_books.json",
        "/home/mserrao/PSSComics/Comics/DatasetDCM/comics100_all/comics_val_books.json",
        "/home/mserrao/PSSComics/Comics/DatasetDCM/comics100_all/comics_test_books.json"
    ]
    combined_data = []
    
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                print(f"Loading {json_path}...")
                data = json.load(f)
                print(f"Found {len(data)} books in {os.path.basename(json_path)}")
                combined_data.extend(data)
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
    
    with open(output_path, 'w') as f:
        print(f"Writing {len(combined_data)} books to {output_path}")
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined JSON created at {output_path}")
    
    splits = {}
    categories = {
        "stories": 0,
        "textstories": 0, 
        "advertisements": 0,
        "covers": 0
    }
    
    for book in combined_data:
        split = book.get("split", "unknown")
        if split not in splits:
            splits[split] = 0
        splits[split] += 1
        
        for category in categories:
            categories[category] += len(book.get(category, []))
    
    print("\nStatistics:")
    print(f"Total books: {len(combined_data)}")
    print("Books by split:")
    for split, count in splits.items():
        print(f"  - {split}: {count} books")
    print("Content items:")
    for category, count in categories.items():
        print(f"  - {category}: {count} items")
        
def split_data(input_file, output_dir, train=0.6, val=0.2, test=0.2, seed=221):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'r') as f:
        comics_data = json.load(f)
    
    random.shuffle(comics_data)
    
    total_comics = len(comics_data)
    train_size = int(total_comics * train)
    val_size = int(total_comics * val)
    
    train_data = comics_data[:train_size]
    val_data = comics_data[train_size:train_size + val_size]
    test_data = comics_data[train_size + val_size:]
    

    with open(os.path.join(output_dir, 'comics_train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'comics_val.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'comics_test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Dataset split complete:")
    print(f"Total comics: {total_comics}")
    print(f"Training set: {len(train_data)} comics ({len(train_data)/total_comics:.1%})")
    print(f"Validation set: {len(val_data)} comics ({len(val_data)/total_comics:.1%})")
    print(f"Test set: {len(test_data)} comics ({len(test_data)/total_comics:.1%})")

        
def split_dataset(dataset, train_size=0.7, val_size=0.15, test_size=0.15, seed=123):

    random.seed(seed)

    num_books = len(dataset.books)
    indices = list(range(num_books))

    random.shuffle(indices)
    
    train_end = int(train_size * num_books)
    val_end = train_end + int(val_size * num_books)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"Total books: {num_books}")
    print(f"Training books: {len(train_indices)}, Validation books: {len(val_indices)}, Test books: {len(test_indices)}")
    
    return train_subset, val_subset, test_subset