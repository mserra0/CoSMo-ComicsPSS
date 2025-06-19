import torch
from torch.utils.data import Dataset
import json
import os
import tqdm
import numpy as np
from PIL import Image

class PSSFusionDataset(Dataset):
    def __init__(self, 
                root_dir, 
                backbones, 
                annotations_path, 
                precompute_dir,
                max_seq_len, 
                filter_unknown = True,
                device = 'cuda' if torch.cuda.is_available() else 'cpu',
                augment_data=False,            
                num_augmented_copies=1,      
                augmentation_shuffle_stories=True, 
                removal_p = 0.0,
                num_synthetic_books=0,       
                min_stories = 3,
                max_stories = 5,
                synthetic_remove_p = 0.15):
        
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.book_lookup = {}
        for book in self.annotations:
            if "hash_code" in book:
                self.book_lookup[book["hash_code"]] = book
            else: 
                raise Exception(f'Book Hash {book} not found!')

        self.class_names = ["advertisement", "cover", "story", "textstory", "first-page"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        self.root_dir = root_dir
        self.backbones = backbones  # { 'name' : feature_dim }
        self.backbone_names = list(backbones.keys())
        self.backbones_dimensions = list(backbones.values())
        self.max_seq_len = max_seq_len
        self.device = device
        self.filter_unknown = filter_unknown
        
        self.book_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.book_dirs.sort()  
        
        self.books = []
        
        for book in self.annotations:
            book_id = book["hash_code"]
            book_dir = book_id 
            
            book_path = os.path.join(root_dir, book_dir)
            if not os.path.isdir(book_path):
                print(f"Warning: Directory not found for book {book_id} at {book_path}")
                continue
            
            image_files = [f for f in os.listdir(book_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                print(f"Warning: No images found for book {book_id}")
                continue
            image_files.sort(key=lambda x: self._get_page_number(x))

            if image_files:
                book_data = self._find_book_annotations(book_dir)
                
                image_paths = [os.path.join(book_path, img) for img in image_files]
                
                page_labels = self._get_page_labels(book_data, len(image_paths), image_files)
                
                filtered_paths = []
                filtered_labels = []

                for path, label in zip(image_paths, page_labels):
                    if not self._is_valid_image(path):
                        print(f'Skipping invalid image {path}')
                        continue
                    
                    if self.filter_unknown and (label == -1 or label is None):
                        print(f'Skipping unknown label image {path}')
                        continue
                    
                    filtered_paths.append(path)
                    filtered_labels.append(label)

                if filtered_paths:
                    self.books.append({
                        'book_id': book_dir,
                        'image_paths': filtered_paths,
                        'page_labels': filtered_labels,
                        'metadata': {'book_data':book_data,
                                     'source':'original'}
                    })
                    
        self.original_books_len = len(self.books)
        
        print(f"Found {self.original_books_len} books")
        
        self.fussion_features = {}
        if precompute_dir:
            for backbone_name, feature_dim in self.backbones.items():
                cache_path = precompute_dir
                if augment_data:
                    base, ext = os.path.splitext(cache_path)
                    aug_suffix = f"_cp{num_augmented_copies}synth{num_synthetic_books}"
                    cache_path = f"{base}{aug_suffix}_{backbone_name}{ext}"

                elif self.filter_unknown:
                    base, ext = os.path.splitext(precompute_dir)
                    cache_path = f"{base}_filtered_{backbone_name}{ext}"
                    
                if os.path.exists(cache_path):
                    try:
                        print(f'Loading precomputed features from {cache_path} for {backbone_name}')
                        self.fussion_features[backbone_name] = torch.load(cache_path, weights_only=True)
                        len_features = len(self.fussion_features[backbone_name])
                        
                        if len_features != len(self.books):
                            print(f"Cache size mismatch ({len_features}) vs dataset size ({len(self.books)})")
                            raise Exception('Size mismatch!')
                    
                    except Exception as e:
                        raise ValueError(f'Error loading precompute features from {cache_path}: {e}')
                        
                else:
                    raise ValueError(f'No features found for {backbone_name} in {cache_path}!')
                
    def _find_book_annotations(self, book_id):
        for key in [book_id, book_id.split('_')[0] if '_' in book_id else None]:
            if key and key in self.book_lookup:
                return self.book_lookup[key]
        
        print(f"Warning: No annotation found for book {book_id}")
        return None
    
        
    def _is_valid_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
                
            return True
        except Exception as e:
            print(f"Skipping corrupted image: {image_path} - {str(e)}")
            return False
           
    def _get_page_labels(self, book_data, num_pages, image_files=None):
        """Generate page labels with proper alignment to the image files"""
        if not book_data:
            return [-1] * num_pages
        
        page_labels = [-1] * num_pages
        first_page_idx = self.class_to_idx["first-page"]
        
        page_map = {}
        if image_files:
            for i, img_file in enumerate(image_files):
                page_num = self._get_page_number(img_file)
                page_map[page_num] = i
        
        category_mapping = {
            "stories": self.class_to_idx["story"],
            "first-pages": self.class_to_idx["first-page"],  
            "textstories": self.class_to_idx["textstory"],
            "advertisements": self.class_to_idx["advertisement"],
            "covers": self.class_to_idx["cover"]
        }
        
        for category, label_idx in category_mapping.items():
            if category not in book_data:
                continue
                
            for item in book_data[category]:
                start_page = item.get("page_start", 0)
                end_page = item.get("page_end", start_page)
                
                if category == "stories":
                    first_page_marked = False
                    
                    for i in range(num_pages):
                        page_num = None
                        for p_num, idx in page_map.items():
                            if idx == i:
                                page_num = p_num
                                break

                        if page_num is not None and start_page <= page_num <= end_page:
                        
                            if page_num == start_page:
                                page_labels[i] = first_page_idx
                                first_page_marked = True
                            else:
                                page_labels[i] = label_idx
                else:
                    
                    for i in range(num_pages):
                        page_num = None
                        for p_num, idx in page_map.items():
                            if idx == i:
                                page_num = p_num
                                break

                        if page_num is not None and start_page <= page_num <= end_page:
                            page_labels[i] = label_idx
        
        return page_labels
    
    def _get_page_number(self, filename: str) -> int:
        try:
            num = ''.join(filter(str.isdigit, filename))
            return int(num) if num else 0
        except:
            print('No image number found.')
            return 0
    
    def __len__(self):
        return len(self.books)
    
    def __getitem__(self, idx): 
        book = self.books[idx]
        
        book_fussion_features = {}
        for bb_name, bb_features in self.fussion_features.items():
            try:
                book_fussion_features[bb_name] = bb_features[idx].to(self.device)
            except IndexError:
                raise RuntimeError(
                    f"IndexError: Failed to get features for backbone '{bb_name}' at index {idx}. "
                    f"Cache length might not match dataset length. Expected length: {len(bb_features)}."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error accessing features for backbone '{bb_name}' at index {idx}: {e}"
                )
                  
        page_labels = book['page_labels']
        page_labels_tensor = torch.tensor(page_labels, dtype=torch.long, device=self.device)
        
        seq_length = min(len(page_labels), self.max_seq_len)
    
        attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
        if seq_length < self.max_seq_len:
            attention_mask[seq_length:] = 0
            for bb_name, bb_features in book_fussion_features.items():
                bb_padded_features = torch.zeros(self.max_seq_len, self.backbones[bb_name], device=self.device)
                bb_padded_features[:seq_length] = bb_features[:seq_length]
                book_fussion_features[bb_name] = bb_padded_features
            
            padded_labels = torch.full((self.max_seq_len,), -1, dtype=torch.long, device=self.device)
            padded_labels[:seq_length] = page_labels_tensor[:seq_length]
            page_labels_tensor = padded_labels
            
        else:
            for bb_name, bb_features in book_fussion_features.items():
                book_fussion_features[bb_name] = bb_features[:self.max_seq_len]
                
            page_labels_tensor = page_labels_tensor[:self.max_seq_len]
        
        return {
            'features': book_fussion_features,
            'attention_mask': attention_mask,
            'book_id': book['book_id'],
            'page_labels': page_labels_tensor 
        }
    
    def get_num_classes(self):
        return len(self.class_names)
    
    def get_class_names(self):
        return self.class_names