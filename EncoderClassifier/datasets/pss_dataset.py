import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
import tqdm
import random 
import copy  
import numpy as np

class PSSDataset(Dataset):
    
    def __init__(
        self,
        root_dir,
        model_id,
        backbone,
        backbone_name,
        feature_dim,
        processor,
        annotations_path,  
        max_seq_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
        precompute_features=True,
        precompute_dir=None,
        transform=None,
        filter_unknown=True,
        # --- Augmentation Parameters ---
        augment_data=False,            
        num_augmented_copies=1,      
        augmentation_shuffle_stories=True, 
        removal_p = 0.05,
        num_synthetic_books=0,       
        min_stories = 3,
        max_stories = 5,
        synthetic_remove_p = 0.15
        ):
        
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
            
        if augment_data and transform is None:
            print("Warning: augment_data=True but transform=None. Synthetic data might not be properly augmented.")
        
        self.book_lookup = {}
        for book in self.annotations:
            if "hash_code" in book:
                self.book_lookup[book["hash_code"]] = book
            else: 
                raise Exception(f'Book Hash {book} not found!')

        self.class_names = ["advertisement", "cover", "story", "textstory", "first-page"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        self.root_dir = root_dir
        self.device = device
        self.max_seq_length = max_seq_length
        self.transform = transform
        self.precompute_features = precompute_features
        self.precompute_dir = precompute_dir
        self.filter_unknown = filter_unknown
        
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.processor = processor
        self.feature_dim = feature_dim
       
        self.book_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.book_dirs.sort()  
        
        self.min_stories = min_stories
        self.max_stories = max_stories
        
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
        
        if augment_data:
            print("Applying data augmentation...")
            if num_augmented_copies > 0:
                augmented_books = self._augment_books(
                    num_copies=num_augmented_copies,
                    shuffle_stories=augmentation_shuffle_stories,
                    removal_p=removal_p
                )
                self.books.extend(augmented_books)
                print(f" -> Added {len(augmented_books)} augmented copies.")
        
            if num_synthetic_books > 0:
                synthetic_books = self._create_synthetic_books(num_books=num_synthetic_books, removal_p=synthetic_remove_p)
                self.books.extend(synthetic_books)
                print(f" -> Added {len(synthetic_books)} synthetic books.")

            print(f"Total books after augmentation: {len(self.books)}")
        
        self._print_stats()
        
        self.features_cache = {}
        if precompute_dir:
            cache_path = precompute_dir
            
            if augment_data:
                 base, ext = os.path.splitext(cache_path)
                 aug_suffix = f"_cp{num_augmented_copies}synth{num_synthetic_books}"
                 cache_path = f"{base}{aug_suffix}_{self.backbone_name}{ext}"

            elif self.filter_unknown:
                base, ext = os.path.splitext(precompute_dir)
                cache_path = f"{base}_filtered_{self.backbone_name}{ext}"
                
            if os.path.exists(cache_path) and not precompute_features:
                try:
                    print(f'Loading precomputed features from {cache_path}')
                    self.features_cache = torch.load(cache_path, weights_only=True)
                    if len(self.features_cache) != len(self.books):
                        print(f"Cache size mismatch ({len(self.features_cache)}) vs dataset size ({len(self.books)}). Recomputing.")
                        raise Exception('Size mismatch!')
                    print(f'Features cache Example shape {self.features_cache[0].shape}')
                except Exception as e:
                    print(f'Error loading precompute features from {cache_path}: {e}')
                    self._precompute_all_features(cache_path)
            else:
                print(f'Perecomputing all {self.backbone_name} features and loading it in {cache_path}!')
                self._precompute_all_features(cache_path)
    
    def _find_book_annotations(self, book_id):
        for key in [book_id, book_id.split('_')[0] if '_' in book_id else None]:
            if key and key in self.book_lookup:
                return self.book_lookup[key]
        
        print(f"Warning: No annotation found for book {book_id}")
        return None
    
    def _print_stats(self):
        page_label_counts = {name: 0 for name in self.class_names}
        if not self.filter_unknown:
            page_label_counts["unknown"] = 0
        
        for book in self.books:
            for label in book['page_labels']:
                if not self.filter_unknown and (label is None or label == -1):
                    page_label_counts["unknown"] += 1
                else:
                    label_name = self.class_names[label]
                    page_label_counts[label_name] += 1
        
        print("Page label distribution:")
        for label, count in page_label_counts.items():
            print(f"  {label}: {count} pages")
        
    
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
        
    def _extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            inputs = {}
            pixel_values = {}
                
            if 'siglip2' in self.backbone_name:
                inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
            else:
                processed = self.processor(images=image, return_tensors="pt")
                pixel_values = processed['pixel_values'].to(self.device)
                
            image_features = torch.zeros(self.feature_dim)
            
            with torch.no_grad():
                if 'clip' in self.backbone_name:
                    image_features = self.backbone.get_image_features(pixel_values=pixel_values)
                    
                elif 'siglip2' in self.backbone_name:
                    image_features = self.backbone.get_image_features(**inputs)
                    
                elif 'siglip' in self.backbone_name:
                    image_features = self.backbone.vision_model(pixel_values=pixel_values).pooler_output

                elif 'dinov2' in self.backbone_name:
                    outputs = self.backbone(pixel_values)
                    image_features = outputs.last_hidden_state[:, 0]
  
            return image_features.squeeze(0).cpu()
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return torch.zeros(self.feature_dim)
    
    def _precompute_all_features(self, save_path=None):
        print(f"Precomputing {self.backbone_name} features for all book pages...")
        
        for book_idx, book in enumerate(tqdm.tqdm(self.books)):
            book_features = []
                    
            for img_path in book['image_paths']:
                book_features.append(self._extract_features(img_path))
                
            if book_features:  
                self.features_cache[book_idx] = torch.stack(book_features)
                
        if save_path:
            print(f"Saving precomputed features to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.features_cache, save_path)
        
        print("Feature precomputation complete!")
        
    def _augment_books(self, num_copies=1, shuffle_stories=True, removal_p = 0.05):
        augmented_books = []
        cover_idx = self.class_to_idx.get("cover", -1)
        story_idx = self.class_to_idx.get("story", -1)
        ad_idx = self.class_to_idx.get("advertisement", -1)
        textstory_idx = self.class_to_idx.get("textstory", -1)
        first_page_idx = self.class_to_idx.get("first-page", -1)
        
        for i, original_book in enumerate(tqdm.tqdm(self.books, desc="Creating augmented copies")):
            for j in range(num_copies):
                new_book = copy.deepcopy(original_book)
                new_book['book_id'] = f"{original_book['book_id']}_aug_{j+1}"
                new_book['metadata']['source'] = 'synthetic'
                
                paths = new_book['image_paths']
                labels = new_book['page_labels']
                
                if not paths:
                    continue
                    
                cover_pages = []
                content_blocks = []
                current_block = []
                current_type = None
                
                for i, (path, label) in enumerate(zip(paths, labels)):
                    if label == cover_idx:
                        cover_pages.append((path, label))
                    elif i == 0 and not cover_pages:
                        cover_pages.append((path, label))
                    else:
                       
                        current_block_type = None
                        if label == story_idx or label == first_page_idx:
                            current_block_type = story_idx 
                        else:
                            current_block_type = label
                            
                        if current_block_type != current_type:
                            if current_block:
                                content_blocks.append((current_type, current_block))
                            current_block = []
                            current_type = current_block_type
                        current_block.append((path, label))
                                
                if current_block:
                    content_blocks.append((current_type, current_block))
            
                processed_blocks = []
                for block_type, block in content_blocks:
                    if block_type == story_idx and len(block) > 1 and shuffle_stories:
                        
                        first_page = block[0]
                        first_page_path, first_page_label = first_page
                        
                        if first_page_label != first_page_idx:
                            first_page = (first_page_path, first_page_idx)
                            
                        rest_pages = block[1:]
                        random.shuffle(rest_pages)
                        rest_pages = [element for element in rest_pages if random.random() >= removal_p]
                        
                        rest_pages = [(path, story_idx) for path, _ in rest_pages]
                        
                        processed_blocks.append((block_type, [first_page] + rest_pages))
                    else:
                        processed_blocks.append((block_type, block))
                
                if shuffle_stories:
                    story_blocks = [(i, block) for i, (block_type, block) in enumerate(processed_blocks) 
                                if block_type == story_idx]
                    
                    if story_blocks:
                        story_indices = [idx for idx, _ in story_blocks]
                        shuffled_indices = story_indices.copy()
                        random.shuffle(shuffled_indices)
                    
                        index_map = {orig: shuffled for orig, shuffled in zip(story_indices, shuffled_indices)}
                    
                        new_block_order = []
                        for i, (block_type, block) in enumerate(processed_blocks):
                            if i in index_map:  
                                for orig_idx, new_idx in index_map.items():
                                    if new_idx == i:
                                        new_block_order.append(processed_blocks[orig_idx])
                                        break
                            else:
                                new_block_order.append((block_type, block))
                        
                        processed_blocks = new_block_order
            
                new_paths = []
                new_labels = []

                for path, label in cover_pages:
                    new_paths.append(path)
                    new_labels.append(label)
                    
                for block_type, block in processed_blocks:
                    for path, label in block:
                        new_paths.append(path)
                        new_labels.append(label)
                
                new_book['image_paths'] = new_paths
                new_book['page_labels'] = new_labels
                
                augmented_books.append(new_book)
        
        return augmented_books
    
    def _create_synthetic_books(self, num_books=100, removal_p=0.1):
        synthetic_books = []
        
        cover_idx = self.class_to_idx["cover"]
        story_idx = self.class_to_idx["story"]
        textstory_idx = self.class_to_idx["textstory"]
        ad_idx = self.class_to_idx["advertisement"]
        first_page_idx = self.class_to_idx["first-page"]
        
        cover_pages = []
        story_blocks = []
        ad_blocks = []
        textstory_blocks = []
        
        current_type = None
        current_block = []
        
        print("Categorizing pages for synthetic book creation...")
        original_books = self.books[:self.original_books_len]
    
        for book in tqdm.tqdm(original_books, desc="Categorizing pages"):
            current_type = None
            current_block = []
            for i, (path, label) in enumerate(zip(book['image_paths'], book['page_labels'])):
                if label == cover_idx:
                    cover_pages.append(path)
                elif i == 0 and not cover_pages:
                    cover_pages.append(path)
                else:
                    if label != current_type:
                        if current_block:
                            if current_type == story_idx:
                                story_blocks.append(current_block)
                            elif current_type == ad_idx:
                                ad_blocks.append(current_block)
                            elif current_type == textstory_idx:
                                textstory_blocks.append(current_block)
                            
                        current_block = []
                        current_type = label
                    current_block.append(path)  
            
            if current_block:
                if current_type == story_idx:
                    story_blocks.append(current_block)
                elif current_type == ad_idx:
                    ad_blocks.append(current_block)
                elif current_type == textstory_idx:
                    textstory_blocks.append(current_block)

        for i in tqdm.tqdm(range(num_books), desc="Creating synthetic books"):
            new_image_paths = []
            new_page_labels = []
            
            # Add cover
            cover_path = random.choice(cover_pages)
            new_image_paths.append(cover_path)
            new_page_labels.append(cover_idx)
            
            # Create copies to avoid modifying originals
            available_stories = story_blocks.copy()
            available_ads = ad_blocks.copy()
            available_textstories = textstory_blocks.copy()
            
            num_stories = min(random.randint(self.min_stories, self.max_stories), len(available_stories))
            
            for _ in range(num_stories):
                # Add a story
                if not available_stories:
                    break
                    
                idx = random.randrange(len(available_stories))
                story_block = available_stories.pop(idx)
                while len(story_block) < 3:
                    idx = random.randrange(len(available_stories))
                    story_block = available_stories.pop(idx)
                    
                if story_block:
                        first_page = story_block[0]
                        rest_pages = story_block[1:] if len(story_block) > 1 else []
                        random.shuffle(rest_pages)
                        rest_pages = [page for page in rest_pages if random.random() >= removal_p]
                        
                        # Add first page with first-page label
                        new_image_paths.append(first_page)
                        new_page_labels.append(first_page_idx)
                        
                        # Add rest of pages with story label
                        new_image_paths.extend(rest_pages)
                        new_page_labels.extend([story_idx] * len(rest_pages))
                
                # After each story, add either ads or text stories
                content_type = random.choice(['ad', 'text'])
                
                if content_type == 'ad' and available_ads:
                    idx = random.randrange(len(available_ads))
                    ad_pages = available_ads.pop(idx)
                    
                    new_image_paths.extend(ad_pages)
                    new_page_labels.extend([ad_idx] * len(ad_pages))
                elif content_type == 'text' and available_textstories:
                    idx = random.randrange(len(available_textstories))
                    text_pages = available_textstories.pop(idx)
                    
                    new_image_paths.extend(text_pages)
                    new_page_labels.extend([textstory_idx] * len(text_pages))

            synthetic_books.append({
                'book_id': f"synthetic_{i+1}",
                'image_paths': new_image_paths,
                'page_labels': new_page_labels,
                'metadata': {'book_data': None, 'source': 'synthetic'}
            })

        return synthetic_books
    
    def __len__(self):
        return len(self.books)
    
    def __getitem__(self, idx): 
        book = self.books[idx]
        
        if self.precompute_dir and idx in self.features_cache:
            features = self.features_cache[idx].to(self.device)
        else:
            print(f'Cache not found! Computing {self.backbone_name} features...')
            features = torch.stack([
                self._extract_features(img_path) 
                for img_path in book['image_paths']
            ]).to(self.device)
        
        page_labels = book['page_labels']
        page_labels_tensor = torch.tensor(page_labels, dtype=torch.long)
        
        seq_length = min(len(features), self.max_seq_length)
    
        attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
        if seq_length < self.max_seq_length:
            attention_mask[seq_length:] = 0
            
            padded_features = torch.zeros(self.max_seq_length, self.feature_dim)
            padded_features[:seq_length] = features[:seq_length]
            features = padded_features
            
            padded_labels = torch.full((self.max_seq_length,), -1, dtype=torch.long)
            padded_labels[:seq_length] = page_labels_tensor[:seq_length]
            page_labels_tensor = padded_labels
            
        else:
            features = features[:self.max_seq_length]
            page_labels_tensor = page_labels_tensor[:self.max_seq_length]
        
        return {
            'features': features,
            'attention_mask': attention_mask,
            'book_id': book['book_id'],
            'page_labels': page_labels_tensor 
        }
    
    def get_num_classes(self):
        return len(self.class_names)
    
    def get_class_names(self):
        return self.class_names