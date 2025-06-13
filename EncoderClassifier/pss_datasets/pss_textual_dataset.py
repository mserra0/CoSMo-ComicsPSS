import torch
import os
import json
from torch.utils.data import Dataset
from PIL import Image
import tqdm

class PSSTextDataset(Dataset):
    def __init__(self,
        root_dir,
        annotations_path,  
        embedding_model,
        features_dim = 1024,
        max_seq_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
        precompute_features=True,
        precompute_dir=None,
        batch_size = 34,
        filter_unknown = True
        ):
        
        self.root_dir = root_dir
        self.max_seq_length = max_seq_length
        self.device = device
        self.emb_model = embedding_model
        self.precompute = precompute_features
        self.precompute_dir = precompute_dir
        self.batch_size = batch_size
        self.filter_unknown = filter_unknown
        self.feature_dim = features_dim
        
        self.class_names = ["advertisement", "cover", "story", "textstory", "first-page"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
    
        self.annotations = self._load_json(annotations_path)
            
        self.book_lookup = {}
        for book in self.annotations:
            if "hash_code" in book:
                self.book_lookup[book["hash_code"]] = book
            else: 
                raise Exception(f'Book Hash {book} not found!')
            
        self.books = []
        for book in self.annotations:
            book_id = book["hash_code"]
            book_dir = book_id 
            
            book_path = os.path.join(root_dir, book_dir)
            if not os.path.isdir(book_path):
                print(f"Warning: Directory not found for book {book_id} at {book_path}")
                continue
            
            image_files = [f for f in os.listdir(book_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            ocr_files = [f for f in os.listdir(book_path) if f.lower().endswith(('.json'))]
            if not image_files:
                print(f"Warning: No images found for book {book_id}")
                continue
            elif not ocr_files:
                print(f"Warning: No OCRs found for book {book_id}")
                continue
            
            image_files.sort(key=lambda x: self._get_page_number(x))
            ocr_files.sort(key=lambda x: self._get_page_number(x))

            if image_files:
                book_data = self._find_book_annotations(book_dir)
                
                image_paths = [os.path.join(book_path, img) for img in image_files]
                ocr_paths = [os.path.join(book_path, ocr) for ocr in ocr_files]
                
                page_labels = self._get_page_labels(book_data, len(image_paths), image_files)
                
                filtered_img_paths = []
                filtered_ocr_paths = []
                filtered_labels = []

                for img_path, label, ocr_path in zip(image_paths, page_labels, ocr_paths):
                    if not self._is_valid_image(img_path):
                        print(f'Skipping invalid image {img_path}')
                        continue
                    
                    if self.filter_unknown and (label == -1 or label is None):
                        print(f'Skipping unknown label image {img_path}')
                        continue
                    
                    if not os.path.exists(ocr_path):
                        print(f'Skipping image {img_path} because JSON file does not exist: {ocr_path}')
                        continue
                    
                    filtered_img_paths.append(img_path)
                    filtered_ocr_paths.append(ocr_path)
                    filtered_labels.append(label)

                if filtered_img_paths:
                    self.books.append({
                        'book_id': book_dir,
                        'image_paths': filtered_img_paths,
                        'ocr_paths' : filtered_ocr_paths,
                        'page_labels': filtered_labels,
                        'metadata': {'book_data':book_data,
                                     'source':'original'}
                    })
                    
        self.embeddings_cache = {}
        if precompute_dir:
            base, ext = os.path.splitext(precompute_dir)
            cache_path = f"{base}_textual_emb{ext}"
            
            if os.path.exists(cache_path) and not precompute_features:
                try:
                    print(f'Loading precomputed features from {cache_path}')
                    self.embeddings_cache = torch.load(cache_path, weights_only=True)
                    if len(self.embeddings_cache) != len(self.books):
                        print(f"Cache size mismatch ({len(self.embeddings_cache)}) vs dataset size ({len(self.books)}). Recomputing.")
                        raise Exception('Size mismatch!')
                    print(f'Features cache Example shape {self.embeddings_cache[0].shape}')
                except Exception as e:
                    print(f'Error loading precompute features from {cache_path}: {e}')
                    self._precompute_embeddings(cache_path)
            else:
                print(f'Perecomputing all embeddings and loading it in {cache_path}!')
                self._precompute_embeddings(cache_path)
                
                
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
        
    def _load_json(self, json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading JSON {json_file}: {e}")
            return None
        
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
                    title = item['title']
                    first_page_marked = False
                    
                    for i in range(num_pages):
                        page_num = None
                        for p_num, idx in page_map.items():
                            if idx == i:
                                page_num = p_num
                                break

                        if page_num is not None and start_page <= page_num <= end_page:
                            # Altough not ideal we will skip storys containing the word 'continue' in the title asstand alone stories
                            if page_num == start_page and 'continue' not in title: 
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
        

    def _extract_emb_batch(self, ocr_paths, batch_size=32):
        all_features = []
        
        for i in tqdm.tqdm(range(0, len(ocr_paths), batch_size), 
                        desc="Extracting features", unit="batch"):
            batch_paths = ocr_paths[i:i + batch_size]
            
            valid_paths = []
            for path in batch_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    all_features.append(torch.zeros(self.feature_dim))
            
            if not valid_paths:
                continue
                
            try:
                batch_text = []
                for path in valid_paths:
                    ocr_data = self._load_json(path)
                    ocr_text = ocr_data['OCRResult']
                    batch_text.append(ocr_text)
                
                with torch.no_grad():                    
                    bacth_embeddings = self.emb_model.encode(batch_text)
                    bacth_emb_tensor = torch.from_numpy(bacth_embeddings)
                    all_features.extend([feat for feat in bacth_emb_tensor])
                    
            except Exception as e:
                print(f"Error processing batch starting at {i}: {str(e)}")
                for _ in valid_paths:
                    all_features.append(torch.zeros(self.feature_dim))
        
        return all_features
    
    def _precompute_embeddings(self, save_path=None):
        print(f"Precomputing OCR embeddings for all book pages...")
        
        all_ocr_paths= []
        book_boundaries = []  # Store (start_idx, end_idx, book_idx) for each book
        current_idx = 0
        
        for book_idx, book in enumerate(self.books):
            start_idx = current_idx
            ocr_paths = book['ocr_paths']
            all_ocr_paths.extend(ocr_paths)
            current_idx += len(ocr_paths)
            book_boundaries.append((start_idx, current_idx, book_idx))
            
        print(f"Processing {len(all_ocr_paths)} OCR files across {len(self.books)} books...")
        
        all_embeddings = self._extract_emb_batch(all_ocr_paths, self.batch_size)
        
        for start_idx, end_idx, book_idx in tqdm.tqdm(book_boundaries, desc="Organizing embeddings by book"):
            book_embd = all_embeddings[start_idx:end_idx]
            
            if book_embd:
                self.embeddings_cache[book_idx] = torch.stack(book_embd)
        
        if save_path:
            print(f"Saving precomputed embeddings to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.embeddings_cache, save_path)
        
        print("Feature precomputation complete!")
        
    def __len__(self):
        return len(self.books)
    
    def __getitem__(self, idx): 
        book = self.books[idx]
        
        if self.precompute_dir and idx in self.embeddings_cache:
            embeddings = self.embeddings_cache[idx].to(self.device)
        else:
            print(f'Cache not found! Computing embeddings...')
            embeddings = torch.stack(
                self._extract_emb_batch(book['ocr_paths'], self.batch_size) 
            ).to(self.device)
        
        page_labels = book['page_labels']
        page_labels_tensor = torch.tensor(page_labels, dtype=torch.long)
        
        seq_length = min(len(embeddings), self.max_seq_length)
    
        attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
        if seq_length < self.max_seq_length:
            attention_mask[seq_length:] = 0
            
            padded_features = torch.zeros(self.max_seq_length, self.feature_dim)
            padded_features[:seq_length] = embeddings[:seq_length]
            embeddings = padded_features
            
            padded_labels = torch.full((self.max_seq_length,), -1, dtype=torch.long)
            padded_labels[:seq_length] = page_labels_tensor[:seq_length]
            page_labels_tensor = padded_labels
            
        else:
            embeddings = embeddings[:self.max_seq_length]
            page_labels_tensor = page_labels_tensor[:self.max_seq_length]
        
        return {
            'features': embeddings,
            'attention_mask': attention_mask,
            'book_id': book['book_id'],
            'page_labels': page_labels_tensor 
        }
    
    def get_num_classes(self):
        return len(self.class_names)
    
    def get_class_names(self):
        return self.class_names