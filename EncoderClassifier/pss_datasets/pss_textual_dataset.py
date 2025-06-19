import torch
import os
import json
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import random
import copy

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
        filter_unknown = True,
        # ------ Agmentations
        augment_data=False,  
        transforms = None,          
        num_augmented_copies=1,      
        augmentation_shuffle_stories=True, 
        removal_p = 0.0,
        num_synthetic_books=0,       
        min_stories = 3,
        max_stories = 5,
        synthetic_remove_p = 0.15
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
        self.min_stories = min_stories
        self.max_stories = max_stories
        self.augment_data = augment_data
        
        self.transforms = transforms
        
        self.phrases = [
            "This page has rotation {x} degrees and fill levels {y}, {z}.",
            "Rotation: {x}°. FillX: {y}, FillY: {z}.",
            "Document skew angle: {x}. Fill intensity detected: {y}/{z}.",
            "Detected layout parameters — rotation: {x}, brightness: {y}, contrast: {z}.",
            "OCR confidence score: {x}%. Fill ratio X: {y}%, Fill ratio Y: {z}%.",
            "Visual features — rotation={x}, fill=({y}, {z}).",
            "Layout info: rot={x}°, fx={y}, fy={z}.",
            "Metadata: rotation {x}, fill-x {y}, fill-y {z}.",
            "Angle={x}, FillX={y}, FillY={z} — auto-generated from scan.",
            "Scan metrics: tilt={x}, fill left={y}, fill right={z}.",
            "Page stats: rotation {x} degrees, fill percentages: {y}%, {z}%."
        ]

        
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
                    
                    
        self.embeddings_cache = {}
        if precompute_dir:
            base, ext = os.path.splitext(precompute_dir)
            cache_path = f"{base}_textual_emb{ext}"
            if augment_data:
                aug_suffix = f"_cp{num_augmented_copies}synth{num_synthetic_books}"
                cache_path = f"{base}{aug_suffix}_textual_emb{ext}"

            
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
    
    def _generate_random_phrase(self):
        phrase = random.choice(self.phrases)
        x = random.randint(-10, 10)           # Example: rotation
        y = random.randint(0, 100)            # Example: fill-x
        z = random.randint(0, 100)            # Example: fill-y
        return phrase.format(x=x, y=y, z=z)
        
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
                    if self.augment_data:
                        description = self._generate_random_phrase()
                        ocr_text = description + ocr_text
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
        
        
    def _augment_books(self, num_copies=1, shuffle_stories=True, removal_p = 0.05):
        augmented_books = []
        cover_idx = self.class_to_idx.get("cover", -1)
        story_idx = self.class_to_idx.get("story", -1)
        
        for i, original_book in enumerate(tqdm.tqdm(self.books, desc="Creating augmented copies")):
            for j in range(num_copies):
                new_book = copy.deepcopy(original_book)
                new_book['book_id'] = f"{original_book['book_id']}_aug_{j+1}"
                new_book['metadata']['source'] = 'synthetic'
                
                ocr_paths = new_book['ocr_paths']
                img_path = new_book['image_paths']
                labels = new_book['page_labels']
                
                if not img_path or not ocr_paths:
                    print(f"WARNING: No paths for {new_book['book_id']}")
                    continue
                    
                cover_pages = []
                content_blocks = []
                current_block = []
                current_type = None
                
                for i, (img_pth, ocr_pth, label) in enumerate(zip(img_path, ocr_paths, labels)):
                    if label == cover_idx:
                        cover_pages.append((img_pth, ocr_pth, label))
                    elif i == 0 and not cover_pages:
                        cover_pages.append((img_pth, ocr_pth, label))
                    else:
                
                        if label != current_type:
                            if current_block:
                                content_blocks.append((current_type, current_block))
                            current_block = []
                            current_type = label
                        current_block.append((img_pth, ocr_pth, label))
                                
                if current_block:
                    content_blocks.append((current_type, current_block))
            
                processed_blocks = []
                for block_type, block in content_blocks:
                    if block_type == story_idx and len(block) > 1 and shuffle_stories:
                        shuffled_block = block.copy()
                        random.shuffle(shuffled_block)
                        shuffled_block = [element for element in shuffled_block if random.random() >= removal_p]
                        processed_blocks.append((block_type, shuffled_block))
                    else: 
                        # including first_page_idx
                        processed_blocks.append((block_type, block))
                
                if shuffle_stories:
                    story_blocks = [(i, block) for i, (block_type, block) in enumerate(processed_blocks) 
                                if block_type == story_idx and len(block) > 3] # in order to keep small stroies in the same position
                    
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
            
                new_img_pth = []
                new_ocr_pth = []
                new_labels = []
                

                for img_pth, ocr_pth, label in cover_pages:
                    new_img_pth.append(img_pth)
                    new_ocr_pth.append(ocr_pth)
                    new_labels.append(label)
                    
                for block_type, block in processed_blocks:
                    for img_pth, ocr_pth, label in block:
                        new_img_pth.append(img_pth)
                        new_ocr_pth.append(ocr_pth)
                        new_labels.append(label)
                
                new_book['ocr_paths'] = new_ocr_pth
                new_book['image_paths'] = new_img_pth
                new_book['page_labels'] = new_labels
                
                augmented_books.append(new_book)
        
        return augmented_books
    
    def _create_synthetic_books(self, num_books=100, removal_p=0.1, min_stories_per_block = 1, max_stories_per_block=3):
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
        
        print("Categorizing pages for synthetic book creation...")
        original_books = self.books[:self.original_books_len]

        for book in tqdm.tqdm(original_books, desc="Categorizing pages"):
            current_type = None
            current_block = []
            
            for i, (img_pth, ocr_pth, label) in enumerate(zip(book['image_paths'], book['ocr_paths'], book['page_labels'])):
                if label == cover_idx:
                    cover_pages.append((img_pth, ocr_pth))
                elif i == 0 and not cover_pages:
                    cover_pages.append((img_pth, ocr_pth))
                else:
                    # Group story and first-page together as story blocks
                    block_type = story_idx if label in [story_idx, first_page_idx] else label
                    
                    if block_type != current_type:
                        if current_block:
                            if current_type == story_idx:
                                story_blocks.append(current_block)
                            elif current_type == ad_idx:
                                ad_blocks.append(current_block)
                            elif current_type == textstory_idx:
                                textstory_blocks.append(current_block)
                            
                        current_block = []
                        current_type = block_type
                    current_block.append((img_pth, ocr_pth))  
            
            if current_block:
                if current_type == story_idx:
                    story_blocks.append(current_block)
                elif current_type == ad_idx:
                    ad_blocks.append(current_block)
                elif current_type == textstory_idx:
                    textstory_blocks.append(current_block)

        for i in tqdm.tqdm(range(num_books), desc="Creating synthetic books"):
            new_image_paths = []
            new_ocr_paths = []
            new_page_labels = []
            
            # Add cover
            img_path, ocr_pth = random.choice(cover_pages)
            new_image_paths.append(img_path)
            new_ocr_paths.append(ocr_pth)
            new_page_labels.append(cover_idx)
            
            available_stories = story_blocks.copy()
            available_ads = ad_blocks.copy()
            available_textstories = textstory_blocks.copy()
            
            num_story_blocks = min(random.randint(self.min_stories, self.max_stories), len(available_stories))
            
            stories_added = 0
            while stories_added < num_story_blocks and available_stories:
                total_story_pages = 0
                stories_added_to_block = 0
                num_stories_per_block = min(random.randint(min_stories_per_block, max_stories_per_block), len(available_stories))
                
                while (stories_added_to_block < num_stories_per_block or total_story_pages < 5) and available_stories:
                    idx = random.randrange(len(available_stories))
                    story_block = available_stories.pop(idx)
                    
                    if story_block:
                        # First page gets first-page label, rest get story label
                        first_page_img, first_page_ocr = story_block[0]
                        rest_pages = story_block[1:] if len(story_block) > 1 else []
                        random.shuffle(rest_pages)
                        rest_pages = [page for page in rest_pages if random.random() >= removal_p]
                        
                        # Add first page with first-page label
                        new_image_paths.append(first_page_img)
                        new_ocr_paths.append(first_page_ocr)
                        new_page_labels.append(first_page_idx)
                        total_story_pages += 1
                        
                        # Add rest of pages with story label
                        new_image_paths.extend([img_path for img_path, ocr in rest_pages])
                        new_ocr_paths.extend([ocr for img_path, ocr in rest_pages])
                        new_page_labels.extend([story_idx] * len(rest_pages))
                        total_story_pages += len(rest_pages)

                        stories_added_to_block +=1
                        
                stories_added += 1
            
                content_type = random.choice(['ad', 'text'])
                
                if content_type == 'ad' and available_ads:
                    idx = random.randrange(len(available_ads))
                    ad_pages = available_ads.pop(idx)
                    
                    new_image_paths.extend([img_path for img_path, ocr in ad_pages])
                    new_ocr_paths.extend([ocr for img_path, ocr in ad_pages])
                    new_page_labels.extend([ad_idx] * len(ad_pages))
                elif content_type == 'text' and available_textstories:
                    idx = random.randrange(len(available_textstories))
                    text_pages = available_textstories.pop(idx)
                    
                    new_image_paths.extend([img_path for img_path, ocr in text_pages])
                    new_ocr_paths.extend([ocr for img_path, ocr in text_pages])
                    new_page_labels.extend([textstory_idx] * len(text_pages))

            synthetic_books.append({
                'book_id': f"synthetic_{i+1}",
                'image_paths': new_image_paths,
                'ocr_paths' : new_ocr_paths,
                'page_labels': new_page_labels,
                'metadata': {'book_data': None, 'source': 'synthetic'}
            })

        return synthetic_books
        
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