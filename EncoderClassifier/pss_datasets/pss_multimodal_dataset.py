import tqdm
import torch
import os
from .pss_dataset import PSSDataset

class PSSMultimodalDataset(PSSDataset):
    def __init__(
        self,
        root_dir,
        # -- Textua Embedding Model
        embedding_model,
        emb_feature_dim,
        precompute_emb,
        precompute_emb_dir,
        # -- Visual Backbone Feature Extractor
        model_id,
        backbone,
        backbone_name,
        bb_feature_dim,
        processor,
        precompute_visual_features=True,
        precompute_visial_featres_dir=None,
        # ---------------
        annotations_path = None,  
        max_seq_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
        transform=None,
        filter_unknown=True,
        batch_size = 32,
        #  --- Augmentation Parameters --- 
        augment_data=False,            
        num_augmented_copies=1,      
        augmentation_shuffle_stories=True, 
        removal_p = 0.05,
        num_synthetic_books=0,       
        min_stories = 3,
        max_stories = 5,
        synthetic_remove_p = 0.15,
    ):
        super().__init__(
            root_dir=root_dir,
            model_id=model_id,
            backbone=backbone,
            backbone_name=backbone_name,
            feature_dim=bb_feature_dim,
            processor=processor,
            annotations_path=annotations_path,
            max_seq_length=max_seq_length,
            device=device,
            precompute_features=precompute_visual_features,
            precompute_dir=precompute_visial_featres_dir,
            transform=transform,
            filter_unknown=filter_unknown,
            augment_data=augment_data,
            num_augmented_copies=num_augmented_copies,
            augmentation_shuffle_stories=augmentation_shuffle_stories,
            removal_p=removal_p,
            num_synthetic_books=num_synthetic_books,
            min_stories=min_stories,
            max_stories=max_stories,
            synthetic_remove_p=synthetic_remove_p
        )
        
        self.root_dir = root_dir
        self.emb_model = embedding_model
        self.emb_feature_dim = emb_feature_dim
        self.precompute_emb = precompute_emb
        self.precompute_emb_dir = precompute_emb_dir
        self.batch_size = batch_size
    
        for book in self.books:
            book_id = book['book_id']
            book_path = os.path.join(root_dir, book_id)
            image_paths = book['image_paths']
            page_labels = book['page_labels']
            
            ocr_files = [f for f in os.listdir(book_path) if f.lower().endswith(('.json'))]
            if not ocr_files:
                print(f"Warning: No OCRs found for book {book_id}")
                continue
            
            ocr_files.sort(key=lambda x: self._get_page_number(x))
            ocr_paths = [os.path.join(book_path, ocr) for ocr in ocr_files]
            
            filtered_img_paths = []
            filtered_ocr_paths = []
            filtered_labels = []

            for img_path, label, ocr_path in zip(image_paths, page_labels, ocr_paths):
                if not os.path.exists(ocr_path):
                    print(f'Skipping image {img_path} because JSON file does not exist: {ocr_path}')
                    continue
                
                filtered_img_paths.append(img_path)
                filtered_ocr_paths.append(ocr_path)
                filtered_labels.append(label)
            
            book['image_paths'] = filtered_img_paths
            book['page_labels'] = filtered_labels
            book['ocr_paths'] = filtered_ocr_paths
                    
        self.embeddings_cache = {}
        if self.precompute_emb_dir:
            base, ext = os.path.splitext(self.precompute_emb_dir)
            cache_path = f"{base}_textual_emb{ext}"
            
            if os.path.exists(cache_path) and not precompute_emb:
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
                    all_features.append(torch.zeros(self.emb_feature_dim))
            
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
                    all_features.append(torch.zeros(self.emb_feature_dim))
        
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
        item = super().__getitem__(idx)
        book = self.books[idx]
        
        if self.precompute_emb_dir and idx in self.embeddings_cache:
            embeddings = self.embeddings_cache[idx].to(self.device)
        else:
            print(f'Cache not found! Computing embeddings...')
            embeddings = torch.stack(
                self._extract_emb_batch(book['ocr_paths'], self.batch_size) 
            ).to(self.device)
        
        
        seq_length = min(len(embeddings), self.max_seq_length)

        if seq_length < self.max_seq_length:
            padded_embeddings = torch.zeros(self.max_seq_length, self.emb_feature_dim)
            padded_embeddings[:seq_length] = embeddings[:seq_length]
            embeddings = padded_embeddings

        else:
            embeddings = embeddings[:self.max_seq_length]

        
        item['textual_features'] = embeddings
        item['visual_features'] = item['features']
        
        del item['features']
        
        return item