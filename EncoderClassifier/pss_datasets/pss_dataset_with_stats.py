import torch
from .pss_dataset import PSSDataset
import os 
import json
import numpy as np
from pathlib import Path

class PSSDatasetWithStats(PSSDataset):
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
        #  --- Augmentation Parameters --- 
        augment_data=False,            
        num_augmented_copies=1,      
        augmentation_shuffle_stories=True, 
        removal_p = 0.05,
        num_synthetic_books=0,       
        min_stories = 3,
        max_stories = 5,
        synthetic_remove_p = 0.15,
        #  --- Statistics Parameters --- 
        detections_json_path = None,
        stats_cache_path = None, 
        precompute_detection_features = False
    ):
        super().__init__(
            root_dir=root_dir,
            model_id=model_id,
            backbone=backbone,
            backbone_name=backbone_name,
            feature_dim=feature_dim,
            processor=processor,
            annotations_path=annotations_path,
            max_seq_length=max_seq_length,
            device=device,
            precompute_features=precompute_features,
            precompute_dir=precompute_dir,
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
        self.detections_json_path = detections_json_path
        self.detection_features_cache_path = stats_cache_path
        self.precompute_detection_features = precompute_detection_features
        self.detection_features = {}

        self.CLS_MAPPING = {
            1: 'panel',
            2: 'character',
            4: 'Text',
        }
        
        self.PANEL_CLS_NAME = 'panel'
        self.CHARACTER_CLS_NAME = 'character'
        self.TEXT_CLS_NAME = 'Text'
        
        self.detection_features_cols = [
            # panel-Related
            'panel_bbox_max_dim_max', 'panel_bbox_max_dim_mean', 'panel_area_max', 'panel_area_mean', 'panel_bbox_count',
            'max_panel_centroid_x', 'max_panel_centroid_y', 'panel_coverage',
            # character-Related
            'character_bbox_max_dim_max', 'character_bbox_max_dim_mean', 'character_bbox_count',
            'total_character_area', 'character_to_text_ratio',
            # Text-Related
            'text_bbox_max_dim_max', 'text_bbox_max_dim_mean', 'Text_bbox_count',
            'total_text_area', 'text_to_panel_ratio', 'max_text_centroid_x', 'max_text_centroid_y',
        ]
        
        self.idx_cols = [
            # Page-Level identifiers 
            'image_id', 
            'file_name' 
        ]
        self.detection_feature_dim = len(self.detection_features_cols)
        self.coco_images_by_filename, self.coco_detections_by_img_id = self._load_coco()

        if stats_cache_path and os.path.exists(stats_cache_path) and not precompute_detection_features:
            try:
                print(f'Loading precomputed statistcs from cache {stats_cache_path}...')
                self.detection_features = torch.load(stats_cache_path, weights_only=True)
                if len(self.detection_features) !=  len(self.books):
                    print(f"Cache size mismatch: {len(self.detection_features)} vs {len(self.books)}. Recomputing.")
                    self._precompute_stats(stats_cache_path)
            except Exception as e:
                print(f'Error loading stats cache: {e}. Trying to recompute...')
                self._precompute_stats(stats_cache_path)
        elif precompute_detection_features:
            self._precompute_stats(stats_cache_path)
    
    def _load_coco(self):
        if not self.detections_json_path or not os.path.exists(self.detections_json_path):
            print(f"Detection JSON file not found or not specified: {self.detections_json_path}")
            return {}, {}

        print(f"Loading COCO detection data from: {self.detections_json_path}")
        with open(self.detections_json_path, 'r') as f:
            coco_data = json.load(f)
            
        images_data_by_filename = {}
        for img_info in coco_data.get('images', []):
            images_data_by_filename[img_info['file_name']] = {
                'id': img_info['id'],
                'width': img_info['width'],
                'height': img_info['height']
            }

        annotations_by_image_id = {} 
        for ann_info in coco_data.get('annotations', []):
            img_id = ann_info['image_id']
            if img_id not in annotations_by_image_id:
                annotations_by_image_id[img_id] = []
            
            bbox = ann_info.get('bbox')
            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox):
                 annotations_by_image_id[img_id].append(ann_info)
            else:
                print(f"Warning: Malformed bbox for ann_id {ann_info.get('id')} on image_id {img_id}. Skipping annotation.")

        print(f"Loaded {len(images_data_by_filename)} image entries and annotations for {len(annotations_by_image_id)} images.")
        return images_data_by_filename, annotations_by_image_id
            
    def _extract_page_coco_features(self, page_annots, page_width, page_height):
        panel_bboxes_data = {'areas': [], 'max_dims': [], 'centroids_x': [], 'centroids_y': []}
        character_bboxes_data = {'areas': [], 'max_dims': []}
        text_bboxes_data = {'areas': [], 'max_dims': [], 'centroids_x': [], 'centroids_y': []}
        
        page_area = float(page_width * page_height)
        
        for annot in page_annots:
            cat_id = annot['category_id']
            cls_name = self.CLS_MAPPING.get(cat_id)
            bbox = annot['bbox'] 
            bbox_area = float(annot['area'])
            
            w, h = float(bbox[2]), float(bbox[3])
            if w <= 0 or h <= 0: 
                print('Skipping invalid BBOX for this image')
                continue
            
            max_dim = max(w, h)
            centroid_x = float(bbox[0]) + w / 2
            centroid_y = float(bbox[1]) + h / 2

            if cls_name == self.PANEL_CLS_NAME:
                panel_bboxes_data['areas'].append(bbox_area)
                panel_bboxes_data['max_dims'].append(max_dim)
                panel_bboxes_data['centroids_x'].append(centroid_x)
                panel_bboxes_data['centroids_y'].append(centroid_y)
            elif cls_name == self.CHARACTER_CLS_NAME:
                character_bboxes_data['areas'].append(bbox_area)
                character_bboxes_data['max_dims'].append(max_dim)
            elif cls_name == self.TEXT_CLS_NAME:
                text_bboxes_data['areas'].append(bbox_area)
                text_bboxes_data['max_dims'].append(max_dim)
                text_bboxes_data['centroids_x'].append(centroid_x)
                text_bboxes_data['centroids_y'].append(centroid_y)
            else:
                print(f'Unexpected class found {cls_name}')
                
        # Panel Features
        panel_bbox_count = len(panel_bboxes_data['areas'])
        panel_bbox_max_dim_max = np.max(panel_bboxes_data['max_dims']) if panel_bboxes_data['max_dims'] else 0.0
        panel_bbox_max_dim_mean = np.mean(panel_bboxes_data['max_dims']) if panel_bboxes_data['max_dims'] else 0.0
        panel_area_max = np.max(panel_bboxes_data['areas']) if panel_bboxes_data['areas'] else 0.0
        panel_area_mean = np.mean(panel_bboxes_data['areas']) if panel_bboxes_data['areas'] else 0.0
        max_panel_centroid_x = np.max(panel_bboxes_data['centroids_x']) if panel_bboxes_data['centroids_x'] else 0.0
        max_panel_centroid_y = np.max(panel_bboxes_data['centroids_y']) if panel_bboxes_data['centroids_y'] else 0.0
        total_panel_area = np.sum(panel_bboxes_data['areas'])
        panel_coverage = total_panel_area / page_area if page_area > 0 else 0.0

        # Character Features
        character_bbox_count = len(character_bboxes_data['areas'])
        character_bbox_max_dim_max = np.max(character_bboxes_data['max_dims']) if character_bboxes_data['max_dims'] else 0.0
        character_bbox_max_dim_mean = np.mean(character_bboxes_data['max_dims']) if character_bboxes_data['max_dims'] else 0.0
        total_character_area = np.sum(character_bboxes_data['areas'])

        # Text Features
        text_bbox_count = len(text_bboxes_data['areas'])
        text_bbox_max_dim_max = np.max(text_bboxes_data['max_dims']) if text_bboxes_data['max_dims'] else 0.0
        text_bbox_max_dim_mean = np.mean(text_bboxes_data['max_dims']) if text_bboxes_data['max_dims'] else 0.0 
        total_text_area = np.sum(text_bboxes_data['areas'])
        max_text_centroid_x = np.max(text_bboxes_data['centroids_x']) if text_bboxes_data['centroids_x'] else 0.0
        max_text_centroid_y = np.max(text_bboxes_data['centroids_y']) if text_bboxes_data['centroids_y'] else 0.0

        # Ratios
        character_to_text_ratio = total_character_area / total_text_area if total_text_area > 0 else 0.0
        text_to_panel_ratio = total_text_area / total_panel_area if total_panel_area > 0 else 0.0
        
        feature_vector = np.array([
            panel_bbox_max_dim_max, panel_bbox_max_dim_mean, panel_area_max, panel_area_mean, panel_bbox_count,
            max_panel_centroid_x, max_panel_centroid_y, panel_coverage,
            character_bbox_max_dim_max, character_bbox_max_dim_mean, character_bbox_count,
            total_character_area, character_to_text_ratio,
            text_bbox_max_dim_max, text_bbox_max_dim_mean, text_bbox_count,
            total_text_area, text_to_panel_ratio, max_text_centroid_x, max_text_centroid_y,
        ], dtype=np.float32)

        return feature_vector
            
    def _precompute_stats(self, stats_cache_path):
        print('Starting precomputation of handcrafted features...')
        self.detection_features = {} 
        
        if not self.coco_detections_by_img_id or not self.coco_images_by_filename:
            print('COCO detections data not loaded!')
            return
        
        print(f'Example of filname as key for COCO images: {next(iter(self.coco_images_by_filename))}')
   
        for book_idx, book in enumerate(self.books):
            book_hash = book['book_id']
            book_pages = book['image_paths']
            book_page_features = []
            
            for page_pth in book_pages:
                full_pth = Path(page_pth)
                short_pth = str(Path(*full_pth.parts[-2:]))
                
                coco_image_info = self.coco_images_by_filename.get(short_pth, [])
                
                if not os.path.exists(page_pth):
                    print(f'Path {page_pth} does not exist because')
                    
                if coco_image_info:
                    image_id = coco_image_info['id']
                    page_width = coco_image_info['width']
                    page_height = coco_image_info['height']
                    page_annots = self.coco_detections_by_img_id.get(image_id, [])
                    if page_annots:
                        features_vector = self._extract_page_coco_features(page_annots, page_width, page_height)
                    else:
                        print(f'Warning: No detetctions for image ID {image_id} located at {page_pth}.')
                        features_vector = np.zeros(self.detection_feature_dim, dtype=np.float32)
                else:
                    print(f'No annotations for {page_pth} or path not found!')
                    features_vector = np.zeros(self.detection_feature_dim, dtype=np.float32)
                    
                book_page_features.append(features_vector)
            
            if book_page_features:
                self.detection_features[book_hash] = np.stack(book_page_features)
            else:
                 
                num_pages_in_book = len(book.get('image_paths', []))
                self.detection_features[book_hash] = np.zeros((num_pages_in_book, self.detection_feature_dim), dtype=np.float32)

        if stats_cache_path:
            print(f"Saving precomputed statistics to {stats_cache_path}...")
            try:
                torch.save(self.detection_features, stats_cache_path)
                print("Statistics saved successfully.")
            except Exception as e:
                print(f"Error saving statistics to cache: {e}")
            
            print("Precomputation of handcrafted statistics finished.")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        book = self.books[idx]
        book_hash = item['book_id']
        
        book_detections_features = []
        if book_hash in self.detection_features:
            book_detections_features = self.detection_features[book_hash]
            if len(book_detections_features) != len(book['image_paths']):
                print(f'Size Missmatch for book {book_hash}!')
        else:
            print(f'No annotations found for {book_hash}')
        
        book_detections_features = torch.from_numpy(book_detections_features).float()     
        seq_length = min(len(book_detections_features), self.max_seq_length)
        attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
        
        if seq_length < self.max_seq_length:
            attention_mask[seq_length:] = 0
            
            padded_detection_features = torch.zeros(self.max_seq_length, self.detection_feature_dim)
            padded_detection_features[:seq_length] = book_detections_features[:seq_length]
            book_detections_features = padded_detection_features
            
        else:
            book_detections_features = book_detections_features[:self.max_seq_length]
        
        item['detection_features'] = book_detections_features
        
        return item