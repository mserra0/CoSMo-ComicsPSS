import torch
from PIL import Image
from pss_dataset import PSSDataset
import Comics.Statistics.PSS_analyzer as pssa
import os 
import json
import pandas as pd

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
        detections_json = None,
        stats_cache_path = None, 
        precompute_stats = False
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
        
        self.detection_json = detections_json
        self.stats_cache_path = stats_cache_path
        self.precompute_stats = precompute_stats
        self.stats = {}

        self.CLS_MAPPING = {
            1: 'Panel',
            2: 'Character',
            4: 'Text',
        }
        
        self.detectios = self._load_handcrafted_features()

        if stats_cache_path and os.path.exists(stats_cache_path) and not precompute_stats:
            try:
                print(f'Loading precomputed statistcs from cache {stats_cache_path}...')
                self.stats = torch.load(stats_cache_path, weights_only=True)
                if len(self.stats) !=  len(self.books):
                    print(f"Cache size mismatch: {len(self.stats)} vs {len(self.books)}. Recomputing.")
                    self._precompute_stats(stats_cache_path)
            except Exception as e:
                print(f'Error loading stats cache: {e}. Trying to recompute...')
                self._precompute_stats(stats_cache_path)
        elif precompute_stats:
            self._precompute_stats(stats_cache_path)
    
    def _load_handcrafted_features(self):
        if not self.detection_json or not os.path.exists(self.detection_json):
            print(f'Warning: No valid detection JSON provided: {self.detection_json}')
            return {}
        
        try:
            print(f'Loading handcrafted features from {self.detection_json}')
            data = pssa.analyze_comic_book(self.detection_json, plot=False)
            images_df = data['images']
            annots_df = data['annotations']      
            handcrafted_df = pssa.calculate_advanced_features(annots_df, images_df)      
        
        except Exception as e:
            print(f'Error loading detections from {self.detection_json}: {e}')
            return {}
            
    def _extract_book_stats(self, image_path):
        book_stats = []
        
        pass
            
    
    def _precompute_stats(self, stats_cache_path):
        pass
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['stats'] = self.stats.get(idx)
        return item