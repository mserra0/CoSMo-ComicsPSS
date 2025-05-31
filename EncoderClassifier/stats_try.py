from datasets.pss_dataset_with_stats import PSSDatasetWithStats
import torch
from transformers import SiglipImageProcessor, AutoProcessor, AutoModel
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def try_data(root_dir, model_id, backbone, backbone_name, 
             processor, gpu_id, data_dir,  coco_json):
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Process using device: {device}")
    
    backbone.to(device)
    
    if 'dinov2' in backbone_name:
        feature_dim = backbone.config.hidden_size
    elif 'clip' in backbone_name:
        feature_dim = backbone.config.vision_config.projection_dim
    elif 'siglip' in backbone_name:
        feature_dim = backbone.config.vision_config.hidden_size
    else:
        raise ValueError(f"Warning: Unknown backbone '{backbone_name}'")
    
    print(f'Loaded {backbone_name} with feature dim {feature_dim}')

   
    val_dataset = PSSDatasetWithStats(root_dir=root_dir, 
                                    model_id = model_id,
                                    backbone=backbone,
                                    backbone_name = backbone_name, 
                                    feature_dim = feature_dim,
                                    processor=processor, 
                                    device=device, 
                                    annotations_path=f'{data_dir}/comics_val.json', 
                                    precompute_features=False,
                                    precompute_dir=f'{data_dir}/features_val.pt', 
                                    augment_data=False,
                                    detections_json_path = coco_json,
                                    stats_cache_path = f'{data_dir}/stats_val.pt', 
                                    precompute_stats = True
                                    )
    
    
    if len(val_dataset) > 0:
        print(f'Loaded dataset with {len(val_dataset)} books')
        
        book_item = val_dataset[0]
        book_hash = book_item['book_id']
        features = book_item['features']
        labels = book_item['page_labels']
        stats = book_item['detection_features']
        attention_mask = book_item['attention_mask']
        seq_lengh = sum(attention_mask != 0)
        
        print(f'Loaded book {book_hash} with {len(labels[:seq_lengh])} pages:')
        print(f'Backbone features shape {features.shape}')
        print(f'Detection features shape {stats.shape}')
        
        print('First page BB features: ', features[0])
        print('First page Detection features: ', stats[0])
        
if __name__ == '__main__':
    
    root_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images'
    annotations_dir = '/home/mserrao/PSSComics/Comics/DatasetDCM/comics_all_430.json'
    precompute_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
    checkpoint_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/checkpoints'
    data_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
    out_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/out'
    coco_json = '/home-local/mserrao/PSSComics/multimodal-comic-pss/data/DCM/magi/val.json'
    model_id='openai/clip-vit-large-patch14'
    gpu_id = 3
    
    parts = model_id.split('/')[1].split('-')
    backbone_name = f'{parts[0]}_{parts[-1]}'
    
    print(f"Loading model: {model_id}")
    
    backbone = AutoModel.from_pretrained(model_id).eval()
    
    if 'siglip2' in backbone_name:
        processor = SiglipImageProcessor.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id)


    try_data(root_dir, model_id, backbone, backbone_name,
             processor, gpu_id, data_dir, coco_json)
        
        