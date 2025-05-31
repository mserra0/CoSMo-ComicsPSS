import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BookBERT(nn.Module):
    def __init__(self, feature_dim=768, bert_input=768, num_hidden_layers = 4, num_attention_heads = 4,
                 positional_embeddings ='absolute', num_classes=4, hidden_dim=256, dropout_p=0.3):
        super(BookBERT, self).__init__()
    
        config = BertConfig(
            num_hidden_layers=num_hidden_layers,
            hidden_size=bert_input,  
            num_attention_heads=num_attention_heads,  
            intermediate_size=4*bert_input,
            position_embedding_type = positional_embeddings 
        )
        
        self.projection = nn.Linear(feature_dim, bert_input)
        
        self.bert_encoder = BertModel(config)
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_input, (bert_input+hidden_dim)//2),
            nn.Linear((bert_input+hidden_dim)//2,hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.GELU(), 
            nn.Dropout(dropout_p),  
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.LayerNorm(hidden_dim // 2), 
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, num_classes) 
        )
    
    def forward(self, features, attention_mask):
        
        projected_features = self.projection(features)

        bert_output = self.bert_encoder(
                inputs_embeds=projected_features,
                attention_mask=attention_mask
                )
        
        sequence_output = bert_output.last_hidden_state
        
        logits = self.classifier(sequence_output) 
        
        return logits
    
class BookBERT2(nn.Module):
    def __init__(self, feature_dim=768, bert_input = 768, num_hidden_layers = 4, num_attention_heads = 4,
                 positional_embeddings ='absolute', num_classes=4, hidden_dim=256, dropout_p=0.3,
                 detections_feature_dim = 20):
        super(BookBERT2, self).__init__()
    
        self.feature_dim = feature_dim
        self.detections_feature_dim = detections_feature_dim
        
        self.detection_projection = nn.Linear(self.detections_feature_dim, self.feature_dim)
        
        self.detection_batch_norm = nn.BatchNorm1d(self.detections_feature_dim)
        self.image_feature_batch_norm = nn.BatchNorm1d(self.feature_dim)

        combined_input_dim = self.feature_dim * 2
        
        self.projection = nn.Linear(combined_input_dim, bert_input)
        
        config = BertConfig(
            num_hidden_layers=num_hidden_layers,
            hidden_size=bert_input,  
            num_attention_heads=num_attention_heads,  
            intermediate_size=4*bert_input,
            position_embedding_type = positional_embeddings 
        )
        
        self.bert_encoder = BertModel(config)
        
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_input, (bert_input+hidden_dim)//2),
            nn.Linear((bert_input+hidden_dim)//2,hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.GELU(), 
            nn.Dropout(dropout_p),  
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.LayerNorm(hidden_dim // 2), 
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, num_classes) 
        )
    
    def forward(self, features, attention_mask, detection_features):
        
        batch_size, seq_len, d_feat_dim = detection_features.shape
        _, _, img_feat_dim = features.shape
        
        # normalizing detection features
        norm_input = detection_features.reshape(batch_size * seq_len, d_feat_dim)
        normalized_detection_features = self.detection_batch_norm(norm_input)
        normalized_detection_features = normalized_detection_features.reshape(batch_size, seq_len, d_feat_dim)
        
        projected_detection_features = self.detection_projection(normalized_detection_features)
        
        # normalizing backbone features
        norm_input_image = features.reshape(batch_size * seq_len, img_feat_dim)
        normalized_image_features = self.image_feature_batch_norm(norm_input_image)
        normalized_image_features = normalized_image_features.reshape(batch_size, seq_len, img_feat_dim)
        
        combined_features = torch.cat((normalized_image_features, projected_detection_features), dim=-1)
        
        projected_features = self.projection(combined_features)
        
        bert_output = self.bert_encoder(
                inputs_embeds=projected_features,
                attention_mask=attention_mask
                )
        
        sequence_output = bert_output.last_hidden_state
        
        logits = self.classifier(sequence_output) 
        
        return logits
    
class BookBERTfusion(nn.Module):
    def __init__(self, backbones, num_hidden_layers = 4, num_attention_heads = 4,
                 positional_embeddings ='absolute', num_classes=4, hidden_dim=256, dropout_p=0.3,
                 projection_dim = 1024, bert_input_dim = 768):
        super(BookBERTfusion, self).__init__()
    
        self.backbones = backbones
        
        self.batch_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        
        for name, dim in self.backbones.items():
            self.batch_norms[name] = nn.BatchNorm1d(dim)
            self.projections[name] = nn.Linear(dim, projection_dim)
        
        concatenated_dim = len(self.backbones) * projection_dim
        intermidiate_fusion_dim = (concatenated_dim + bert_input_dim) // 2
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concatenated_dim, intermidiate_fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(intermidiate_fusion_dim),
            nn.Dropout(dropout_p),
            nn.Linear(intermidiate_fusion_dim, bert_input_dim),
            nn.LayerNorm(bert_input_dim), 
            nn.ReLU() 
        )
        
        config = BertConfig(
            num_hidden_layers=num_hidden_layers,
            hidden_size=bert_input_dim,  
            num_attention_heads=num_attention_heads,  
            intermediate_size=4*bert_input_dim,
            position_embedding_type = positional_embeddings 
        )
        
        self.bert_encoder = BertModel(config)
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_input_dim, (bert_input_dim+hidden_dim)//2),
            nn.Linear((bert_input_dim+hidden_dim)//2,hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.GELU(), 
            nn.Dropout(dropout_p),  
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.LayerNorm(hidden_dim // 2), 
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, num_classes) 
        )
    
    def forward(self, fusion_features, attention_mask):
        projected_features_list = []
        
        for bb_name, bb_featrues in fusion_features.items():
            batch_size, seq_len, feat_dim = bb_featrues.shape
            
            norm_input = bb_featrues.reshape(batch_size * seq_len, feat_dim)
            normalized_features = self.batch_norms[bb_name](norm_input)
            normalized_features = normalized_features.reshape(batch_size, seq_len, feat_dim)
            
            projected_features = self.projections[bb_name](normalized_features)
            projected_features_list.append(projected_features)
            
        concat_features = torch.cat(projected_features_list, dim=-1)
        
        fused_features = self.fusion_mlp(concat_features)
        
        bert_output = self.bert_encoder(
                inputs_embeds=fused_features,
                attention_mask=attention_mask
                )
        
        sequence_output = bert_output.last_hidden_state
        
        logits = self.classifier(sequence_output) 
        
        return logits