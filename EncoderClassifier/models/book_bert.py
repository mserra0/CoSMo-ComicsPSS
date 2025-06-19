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
            position_embedding_type = positional_embeddings,
            max_position_embeddings = 1024
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
            nn.GELU(),
            nn.LayerNorm(intermidiate_fusion_dim),
            nn.Dropout(dropout_p),
            nn.Linear(intermidiate_fusion_dim, bert_input_dim),
            nn.LayerNorm(bert_input_dim), 
        )
        
        config = BertConfig(
            num_hidden_layers=num_hidden_layers,
            hidden_size=bert_input_dim,  
            num_attention_heads=num_attention_heads,  
            intermediate_size=4*bert_input_dim,
            position_embedding_type = positional_embeddings, 
            max_position_embeddings=2048
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
    
class BookBERTMultimodal(BookBERT):
    def __init__(self, 
                 textual_feature_dim,
                 visual_feature_dim=768, 
                 bert_input_dim=768,
                 projection_dim = 1024,
                 num_hidden_layers=4,
                 num_attention_heads=4,
                 positional_embeddings='absolute',
                 num_classes=4,
                 hidden_dim=256,
                 dropout_p=0.3
    ):
        super(BookBERTMultimodal, self).__init__(
            feature_dim=visual_feature_dim,
            bert_input=bert_input_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            positional_embeddings=positional_embeddings,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p
        )
        
        self.textual_feature_dim = textual_feature_dim
        self.visual_feature_dim = visual_feature_dim
        self.projection_dim = projection_dim
        
        self.visual_projection = nn.Linear(self.visual_feature_dim, self.projection_dim)
        self.textual_projection = nn.Linear(self.textual_feature_dim, self.projection_dim)
        
        self.norm = nn.LayerNorm(self.projection_dim)

        concat_dim = 2 * self.projection_dim 
        hidden_dims = [concat_dim * 2, concat_dim // 2, bert_input_dim]
        
        self.fusionMLP = nn.Sequential(
            nn.Linear(concat_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),  
            nn.GELU(), 
            nn.Dropout(dropout_p),  
            nn.Linear(hidden_dims[0], hidden_dims[1]), 
            nn.LayerNorm(hidden_dims[1]), 
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dims[1], hidden_dims[2]) 
        )
        
    def forward(self, textual_features, visual_features, attention_mask):
        
        projected_txt_features = self.textual_projection(textual_features)
        projected_vis_features = self.visual_projection(visual_features)
        
        norm_projected_txt_features = self.norm(projected_txt_features)
        norm_projected_vis_features = self.norm(projected_vis_features)
        
        concat_features = torch.cat([norm_projected_txt_features, norm_projected_vis_features], dim=-1)
        
        fused_features = self.fusionMLP(concat_features)

        bert_output = self.bert_encoder(
                inputs_embeds=fused_features,
                attention_mask=attention_mask
                )
        
        sequence_output = bert_output.last_hidden_state
        
        logits = self.classifier(sequence_output) 
        
        return logits
    
class BookBERTMultiVision(BookBERTfusion):
    def __init__(self, backbones, num_hidden_layers=4, num_attention_heads=4, positional_embeddings='absolute', 
                 num_classes=5, hidden_dim=256, dropout_p=0.3, projection_dim=1024, bert_input_dim=768, max_seq_length = 2048):
        super(BookBERTMultiVision, self).__init__(
            backbones = backbones, num_hidden_layers = num_hidden_layers, num_attention_heads = num_attention_heads,
            positional_embeddings =positional_embeddings, num_classes=num_classes, hidden_dim=hidden_dim, dropout_p=dropout_p,
            projection_dim = 1024, bert_input_dim = bert_input_dim
        )
        self.dropout_p = dropout_p
        self.bert_input_dim = bert_input_dim
        self.projections = nn.ModuleDict()
        for name, dim in self.backbones.items():
            self.projections[name] = self.projector(dim)
            
        self.learnable_tokens_emb =nn.Embedding(512, bert_input_dim)
        self.register_buffer('token_indices', torch.arange(512))
        
    def projector(self, feature_dim):    
        intermidate_size1 = (feature_dim + self.bert_input_dim) * 2 
        intermidate_size2 = intermidate_size1 // 2
        
        projection = nn.Sequential(
            nn.Linear(feature_dim, intermidate_size1),
            nn.LayerNorm(intermidate_size1),  
            nn.GELU(), 
            nn.Dropout(self.dropout_p),  
            nn.Linear(intermidate_size1, intermidate_size2), 
            nn.LayerNorm(intermidate_size2), 
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(intermidate_size2, self.bert_input_dim) 
        )
        
        return projection
    
    def forward(self, fusion_features, attention_mask):
        backbone_count = len(fusion_features)
        batch_size, seq_len = attention_mask.shape
        processed_features = []
        mask = attention_mask.bool()

        for bb_name, bb_features in fusion_features.items():
            valid_features = bb_features[mask]
            normalized = torch.zeros_like(bb_features)

            if valid_features.numel() > 0:
                normalized_valid = self.batch_norms[bb_name](valid_features)
                normalized[mask] = normalized_valid
                
            projected = self.projections[bb_name](normalized)
            processed_features.append(projected)
        
        token_indices_batch = self.token_indices.unsqueeze(0).expand(batch_size, -1)
        learnable_tokens = self.learnable_tokens_emb(token_indices_batch)
        
        processed_features.append(learnable_tokens)
        tokens_count = backbone_count+1
        
        stacked_features = torch.stack(processed_features, dim=2)
        combined_features = stacked_features.view(batch_size, seq_len * tokens_count, -1)

        expanded_mask = attention_mask.unsqueeze(2).expand(-1, -1, tokens_count)
        expanded_mask = expanded_mask.reshape(batch_size, seq_len * tokens_count)
        
        bert_output = self.bert_encoder(
            inputs_embeds=combined_features,
            attention_mask=expanded_mask
        )
        
        sequence_output = bert_output.last_hidden_state
        
        reshaped_output = sequence_output.view(batch_size, seq_len, tokens_count, -1)

        last_modality_features = reshaped_output[:, :, -1, :]
        
        logits = self.classifier(last_modality_features)
        
        return logits
    
    
class BookBERTMultimodal2(BookBERT):
    def __init__(self, 
                 textual_feature_dim,
                 visual_feature_dim=768, 
                 bert_input_dim=768,
                 projection_dim = 1024,
                 num_hidden_layers=4,
                 num_attention_heads=4,
                 positional_embeddings='absolute',
                 num_classes=4,
                 hidden_dim=256,
                 dropout_p=0.3
    ):
        self.dropout_p = dropout_p
        self.textual_feature_dim = textual_feature_dim
        self.visual_feature_dim = visual_feature_dim
        self.bert_input_dim = bert_input_dim
        self.projection_dim = projection_dim
        
        super(BookBERTMultimodal2, self).__init__(
            feature_dim=visual_feature_dim,
            bert_input=bert_input_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            positional_embeddings=positional_embeddings,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p
        )


        intermidate_size1 = (self.visual_feature_dim + bert_input_dim) * 2
        intermidate_size2 = intermidate_size1 // 2
        
        
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_dim, intermidate_size1),
            nn.LayerNorm(intermidate_size1),  
            nn.GELU(), 
            nn.Dropout(self.dropout_p),  
            nn.Linear(intermidate_size1, intermidate_size2), 
            nn.LayerNorm(intermidate_size2), 
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(intermidate_size2, self.bert_input_dim) 
        )
        
        intermidate_size1 = (self.textual_feature_dim + bert_input_dim) * 2
        intermidate_size2 = intermidate_size1 // 2
        
        self.textual_projection = nn.Sequential(
            nn.Linear(self.textual_feature_dim, intermidate_size1),
            nn.LayerNorm(intermidate_size1),  
            nn.GELU(), 
            nn.Dropout(self.dropout_p),  
            nn.Linear(intermidate_size1, intermidate_size2), 
            nn.LayerNorm(intermidate_size2), 
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(intermidate_size2, self.bert_input_dim) 
        )
        
        self.norm = nn.LayerNorm(self.bert_input_dim)
        
    def forward(self, textual_features, visual_features, attention_mask):
        batch_size, seq_len, feat_dim = textual_features.shape
        
        projected_txt_features = self.textual_projection(textual_features)
        projected_vis_features = self.visual_projection(visual_features)
        
        norm_projected_txt_features = self.norm(projected_txt_features)
        norm_projected_vis_features = self.norm(projected_vis_features)
        
        processed_feat = [norm_projected_txt_features, norm_projected_vis_features]
        
        stacked_features = torch.stack(processed_feat, dim=2)
        
        combined_features = stacked_features.view(batch_size, seq_len * 2, -1)
        
        expanded_mask = attention_mask.unsqueeze(2).expand(-1, -1, 2)
        expanded_mask = expanded_mask.reshape(batch_size, seq_len * 2)

        bert_output = self.bert_encoder(
                inputs_embeds=combined_features,
                attention_mask=expanded_mask
                )
        
        sequence_output = bert_output.last_hidden_state
        
        reshaped_output = sequence_output.view(batch_size, seq_len, 2, -1)
        
        last_token_features = reshaped_output[:, :, -1, :]
        
        logits = self.classifier(last_token_features)
        
        return logits