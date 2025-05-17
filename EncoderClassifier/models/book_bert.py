import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BookBERT(nn.Module):
    def __init__(self, feature_dim=768, num_hidden_layers = 4, num_attention_heads = 4,
                 positional_embeddings ='absolute', num_classes=4, hidden_dim=256, dropout_p=0.3):
        super(BookBERT, self).__init__()
    
        config = BertConfig(
            num_hidden_layers=num_hidden_layers,
            hidden_size=feature_dim,  
            num_attention_heads=num_attention_heads,  
            intermediate_size=512,
            position_embedding_type = positional_embeddings 
        )
        
        self.bert_encoder = BertModel(config)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, (feature_dim+hidden_dim)//2),
            nn.Linear((feature_dim+hidden_dim)//2,hidden_dim),
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
        bert_output = self.bert_encoder(
                inputs_embeds=features,
                attention_mask=attention_mask
                )
        
        sequence_output = bert_output.last_hidden_state
        
        logits = self.classifier(sequence_output) 
        
        return logits