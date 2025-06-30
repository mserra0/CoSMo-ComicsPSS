#coding:utf8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from typing import Dict, Optional, List, Union
import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers.utils import is_flash_attn_2_available
import numpy as np
from collections import defaultdict

from sentence_transformers import SentenceTransformer

class Qwen3Embedding():
    def __init__(self, model_name_or_path, instruction=None,  use_fp16: bool = True, use_cuda: bool = True, max_length=8192):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        self.instruction = instruction
        if is_flash_attn_2_available() and use_cuda:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
        if use_cuda:
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side='left')
        self.max_length=max_length
    
    def last_token_pool(self, last_hidden_states: Tensor,
        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        if task_description is None:
            task_description = self.instruction
        return f'Instruct: {task_description}\nQuery:{query}'

    def encode(self, sentences: Union[List[str], str], is_query: bool = False, instruction=None, dim: int = -1):
        if isinstance(sentences, str):
            sentences = [sentences]
        if is_query:
            sentences = [self.get_detailed_instruct(instruction, sent) for sent in sentences]
        inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        inputs.to(self.model.device)
        model_outputs = self.model(**inputs)
        output = self.last_token_pool(model_outputs.last_hidden_state, inputs['attention_mask'])
        if dim != -1:
            output = output[:, :dim]
        output  = F.normalize(output, p=2, dim=1)
        return output

if __name__ == "__main__":
    model_path = "Qwen/Qwen3-Embedding-0.6B"
    model1 = Qwen3Embedding(model_path, use_cuda=True, use_fp16=True)
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    doc_outputs = model1.encode(documents)

    print('doc outputs Method 1', sum(sum(doc_outputs)))

    model2 = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device_map": "auto"},
    tokenizer_kwargs={"padding_side": "left"},
    )
    
    doc_out = model2.encode(documents)
    print('doc embedings Method 2', sum(sum(doc_out)))