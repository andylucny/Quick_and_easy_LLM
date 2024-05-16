import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from download import download_LaMini_model; download_LaMini_model()
from pprint import pprint

checkpoint = "./LaMini/"  # LaMini-Flan-T5-248M
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device='cpu')
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to('cpu')

def embed(input_text):
    # tokenizer
    tokens = tokenizer.encode(input_text, return_tensors="pt")
    
    # add start
    pad = base_model.config.pad_token_id # 0 
    start = torch.tensor([[pad]])
    input_tokens = torch.concatenate([start,tokens],dim=1)

    # embedding
    embed = base_model.shared(input_tokens) 
    
    return input_tokens, embed

text = "Ivan won a car in Moscow"

token_ids, token_embeddings = embed(text)

pprint(token_ids)
print()
#pprint(token_embeddings)
for token in token_embeddings[0]:
    print(f"{token[0]:.2f}, {token[1]:.2f}, {token[2]:.2f}, ..., {token[-1]:.2f}")


