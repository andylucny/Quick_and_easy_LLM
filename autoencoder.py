import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from download import download_LaMini_model; download_LaMini_model()

checkpoint = "./LaMini/"  # LaMini-Flan-T5-248M
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device='cpu')
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to('cpu')

input_text = "What is the capital of the USA?"
print("Input:",input_text)

# tokenizer
tokens = tokenizer.encode(input_text, return_tensors="pt") # shape [1, 9]

# add start
pad = base_model.config.pad_token_id # 0 
eos = base_model.config.eos_token_id # 1
start = torch.tensor([[pad]])
input_tokens = torch.concatenate([start,tokens],dim=1) # shape [1, 10]

# embedding
embed = base_model.shared(input_tokens) # shape [1, 10, 768]

# encode
def encode(x, mask=None):
    for block in base_model.encoder.block:
        x, mask = block(x, mask)
    return base_model.encoder.final_layer_norm(x)

# hidden
hidden = encode(embed) # shapes [1, 10, 768], [1, 12, 10, 10]

# decode
def decode(x, mask, crossx, crossmask):
    for block in base_model.decoder.block:
        x, mask, crossmask = block(x, mask, None, crossx, crossmask)
    return base_model.decoder.final_layer_norm(x)

# decode back
output = decode(embed, None, hidden, torch.ones(hidden.shape[:2])) # shape [1, x, 768]
logits = base_model.lm_head(output) # shape [x, 32128]
output_tokens = torch.argmax(logits[:,:,:],dim=2) # 0-32127 

# wiping out
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("Output:",output_text) # ~ 'What is the capital of the USA?'
