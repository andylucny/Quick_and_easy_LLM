import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from download import download_LaMini_model; download_LaMini_model()

checkpoint = "./LaMini/"  # LaMini-Flan-T5-248M
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device='cpu')
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to('cpu')

def embed_and_encode(input_text):
    # tokenizer
    tokens = tokenizer.encode(input_text, return_tensors="pt")

    # add start
    pad = base_model.config.pad_token_id # 0 
    start = torch.tensor([[pad]])
    input_tokens = torch.concatenate([start,tokens],dim=1)

    # embedding
    embed = base_model.shared(input_tokens) 

    # encode
    def encode(x, mask=None):
        for block in base_model.encoder.block:
            x, mask = block(x, mask)
        return base_model.encoder.final_layer_norm(x)

    # hidden
    hidden = encode(embed) 

    cls = 0 # classification token
    return hidden[0][cls] 

def cosine_similarity(u,v):
    return (u @ v) / (torch.norm(u)*torch.norm(v))
    
texts = [
    "Move with your hand to hit the blue ball!",
    "Move with your hands to hit the ball!",
    "Hit the ball by your hand!",
    "Hit the ball by your hands!",
    "Remove the ball from the table somehow!",
    "Touch your head!",
    "Put yout hand on your head!",
    "Smile!",
    "Express happiness with your face!",
    "Be neutral!",
]

codes = []
for text in texts:
    code = embed_and_encode(text)
    codes.append(code)

for i, codei in enumerate(codes):
    for j, codej in enumerate(codes):
        similarity = cosine_similarity(codei, codej)
        print(f"{similarity:6.3f}",end=" ")
    print()
