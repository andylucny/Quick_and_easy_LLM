import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from download import download_LaMini_model; download_LaMini_model()

checkpoint = "./LaMini/"  # LaMini-Flan-T5-248M
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device='cpu')
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to('cpu')

def tare(starting_text, n):
    # tokenizer
    tokens = tokenizer.encode(starting_text, return_tensors="pt")

    # decode
    def decode(x, mask):
        for block in base_model.decoder.block:
            x, mask = block(x, mask, None, None, None) 
        return base_model.decoder.final_layer_norm(x)

    # generate
    pad = base_model.config.pad_token_id # 0 
    eos = base_model.config.eos_token_id # 1
    start = torch.tensor([[pad]])
    output_tokens = torch.concatenate([start,tokens],dim=1)
    for _ in range(n):
        print(".",end="")
        embed = base_model.shared(output_tokens) 
        output = decode(embed, None) 
        logits = base_model.lm_head(output) 
        next_token = torch.argmax(logits[:,-1,:]) # 0-32127 
        output_tokens = torch.concatenate([output_tokens,torch.tensor([[next_token]])],dim=1)
        if next_token == eos:
            break

    # wipe out
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    return output_text

generated_text = tare("We all live in ",30)

print()
print(generated_text)
