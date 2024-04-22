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

    return hidden

def decode_and_wipeout(hidden):
    # decode
    def decode(x, mask, crossx, crossmask):
        for block in base_model.decoder.block:
            x, mask, crossmask = block(x, mask, None, crossx, crossmask)
        return base_model.decoder.final_layer_norm(x)

    # generate
    pad = base_model.config.pad_token_id # 0 
    eos = base_model.config.eos_token_id # 1
    start = torch.tensor([[pad]])
    output_tokens = start
    while True:
        embed = base_model.shared(output_tokens) 
        output = decode(embed, None, hidden, torch.ones(hidden.shape[:2])) 
        logits = base_model.lm_head(output) 
        next_token = torch.argmax(logits[:,-1,:]) # 0-32127 
        output_tokens = torch.concatenate([output_tokens,torch.tensor([[next_token]])],dim=1)
        if next_token == eos:
            break

    # wipe out
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    return output_text

fact = "Nikita's bicycle was stolen in Leningrad."
news = "Ivan Jefimovic won a new car in Moscow."
request = " What happened in Russia?"

hidden1 = embed_and_encode(fact+request)
hidden2 = embed_and_encode(news+request)
n = hidden1.shape[1] # == hidden2.shape[1], texts selected to avoid padding

response1 = decode_and_wipeout(hidden1)
response2 = decode_and_wipeout(hidden2)
response3 = decode_and_wipeout(hidden1*0.1+hidden2*0.9)
response4 = decode_and_wipeout(hidden1*0.3+hidden2*0.7)
response5 = decode_and_wipeout(hidden1*0.93+hidden2*0.07)
response6 = decode_and_wipeout(hidden1*0.9+hidden2*0.1)

print(response1) # == fact
print(response2) # == news
print(response3) # == news
print(response4) # not enough info
print(response5) # == fact
print(response6) # not enough info

# combine the fact and the news
def chimerize(indices):
    hidden = []
    for i, index in enumerate(indices):
        hidden.append((hidden1[0,i] if index == 0 else hidden2[0,i]).unsqueeze(0))
    hidden = torch.concatenate(hidden,dim=0).unsqueeze(0)
    response = decode_and_wipeout(hidden)
    return response 

# random crossing of the fact and the news
print()
for _ in range(40):
    indices = (torch.rand(n)>0.5).int()
    print(chimerize(indices))

# from the fact to the news the gradually
print()
for i in range(n+1):
    indices = torch.zeros(n)
    indices[:i] = 1
    print(chimerize(indices))

# from the news to the fact the gradually
print()
for i in range(n+1):
    indices = torch.ones(n)
    indices[:i] = 0
    print(chimerize(indices))
