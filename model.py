import torch
import numpy as np
from transformers import AutoModel,AutoTokenizer,AutoModelForSequenceClassification


num_labels=6
device="cuda" if torch.cuda.is_available() else "cpu"
model_ckpt = "distilbert-base-uncased"

tokenizer=AutoTokenizer.from_pretrained(model_ckpt)
distilbert_model = AutoModel.from_pretrained(model_ckpt).to(device)
model=AutoModelForSequenceClassification.from_pretrained(model_ckpt,num_labels=num_labels).to(device)

def tokenise(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


def extract_hidden_states(batch):
    
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    
    with torch.no_grad():
        last_hidden_state = distilbert_model(**inputs).last_hidden_state
        
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}