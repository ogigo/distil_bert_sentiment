from transformers import AutoTokenizer,AutoModelForSequenceClassification,pipeline
import torch

model_ckpt = "distilbert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_ckpt)
best_model=AutoModelForSequenceClassification.from_pretrained("kajol/distil_bert_sentiment")

distil_classifier = pipeline("text-classification", model="kajol/distil_bert_sentiment")

new_data = 'I watched a movie last night, it was quite brilliant'
preds = distil_classifier(new_data, return_all_scores=True)

def predict_hate_speech(text):
    predictions=[]
    
    emotion={0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}

    #tokenizer(text, padding=True, truncation=True)
    encode_text=tokenizer.encode_plus(text,
                                      add_special_tokens=True,
                                      max_length=128,
                                      truncation=True,
                                      padding="max_length",
                                      return_tensors="pt")
    
    ids=encode_text["input_ids"].squeeze(1)
    mask=encode_text["attention_mask"].squeeze(1)
    
    with torch.no_grad():
        out=best_model(ids,mask)
        logits=out["logits"]
        pred=torch.argmax(logits,dim=-1)
        predictions.extend(pred)
        
    ensemble_predictions = torch.tensor(predictions)
    
    prediction=ensemble_predictions.float().mean().round().long().tolist()
    
    result_speech=emotion[prediction]
    
    return result_speech

if __name__=="__main__":
    result=predict_hate_speech(new_data)
    print(result)