import os
from model import model,tokenizer
from metrics import compute_metrics
os.environ['WANDB_DISABLED'] = 'true'
from transformers import Trainer, TrainingArguments
from dataset import train_dataset,valid_dataset
model_ckpt="distilbert-base-uncased"

bs = 64 # batch size
logging_steps = len(train_dataset) // bs
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=3,             # number of training epochs
                                  learning_rate=2e-5,             # model learning rate
                                  per_device_train_batch_size=bs, # batch size
                                  per_device_eval_batch_size=bs,  # batch size
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False, 
                                  report_to="none",
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level="error")

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=train_dataset,
                  eval_dataset=valid_dataset,
                  tokenizer=tokenizer)

if __name__=="__main__":
    trainer.train()

