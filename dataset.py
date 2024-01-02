import pandas as pd
from datasets import Dataset
from model import tokenise,extract_hidden_states

train_df=pd.read_csv("data path")
test_df=pd.read_csv("data path")
valid_df=pd.read_csv("data path")

train_dataset=Dataset.from_pandas(train_df)
test_dataset=Dataset.from_pandas(test_df)
valid_dataset=Dataset.from_pandas(valid_df)

train_dataset=train_dataset.map(tokenise,batched=True, batch_size=None)
valid_dataset=valid_dataset.map(tokenise,batched=True, batch_size=None)
test_dataset=test_dataset.map(tokenise,batched=True, batch_size=None)

train_dataset=train_dataset.map(extract_hidden_states,batched=True)
test_dataset=test_dataset.map(extract_hidden_states,batched=True)
valid_dataset=valid_dataset.map(extract_hidden_states,batched=True)