import logging
import os
import torch
import yaml
import numpy as np
import pandas as pd
import argparse
from shorten_data_sbert import summarizeIfLongerThan
import textwrap

# argparser = argparse.ArgumentParser()
# argparser.add_argument('--hf_token', type=str, default=os.environ.get('HUGGING_FACE_HUB_TOKEN'))
# argparser.add_argument('--cache_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'cache'))
# args = argparser.parse_args()
hf_token=os.environ.get('HUGGING_FACE_HUB_TOKEN')
cache_dir=os.path.join(os.path.dirname(__file__), 'cache')

os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_MODELS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir


assert os.environ["HUGGING_FACE_HUB_TOKEN"]
assert os.environ["HF_DATASETS_CACHE"]
assert os.environ["HF_MODELS_CACHE"]
assert os.environ["HF_HOME"]


# Set seeds
torch.manual_seed(2023)
np.random.seed(2023)

from ludwig.api import LudwigModel

df=pd.DataFrame(columns=['user_input', 'context', 'output', 'data_idx'])
i=0
model= LudwigModel.load("../animikh/ElevateMind/models/50k-samples-5-epochs/")
print("Hi, how can I help you?")
context_placeholder=""
summ=""
while True:
    print(i)
    if i%2==0:
        input_placeholder = input("usr:")
        # new_row = pd.DataFrame([[input_placeholder, context_placeholder, text, i]], columns=['user_input', 'context', 'output', 'data_idx'])
        # text_test = pd.concat([df, new_row], ignore_index=True)
    else:
        test_text= pd.DataFrame([
        {
            "user_input": input_placeholder,
            "context": summ
        }])
        if input_placeholder=="Bye!":
          response="Bye!"
          print(response)
          context_placeholder += f"usr:{input_placeholder}\nsys:{response}\n"
          summ=summarizeIfLongerThan(context_placeholder)
          new_row = pd.DataFrame([[input_placeholder, summ,response , i]], columns=['user_input', 'context', 'output', 'data_idx'])
          df = pd.concat([df, new_row], ignore_index=True)
          break
        else:
          predictions = model.predict(test_text,output_directory=os.path.join(os.path.dirname(__file__), 'Conversationwithcontext2'))[0]
          input_with_prediction = (test_text['user_input'][0], test_text['context'][0], predictions['output_response'][0])
          print("sys:"+input_with_prediction[2][0])
          context_placeholder += f"usr:{input_with_prediction[0]}\nsys:{input_with_prediction[2][0]}\n"
          summ=summarizeIfLongerThan(context_placeholder)
          #print(summ)
          new_row = pd.DataFrame([[input_with_prediction[0], summ, input_with_prediction[2][0], i]], columns=['user_input', 'context', 'output', 'data_idx'])
          df = pd.concat([df, new_row], ignore_index=True)
    i+=1

path = r'Conversationwithcontext50.txt'
#print(df)
with open(path, 'a') as f:
    df_string = df.to_string(header=True, index=False, justify='left')
    f.write(df_string)

new_df = df[['user_input', 'output']].copy()

# Saving the new DataFrame to a file
new_path = r'Conversationwithcontext50_print.txt'
with open(new_path, 'a') as file:
    df_string_new = new_df.to_string(header=True, index=False, justify='left')
    file.write(df_string_new)