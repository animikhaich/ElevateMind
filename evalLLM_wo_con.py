import logging
import os
import torch
import yaml
import numpy as np
import pandas as pd
import argparse
import shutil
from shorten_data_sbert import summarizeIfLongerThan

argparser = argparse.ArgumentParser()

argparser.add_argument('--data_path', type=str, default='../animikh/ElevateMind/data/conversations_summarized_25K.csv')
argparser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
argparser.add_argument('--lr', type=float, default=0.0004)
argparser.add_argument('--batch_size', type=int, default=1)
argparser.add_argument('--epochs', type=int, default=3)
argparser.add_argument('--grad_acc_steps', type=int, default=16)
argparser.add_argument('--seed', type=int, default=2023)
argparser.add_argument('--temperature', type=float, default=0.1)
argparser.add_argument('--gen_seq_len', type=int, default=512)
argparser.add_argument('--input_seq_len', type=int, default=512)
argparser.add_argument('--cap_dsize', type=int, default=None)
argparser.add_argument('--save_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'models/LLMmodel'))
argparser.add_argument('--hf_token', type=str, default=os.environ.get('HUGGING_FACE_HUB_TOKEN'))
argparser.add_argument('--cache_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'cache'))

args = argparser.parse_args()

os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
os.environ["HF_MODELS_CACHE"] = args.cache_dir
os.environ["HF_HOME"] = args.cache_dir


assert os.environ["HUGGING_FACE_HUB_TOKEN"]
assert os.environ["HF_DATASETS_CACHE"]
assert os.environ["HF_MODELS_CACHE"]
assert os.environ["HF_HOME"]


# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

from ludwig.api import LudwigModel

# Load data
df = pd.read_csv(args.data_path)
df = df.where(df.notna(), "") # Replace NaN with empty string

dsize = len(df)

# Cap dataset size
if args.cap_dsize is not None and args.cap_dsize > 0:
    df = df.sample(args.cap_dsize)

#print(f"Dataset size reduced from: {dsize} -> {len(df)}")


# Shuffle and Split data into train, val and test: 80%, 10%, 10%
total_rows = len(df)
split_0_count = int(total_rows * 0.8)
split_1_count = int(total_rows * 0.10)
split_2_count = total_rows - split_0_count - split_1_count

#print(f"Counts for train, val and test: {split_0_count}, {split_1_count}, {split_2_count}")

split_values = np.concatenate([
    np.zeros(split_0_count),
    np.ones(split_1_count),
    np.full(split_2_count, 2)
])
np.random.shuffle(split_values)
df['split'] = split_values
df['split'] = df['split'].astype(int)

#print(df.head(20))


# You are an expert who will summarize the following conversation between a 'usr' and a 'sys' 
# without losing information and without adding new information.

# Train model
# df = pd.DataFrame([
#       {
#             "user_input": "Hi, I am really upset I am not doing good in my career.",
#             "context": "",
#       },
#       {
#             "user_input": "Hi, I am really upset I am not doing good in my career.",
#             "context": "",
#       },
#       {
#             "user_input": "Hi, I am really upset I am not doing good in my career.",
#             "context": "",
#       },
# ])


input_placeholder = '{user_input}'
context_placeholder = '{context}'

zero_shot_config = yaml.safe_load(
f"""
model_type: llm
base_model: meta-llama/Llama-2-7b-hf

input_features:
  - name: user_input
    type: text

output_features:
  - name: output
    type: text

prompt:
  template: >-
    As an expert caregiver, provide supportive positive feedback based on prior conversations with the user ('usr') and your previous responses ('sys') to address their concerns.

    ### Input: {input_placeholder}

    ### Context: {context_placeholder}

    ### Response:

generation:
  temperature: {args.temperature}
  max_new_tokens: {args.gen_seq_len}

quantization:
  bits: 4

preprocessing:
  global_max_sequence_length: {args.input_seq_len}
  split:
    type: fixed
"""
)

model = LudwigModel(config=zero_shot_config, logging_level=logging.INFO)
results = model.train(dataset=df[:10])

test_set=pd.DataFrame([
      {
            "user_input": "Hi, I am really upset I am not doing good in my career.",
            "context": "",
      },
])

# preds, _ = model.predict(test_set, skip_save_predictions=False)
# print(type(preds['output_response']))
# print(preds['output_response'].iloc[0][0].split("\n")[0])
# print(preds['output_response'].split("\n")[1])
# path = r'ConversationLLMwithcontext1.txt'
# #print(df)
# with open(path, 'a') as f:
#     df_string = df.to_string(header=True, index=False, justify='left')
#     f.write(df_string)

# new_df = df[['user_input', 'output']].copy()

# # Saving the new DataFrame to a file
# new_path = r'ConversationLLMwithcontext1_print.txt'
# with open(new_path, 'a') as file:
#     df_string_new = new_df.to_string(header=True, index=False, justify='left')
#     file.write(df_string_new)

df_LLM=pd.DataFrame(columns=['user_input', 'context', 'output', 'data_idx'])
i=0
# model= LudwigModel.load("../animikh/ElevateMind/models/25k-samples-3-epochs/")
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
        #   context_placeholder += f"usr:{input_placeholder}\nsys:{response}\n"
        #   summ=summarizeIfLongerThan(context_placeholder)
          new_row = pd.DataFrame([[input_placeholder, summ,response , i]], columns=['user_input', 'context', 'output', 'data_idx'])
          df_LLM = pd.concat([df_LLM, new_row], ignore_index=True)
          break
        else:
          #predictions = model.predict(test_text)[0]
          preds, _ = model.predict(test_set, skip_save_predictions=False)
          input_with_prediction = (test_text['user_input'][0], test_text['context'][0], preds['output_response'].iloc[0][0].split("\n")[0])
          print("sys:"+input_with_prediction[2])
        #   context_placeholder += f"usr:{input_with_prediction[0]}\nsys:{input_with_prediction[2]}\n"
        #   summ=summarizeIfLongerThan(context_placeholder)
          #print(summ)
          new_row = pd.DataFrame([[input_with_prediction[0], summ, input_with_prediction[2], i]], columns=['user_input', 'context', 'output', 'data_idx'])
          df_LLM = pd.concat([df_LLM, new_row], ignore_index=True)
    i+=1

path = r'ConversationLLMwithoutcontext1.txt'
#print(df)
with open(path, 'a') as f:
    df_string = df_LLM.to_string(header=True, index=False, justify='left')
    f.write(df_string)

new_df = df_LLM[['user_input', 'output']].copy()

# Saving the new DataFrame to a file
new_path = r'ConversationLLMwithoutcontext1_print.txt'
with open(new_path, 'a') as file:
    df_string_new = new_df.to_string(header=True, index=False, justify='left')
    file.write(df_string_new)