import logging
import os
import torch
import yaml
import numpy as np
import pandas as pd
import argparse
argparser = argparse.ArgumentParser()

argparser.add_argument('--data_path', type=str, default='../animikh/ElevateMind/data/conversations.csv')
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
argparser.add_argument('--save_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'models'))
argparser.add_argument('--hf_token', type=str, default=os.environ.get('HUGGING_FACE_HUB_TOKEN'))
argparser.add_argument('--cache_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'cache'))

args = argparser.parse_args()
# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# Load data
df = pd.read_csv(args.data_path)
df = df.where(df.notna(), "") # Replace NaN with empty string

dsize = len(df)

# Cap dataset size
if args.cap_dsize is not None and args.cap_dsize > 0:
    df = df.sample(args.cap_dsize)

print(f"Dataset size reduced from: {dsize} -> {len(df)}")

# Shuffle and Split data into train, val and test: 80%, 10%, 10%
total_rows = len(df)
split_0_count = int(total_rows * 0.8)
split_1_count = int(total_rows * 0.10)
split_2_count = total_rows - split_0_count - split_1_count

print(f"Counts for train, val and test: {split_0_count}, {split_1_count}, {split_2_count}")

# split_values = np.concatenate([
#     np.zeros(split_0_count),
#     np.ones(split_1_count),
#     np.full(split_2_count, 2)
# ])
# #np.random.shuffle(split_values)
# df['split'] = split_values
# df['split'] = df['split'].astype(int)

# #print(df['split'])

#df_new = df[df['split'] == 2].copy()
df_new=df[-2500:]
#print(df_new.iloc[0])

# empty_string_index = df_new.index[df_new['context'] == ""].tolist()[0]
# print(empty_string_index,"Hi")
#missing_context = df_new[df_new['context']==""]
# # Remove rows until the empty string is encountered
#df_new = df_new.iloc[empty_string_index+1:]
#df_new.to_csv('test_set.txt', index=False)
path = r'test_set.txt'
#print(df)
with open(path, 'a') as f:
    df_string = df_new.to_string(header=True, index=False, justify='left')
    f.write(df_string)