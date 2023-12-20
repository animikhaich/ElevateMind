import logging
import os
import torch
import yaml
import numpy as np
import pandas as pd
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--data_path', type=str, default='data/conversations_summarized_25K.csv')
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

os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
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

print(f"Dataset size reduced from: {dsize} -> {len(df)}")


# Shuffle and Split data into train, val and test: 80%, 10%, 10%
total_rows = len(df)
split_0_count = int(total_rows * 0.8)
split_1_count = int(total_rows * 0.10)
split_2_count = total_rows - split_0_count - split_1_count

print(f"Counts for train, val and test: {split_0_count}, {split_1_count}, {split_2_count}")

split_values = np.concatenate([
    np.zeros(split_0_count),
    np.ones(split_1_count),
    np.full(split_2_count, 2)
])
np.random.shuffle(split_values)
df['split'] = split_values
df['split'] = df['split'].astype(int)

print(df.head(20))

# You are an expert who will summarize the following conversation between a 'usr' and a 'sys' 
# without losing information and without adding new information.

# Train model

input_placeholder = '{user_input}'
context_placeholder = '{context}'

qlora_fine_tuning_config = yaml.safe_load(
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

adapter:
  type: lora

quantization:
  bits: 4

preprocessing:
  global_max_sequence_length: {args.input_seq_len}
  split:
    type: random
    probabilities:
    - 1
    - 0
    - 0

trainer:
  type: finetune
  epochs: {args.epochs}
  batch_size: {args.batch_size}
  eval_batch_size: {args.batch_size}
  max_memory: 0.85
  gradient_accumulation_steps: {args.grad_acc_steps}
  learning_rate: {args.lr}
  learning_rate_scheduler:
    warmup_fraction: 0.03
"""
)

model = LudwigModel(config=qlora_fine_tuning_config, logging_level=logging.INFO)
results = model.train(dataset=df)

# Save model
model.save(args.save_dir)