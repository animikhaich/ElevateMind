from datasets import load_dataset
import os
import json
import pandas as pd
from tqdm import tqdm


data = load_dataset(
    'thu-coai/augesc', split='train', cache_dir="./.huggingface_cache", num_proc=os.cpu_count()
)

# Empty DataFrame
df = pd.DataFrame(columns=['user_input', 'context', 'output', 'data_idx'])

idx = 0

pbar = tqdm(total=len(data))

for element in data:
    context = ""
    user_input = ""
    for conv in json.loads(element['text']):
        speaker = conv[0]
        text = conv[1]
        
        if speaker == 'usr':
            user_input = text
        if speaker == 'sys':
            new_row = pd.DataFrame([[user_input, context, text, idx]], columns=['user_input', 'context', 'output', 'data_idx'])
            df = pd.concat([df, new_row], ignore_index=True)
            context += f"usr:{user_input}\nsys:{text}\n"
            # if len(context + f"usr:{user_input}\nsys:{text}\n") > 3500:
            #     content = chatGPTSummarize(context)
            user_input = ""
    
    pbar.update(1)

os.makedirs('data', exist_ok=True)
df.to_csv('data/conversations.csv', index=False)
