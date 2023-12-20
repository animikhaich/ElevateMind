import pandas as pd
from tqdm import tqdm
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser()

argparser.add_argument('--cache_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'cache'))
argparser.add_argument('--data_path', type=str, default='data/conversations.csv')
argparser.add_argument('--data_size', type=int, default=25_000)
argparser.add_argument('--num_sent', type=int, default=3)
argparser.add_argument('--max_len', type=int, default=70)

args = argparser.parse_args()

os.environ["HF_HOME"] = args.cache_dir


from summarizer.sbert import SBertSummarizer

model = SBertSummarizer('paraphrase-MiniLM-L6-v2')



def summarize(context, num_sent=args.num_sent):
    global api_call_counter

    response = model(context, num_sentences=num_sent, max_length=args.max_len)
    if len(response) > 350:
        response = summarize(context, num_sent=num_sent-1)
    
    return response

def summarizeIfLongerThan(context, max_len=350):
    try:
        if len(context) > max_len:
            return summarize(context)
        else:
            return context
    except Exception as e:
        print(e)
        return context

df = pd.read_csv(args.data_path)
df = df.sample(args.data_size)
df = df.where(df.notna(), "") 

# Enable tqdm for pandas apply
tqdm.pandas(desc="Summarizing Conversations", ascii=True)

# Apply the function with progress bar
df['context'] = df['context'].progress_apply(summarizeIfLongerThan)


df.to_csv('data/conversations_summarized_25K.csv', index=False)
