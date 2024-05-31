import tiktoken
import pandas as pd
from utils import getDataPath

tokenizer = tiktoken.get_encoding("cl100k_base")
df = pd.read_csv(f'{getDataPath()}/titulo_e_texto.csv')


def process_long_text(text, max_tokens=1024):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = tokenizer.decode(truncated_tokens)
        return truncated_text
    return text


def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


df['processed_text'] = df['text'].apply(process_long_text)
df['num_tokens'] = df['processed_text'].apply(count_tokens)

df.to_csv(f'{getDataPath()}/token.csv', index=False)
