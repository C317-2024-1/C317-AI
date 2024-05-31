import pandas as pd
from openai import OpenAI
from utils import getDataPath

api_key = 'sk-proj-I3tjkHpgKcScB38bl1QjT3BlbkFJc3PBfRNm0VeOtT77QNnP'
client = OpenAI(api_key=api_key)


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


df = pd.read_csv(f'{getDataPath()}/token.csv')

df['embedding'] = df['processed_text'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

coluna1 = df['text']
coluna2 = df['embedding']
novo_df = pd.DataFrame({'text': coluna1, 'embedding': coluna2})
novo_df.to_csv(f'{getDataPath()}/intellicash.csv', index=False)
