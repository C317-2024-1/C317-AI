import ast
import os
import pandas as pd
import tiktoken
from openai import OpenAI
from scipy import spatial
from utils import getDataPath
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
client = OpenAI(api_key=API_KEY)
EMBEDDING_MODEL = 'text-embedding-3-small'
GPT_MODEL = 'gpt-3.5-turbo-0125'

df = pd.read_csv(f'{getDataPath()}/intellicash.csv')
df['embedding'] = df['embedding'].apply(ast.literal_eval)

texto = 'Desculpe-me, mas não consigo resolver seu problema com os dados que possuo. Por favor, entre em contato conosco atarvés do site (https://www.iws.com.br/contato.php) ou nosso canal de atendimento (https://www.iws.com.br/central-de-relacionamento.php).'

def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = f'Use a documentacao abaixo sobre o Sistema de Gestao Intellicash da empresa IWS Sistemas para responder a seguinte pergunta. Caso não consiga responder, escreva "{texto}."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nDocumentacao Intellicash:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
        query: str,
        df: pd.DataFrame = df,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "Voce responde perguntas sobre o Sistema de Gestao Intellicash da empresa IWS Sistemas."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message
