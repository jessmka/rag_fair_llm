import pandas as pd
import numpy as np
import chromadb
import ast
import json

import chromadb

# Connect using absolute path
client = chromadb.PersistentClient(
    path="/Users/jessicakahn/Documents/repos/MIND/mind_chroma_db"
)
collection = client.get_collection(name="mind_news")

def prompt(article_list):
    return f"""Retrieved articles (top 8 for this user, based on history):
    {article_list}

    Task: Write a 200-word morning briefing covering these stories for a reader 
    interested in these articles. Group related stories, 
    skip redundant coverage, note if multiple sources are covering the same event.

    """


# Load data
# df = pd.read_csv('../data/news_processed.csv')
user_df = pd.read_csv('./data/behavior_processed.csv')

# Make history column a list
print('1. Making history list')
user_df['history_list'] = user_df['History'].str.split(" ")
# Filter users with no history
print('2. Filter users with no history')
user_df = user_df[user_df['history_list'].apply(lambda x: isinstance(x, list))]
# Get embeddings of history items
print('3. Get embeddings of history items')
user_df['records'] = user_df.apply(lambda row:collection.get(ids=list(set(row['history_list'])),include=['embeddings']), axis=1)
# Average the user history into a single embedding vector
print('4. Average the embedding vector')
user_df['averaged_embed'] = user_df.apply(lambda row: np.mean(row['records']['embeddings'], axis=0, keepdims=True)[0],axis=1)
# Query for top-k based on averaged embedding vector - returns 
print('5. Query top 10 results based on average embedding vector')
user_df['topk'] = user_df.apply(lambda row: collection.query(query_embeddings=row['averaged_embed'],include=['documents']), axis=1)
print('6. Format the prompt')
user_df['prompt'] = user_df.apply(lambda row:prompt(row['topk']['documents']), axis=1)

# Export to a JSON file
print('7. Convert to dict')
prompt_dict = user_df.set_index('ImpressionID')['prompt'].to_dict()

print('8. Write to file')
with open("./data/prompts.json", "w") as file:
    json.dump(prompt_dict, file, indent=4)