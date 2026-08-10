import pandas as pd
import numpy as np
# import chromadb
from sentence_transformers import SentenceTransformer


# class GetDB():
#     def __init__(self):
#         # Initialize the client with a specific local directory path
#         self.client = chromadb.PersistentClient(path="./mind_chroma_db")
#         self.collection = self.client.get_collection(name="mind_news")

class Data():
    def __init__(self):
        behavior = pd.read_csv('/Users/jessicakahn/Documents/repos/MIND/data/MINDsmall_train/behaviors.tsv',sep='\t', header=None)
        behavior.columns = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
        self.df = (
            behavior.assign(
                impression=behavior["Impressions"].str.split()
            )
            .explode("impression")
            .assign(
                article_id=lambda x: x["impression"].str.rsplit("-", n=1).str[0],
                clicked=lambda x: x["impression"].str.rsplit("-", n=1).str[1].astype(int)
            )
            .drop(columns=["impression"])
            .reset_index(drop=True)
        )
        
    def get_user_profile_vector(user_id, behavior_df, id_to_embedding):
        # Fetch history string for the user
        # TODO []: update this function, I think it uses unexploded version of dataset, maybe don't need to explode?
        user_rows = behavior_df[behavior_df['user_id'] == user_id]
        if user_rows.empty or pd.isna(user_rows.iloc[0]['history']):
            return None
        
        history_ids = user_rows.iloc[0]['history'].split()
        
        # Retrieve embeddings for all articles in user history
        history_embeddings = [id_to_embedding[nid] for nid in history_ids if nid in id_to_embedding]
        
        if not history_embeddings:
            return None
        
        # Compute the average vector representing user interests
        user_vector = np.mean(history_embeddings, axis=0)
        return user_vector.reshape(1, -1)
class Embed():
    def __init__(self):
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def query_db(news_id):
    # Fetch embedding of article based on id?
    pass

def main():
    # db = GetDB()
