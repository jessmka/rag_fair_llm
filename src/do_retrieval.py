import pandas as pd
import numpy as np
import json
import chromadb
from tqdm.auto import tqdm
from utilities import *
tqdm.pandas()

MIN_HISTORY_ITEMS = 10
NUM_HISTORY_ITEMS = 10

class Retrieve:
    """This class will retrieve the user recs based on average embeddings"""
    def __init__(self, build_data=True, save_data=True):
        # Load chromadb data
        user_df_pickle_path = '/Users/jessicakahn/Documents/repos/MIND/data/retrieved_user.pkl'
        news = pd.read_csv('/Users/jessicakahn/Documents/repos/MIND/data/news_processed.csv')
        self.gender_dict = news.set_index('NewsID')['Gender'].to_dict()
        if build_data:
            client = chromadb.PersistentClient(path="/Users/jessicakahn/Documents/repos/MIND/mind_chroma_db")
            self.collection = client.get_collection(name="mind_news")
            # self.responses_path = '/Users/jessicakahn/Documents/repos/MIND/data/responses.json'
            
            user_df = self.build_from_raw_data()
            # filter out users with no history
            user_df['history_len'] = user_df['history_list'].str.len()
            # filter to users with more than 10 items in history
            print('Before filter: ', user_df.shape)
            
            user_df = user_df[user_df['history_list'].apply(lambda x: isinstance(x, list))].copy()
            user_df = user_df[user_df['history_len']>=MIN_HISTORY_ITEMS].copy()
            # Take only the last N items from their history for averaging
            user_df['hist_'] = user_df['history_list'].str[-NUM_HISTORY_ITEMS:]
            print('After filter: ', user_df.shape)

            user_df['history_gender'] = user_df['hist_'].apply(
                lambda x: [self.gender_dict.get(i, 'Unknown') for i in x]
            )
            print('Getting history embeddings')
            # Get
            user_df['records'] = user_df.progress_apply(lambda row:self.collection.get(ids=list(set(row['hist_'])),include=['embeddings']), axis=1)
            # Average the embeddings retrieved in the last step
            user_df['averaged_embed'] = user_df.progress_apply(lambda row: np.mean(row['records']['embeddings'], axis=0, keepdims=True).flatten(),axis=1)
            print(user_df.head())
            print('Retrieving from average embeddings')
            user_df['retrieved'] = user_df.progress_apply(lambda row: self.collection.query(query_embeddings=row['averaged_embed']),axis=1)
            
            user_df['retrieved_gender'] = user_df.apply(
                                lambda row: [self.gender_dict.get(i, 'Unknown') for i in row['retrieved']['ids'][0]],
                                axis=1
                            )
            self.user_df = user_df
            if save_data:
                print(f'Pickling to {user_df_pickle_path}')
                self.user_df.to_pickle(user_df_pickle_path)
        else:
            # Load data if not building it
            print('Unpickling')
            self.user_df = pd.read_pickle(user_df_pickle_path)
    
    def get_user_df(self):
        return self.user_df
 

    def build_from_raw_data(self):
        # Load raw behavior data and process
        behavior = pd.read_csv('/Users/jessicakahn/Documents/repos/MIND/data/MINDsmall_train/behaviors.tsv',sep='\t', header=None)
        behavior.columns = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
        behavior['history_list'] = behavior['History'].str.split(" ")
        behavior['history_len'] = behavior['history_list'].str.len()
        return behavior

class GenderSkew:
    def __init__(self, df):
        # Assumes df has columns history_gender and retrieved_gender
        pass






if __name__ == '__main__':
    ## Build the dataset
    ret = Retrieve(build_data=True, save_data=True)
    ## Load the data once built 
    # ret = Retrieve(build_data=False, save_data=False)
    # user_df = ret.get_user_df()
   

    # print(user_df.head(5))
    # print(user_df.columns)
    


        


