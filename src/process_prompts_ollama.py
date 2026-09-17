import ollama
import json
import pandas as pd
import random
import argparse

"""
Updated 10/9/26
This script takes as input user DF and retrieved ids - the retrieved IDs can be 
from the user df or in a separate file given and then runs them through ollama and 
returns the response
"""



N = 5000

llm_model = "qwen2.5:7b"
# Path to your file containing prompts (one per line)


def prompt1(history_list, article_list):
    return f"""Retrieved articles (top 10 for this user, based on history):
    {article_list}

    Task: Write a 200-word morning briefing covering these stories for a reader 
    interested in these articles. Group related stories, 
    skip redundant coverage, note if multiple sources are covering the same event.

    """

def prompt2(history_list, article_list, news_df):

    history_str = "\n\n".join(
        f"[{aid}] Title: {news_df.loc[aid, 'Title']}\n"
        f"Abstract: {news_df.loc[aid, 'Abstract'] or 'N/A'}"
        for aid in history_list[-5:]
    )
    candidates_str = "\n\n".join(
        f"[{aid}] Title: {news_df.loc[aid, 'Title']}\n"
        f"Abstract: {news_df.loc[aid, 'Abstract'] or 'N/A'}"
        for aid in article_list
    )
    return f"""You are a news recommendation reranker. Given a user's reading history 
    and a list of candidate articles, rank the candidates from most to 
    least relevant to the user's interests.

    User's recent reading history:
    {history_str}

    Candidate articles to rank:
    {candidates_str}

    Instructions:
    - Rank ALL candidate articles from most to least relevant.
    - Return ONLY a JSON array containing exactly the {len(article_list)} candidate article IDs listed above, reordered by relevance.
    - Use the exact IDs provided above (e.g. N45527) — do not invent new IDs.
    - Do not include any explanation, markdown, or additional text.
    - Do not include any IDs other than the ones explicitly listed in the candidates section above.


    Output format: a JSON array of exactly {len(article_list)} IDs from the candidates listed above. """

def get_candidates_from_impression(impressions_str):
    """Extract just the article IDs (ignore -0/-1 labels) as the candidate set for reranking."""
    pairs = impressions_str.strip().split()
    return [p.split("-")[0] for p in pairs]

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_type", 
        type=str, 
        help="from_df/new_ids/from_impressions: Where retrieved candidate IDs come from"
    )
    parser.add_argument(
        "prompt_type",             
        type=str,             
        help="summary/rank: Prompt type is summary or ranker"
        )
    args = parser.parse_args()
    

    if args.prompt_type == 'summary':
        prompt_func = prompt1
    elif args.prompt_type == 'rank':
        prompt_func = prompt2
    
    user_df_pickle_path = '/Users/jessicakahn/Documents/repos/MIND/data/retrieved_user.pkl'
    user_df = pd.read_pickle(user_df_pickle_path)
    news_df = pd.read_csv('/Users/jessicakahn/Documents/repos/MIND/data/MINDsmall_train/news.tsv',sep='\t',header=None)
    news_df.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']

    news_df = news_df.set_index("NewsID")  # or whatever the actual ID column is called

    # article_title_dict = news_df.set_index('NewsID')['Title'].to_dict()
    # abstract_dict = news_df.set_index('NewsID')['Abstract'].to_dict()
    # user_df['abstract_history_list'] = user_df['history_list'].apply(lambda x:[abstract_dict[i] for i in x])
    # user_df['title_history_list'] = user_df['history_list'].apply(lambda x:[article_title_dict[i] for i in x])
    user_df['uuid'] = user_df['UserID'].astype(str)+':'+user_df['ImpressionID'].astype(str)

    input_type = args.input_type
    # input_file_path = "/Users/jessicakahn/Documents/repos/MIND/data/prompts.json"
    output_file_path = f"/Users/jessicakahn/Documents/repos/MIND/data/responses_{llm_model}_{N}_{input_type}.json"
    if args.input_type == 'from_df':
        user_df['prompt'] = user_df.apply(lambda row:prompt_func(row['title_history_list'], row['retrieved']['documents'], news_df), axis=1)
        
        

    elif args.input_type == 'new_ids':
        with open("/Users/jessicakahn/Documents/repos/MIND/data/output_user_retrievals.json", "r") as file:
            retrieval_dict = json.load(file)
        # Drop impression ids for this type since retrieval happened in a separate script on the user level
        user_df = user_df.drop_duplicates(subset=['UserID'])

        # Get retrieved IDs from file and map to existing user_df
        user_df['retrieved_from_dict'] = user_df['UserID'].map(retrieval_dict)
        # user_df['retrieved_abstracts'] = user_df['retrieved_from_dict'].apply(lambda x:[abstract_dict[i] for i in x])
        user_df['prompt'] = user_df.apply(lambda row:prompt_func(row['history_list'],row['retrieved_from_dict'], news_df),axis=1)

    elif args.input_type == 'from_impressions':
        user_df['candidate_ids'] = user_df['Impressions'].apply(get_candidates_from_impression)
        user_df['prompt'] = user_df.apply(
            lambda row: prompt_func(row['history_list'], row['candidate_ids'], news_df), axis=1
        )

    prompt_dict = user_df.set_index('uuid')['prompt'].to_dict()

    random.seed(42)
    sampled_items = dict(random.sample(list(prompt_dict.items()), N))
    output_dict = {}

    counter = 0
    with open(output_file_path, "w", encoding="utf-8") as out_f:
        for k, prompt in sampled_items.items():
            print(f"Processing prompt {counter}/{len(sampled_items.keys())}...")
            
            # Structure your chat template message array
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Call the optimized local model
            response = ollama.chat(
                model=llm_model,  # Or llama3.2, deepseek-r1:8b, etc.
                messages=messages,
                options={
                    "num_predict": 4096,
                    "num_ctx": 8192,  # or higher, depending on your prompt length
                    "temperature": 0.2,
                }
            )
            
            # Extract text response
            answer = response['message']['content']
            output_dict[k] = answer
            counter += 1

    # Write to file
    with open(output_file_path, "w") as file:
        json.dump(output_dict, file, indent=4)


    print(f"Done! All responses saved to {output_file_path}")



if __name__ == '__main__':
    main()

    
    