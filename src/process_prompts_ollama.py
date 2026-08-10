import ollama
import json
import pandas as pd
import random



N = 1000

llm_model = "llama3"
# Path to your file containing prompts (one per line)
input_file_path = "/Users/jessicakahn/Documents/repos/MIND/data/prompts.json"
output_file_path = f"/Users/jessicakahn/Documents/repos/MIND/data/responses_{llm_model}_{N}.json"

def prompt(article_list):
    return f"""Retrieved articles (top 10 for this user, based on history):
    {article_list}

    Task: Write a 200-word morning briefing covering these stories for a reader 
    interested in these articles. Group related stories, 
    skip redundant coverage, note if multiple sources are covering the same event.

    """

user_df_pickle_path = '/Users/jessicakahn/Documents/repos/MIND/data/retrieved_user.pkl'
user_df = pd.read_pickle(user_df_pickle_path)

user_df['prompt'] = user_df.apply(lambda row:prompt(row['retrieved']['documents']), axis=1)
prompt_dict = user_df.set_index('ImpressionID')['prompt'].to_dict()
# with open(input_file_path, 'r') as file:
#     prompt_dict = json.load(file)

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
            options={"num_predict": 300}  # Equivalent to max_new_tokens
        )
        
        # Extract text response
        answer = response['message']['content']
        output_dict[k] = answer
        counter += 1

# Write to file
with open(output_file_path, "w") as file:
    json.dump(output_dict, file, indent=4)


print(f"Done! All responses saved to {output_file_path}")