import ollama

# Path to your file containing prompts (one per line)
input_file_path = "prompt.txt"
output_file_path = "responses.txt"

# 1. Read your prompts from the text file
with open(input_file_path, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# 2. Process each prompt using Ollama's local engine
with open(output_file_path, "w", encoding="utf-8") as out_f:
    for idx, prompt in enumerate(prompts, 1):
        print(f"Processing prompt {idx}/{len(prompts)}...")
        
        # Structure your chat template message array
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Call the optimized local model
        response = ollama.chat(
            model="qwen2.5",  # Or llama3.2, deepseek-r1:8b, etc.
            messages=messages,
            options={"num_predict": 300}  # Equivalent to max_new_tokens
        )
        
        # Extract text response
        answer = response['message']['content']
        
        # 3. Write results to your output file
        out_f.write(f"--- Prompt {idx} ---\n{prompt}\n")
        out_f.write(f"--- Response ---\n{answer}\n\n")

print(f"Responses saved to {output_file_path}")