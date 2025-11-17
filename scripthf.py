import os
os.environ["HF_HOME"] = "/home/lukewang/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/lukewang/.cache/huggingface"

from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen3-14B"

import json

inputjson = 'data.json'
systempromptfile = 'systemprompt.txt'

llm = LLM(
    model=model_name,
    dtype="auto",
    trust_remote_code=True,
    max_model_len=8192, 
)

try:
    with open(systempromptfile) as f:
        system_prompt = f.read()
except FileNotFoundError:
    print("System prompt not found")

try:
    with open(inputjson, 'r', encoding="utf8") as f:
        data = json.load(f)
    print("Data loaded from file:")
except FileNotFoundError:
    print("Error: '" + inputjson + "' not found.")
except json.JSONDecodeError:
    print("Error: Invalid JSON format")

thinking_dict = {}

# Configure sampling parameters
sampling_params = SamplingParams(
    max_tokens=32768,
    temperature=0.7,  # Adjust as needed
    top_p=0.9,        # Adjust as needed
    stop_token_ids=[151668],  # Special token from original code
)

count = 1
for question in data:
    arr = []
    partial_solution = ""
    for i in range(len(data[question]) - 1):
        partial_solution += data[question][i]
        prompt = f"""Question: {question}

Solution:
{partial_solution}
"""
        messages=[
            {
                "content": system_prompt,
                "role": "system",
            },
            {
                "content": prompt,
                'role': 'user',
            },
        ]

        # Use vLLM's chat method for inference
        outputs = llm.chat(
            messages=[messages],
            sampling_params=sampling_params,
            use_tqdm=False
        )

        # Extract the generated text
        thinking_content = outputs[0].outputs[0].text.strip("\n")

        print("Completed step " + str(i) + " for question " + str(count))
        arr.append(thinking_content)
    thinking_dict[question] = arr
    print("COMPLETED: " + str(count) + "/10")
    if count == 1: break
    else: count += 1

outputfile = 'output_' + ''.join([char for char in model_name if char.isalnum()]) + '.json'
with open(outputfile, 'w') as f:
    json.dump(thinking_dict, f)