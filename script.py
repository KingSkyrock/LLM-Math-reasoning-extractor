import os
os.environ["HF_HOME"] = "/home/lukewang/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/lukewang/.cache/huggingface"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-14B"

import json

inputjson = 'data.json'
systempromptfile = 'systemprompt.txt'

# Detect available GPUs
num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPU(s)")

# Load model with device_map to automatically use all GPUs
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    device_map="auto",  # Automatically distribute across all GPUs
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
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768,
            max_time=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print("Completed step " + str(i) + " for question " + str(count))
        arr.append(thinking_content)
    thinking_dict[question] = arr
    print("COMPLETED: " + str(count) + "/10")
    if count == 1: break
    else: count += 1

outputfile = 'output_' + ''.join([char for char in model_name if char.isalnum()]) + '.json'
with open(outputfile, 'w') as f:
    json.dump(thinking_dict, f)
