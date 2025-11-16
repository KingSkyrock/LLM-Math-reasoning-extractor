from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-14B"

import json

inputjson = 'data.json'
llm = 'llama3.1:8b'
systempromptfile = 'systemprompt.txt'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
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
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        print("Completed step " + str(i) + " for question " + str(count))
        arr.append(thinking_content)
    thinking_dict[question] = arr
    print("COMPLETED: " + str(count) + "/10")
    if count == 10: break
    else: count += 1

outputfile = 'output_' + ''.join([char for char in llm if char.isalnum()]) + '.json'
with open(outputfile, 'w') as f:
    json.dump(thinking_dict, f)