from ollama import chat
from ollama import ChatResponse
import json

inputjson = 'data.json'
llm = 'llama3.1:8b'
systempromptfile = 'systemprompt.txt'

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
        response: ChatResponse = chat(model=llm, messages=[
            {
                "content": system_prompt,
                "role": "system",
            },
            {
                "content": prompt,
                'role': 'user',
            },
        ])
        print("Completed step " + str(i) + " for question " + str(count))
        arr.append(response.message.content)
    thinking_dict[question] = arr
    print("COMPLETED: " + str(count) + "/10")
    if count == 10: break
    else: count += 1

outputfile = 'output_' + ''.join([char for char in llm if char.isalnum()]) + '.json'
with open(outputfile, 'w') as f:
    json.dump(thinking_dict, f)