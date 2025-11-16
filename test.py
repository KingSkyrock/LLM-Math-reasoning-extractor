from ollama import chat
from ollama import ChatResponse
import json

inputjson = 'data.json'
outputjson = 'output.json'

try:
    with open(inputjson, 'r', encoding="utf8") as f:
        data = json.load(f)
    print("Data loaded from file:")
except FileNotFoundError:
    print("Error: '" + inputjson + "' not found.")
except json.JSONDecodeError:
    print("Error: Invalid JSON format")

thinking_dict = {}

for question in data:
    thinking_dict[question] = "thinking process here"

with open(outputjson, 'w') as f:
    json.dump(thinking_dict, f)


#response: ChatResponse = chat(model='llama3.1:8b', messages=[
#  {
#    'role': 'user',
#    'content': 'Why is the sky blue?',
#  },
#])
#print(response.message.content)