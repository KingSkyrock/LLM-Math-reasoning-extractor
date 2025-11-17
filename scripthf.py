import os
# Ensure the cache directory is writable
cache_dir = "/home/lukewang/.cache/huggingface"
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir

# Set environment to work around CUDA compute_89 compilation issues
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;9.0"  # Skip 8.9 (H100) to avoid unsupported arch
os.environ["VLLM_USE_V1"] = "1"  # Must use V1 with this vLLM version
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_SKIP_WARMUP"] = "true"  # Skip profiling run that triggers compilation

# Create cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

from vllm import LLM, SamplingParams
import torch
import json

model_name = "Qwen/Qwen3-14B"

inputjson = 'data.json'
systempromptfile = 'systemprompt.txt'

if __name__ == '__main__':
    # Automatically detect the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    llm = LLM(
        model=model_name,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.45,  # Reduced to fit in available memory (22.2 GiB free per GPU)
        tensor_parallel_size=num_gpus,  # Automatically use all available GPUs
        enforce_eager=False,  # Disable CUDA graphs to avoid compilation
        download_dir=cache_dir,  # Force use of writable cache directory
        disable_custom_all_reduce=True,  # Disable custom kernels that require compilation
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
