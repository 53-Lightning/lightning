import time
from vllm import LLM, SamplingParams

# Parameters
number_gpus = 4
max_model_len = 8192
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)
runs = 100  # Number of runs

# Prepare prompts
prompts = [
    "My dream vacation is"
]

# Paths to models
quantized_model_id = r"/home/bizon/Desktop/models/quantized/Llama-3.1-8B-W8A16"

# Initialize models
quantized_llm = LLM(model=quantized_model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len, gpu_memory_utilization=0.95)

# Variables to store results
quantized_times = []
quantized_outputs = []

# Quantized model runs
for i in range(runs):
    start_time = time.time()
    quantized_output = quantized_llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    tokens_generated = len(quantized_output[0].outputs[0].token_ids)
    tokens_per_second = tokens_generated / elapsed_time

    quantized_times.append(tokens_per_second)
    quantized_outputs.append(quantized_output[0].outputs[0].text)

    print(f"Quantized Run {i + 1}: {tokens_per_second:.2f} tokens/s, Total tokens produced: {tokens_generated}, Total Time Taken: {elapsed_time}")

    if i == (runs - 1):

        average_quantized_tps = sum(quantized_times) / len(quantized_times)
        print(f"\nAverage tokens/s for Quantized Model: {average_quantized_tps:.2f}")

