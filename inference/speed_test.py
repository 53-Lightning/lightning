import time
from vllm import LLM, SamplingParams

# Parameters
number_gpus = 1
max_model_len = 8192
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)
runs = 1  # Number of runs

# Prepare prompt
prompts = [
    "Hello world!"
]

# Paths to model
model_id = "/home/bizon/Desktop/models/llama3.1/Llama-3.1-8B-Instruct"

# Initialize model
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)

# Variables to store results
unquantized_times = []
unquantized_outputs = []

# Unquantized model runs
for i in range(runs):
    start_time = time.time()
    output = llm.generate(prompts, sampling_params)
    end_time = time.time()

    elapsed_time = end_time - start_time
    tokens_generated = len(output[0].outputs[0].token_ids)
    tokens_per_second = tokens_generated / elapsed_time

    unquantized_times.append(tokens_per_second)
    unquantized_outputs.append(output[0].outputs[0].text)

    print(f"Unquantized Run {i + 1}: {tokens_per_second:.2f} tokens/s, Total tokens produced: {tokens_generated}, Total Time Taken: {elapsed_time}")

    generated_text = output[0].outputs[0].text
    print(f"Generated text: {generated_text!r}\n")

    if i == (runs - 1):

        average_unquantized_tps = sum(unquantized_times) / len(unquantized_times)
        print(f"\nAverage tokens/s for Model {model_id}: {average_unquantized_tps:.2f}")

