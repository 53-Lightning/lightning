import time
from vllm import LLM, SamplingParams

prompts = [
    "Hello world!",
    "How are you?",
    "What is 2+2",
    "What color is the sky?"
]
sampling_params = SamplingParams(temperature=0, top_p=0.1, max_tokens=100000)

llm = LLM(
    model="/home/bizon/Desktop/models/quantized/Llama-3.1-70B-Instruct-awq",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95,  # Slightly reduce to 0.95
    max_model_len=108144,
    enforce_eager=True  # Set to False, can improve performance
)

# Start the timer
start_time = time.time()

# Generate the outputs
outputs = llm.generate(prompts, sampling_params)

# End the timer
end_time = time.time()
total_time = end_time - start_time

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\n")
    print(f"Generated text: {generated_text!r}\n")
    
    # Get the number of tokens generated
    num_tokens_generated = len(output.outputs[0].token_ids)
    print(f"Number of tokens generated: {num_tokens_generated}")
    print(f"Time taken: {total_time:.2f} seconds")
    tokens_per_second = num_tokens_generated / total_time
    print(f"Tokens per second: {tokens_per_second:.2f}")