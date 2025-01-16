import asyncio
import time
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# Parameters
number_gpus = 1
max_model_len = 8192
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)
runs = 20  # Number of runs

# Prepare prompt
prompts = [
    "Hello world!"
]

# Paths to model
model_id = r"/home/bizon/Desktop/models/llama3.1/Llama-3.1-70B-Instruct"

# Initialize AsyncLLMEngine
engine_args = AsyncEngineArgs(model=model_id, tensor_parallel_size=number_gpus, pipeline_parallel_size=6, max_model_len=max_model_len)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Variables to store results
unquantized_times = []
unquantized_outputs = []

async def run_async():
    for i in range(runs):
        start_time = time.time()
        request_id = f"run_{i}"
        outputs = engine.generate(prompts[0], sampling_params, request_id=request_id)
        async for output in outputs:
            final_output = output
        end_time = time.time()

        elapsed_time = end_time - start_time
        tokens_generated = len(final_output.outputs[0].token_ids)
        tokens_per_second = tokens_generated / elapsed_time

        unquantized_times.append(tokens_per_second)
        unquantized_outputs.append(final_output.outputs[0].text)

        print(f"Unquantized Run {i + 1}: {tokens_per_second:.2f} tokens/s, Total tokens produced: {tokens_generated}, Total Time Taken: {elapsed_time}")

        generated_text = final_output.outputs[0].text
        print(f"Generated text: {generated_text!r}\n")

    average_unquantized_tps = sum(unquantized_times) / len(unquantized_times)
    print(f"\nAverage tokens/s for Model {model_id}: {average_unquantized_tps:.2f}")

# Run the async function
asyncio.run(run_async())
