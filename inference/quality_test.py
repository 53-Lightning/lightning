from vllm import LLM, SamplingParams

# Parameters
number_gpus = 4
max_model_len = 8192
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

# Cloze tasks prompts
# List of cloze task prompts
cloze_prompts = [
    "The process by which plants convert sunlight into chemical energy is called ",
    "The treaty that marked the end of the American Revolutionary War is known as the Treaty of ",
    "The effect describes the apparent deflection of moving objects due to Earth's rotation is called the ",
    "In Greek mythology, the figure condemned to push a boulder uphill for eternity is ",
    "The scale used to measure the intensity of earthquakes based on observed effects is the ",
    "The chemical element with the atomic number 82 is ",
    "The mathematical constant approximately equal to 2.718 is ",
    "The peninsula in Europe that includes Denmark and northern Germany is called the the ",
    "The painting by Grant Wood depicting a farmer and his daughter is ",
    "In physics, the principle stating that energy cannot be created or destroyed is the law of "
]

# Cloze task answers
cloze_answers = [
    "photosynthesis",
    "Paris",
    "Coriolis",
    "Sisyphus",
    "Mercalli",
    "lead",
    "e",
    "Jutland",
    "American Gothic",
    "conservation"
]

# Paths to model
model_id = r"/home/bizon/Desktop/models/llama3.1/Llama-3.1-8B"
#model_id = r"/home/bizon/Desktop/models/quantized/Llama-3.1-8B-W8A16"
#model_id = r"/home/bizon/Desktop/models/quantized/Llama-3.1-8B-W4A16"

# Initialize model
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)

# generating the output from the LLM
output = llm.generate(cloze_prompts, sampling_params)

# Extracting the text from the output
generated_text = output[0].outputs[0].text
print(f"Generated text: {generated_text!r}\n")

