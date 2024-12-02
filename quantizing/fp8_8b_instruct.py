from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from vllm import LLM

# loading the model

MODEL_ID = r"/home/bizon/VLLM/Llama-3.1-8B-Instruct"
QUANT = r"/home/bizon/VLLM/Llama-3.1-8B-Instruct-FP8-Dynamic"

model = AutoModelForCausalLM.from_pretrained(
  MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

input_ids = tokenizer("Hello my name is", return_tensors = "pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens = 50)
print(tokenizer.decode(output[0]))

# Save the model.
SAVE_DIR = MODEL_ID.split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

model = LLM(SAVE_DIR,
    gpu_memory_utilization=0.95,  # Slightly reduce to 0.95
    max_model_len=108144,
    enforce_eager=True  # Set to False, can improve performance
)
model.generate("Hello my name is")