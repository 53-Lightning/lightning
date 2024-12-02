import torch
import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Explicitly set which GPUs to use (0-3)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model_path = '/home/bizon/VLLM/Llama-3.1-70B-Instruct'
quant_path = '/home/bizon/VLLM/Llama-3.1-70B-Instruct-awq'
quant_config = { 
    "zero_point": True, 
    "q_group_size": 128, 
    "w_bit": 4, 
    "version": "GEMM" 
}

# Create device map for 4 GPUs
device_map = {
    f"model.layers.{i}": i % 4 for i in range(80)  # Adjust number of layers if needed
}
device_map["model.embed_tokens"] = 0
device_map["model.norm"] = 3
device_map["lm_head"] = 3

# Load model with explicit device map
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    **{
        "low_cpu_mem_usage": True,
        "use_cache": False,
    }
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)