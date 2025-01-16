import torch
import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Quantization configuration for improved performance
quant_config = { 
    "zero_point": True,  # Enable zero-point quantization
    "q_group_size": 64,  # Reduced group size for potentially better precision
    "w_bit": 4,          # 4-bit quantization
    "version": "GEMM"    # GEMM-optimized quantization
}

def quantize_model(
    model_path='/home/bizon/Desktop/models/llama3.1/Llama-3.1-8B-Instruct', 
    quant_path='/home/bizon/Desktop/models/quantized/Llama-3.1-8B-Instruct-awq'
):
    # Optimize GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # Create an intelligent device map
    device_map = {
        f"model.layers.{i}": i % 4 for i in range(80)  # Distribute layers across GPUs
    }
    device_map.update({
        "model.embed_tokens": 0,
        "model.norm": 3,
        "lm_head": 3
    })

    # Load model with optimized settings
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize the model
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model and tokenizer
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f"Quantized model saved to: {quant_path}")

if __name__ == "__main__":
    quantize_model()