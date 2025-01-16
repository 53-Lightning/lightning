from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from peft import get_peft_model
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# set variables 
model_name = "/home/bizon/Desktop/models/llama3.1/Llama-3.1-8B-Instruct" 
max_seq_length = 2048 
dtype = None 
load_in_4bit = True 

# using unsloth 
model, tokenizer = FastLanguageModel.from_pretrained(
	model_name = model_name,
	max_seq_length = max_seq_length,
	dtype = dtype, 
	load_in_4bit= load_in_4bit)

# baseline model for comparison 
model2, tokenizer2 = FastLanguageModel.from_pretrained(
	model_name = model_name, 
	max_seq_length = max_seq_length,
	dtype = dtype, 
	load_in_4bit=load_in_4bit)

# LoRA model 
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0. Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.1, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

# write prompt for LLM
prompt = """
### Input:
{}

### Response:
{}"""

# must add EOS_TOKEN or else the generation will go on forever 
EOS_TOKEN = tokenizer.eos_token 

# formatting without including instructions to validate fine-tuning works 
def formatting_prompts_func(examples):
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = prompt.format(input, output) + EOS_TOKEN # fine-tuning vs programmatic functionality? 
        texts.append(text)
    return { "text" : texts, }
pass

# load dataset
dataset = load_dataset("ayaan04/shakespeare-text", split = "train").select(range(1000)) # starting with 1000 to test 
dataset = dataset.map(formatting_prompts_func, batched = True,)

# spread across multiple gpus
print(torch.cuda.get_device_name()) # not sure why this is only saying 1 

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        dataloader_num_workers=4,
        ddp_find_unused_parameters= False, 
        #device="cuda:0,1,2,3",
        warmup_steps = 5,
        num_train_epochs = 1, # set for 1 full training run.
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer.train()

# alpaca_prompt = copied from above
inference_model, inference_tokenizer = model, tokenizer
FastLanguageModel.for_inference(inference_model) # Enable native 2x faster inference
inputs = tokenizer(
[
    prompt.format(
        "How are you doing today", # input
        "", # output - leave blank for generation
    )
], return_tensors = "pt").to("cuda")

outputs = inference_model.generate(**inputs, max_new_tokens = 64, temperature = 0.5, use_cache = True)
inference_tokenizer.batch_decode(outputs)