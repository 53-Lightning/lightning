import random 
import numpy as np 
import torch 
from trl import SFTConfig, SFTTrainer
from textwrap import dedent
from peft import (LoraConfig, get_peft_model, TaskType)
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset

# set a seed for reproducability
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(0)

# set up model / tokenizer 
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("/media/bizon/53Drive/Llama-3.1-8B-Instruct", use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    "/media/bizon/53Drive/Llama-3.1-8B-Instruct"
)

# padding token 
PAD_TOKEN = "<|pad|>"

tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

# extend the embeddings to include padded tokenizer
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

print(tokenizer.pad_token, tokenizer.pad_token_id)
# output: ('<|pad|>', 128256)

# formatting df in specific way
def format_example(row: dict):
    prompt = dedent(
        f"""
        {row["input"]}
        """
    )
    messages = [
        # the system prompt is VERY important to adjust/control the behavior of the model, make sure to use it properly according to your task
        {"role": "system", "content": row["instruction"]},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["output"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# load dataset
ds_t = load_dataset("ayaan04/shakespeare-text", split = "train").select(range(0,90)) # starting with 1000 to test 
ds_v = load_dataset("ayaan04/shakespeare-text", split = "train").select(range(90,100))

# in order to only evaluate the generation of the model, we shouldn't consider the text that were already inputed, we will use the end header id token to get the generated text only, and mask everything else
response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, # rank for matrix decomposition
    lora_alpha=32, 
    lora_dropout=0.1
)

# wrap base model and peft config 
#model2 = AutoModelForCausalLM.from_pretrained(model_name)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # trainable params: 3,407,872 || all params: 8,033,669,120 || trainable%: 0.0424

# where to store output for now
OUTPUT_DIR = "/home/bizon/Downloads"

# configurations set up 
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field='text',  # this is the final text example we formatted
    max_seq_length=4096,
    num_train_epochs=1, # full forward and backward  pass through entire training dataset
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,  
    gradient_accumulation_steps=4,  # by using gradient accum, we updating weights every: batch_size * gradient_accum_steps = 4 * 2 = 8 steps
    optim="paged_adamw_8bit",  # paged adamw
    eval_strategy='steps',
    eval_steps=0.2,  # evalaute every 20% of the trainig steps
    save_steps=0.2,  # save every 20% of the trainig steps
    logging_steps=10,
    learning_rate=1e-4,
    fp16=False,  # also try bf16=True
    save_strategy='steps',
    warmup_ratio=0.1,  # learning rate warmup
    save_total_limit=2,
    lr_scheduler_type="cosine",  # scheduler
    save_safetensors=True,  # saving to safetensors
    dataset_kwargs={
        "add_special_tokens": False,  # we template with special tokens already
        "append_concat_token": False,  # no need to add additional sep token
    },
    seed=1
)

# set up trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds_t,
    eval_dataset=ds_v,
    tokenizer=tokenizer,
    data_collator=collator,
)

# fine-tune model 
trainer.train()

  	
# loading and merging the model 
from peft import PeftModel

NEW_MODEL = "C:/Users/Bizon/Downloads/checkpoint-2"

# load trained/resized tokenizer
tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)

# here we are loading the raw model, if you can't load it on your GPU, you can just change device_map to cpu
# we won't need gpu here anyway
ft_model = AutoModelForCausalLM.from_pretrained("C:/Models/Llama-3.1-8B-Instruct")
        
ft_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
ft_model = PeftModel.from_pretrained(ft_model, NEW_MODEL)
ft_model = ft_model.merge_and_unload()
