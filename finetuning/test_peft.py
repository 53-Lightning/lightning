import torch 
from trl import SFTConfig, SFTTrainer
from peft import (LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel)
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset

# set up quantization
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True, # 4-bit quantization
#     bnb_4bit_quant_type='nf4', # type of 4-bit quantization (normal float 4)
#     bnb_4bit_compute_dtype=torch.bfloat16, # type used for computations during inference/training
#  )

# set up tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/bizon/Desktop/models/llama3.1/Llama-3.1-8B-Instruct", use_fast=True)

# set up model 
model = AutoModelForCausalLM.from_pretrained(
    "/home/bizon/Desktop/models/llama3.1/Llama-3.1-8B-Instruct", 
    # quantization_config=quantization_config, # quantize model 
    device_map={"":0}
)

# padding token 
PAD_TOKEN = "<|pad|>"

tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

# extend the embeddings to include padded tokenizer
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

# eos_token to train model to shorten responses 
EOS_TOKEN = tokenizer.eos_token # <|eot_id|>


# define how you want the model to read responses ** important to format correctly **
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"### Input: {example['input'][i]}\n ### Response: {example['output'][i]} {EOS_TOKEN}" # make sure to add eos token 
        output_texts.append(text)
    return output_texts

# train model on generated prompts only 
response_template = " ### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

# load dataset
ds_t = load_dataset("ayaan04/shakespeare-text", split = "train").select(range(0,1000)) 
ds_v = load_dataset("ayaan04/shakespeare-text", split = "train").select(range(1000,1200))

# LoRA
lora_config = LoraConfig(
    r=32,  # rank for matrix decomposition, where higher increases number of learned parameters
    lora_alpha=16, # scaling factor, where higher allows model to adapt more aggressively 
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        # "mlp.gate_proj",
        # "mlp.up_proj",
        # "mlp.down_proj"
    ],
    lora_dropout=0.1,
    bias='none',
    task_type=TaskType.CAUSAL_LM
)

# wrap base model and peft config 
# model2 = prepare_model_for_kbit_training(model)
model2 = get_peft_model(model, lora_config)
# model2.print_trainable_parameters() # trainable params: 83,886,080 || all params: 8,114,212,864 || trainable%: 1.0338

# where to store output 
OUTPUT_DIR = "/home/bizon/Desktop/models/fine-tuned/12-6-24/run1"

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR, 
    max_seq_length=512,
    num_train_epochs=1,
    per_device_train_batch_size=2, # how many samples each training batch processes
    per_device_eval_batch_size=2,  
    gradient_accumulation_steps=4,  # by using gradient accum, you update the weights every: batch_size * gradient_accum_steps 
    optim="paged_adamw_8bit",  # paged adamw
    eval_strategy='steps',
    eval_steps=0.2,  # evalaute every 20% of the trainig steps
    save_steps=0.2,  # save every 20% of the trainig steps
    logging_steps=10,
    learning_rate=1e-4,
    fp16=False,  # also try bf16=True
    save_strategy='steps',
    warmup_ratio=0.1,  # learning rate warmup
    save_total_limit=5, # number of checkpoints saved to folder 
    lr_scheduler_type="cosine",  # scheduler
    save_safetensors=True,  # saving to safetensors
    dataset_kwargs={
        "add_special_tokens": False,  # we template with special tokens already
        "append_concat_token": False,  # no need to add additional sep token
    },
 )



# set up trainer
trainer_sft = SFTTrainer(
    model=model2,
    args=sft_config,
    train_dataset=ds_t,
    eval_dataset=ds_v,
    formatting_func=formatting_prompts_func,
    data_collator=collator)

# fine-tune model 
trainer_sft.train()

# save model 
trainer_sft.save_model(OUTPUT_DIR)
trainer_sft.save_pretrained(OUTPUT_DIR)

# loading and merging the model 
NEW_MODEL = "/home/bizon/Desktop/models/fine-tuned/12-5-24/run2/checkpoint-52"

# load trained/resized tokenizer
tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)

# load the raw model *** NEED TO FIX***
ft_model = AutoModelForCausalLM.from_pretrained("/home/bizon/Desktop/models/llama3.1/Llama-3.1-8B-Instruct")
ft_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
ft_model = PeftModel.from_pretrained(ft_model, NEW_MODEL)
ft_model = ft_model.merge_and_unload()

input = "Hello how are you today?"

# prepare input in the same format as training
input_prompt = f"### Input: {input} \n ### Response:"

# tokenize the input
inputs = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=True)

# generate response
outputs = ft_model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=100,    
    temperature=0.9,  
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

# decode the generated response
generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# print the full response
print(f"### Input: {input} \n ### Response: {generated_text}")