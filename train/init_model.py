from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

model_id = "/shared/models/Qwen/Qwen3-0.6B"
output_dir = "model/Qwen3-0.6B-init"

print(f"Loading config from {model_id} ...")
config = AutoConfig.from_pretrained(model_id)

print("Initializing model with random weights ...")
model = AutoModelForCausalLM.from_config(config, dtype="bfloat16")

print(f"Saving model to {output_dir} ...")
model.save_pretrained(output_dir)

print(f"Saving tokenizer to {output_dir} ...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(output_dir)

print(f"Done! Random initialized model saved to {output_dir}")
