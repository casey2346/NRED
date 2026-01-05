import torch

def latent_reasoning(prompt, tokenizer, model, device):
    internal_prompt = prompt + "\n<think>"
    inputs = tokenizer(internal_prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    latent = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return latent
