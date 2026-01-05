def baseline_decode(prompt, tokenizer=None, model=None, device="cpu"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
