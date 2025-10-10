from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save to local dir
save_path = "../embedding_model/all-MiniLM-L6-v2"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Model saved at {save_path}")
