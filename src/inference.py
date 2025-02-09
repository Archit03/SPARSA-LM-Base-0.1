import torch
from transformers import PreTrainedTokenizerFast
from model import Transformer, TransformerConfig
import yaml
import os

# Load config
config_path = "config/training_config.yaml"  # Adjust if needed
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load tokenizer
tokenizer_path = config["tokenizer"]["path"]
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Load model config
model_config = TransformerConfig(
    vocab_size=config["model"]["vocab_size"],
    d_model=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    num_heads=config["model"]["num_heads"],
    d_ff=config["model"]["ff_dim"],
    max_seq_len=config["model"]["max_seq_len"],
    dropout=config["model"]["dropout"],
    activation=config["model"]["activation"],
    use_checkpointing=config["model"]["use_checkpointing"],
    tie_embeddings=config["model"]["tie_embeddings"],
    window_size=config["model"]["window_size"],
    global_tokens=config["model"]["global_tokens"],
    use_reentrant=config["model"]["use_reentrant"],
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(model_config).to(device)

# Load checkpoint
checkpoint_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pt")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Model loaded from {checkpoint_path}")
else:
    raise FileNotFoundError(f"❌ Checkpoint not found at {checkpoint_path}")

# Set to evaluation mode
model.eval()

def generate_text(prompt, max_length=50):
    """Generate text given an input prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Ensure it fits model input size
    if input_ids.size(1) > config["model"]["max_seq_len"]:
        input_ids = input_ids[:, -config["model"]["max_seq_len"]:]

    with torch.no_grad():
        output = model(src=input_ids)

    # Convert to text
    generated_ids = torch.argmax(output, dim=-1).squeeze()
    generated_text = tokenizer.decode(generated_ids.cpu().tolist(), skip_special_tokens=True)
    
    return generated_text

# CLI loop for user input
print("\n🔥 Welcome to LuminaLM CLI Inference! 🔥")
print("Type your prompt and press Enter to generate text.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("💬 Enter prompt: ")
    if user_input.lower() == "exit":
        print("\n👋 Exiting... Thank you for using LuminaLM!")
        break

    generated_output = generate_text(user_input, max_length=50)
    print("\n📝 Generated Output:")
    print(generated_output)
    print("\n" + "="*50 + "\n")
