import torch
from transformers import PreTrainedTokenizerFast
from model import Transformer, TransformerConfig
import yaml
import os

# -----------------------------------------------
# Load Configuration
# -----------------------------------------------
def load_config(config_path="config/inference_config.yaml"):
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# -----------------------------------------------
# Load Tokenizer
# -----------------------------------------------
def load_tokenizer(tokenizer_path):
    """Load the tokenizer from the given path and ensure special tokens are set."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Ensure special tokens are correctly set
    special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
    for key, value in special_tokens.items():
        if getattr(tokenizer, key) is None:
            setattr(tokenizer, key, value)

    # Debugging: Verify tokenizer properties
    print(f"✅ Tokenizer loaded from {tokenizer_path}")
    print(f"📝 Vocab Size: {tokenizer.vocab_size}")
    print(f"🔹 Special Tokens: PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}, BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}")

    # Tokenization test
    test_sentence = "Hello, this is a test!"
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)
    print(f"🧪 Tokenization Test: '{test_sentence}' → {encoded}")
    print(f"📝 Decoded Output: {decoded}")

    return tokenizer

# -----------------------------------------------
# Load Model Configuration
# -----------------------------------------------
def get_model_config(config):
    """Load model hyperparameters from the configuration file."""
    return TransformerConfig(
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

# -----------------------------------------------
# Load Model & Checkpoint
# -----------------------------------------------
def load_model(config, device):
    """Load Transformer model from the best checkpoint."""
    model = Transformer(get_model_config(config)).to(device)

    checkpoint_dir = config["training"]["checkpoint_dir"]
    model_checkpoint = config["training"].get("model_checkpoint", "best_LuminaLM_model.pt")  # Default to best model
    checkpoint_path = os.path.join(checkpoint_dir, model_checkpoint)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Model loaded from {checkpoint_path}")

    model.eval()
    return model

# -----------------------------------------------
# Text Generation Function
# -----------------------------------------------
def generate_text(model, tokenizer, config, prompt, max_length=50):
    """Generate text given an input prompt."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Ensure input fits within model's max sequence length
    if input_ids.size(1) > config["model"]["max_seq_len"]:
        input_ids = input_ids[:, -config["model"]["max_seq_len"]:]  # Trim input

    with torch.no_grad():
        output = model(src=input_ids)

    # Debugging: Inspect First Token Output
    first_token_id = torch.argmax(output[0, 0]).item()
    print(f"🔍 First Generated Token ID: {first_token_id} ({tokenizer.decode([first_token_id])})")

    # Convert logits to token indices with improved decoding
    generated_ids = torch.argmax(output, dim=-1).squeeze().tolist()
    filtered_ids = [token for token in generated_ids if token not in [tokenizer.pad_token_id, tokenizer.unk_token_id]]

    generated_text = tokenizer.decode(filtered_ids, skip_special_tokens=True)
    
    return generated_text

# -----------------------------------------------
# Main Function for CLI Inference
# -----------------------------------------------
def main():
    """CLI loop for user input to generate text from the model."""
    try:
        config = load_config()
        tokenizer = load_tokenizer(config["tokenizer"]["path"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(config, device)

        print("\n🔥 Welcome to LuminaLM CLI Inference! 🔥")
        print("Type your prompt and press Enter to generate text.")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("💬 Enter prompt: ")
            if user_input.lower() == "exit":
                print("\n👋 Exiting... Thank you for using LuminaLM!")
                break

            generated_output = generate_text(model, tokenizer, config, user_input, max_length=50)
            print("\n📝 Generated Output:")
            print(generated_output)
            print("\n" + "="*50 + "\n")

    except Exception as e:
        print(f"⚠️ Error: {str(e)}")

# -----------------------------------------------
# Entry Point
# -----------------------------------------------
if __name__ == "__main__":
    main()
