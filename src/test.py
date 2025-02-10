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
    """Load and validate the tokenizer."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"\n✅ Tokenizer loaded from: {tokenizer_path}")
    print(f"📝 Vocab Size: {tokenizer.vocab_size}")

    # Ensure special tokens are set correctly
    special_tokens = {
        "PAD": tokenizer.pad_token_id,
        "UNK": tokenizer.unk_token_id,
        "BOS": tokenizer.bos_token_id,
        "EOS": tokenizer.eos_token_id,
    }
    
    print(f"🔹 Special Tokens: {special_tokens}")

    # Fix missing special tokens
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({"unk_token": "[UNK]"})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "[BOS]"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})

    # Save tokenizer after modification
    tokenizer.save_pretrained(tokenizer_path)

    # Run tokenization test
    sample_text = "Hello, how are you?"
    tokenized = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokenized)
    
    print("\n🧪 Tokenization Test:")
    print(f"📝 Original Text: {sample_text}")
    print(f"🔢 Tokenized: {tokenized}")
    print(f"📝 Decoded: {decoded}")

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

    checkpoint_path = config["inference"]["model_path"]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n✅ Model loaded from {checkpoint_path}")

    model.eval()
    return model

# -----------------------------------------------
# Check Model Logits (Output Distribution)
# -----------------------------------------------
def check_model_logits(model, tokenizer, device):
    """Check if model output logits are reasonable."""
    print("\n🔍 Checking model logits...")

    # Sample prompt
    test_prompt = "Hello, how are you?"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        output_logits = model(src=input_ids)

    # Inspect raw output values
    print("🔍 Model Logits (First Token):", output_logits[:, 0, :])
    print("🔍 Model Logits Stats: Mean:", output_logits.mean().item(), "Std:", output_logits.std().item())

    # Check if logits contain NaN or Inf
    if torch.isnan(output_logits).any() or torch.isinf(output_logits).any():
        print("🚨 Warning: Logits contain NaN or Inf values!")
    else:
        print("✅ Logits are valid.")

# -----------------------------------------------
# Check Token Embeddings
# -----------------------------------------------
def check_token_embeddings(model):
    """Check if token embeddings are properly learned."""
    print("\n🔍 Checking token embeddings...")

    # Extract embedding weights
    embedding_weights = model.embedding.weight.detach().cpu()

    # Print embedding stats
    print("🔍 Token Embedding Stats:")
    print("   Mean:", embedding_weights.mean().item())
    print("   Std:", embedding_weights.std().item())
    print("   Min:", embedding_weights.min().item())
    print("   Max:", embedding_weights.max().item())

    # Check for NaN values
    if torch.isnan(embedding_weights).any() or torch.isinf(embedding_weights).any():
        print("🚨 Warning: Token embeddings contain NaN or Inf values!")
    else:
        print("✅ Token embeddings are valid.")

# -----------------------------------------------
# Text Generation with Sampling
# -----------------------------------------------
def generate_text(model, tokenizer, config, prompt, max_length=50, temperature=0.9, top_k=10):
    """Generate text with temperature and top-k sampling."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Ensure input fits within model's max sequence length
    if input_ids.size(1) > config["model"]["max_seq_len"]:
        input_ids = input_ids[:, -config["model"]["max_seq_len"]:]  # Trim input

    with torch.no_grad():
        output_logits = model(src=input_ids)

    # Apply softmax and sampling
    probabilities = torch.nn.functional.softmax(output_logits / temperature, dim=-1)
    sampled_ids = torch.multinomial(probabilities.squeeze(), num_samples=1).squeeze()
    generated_text = tokenizer.decode(sampled_ids.cpu().tolist(), skip_special_tokens=True)

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

        check_model_logits(model, tokenizer, device)
        check_token_embeddings(model)

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
