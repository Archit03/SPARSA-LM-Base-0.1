import os
import yaml
import math
import torch
from torch import no_grad
from transformers import PreTrainedTokenizerFast

# Import your Transformer classes as defined in your training/inference code
from model import Transformer, TransformerConfig

def load_config(config_path="config/inference_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    # Ensure special tokens exist
    special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
    for key, value in special_tokens.items():
        if getattr(tokenizer, key, None) is None:
            setattr(tokenizer, key, value)
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.trim_offsets = True
    return tokenizer

def setup_model_for_inference(config):
    model_config = TransformerConfig(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['ff_dim'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        use_checkpointing=config['model']['use_checkpointing'],
        tie_embeddings=config['model']['tie_embeddings'],
        window_size=config['model']['window_size'],
        global_tokens=config['model']['global_tokens'],
        use_reentrant=config['model']['use_reentrant']
    )
    model = Transformer(model_config)

    checkpoint_path = config['inference']['model_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if 'model_state_dict' not in checkpoint:
        raise RuntimeError("Checkpoint missing 'model_state_dict' key")

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")

    # Debug: Check embedding matrix dimensions using the actual attribute
    embedding_matrix = model.embedding.weight  # Use model.embedding instead of token_embeddings
    print("Embedding matrix shape:", embedding_matrix.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model

def debug_generate_text(model, tokenizer, prompt, config, max_length=None, temperature=None, top_k=None, top_p=None):
    # Set default generation parameters from config if not provided
    gen_cfg = config.get('generation', {})
    max_length = max_length or gen_cfg.get('max_length', 50)
    temperature = temperature or gen_cfg.get('temperature', 1.0)
    top_k = top_k or gen_cfg.get('top_k', 50)
    top_p = top_p or gen_cfg.get('top_p', 1.0)

    device = next(model.parameters()).device

    # Ensure prompt begins with the BOS token
    if not prompt.startswith(tokenizer.bos_token):
        prompt = f"{tokenizer.bos_token} {prompt}".strip()

    # Encode prompt
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False
    ).to(device)

    # Convert to list of IDs and remove EOS token at the end if present
    generated_ids = input_ids[0].tolist()
    if generated_ids and generated_ids[-1] == tokenizer.eos_token_id:
        print("Prompt already contains EOS token at the end, removing it to allow generation.")
        generated_ids = generated_ids[:-1]
    print("Initial token IDs:", generated_ids)

    # Trim input if it exceeds model's max sequence length
    model_max_len = config['model'].get('max_seq_len', 512)
    if len(generated_ids) > model_max_len:
        generated_ids = generated_ids[-model_max_len:]

    with no_grad():
        for i in range(max_length):
            # Get the most recent tokens (up to model_max_len)
            curr_ids = generated_ids[-model_max_len:]
            curr_input = torch.tensor([curr_ids], device=device)

            outputs = model(curr_input)
            # Assume outputs shape is [batch, seq_len, vocab_size]
            logits = outputs[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                vals, idx = torch.topk(logits, top_k)
                keep_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, idx, True)
                logits[~keep_mask] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 0] = False  # Always keep the first token
                for row_i, row in enumerate(sorted_indices_to_remove):
                    remove_idx = sorted_indices[row_i, row]
                    logits[row_i, remove_idx] = float('-inf')

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Debug prints for the current step
            print(f"\nStep {i + 1}:")
            print("Current token IDs:", generated_ids)
            print("Logits for next token:", logits)
            print("Probabilities for next token:", probs)

            # Sample next token
            next_token_id = torch.multinomial(probs, 1).item()
            print("Sampled token ID:", next_token_id)

            # If the sampled token is EOS, end generation
            if next_token_id == tokenizer.eos_token_id:
                print("EOS token found during generation. Ending generation.")
                break

            generated_ids.append(next_token_id)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

def main():
    config = load_config("config/inference_config.yaml")
    tokenizer = load_tokenizer(config['tokenizer']['path'])
    model = setup_model_for_inference(config)

    prompt = input("Enter a prompt: ")
    debug_text = debug_generate_text(model, tokenizer, prompt, config)
    print("\nFinal Generated Text:")
    print(debug_text)

if __name__ == "__main__":
    main()
