import os
import yaml
import torch
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional
from torch import inference_mode
from transformers import PreTrainedTokenizerFast

###############################################################################
# 1. CONFIGURATION & TOKENIZER LOADING
###############################################################################
def load_config(config_path: str = "config/inference_config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """
    Load and configure the tokenizer for inference.
    Ensures essential special tokens are set and saved.
    """
    # Ensure we have a directory; if tokenizer_path points to a file, use its parent
    tokenizer_dir = os.path.dirname(tokenizer_path) if os.path.isfile(tokenizer_path) else tokenizer_path

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    except Exception as e:
        logging.error(f"Error loading tokenizer from {tokenizer_dir}: {e}")
        raise

    # Define expected special tokens
    expected_special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]"
    }

    # Check and set special tokens if missing; also build a mapping for logging
    special_mapping = {}
    missing_tokens = {}
    for key, token in expected_special_tokens.items():
        current = getattr(tokenizer, key, None)
        if current is None:
            setattr(tokenizer, key, token)
            missing_tokens[key] = token
            token_id = None
        else:
            token_id = tokenizer.convert_tokens_to_ids(current)
        special_mapping[key] = (token, token_id)
    
    if missing_tokens:
        logging.info(f"Missing special tokens detected: {missing_tokens}. Adding them...")
        tokenizer.add_special_tokens(missing_tokens)
        vocab = tokenizer.get_vocab()
        for key, token in expected_special_tokens.items():
            token_id = vocab.get(token)
            special_mapping[key] = (token, token_id)
        logging.info(f"Special tokens mapping (after adding missing tokens): {special_mapping}")
    else:
        logging.info(f"All special tokens are present with IDs: {special_mapping}")

    # Explicitly set attributes and disable extra cleanup for inference
    tokenizer.pad_token = expected_special_tokens["pad_token"]
    tokenizer.unk_token = expected_special_tokens["unk_token"]
    tokenizer.bos_token = expected_special_tokens["bos_token"]
    tokenizer.eos_token = expected_special_tokens["eos_token"]
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.trim_offsets = True

    # Save updated tokenizer for future loads
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer

def test_tokenizer_functionality(tokenizer: PreTrainedTokenizerFast, test_text: str = "Hello, how are you?") -> bool:
    """
    Test tokenizer encoding and decoding to verify that special tokens and vocabulary
    are loaded correctly.
    """
    print("\n=== Testing Tokenizer ===")
    try:
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Special tokens mapping: {tokenizer.special_tokens_map}")
        print(f"\nTest text: {test_text}")
        encoded_ids = tokenizer.encode(test_text)
        decoded_text = tokenizer.decode(encoded_ids)
        print(f"Encoded IDs: {encoded_ids}")
        print(f"Decoded text: {decoded_text}")

        # Also test by manually adding BOS and EOS tokens
        special_text = f"{tokenizer.bos_token} {test_text} {tokenizer.eos_token}"
        special_ids = tokenizer.encode(special_text)
        special_decoded = tokenizer.decode(special_ids, skip_special_tokens=False)
        print("\nWith special tokens:")
        print(f"Input: {special_text}")
        print(f"Encoded: {special_ids}")
        print(f"Decoded: {special_decoded}")

        return True
    except Exception as e:
        print(f"Tokenizer test failed: {e}")
        return False

###############################################################################
# 2. MODEL LOADING FOR INFERENCE
###############################################################################
def setup_model_for_inference(config: Dict[str, Any]):
    """
    Setup model for inference using the custom Transformer class.
    Assumes the checkpoint contains a 'model_state_dict' key.
    """
    from model import Transformer, TransformerConfig

    try:
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
    except Exception as e:
        logging.error(f"Error creating model configuration: {e}")
        raise

    model = Transformer(model_config)

    checkpoint_path = config['inference']['model_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError("Checkpoint missing 'model_state_dict' key")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading model checkpoint: {e}")
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model

###############################################################################
# 3. TEXT GENERATION
###############################################################################
def generate_text(
    model,
    tokenizer,
    prompt: str,
    config: Dict[str, Any],
    max_length: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> str:
    """
    Generate text using a token-by-token loop.
    No noise is injected during inference.
    """
    gen_cfg = config.get('generation', {})
    max_length = max_length or gen_cfg.get('max_length', 50)
    temperature = temperature or gen_cfg.get('temperature', 1.0)
    top_k = top_k or gen_cfg.get('top_k', 50)
    top_p = top_p or gen_cfg.get('top_p', 1.0)

    device = next(model.parameters()).device

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Invalid prompt")

    # Prepend BOS token if not present
    if not prompt.startswith(tokenizer.bos_token):
        prompt = f"{tokenizer.bos_token} {prompt}".strip()

    # Remove EOS at the end to allow generation to continue
    if prompt.endswith(tokenizer.eos_token):
        print("Prompt already contains EOS token at the end, removing it to allow generation.")
        prompt = prompt[:-len(tokenizer.eos_token)].strip()

    # Encode the prompt
    try:
        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False
        ).to(device)
    except Exception as e:
        logging.error(f"Tokenization failed: {e}")
        raise

    model_max_len = config['model'].get('max_seq_len', 128)
    if input_ids.size(1) > model_max_len:
        input_ids = input_ids[:, -model_max_len:]

    prompt_length = input_ids.size(1)
    generated_ids = input_ids[0].tolist()

    with inference_mode():
        for _ in tqdm(range(max_length), desc="Generating"):
            curr_ids = generated_ids[-model_max_len:]
            curr_input = torch.tensor([curr_ids], device=device)
            outputs = model(curr_input)
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
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 0] = False  # Always keep the highest probability token
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()

            if next_token_id == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token_id)

    try:
        generated_text = tokenizer.decode(generated_ids[prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if not isinstance(generated_text, str):
            raise ValueError("Decoded text is not a string")
        return generated_text.strip()
    except Exception as e:
        logging.error(f"Error decoding generated tokens: {e}")
        raise

###############################################################################
# 4. MAIN INTERACTIVE LOOP & TEST
###############################################################################
def main():
    try:
        # Load configuration
        config = load_config("config/inference_config.yaml")

        # Load tokenizer & model
        tokenizer = load_tokenizer(config['tokenizer']['path'])
        model = setup_model_for_inference(config)

        # Test the tokenizer functionality
        if not test_tokenizer_functionality(tokenizer):
            raise RuntimeError("Tokenizer verification failed")

        print("\n🔥 Welcome to LuminaLM Text Generation! 🔥")
        print("Type your prompt and press Enter to generate text.")
        print("Type 'exit' to quit.\n")

        while True:
            prompt = input("💬 Enter prompt: ").strip()
            if prompt.lower() == 'exit':
                print("\n👋 Goodbye!")
                break

            try:
                output = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    config=config
                )
                print(f"\n📝 Generated Output:\n{'-'*50}")
                print(output)
                print('-'*50)
            except Exception as e:
                print(f"⚠️ Error during generation: {str(e)}")
                logging.error("Generation error:", exc_info=True)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logging.error("Fatal error:", exc_info=True)
        print(f"⚠️ Fatal error: {str(e)}")
        raise

###############################################################################
# 5. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
