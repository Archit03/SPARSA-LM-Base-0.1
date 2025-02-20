import os
import yaml
import torch
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional
from torch import inference_mode
from transformers import PreTrainedTokenizerFast

###############################################################################
# 1. CONFIG & TOKENIZER LOADING
###############################################################################
def load_config(config_path: str = "config/inference_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """Load and configure the tokenizer with proper cleanup settings."""
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

    # Adjust tokenizer settings
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.trim_offsets = True

    return tokenizer

###############################################################################
# 2. MODEL LOADING FOR INFERENCE
###############################################################################
def setup_model_for_inference(config: Dict[str, Any]):
    """
    Setup model for inference using the custom 'Transformer' class.
    Assumes the checkpoint has 'model_state_dict' under the key as in training.
    """
    from model import Transformer, TransformerConfig

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

    # Load checkpoint
    checkpoint_path = config['inference']['model_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' not in checkpoint:
        raise RuntimeError("Checkpoint missing 'model_state_dict' key")

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from {checkpoint_path}")

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
    Generate text with a token-by-token loop.
    NOTE: This version does not inject any noise during inference.
    """
    # 3.1. Load defaults from config if not given
    gen_cfg = config.get('generation', {})
    max_length = max_length or gen_cfg.get('max_length', 50)
    temperature = temperature or gen_cfg.get('temperature', 1.0)
    top_k = top_k or gen_cfg.get('top_k', 50)
    top_p = top_p or gen_cfg.get('top_p', 1.0)

    device = next(model.parameters()).device

    # Format prompt with BOS if needed
    if not prompt.startswith(tokenizer.bos_token):
        prompt = f"{tokenizer.bos_token} {prompt}".strip()

    # If prompt ends with EOS, remove it to allow generation
    if prompt.endswith(tokenizer.eos_token):
        print("Prompt already contains EOS token at the end, removing it to allow generation.")
        prompt = prompt[:-len(tokenizer.eos_token)].strip()

    # 3.2. Encode prompt
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False  # We only have one prompt here
    ).to(device)

    # Trim if exceeds model's max_seq_len
    model_max_len = config['model'].get('max_seq_len', 128)
    if input_ids.size(1) > model_max_len:
        input_ids = input_ids[:, -model_max_len:]

    # Record the length of the prompt tokens so we can return only the generated part
    prompt_length = input_ids.size(1)

    # We'll store all generated tokens in this list
    generated_ids = input_ids[0].tolist()

    # 3.3. Token-by-token generation
    with inference_mode():
        for _ in tqdm(range(max_length), desc="Generating"):
            # Prepare current input (truncate to model_max_len)
            curr_ids = generated_ids[-model_max_len:]
            curr_input = torch.tensor([curr_ids], device=device)

            outputs = model(curr_input)
            # Assume the shape is [batch, seq_len, vocab_size]
            logits = outputs[:, -1, :]

            # 3.4. Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # 3.5. Top-k filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                vals, idx = torch.topk(logits, top_k)
                keep_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, idx, True)
                logits[~keep_mask] = float('-inf')

            # 3.6. Top-p (nucleus) filtering
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 0] = False  # Always keep the first token
                for row_i, row in enumerate(sorted_indices_to_remove):
                    remove_idx = sorted_indices[row_i, row]
                    logits[row_i, remove_idx] = float('-inf')

            # 3.7. Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()

            # Stop if EOS token is generated
            if next_token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)

    # 3.8. Decode only the generated portion (after the prompt)
    generated_text = tokenizer.decode(generated_ids[prompt_length:], skip_special_tokens=True)
    generated_text = generated_text.strip()

    return generated_text

###############################################################################
# 4. MAIN INTERACTIVE LOOP
###############################################################################
def main():
    try:
        # Load config
        config = load_config("config/inference_config.yaml")

        # Load tokenizer & model
        tokenizer = load_tokenizer(config['tokenizer']['path'])
        model = setup_model_for_inference(config)

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

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"⚠️ Fatal error: {str(e)}")
        raise

###############################################################################
# 5. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
