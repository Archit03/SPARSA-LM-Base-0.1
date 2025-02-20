import sys
import os
import yaml
import logging
from transformers import PreTrainedTokenizerFast

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_special_tokens_mapping(tokenizer, expected_special_tokens: dict):
    mapping = {}
    vocab = tokenizer.get_vocab()
    for key, token in expected_special_tokens.items():
        # First try to get the token string from special_tokens_map
        token_str = tokenizer.special_tokens_map.get(key)
        # If not set, use our expected token string
        if token_str is None:
            token_str = token
        token_id = vocab.get(token_str)
        mapping[key] = (token_str, token_id)
    return mapping

def ensure_special_token_attributes(tokenizer, expected_special_tokens: dict):
    # Explicitly set the tokenizer's attributes using the mapping from its special_tokens_map
    tokenizer.pad_token = tokenizer.special_tokens_map.get("pad_token", "[PAD]")
    tokenizer.unk_token = tokenizer.special_tokens_map.get("unk_token", "[UNK]")
    tokenizer.cls_token = tokenizer.special_tokens_map.get("cls_token", "[CLS]")
    tokenizer.sep_token = tokenizer.special_tokens_map.get("sep_token", "[SEP]")
    tokenizer.mask_token = tokenizer.special_tokens_map.get("mask_token", "[MASK]")
    tokenizer.bos_token = tokenizer.special_tokens_map.get("bos_token", "[BOS]")
    tokenizer.eos_token = tokenizer.special_tokens_map.get("eos_token", "[EOS]")

def test_tokenizer(tokenizer_path: str):
    # If the path is a file, use its directory
    if os.path.isfile(tokenizer_path):
        tokenizer_dir = os.path.dirname(tokenizer_path)
    else:
        tokenizer_dir = tokenizer_path

    # Load tokenizer from directory
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("TokenizerTest")

    logger.info(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")

    # Define the expected special tokens
    expected_special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]"
    }

    # Retrieve mapping derived from the tokenizer's vocabulary
    special_mapping = get_special_tokens_mapping(tokenizer, expected_special_tokens)
    logger.info(f"Special tokens mapping (derived from vocabulary): {special_mapping}")

    # Set special token attributes explicitly
    ensure_special_token_attributes(tokenizer, expected_special_tokens)

    # Encode a sample text and convert the IDs back to tokens
    sample_text = "Hello world! This is a test."
    encoded_ids = tokenizer.encode(sample_text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
    print("Encoded IDs:", encoded_ids)
    print("Decoded tokens:", tokens)
    
    # Log the final special tokens mapping by checking the attributes directly
    final_mapping = {}
    for key in expected_special_tokens:
        token_str = getattr(tokenizer, key, None)
        token_id = getattr(tokenizer, f"{key}_id", None)
        final_mapping[key] = (token_str, token_id)
    logger.info(f"Final special tokens mapping: {final_mapping}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_tokenizer.py <tokenizer_path>")
        sys.exit(1)
    
    tokenizer_path = sys.argv[1]
    test_tokenizer(tokenizer_path)
