import logging
from tokenizers import Tokenizer

def setup_logging():
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def test_tokenizer(tokenizer_path):
    """Test the tokenizer functionality."""
    # Load the tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    logging.info("Tokenizer loaded successfully.")

    # Test tokenization
    test_sentence = "This is a [MASK] example."
    encoded = tokenizer.encode(test_sentence)
    logging.info(f"Tokens: {encoded.tokens}")
    logging.info(f"Token IDs: {encoded.ids}")

    # Test padding
    encoded.pad(128)
    logging.info(f"Padded Tokens: {encoded.tokens}")
    logging.info(f"Padded Token IDs: {encoded.ids}")

    # Test truncation with long input
    long_sentence = (
        "This is a very long sentence that exceeds the maximum length set for the tokenizer. "
        "The tokenizer should truncate this input correctly when truncation is enabled."
    )
    encoded_long = tokenizer.encode(long_sentence)
    tokenizer.enable_truncation(max_length=128)
    encoded_long = tokenizer.encode(long_sentence)
    logging.info(f"Truncated Tokens: {encoded_long.tokens}")
    logging.info(f"Truncated Token IDs: {encoded_long.ids}")

    # Verify special tokens
    special_sentence = "[CLS] This is a [MASK] example. [SEP]"
    encoded_special = tokenizer.encode(special_sentence)
    logging.info(f"Special Tokens: {encoded_special.tokens}")
    logging.info(f"Special Token IDs: {encoded_special.ids}")

if __name__ == "__main__":
    setup_logging()
    tokenizer_path = r"C:\Users\ASUS\Desktop\SPARSA-LM-Base 0.1\data\processed\tokenizer\tokenizer.json"
    test_tokenizer(tokenizer_path)
