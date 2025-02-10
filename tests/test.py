import logging
from tokenizers import Tokenizer

def setup_logging():
    """Setup logging for the script with clear formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def test_tokenizer(tokenizer_path):
    """Test and debug the tokenizer functionality."""
    try:
        # Load tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logging.info(f"✅ Tokenizer loaded successfully from: {tokenizer_path}")

        # Test tokenization
        test_sentence = "This is a [MASK] example."
        encoded = tokenizer.encode(test_sentence)
        logging.info(f"📝 Test Sentence: {test_sentence}")
        logging.info(f"🔢 Tokens: {encoded.tokens}")
        logging.info(f"🔢 Token IDs: {encoded.ids}")

        # Verify special tokens
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
        missing_tokens = [token for token in special_tokens if token not in tokenizer.get_vocab()]

        if missing_tokens:
            logging.warning(f"🚨 Missing Special Tokens: {missing_tokens}")
        else:
            logging.info(f"✅ All special tokens present: {special_tokens}")

        # Test padding
        tokenizer.enable_padding(length=128)  # Ensure tokenizer has padding enabled
        encoded.pad(128)
        logging.info(f"🔢 Padded Tokens: {encoded.tokens}")
        logging.info(f"🔢 Padded Token IDs: {encoded.ids}")

        # Test truncation
        long_sentence = (
            "This is a very long sentence that exceeds the maximum length set for the tokenizer. "
            "The tokenizer should truncate this input correctly when truncation is enabled."
        )
        tokenizer.enable_truncation(max_length=128)
        encoded_long = tokenizer.encode(long_sentence)
        logging.info(f"🔢 Truncated Tokens: {encoded_long.tokens}")
        logging.info(f"🔢 Truncated Token IDs: {encoded_long.ids}")

        # Test BOS & EOS tokens
        bos_eos_sentence = "[BOS] This is a test. [EOS]"
        encoded_bos_eos = tokenizer.encode(bos_eos_sentence)
        logging.info(f"🔢 BOS/EOS Test Sentence: {bos_eos_sentence}")
        logging.info(f"🔢 BOS/EOS Tokens: {encoded_bos_eos.tokens}")
        logging.info(f"🔢 BOS/EOS Token IDs: {encoded_bos_eos.ids}")

        logging.info("✅ Tokenizer test completed successfully!")

    except Exception as e:
        logging.error(f"❌ Error during tokenizer test: {str(e)}")

if __name__ == "__main__":
    setup_logging()
    tokenizer_path = r"C:\Users\ASUS\Desktop\SPARSA-LM-Base 0.1\data\processed\tokenizer\tokenizer.json"
    test_tokenizer(tokenizer_path)
