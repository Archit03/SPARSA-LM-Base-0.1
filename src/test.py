import torch
from transformers import PreTrainedTokenizerFast
import os

# ✅ Update with your tokenizer path
TOKENIZER_PATH = "C:/Users/ASUS/Desktop/SPARSA-LM-Base 0.1/data/processed/tokenizer"

# 🔹 Load Tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH, add_prefix_space=True)

# 🔹 Ensure special tokens are set
special_tokens = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
}
for key, value in special_tokens.items():
    if getattr(tokenizer, key) is None:
        setattr(tokenizer, key, value)

# ✅ Debug Tokenizer Properties
print(f"\n✅ Tokenizer loaded from: {TOKENIZER_PATH}")
print(f"📝 Vocab Size: {tokenizer.vocab_size}")
print(f"🔹 Special Tokens: PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}, BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}")

# 🧪 Test Sentences
test_sentences = [
    "Hello, how are you?",
    "The AI revolution is here!",
    "Can LuminaLM generate meaningful text?",
    "This is a tokenizer test with BPE encoding.",
]

# 🔹 Tokenization & Decoding Test
for sentence in test_sentences:
    encoded = tokenizer.encode(sentence)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)  # 🔹 FIXED HERE!

    print("\n==============================")
    print(f"📝 Original Text: {sentence}")
    print(f"🔢 Tokenized: {encoded}")
    print(f"📝 Decoded: {decoded}")  # ✅ Should now be clean

# ✅ Test Passed
print("\n✅ Tokenizer test completed successfully!")
