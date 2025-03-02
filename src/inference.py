import os
import yaml
import torch
import logging
import time
from tqdm import tqdm
from typing import Dict, Any, Optional
from torch import inference_mode
from transformers import PreTrainedTokenizerFast
import io
import sys
import json

###############################################################################
# 0. ADVANCED LOGGING SETUP
###############################################################################
def setup_logging(log_file="lumina_inference.log", console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Set up detailed logging to both console and file, forcing UTF-8 output to avoid
    UnicodeEncodeError on Windows consoles with CP1252 or similar code pages.
    """
    # Force stdout and stderr to use UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
    )

    # Console handler (less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (more detailed)
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logging.debug(
        "Logging system initialized with console level %s and file level %s",
        logging.getLevelName(console_level),
        logging.getLevelName(file_level)
    )
    return logger

###############################################################################
# 1. CONFIGURATION & TOKENIZER LOADING WITH DETAILED LOGGING
###############################################################################
def load_config(config_path: str = "config/inference_config.yaml") -> Dict[str, Any]:
    logging.info(f"Attempting to load config from {config_path}")
    start_time = time.time()
    if not os.path.exists(config_path):
        logging.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.debug(f"Config loaded successfully in {time.time() - start_time:.2f} seconds")
        logging.info(f"Config sections found: {list(config.keys())}")
        if 'model' in config:
            logging.info(f"Model config: vocab_size={config['model'].get('vocab_size')}, "
                         f"hidden_dim={config['model'].get('hidden_dim')}, "
                         f"layers={config['model'].get('num_layers')}, "
                         f"heads={config['model'].get('num_heads')}")
        if 'generation' in config:
            logging.info(f"Generation config: max_length={config['generation'].get('max_length')}, "
                         f"temperature={config['generation'].get('temperature')}, "
                         f"top_k={config['generation'].get('top_k')}, "
                         f"top_p={config['generation'].get('top_p')}, "
                         f"repetition_penalty={config['generation'].get('repetition_penalty')}, "
                         f"num_beams={config['generation'].get('num_beams')}, "
                         f"no_repeat_ngram_size={config['generation'].get('no_repeat_ngram_size')}, "
                         f"length_penalty={config['generation'].get('length_penalty')}, "
                         f"early_stopping={config['generation'].get('early_stopping')}")
        if 'inference' in config:
            logging.info(f"Inference config: model_path={config['inference'].get('model_path')}")
        if 'tokenizer' in config:
            logging.info(f"Tokenizer config: path={config['tokenizer'].get('path')}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}", exc_info=True)
        raise

def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    start_time = time.time()
    tokenizer_dir = os.path.dirname(tokenizer_path) if os.path.isfile(tokenizer_path) else tokenizer_path
    logging.debug(f"Using tokenizer directory: {tokenizer_dir}")
    if not os.path.exists(tokenizer_dir):
        logging.error(f"Tokenizer directory not found: {tokenizer_dir}")
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    try:
        logging.debug("Initializing PreTrainedTokenizerFast")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        logging.debug(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
        logging.info(f"Initial tokenizer vocabulary size: {len(tokenizer.get_vocab())}")
    except Exception as e:
        logging.error(f"Error loading tokenizer from {tokenizer_dir}: {e}", exc_info=True)
        raise

    # Define expected special tokens
    expected_special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]"
    }
    logging.debug(f"Checking for expected special tokens: {expected_special_tokens}")

    special_mapping = {}
    missing_tokens = {}
    for key, token in expected_special_tokens.items():
        current = getattr(tokenizer, key, None)
        logging.debug(f"Checking '{key}': current={current}")
        if current is None:
            logging.warning(f"Special token '{key}' is missing")
            setattr(tokenizer, key, token)
            missing_tokens[key] = token
            token_id = None
        else:
            token_id = tokenizer.convert_tokens_to_ids(current)
            logging.debug(f"Token '{current}' has ID {token_id}")
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

    logging.debug("Setting explicit tokenizer attributes")
    tokenizer.pad_token = expected_special_tokens["pad_token"]
    tokenizer.unk_token = expected_special_tokens["unk_token"]
    tokenizer.bos_token = expected_special_tokens["bos_token"]
    tokenizer.eos_token = expected_special_tokens["eos_token"]
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.trim_offsets = True

    logging.debug(f"Saving updated tokenizer to {tokenizer_dir}")
    tokenizer.save_pretrained(tokenizer_dir)
    logging.info(f"Tokenizer configuration complete in {time.time() - start_time:.2f} seconds")
    logging.info(f"Final tokenizer vocabulary size: {len(tokenizer.get_vocab())}")
    return tokenizer

def test_tokenizer_functionality(tokenizer: PreTrainedTokenizerFast, test_text: str = "Hello, how are you?") -> bool:
    logging.info("Testing tokenizer functionality")
    print("\n=== Testing Tokenizer ===")
    start_time = time.time()
    try:
        vocab_size = tokenizer.vocab_size
        logging.debug(f"Vocabulary size: {vocab_size}")
        print(f"Vocabulary size: {vocab_size}")
        special_tokens = tokenizer.special_tokens_map
        logging.debug(f"Special tokens mapping: {special_tokens}")
        print(f"Special tokens mapping: {special_tokens}")
        print(f"\nTest text: {test_text}")
        encoded_ids = tokenizer.encode(test_text)
        decoded_text = tokenizer.decode(encoded_ids)
        logging.debug(f"Encoded IDs: {encoded_ids}")
        logging.debug(f"Decoded text: {decoded_text}")
        print(f"Encoded IDs: {encoded_ids}")
        print(f"Decoded text: {decoded_text}")
        special_text = f"{tokenizer.bos_token} {test_text} {tokenizer.eos_token}"
        special_ids = tokenizer.encode(special_text)
        special_decoded = tokenizer.decode(special_ids, skip_special_tokens=False)
        logging.debug(f"Special text input: {special_text}")
        logging.debug(f"Special text encoded: {special_ids}")
        logging.debug(f"Special text decoded: {special_decoded}")
        print("\nWith special tokens:")
        print(f"Input: {special_text}")
        print(f"Encoded: {special_ids}")
        print(f"Decoded: {special_decoded}")
        bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        logging.debug(f"BOS token ID: {bos_id}, EOS token ID: {eos_id}")
        if bos_id in special_ids and eos_id in special_ids:
            logging.debug("Special tokens correctly identified in encoded sequence")
        else:
            logging.warning("Special tokens not correctly identified in encoded sequence")
        roundtrip = tokenizer.decode(tokenizer.encode(test_text))
        logging.debug(f"Roundtrip text: {roundtrip}")
        if test_text in roundtrip:
            logging.debug("Roundtrip encoding/decoding successful")
        else:
            logging.warning(f"Roundtrip encoding/decoding issue: {test_text} -> {roundtrip}")
        logging.info(f"Tokenizer test completed successfully in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        logging.error(f"Tokenizer test failed: {e}", exc_info=True)
        print(f"Tokenizer test failed: {e}")
        return False

###############################################################################
# 2. MODEL LOADING FOR INFERENCE WITH DETAILED LOGGING
###############################################################################
def setup_model_for_inference(config: Dict[str, Any]):
    logging.info("Setting up model for inference")
    start_time = time.time()
    from model import Transformer, TransformerConfig
    logging.debug("Imported Transformer and TransformerConfig classes")
    try:
        logging.debug("Creating TransformerConfig from parameters")
        model_cfg = config.get('model', {})
        for key, value in model_cfg.items():
            logging.debug(f"Model config parameter: {key} = {value}")
        model_config = TransformerConfig(
            vocab_size=model_cfg.get('vocab_size'),
            d_model=model_cfg.get('hidden_dim'),
            num_layers=model_cfg.get('num_layers'),
            num_heads=model_cfg.get('num_heads'),
            d_ff=model_cfg.get('ff_dim'),
            max_seq_len=model_cfg.get('max_seq_len'),
            dropout=model_cfg.get('dropout'),
            activation=model_cfg.get('activation'),
            use_checkpointing=model_cfg.get('use_checkpointing'),
            tie_embeddings=model_cfg.get('tie_embeddings'),
            window_size=model_cfg.get('window_size'),
            global_tokens=model_cfg.get('global_tokens'),
            use_reentrant=model_cfg.get('use_reentrant')
        )
        logging.info(f"Model configuration created with {model_cfg.get('num_layers')} layers, "
                     f"{model_cfg.get('num_heads')} heads, {model_cfg.get('hidden_dim')} hidden dim")
    except Exception as e:
        logging.error(f"Error creating model configuration: {e}", exc_info=True)
        raise
    logging.debug("Initializing Transformer model")
    model = Transformer(model_config)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    checkpoint_path = config['inference']['model_path']
    logging.info(f"Loading model checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logging.error(f"Model checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    try:
        logging.debug("Loading checkpoint file")
        checkpoint_load_start = time.time()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        logging.debug(f"Checkpoint loaded in {time.time() - checkpoint_load_start:.2f} seconds")
        if isinstance(checkpoint, dict):
            logging.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
        else:
            logging.warning(f"Checkpoint is not a dictionary, type: {type(checkpoint)}")
        if 'model_state_dict' not in checkpoint:
            logging.error("Checkpoint missing 'model_state_dict' key")
            raise RuntimeError("Checkpoint missing 'model_state_dict' key")
        state_dict = checkpoint['model_state_dict']
        logging.debug(f"State dict contains {len(state_dict)} keys")
        model_state_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        missing_keys = model_state_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_state_keys
        if missing_keys:
            logging.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        shape_mismatches = []
        for key in model_state_keys.intersection(checkpoint_keys):
            model_shape = model.state_dict()[key].shape
            checkpoint_shape = state_dict[key].shape
            if model_shape != checkpoint_shape:
                shape_mismatches.append((key, model_shape, checkpoint_shape))
        if shape_mismatches:
            logging.warning(f"Shape mismatches found in {len(shape_mismatches)} tensors")
            for key, model_shape, checkpoint_shape in shape_mismatches:
                logging.warning(f"Shape mismatch for {key}: model has {model_shape}, checkpoint has {checkpoint_shape}")
        load_start = time.time()
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.debug(f"State dict loaded in {time.time() - load_start:.2f} seconds")
        if 'epoch' in checkpoint:
            logging.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'train_loss' in checkpoint:
            logging.info(f"Final training loss: {checkpoint['train_loss']}")
        if 'val_loss' in checkpoint:
            logging.info(f"Final validation loss: {checkpoint['val_loss']}")
        if 'timestamp' in checkpoint:
            logging.info(f"Checkpoint timestamp: {checkpoint['timestamp']}")
        print(f"✅ Model loaded from {checkpoint_path}")
        logging.info("Model checkpoint loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model checkpoint: {e}", exc_info=True)
        raise
    device_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        cuda_id = device.index or 0
        props = torch.cuda.get_device_properties(cuda_id)
        logging.info(f"GPU Device: {props.name} with {props.total_memory / 1024**3:.2f} GB memory")
        logging.debug(f"CUDA capability: {props.major}.{props.minor}")
        logging.debug(f"CUDA device count: {torch.cuda.device_count()}")
    model.to(device)
    logging.debug(f"Model moved to {device} in {time.time() - device_start:.2f} seconds")
    model.eval()
    logging.debug("Model set to evaluation mode")
    logging.info(f"Model setup completed in {time.time() - start_time:.2f} seconds")
    return model

###############################################################################
# 3. TEXT GENERATION WITH DETAILED LOGGING (UPDATED FOR NEW HYPERPARAMS)
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
    repetition_penalty: Optional[float] = None,
    num_beams: Optional[int] = None,
    no_repeat_ngram_size: Optional[int] = None,
    length_penalty: Optional[float] = None,
    early_stopping: Optional[bool] = None,
) -> str:
    """
    Generate text using either beam search (if num_beams > 1) or a token-by-token loop
    with detailed logging and new generation hyperparameters.
    """
    logging.info(f"Starting text generation for prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    generation_start = time.time()
    
    gen_cfg = config.get('generation', {})
    max_length = max_length or gen_cfg.get('max_length', 128)
    temperature = temperature or gen_cfg.get('temperature', 0.65)
    top_k = top_k or gen_cfg.get('top_k', 50)
    top_p = top_p or gen_cfg.get('top_p', 0.9)
    repetition_penalty = repetition_penalty or gen_cfg.get('repetition_penalty', 1.25)
    num_beams = num_beams or gen_cfg.get('num_beams', 1)
    no_repeat_ngram_size = no_repeat_ngram_size or gen_cfg.get('no_repeat_ngram_size', 3)
    length_penalty = length_penalty or gen_cfg.get('length_penalty', 1.0)
    early_stopping = early_stopping if early_stopping is not None else gen_cfg.get('early_stopping', True)
    
    logging.info(f"Generation parameters: max_length={max_length}, temperature={temperature}, top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}, num_beams={num_beams}, no_repeat_ngram_size={no_repeat_ngram_size}, length_penalty={length_penalty}, early_stopping={early_stopping}")

    device = next(model.parameters()).device
    logging.debug(f"Using device: {device}")

    if not isinstance(prompt, str) or not prompt.strip():
        logging.error("Invalid prompt provided: empty or not a string")
        raise ValueError("Invalid prompt")

    if not prompt.startswith(tokenizer.bos_token):
        logging.debug(f"Adding BOS token to prompt: '{tokenizer.bos_token}'")
        prompt = f"{tokenizer.bos_token} {prompt}".strip()
    if prompt.endswith(tokenizer.eos_token):
        logging.debug(f"Removing EOS token from prompt end: '{tokenizer.eos_token}'")
        print("Prompt already contains EOS token at the end, removing it to allow generation.")
        prompt = prompt[:-len(tokenizer.eos_token)].strip()

    try:
        logging.debug("Tokenizing prompt")
        encode_start = time.time()
        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False
        ).to(device)
        logging.debug(f"Prompt encoded in {time.time() - encode_start:.4f} seconds")
        logging.info(f"Encoded prompt length: {input_ids.size(1)} tokens")
        logging.debug(f"Prompt token IDs: {input_ids.tolist()}")
    except Exception as e:
        logging.error(f"Tokenization failed: {e}", exc_info=True)
        raise

    model_max_len = config['model'].get('max_seq_len', 128)
    if input_ids.size(1) > model_max_len:
        logging.warning(f"Prompt length ({input_ids.size(1)}) exceeds model max sequence length ({model_max_len})")
        logging.debug("Truncating prompt to model max sequence length")
        input_ids = input_ids[:, -model_max_len:]
        logging.info(f"Truncated prompt length: {input_ids.size(1)} tokens")

    prompt_length = input_ids.size(1)
    
    # If beam search is requested and the model supports generate(), use it.
    if num_beams > 1 and hasattr(model, "generate"):
        logging.info("Using beam search generation")
        generation_args = {
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
        }
        logging.debug(f"Beam search generation args: {generation_args}")
        with inference_mode():
            output_ids = model.generate(input_ids, **generation_args)
        generated_ids = output_ids[0].tolist()
    else:
        # Otherwise, use token-by-token generation with repetition penalty.
        generated_ids = input_ids[0].tolist()
        logging.info(f"Beginning token generation loop for {max_length} tokens")
        with inference_mode():
            for i in tqdm(range(max_length), desc="Generating"):
                curr_ids = generated_ids[-model_max_len:]
                curr_input = torch.tensor([curr_ids], device=device)
                outputs = model(curr_input)
                # Get the last hidden state
                last_hidden = outputs[:, -1, :]  # shape: [1, hidden_dim]
                # Project hidden state into vocab space if generator exists.
                if hasattr(model, "generator"):
                    logits = torch.matmul(last_hidden, model.generator.weight.T) + model.generator.bias
                else:
                    logits = outputs[:, -1, :]
                logging.debug(f"Logits shape after projection: {logits.shape}")
                
                # Apply temperature scaling.
                logits = logits / temperature

                # Apply repetition penalty.
                for token_id in set(generated_ids):
                    if token_id >= logits.size(1):
                        logging.warning(f"Token ID {token_id} is out-of-bounds for logits of shape {logits.shape}")
                        continue
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

                # Top-k filtering.
                if top_k > 0:
                    vals, idx = torch.topk(logits, top_k)
                    keep_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, idx, True)
                    logits[~keep_mask] = float('-inf')

                # Top-p filtering.
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
                if next_token_id == tokenizer.eos_token_id:
                    logging.info(f"Generated EOS token at position {i+1}, stopping generation")
                    break
                generated_ids.append(next_token_id)

    try:
        logging.debug("Decoding generated tokens")
        generated_text = tokenizer.decode(generated_ids[prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        result = generated_text.strip()
        logging.info(f"Generated {len(generated_ids) - prompt_length} tokens in {time.time() - generation_start:.2f} seconds")
        logging.info(f"Final text length: {len(result)} characters")
        logging.debug(f"Generation result: '{result[:100]}{'...' if len(result) > 100 else ''}'")
        return result
    except Exception as e:
        logging.error(f"Error decoding generated tokens: {e}", exc_info=True)
        raise

###############################################################################
# 4. MAIN INTERACTIVE LOOP & TEST WITH DETAILED LOGGING
###############################################################################
def main():
    setup_logging("logs/lumina_inference.log")
    logging.info("=" * 80)
    logging.info("LuminaLM Text Generation System Starting")
    logging.info("=" * 80)
    
    start_time = time.time()
    try:
        logging.info(f"Python version: {os.sys.version}")
        logging.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logging.info(f"CUDA available: Yes, version {torch.version.cuda}")
            logging.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logging.info(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
        else:
            logging.info("CUDA available: No")
            
        logging.debug(f"Current working directory: {os.getcwd()}")
        logging.info("Loading configuration")
        config = load_config("config/inference_config.yaml")
        logging.info("Setting up tokenizer")
        tokenizer = load_tokenizer(config['tokenizer']['path'])
        logging.info("Setting up model")
        model = setup_model_for_inference(config)
        logging.info("Testing tokenizer functionality")
        if not test_tokenizer_functionality(tokenizer):
            logging.error("Tokenizer verification failed")
            raise RuntimeError("Tokenizer verification failed")
        logging.info("System initialization complete")
        logging.info(f"Total startup time: {time.time() - start_time:.2f} seconds")
        print("\n🔥 Welcome to LuminaLM Text Generation! 🔥")
        print("Type your prompt and press Enter to generate text.")
        print("Type 'exit' to quit.\n")
        while True:
            prompt = input("💬 Enter prompt: ").strip()
            logging.info(f"User prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            if prompt.lower() == 'exit':
                logging.info("User requested exit")
                print("\n👋 Goodbye!")
                break
            try:
                logging.info("Starting text generation")
                gen_start = time.time()
                output = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    config=config
                )
                gen_time = time.time() - gen_start
                logging.info(f"Text generation completed in {gen_time:.2f} seconds")
                logging.info(f"Generated {len(output)} characters")
                logging.debug(f"Generated output: '{output[:100]}{'...' if len(output) > 100 else ''}'")
                print(f"\n📝 Generated Output:\n{'-'*50}")
                print(output)
                print('-'*50)
            except Exception as e:
                logging.error(f"Generation error: {str(e)}", exc_info=True)
                print(f"⚠️ Error during generation: {str(e)}")
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, exiting")
        print("\n👋 Goodbye!")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"⚠️ Fatal error: {str(e)}")
        raise
    finally:
        logging.info("=" * 80)
        logging.info("LuminaLM Text Generation System Shutting Down")
        logging.info("=" * 80)

if __name__ == "__main__":
    main()
