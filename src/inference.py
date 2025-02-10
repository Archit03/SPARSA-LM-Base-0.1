import torch
from transformers import PreTrainedTokenizerFast
from typing import Optional, Dict, Any
import yaml
import os
import logging
from tqdm import tqdm

def load_config(config_path: str = "config/inference_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """Load and configure the tokenizer with proper cleanup settings."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Ensure special tokens are set
    special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
    
    for key, value in special_tokens.items():
        if getattr(tokenizer, key) is None:
            setattr(tokenizer, key, value)
    
    # Configure tokenizer settings for better output
    tokenizer.clean_up_tokenization_spaces = True
    tokenizer.trim_offsets = True
    
    return tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    config: Dict[str, Any],
    max_length: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> str:
    """Generate text with improved token handling and quality."""
    # Use config defaults if parameters not specified
    max_length = max_length or config['generation']['max_length']
    temperature = temperature or config['generation']['temperature']
    top_k = top_k or config['generation']['top_k']
    top_p = top_p or config['generation']['top_p']
    
    device = next(model.parameters()).device
    
    # Properly format prompt with special tokens
    if not prompt.startswith(tokenizer.bos_token):
        prompt = f"{tokenizer.bos_token} {prompt}"
    
    # Encode with proper handling of special tokens
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    ).to(device)
    
    # Ensure input length doesn't exceed model's max_seq_len
    if input_ids.size(1) > config['model']['max_seq_len']:
        input_ids = input_ids[:, -config['model']['max_seq_len']:]
    
    generated_ids = input_ids[0].tolist()
    
    # Generate tokens one at a time
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(max_length), desc="Generating"):
            # Prepare input for model
            curr_input = torch.tensor([generated_ids[-config['model']['max_seq_len']:]]).to(device)
            
            # Get model predictions
            outputs = model(curr_input)
            
            # Get logits for next token prediction
            if len(outputs.shape) == 3:
                logits = outputs[0, -1]
            else:
                logits = outputs[-1]
            
            # Apply temperature scaling
            logits = logits / (temperature if temperature > 0 else 1.0)
            
            # Apply top-k filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[0] = 0
                logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
            
            # Sample next token with adjusted probabilities
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if we generate EOS token
            if next_token == tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token)
    
    # Decode with special handling for better text quality
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    # Clean up any remaining artifacts
    generated_text = ' '.join(generated_text.split())
    
    return generated_text

def setup_model_for_inference(config: Dict[str, Any]):
    """Setup model for inference with correct parameter mapping."""
    from model import Transformer, TransformerConfig
    
    # Map config parameters to TransformerConfig parameters
    model_config = TransformerConfig(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['hidden_dim'],  # Map hidden_dim to d_model
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['ff_dim'],  # Map ff_dim to d_ff
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
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint: {str(e)}")
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model

def main():
    """Main inference loop with improved parameters."""
    try:
        # Load configuration
        config = load_config()
        
        # Adjust generation parameters for better quality
        config['generation'] = {
            'max_length': 100,
            'temperature': 0.8,  # Slightly higher for more natural text
            'top_k': 40,        # Lower for more focused sampling
            'top_p': 0.95       # Higher for more natural language
        }
        
        # Load tokenizer and model
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
                generated_text = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    config
                )
                print(f"\n📝 Generated Output:")
                print("-" * 50)
                print(generated_text)
                print("-" * 50)
            
            except Exception as e:
                print(f"⚠️ Error during generation: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"⚠️ Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()