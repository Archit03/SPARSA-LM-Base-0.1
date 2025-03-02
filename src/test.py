import torch
from pathlib import Path

# Load the saved model checkpoint
model_path = r"C:\Users\ASUS\Desktop\SPARSA-LM-Base 0.1\model\best_LuminaLM_model.pt."  # Replace with your model path
checkpoint = torch.load(model_path, weights_only=True)

# Print model weights
print("Model Weights:")
for name, param in checkpoint['model_state_dict'].items():
    print(f"{name}: {param.shape}")
    print(f"Values: {param.flatten()[:5]}...")  # Print first 5 values of each parameter

# Print loss
if 'loss' in checkpoint:
    print(f"\nFinal Loss: {checkpoint['loss']:.4f}")

# Calculate and print perplexity (if loss is cross-entropy)
if 'loss' in checkpoint:
    perplexity = torch.exp(torch.tensor(checkpoint['loss']))
    print(f"Perplexity: {perplexity:.4f}")