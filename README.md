# SPARSA-LM v0.2.0

## AutoRegressive Language Model with DAPO/VAPO RL Finetuning

**SPARSA-LM** is a modern AutoRegressive Language Model built with state-of-the-art techniques including RMSNorm, Rotary Position Embeddings (RoPE), Grouped Query Attention (GQA), and SwiGLU activation. It features separate pipelines for pretraining and RL-based finetuning using DAPO and VAPO algorithms.

---

## Features

### Model Architecture
- **AutoRegressive Transformer** - Decoder-only architecture for causal language modeling
- **RMSNorm** - Efficient normalization without computing mean
- **Rotary Position Embeddings (RoPE)** - Better length generalization
- **Grouped Query Attention (GQA)** - Reduced memory and computation
- **SwiGLU Activation** - Improved feed-forward performance
- **Flash Attention 2** - Optional optimized attention
- **Sliding Window Attention** - Efficient long-context handling
- **KV-Cache** - Fast autoregressive generation

### Training Pipelines

#### Pretraining
- DeepSpeed ZeRO optimization (stages 0-3)
- Mixed precision training (BF16/FP16)
- Gradient checkpointing
- Distributed data parallel training
- Streaming datasets support

#### RL Finetuning
- **DAPO** (Decoupled Clip and Dynamic Sampling Policy Optimization)
  - Decoupled upper/lower clip bounds
  - Dynamic sampling temperature
  - Entropy targeting to prevent collapse

- **VAPO** (Value-model Augmented Proximal Policy Optimization)
  - Dense per-token rewards from value model
  - Reward smoothing for stability
  - Value function clipping

---

## Project Structure

```
SPARSA-LM-Base-0.1/
├── src/
│   ├── model/
│   │   ├── config.py          # Model configuration
│   │   ├── layers.py          # Core layers (RMSNorm, RoPE, GQA, SwiGLU)
│   │   ├── architecture.py    # AutoRegressive model
│   │   └── generation.py      # Text generation utilities
│   ├── data/
│   │   ├── dataset.py         # Dataset classes
│   │   └── collator.py        # Data collation
│   ├── pretrain/
│   │   ├── config.py          # Pretraining configuration
│   │   └── trainer.py         # Pretraining trainer
│   └── finetune/
│       ├── config.py          # Finetuning configuration
│       ├── dapo.py            # DAPO implementation
│       ├── vapo.py            # VAPO implementation
│       └── trainer.py         # RL finetuning trainer
├── scripts/
│   ├── pretrain.py            # Pretraining entry point
│   └── finetune.py            # Finetuning entry point
├── config/
│   ├── pretrain.yaml          # Pretraining configuration
│   ├── finetune.yaml          # Finetuning configuration
│   └── deepspeed.json         # DeepSpeed configuration
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (recommended)

### Install
```bash
git clone https://github.com/Archit03/SPARSA-LM-Base-0.1
cd SPARSA-LM-Base-0.1
pip install -e .
```

### With Flash Attention (optional)
```bash
pip install -e ".[flash-attn]"
```

### For Development
```bash
pip install -e ".[dev]"
```

---

## Usage

### Pretraining

```bash
# Single GPU
python scripts/pretrain.py \
    --model_size base \
    --tokenizer_path path/to/tokenizer \
    --output_dir outputs/pretrain \
    --num_epochs 3 \
    --batch_size 4

# Multi-GPU with DeepSpeed
deepspeed scripts/pretrain.py \
    --model_size base \
    --tokenizer_path path/to/tokenizer \
    --output_dir outputs/pretrain \
    --deepspeed \
    --deepspeed_stage 2
```

### RL Finetuning

#### DAPO (Recommended)
```bash
python scripts/finetune.py \
    --model_path outputs/pretrain/final \
    --tokenizer_path path/to/tokenizer \
    --algorithm dapo \
    --train_data data/prompts.jsonl \
    --output_dir outputs/finetune-dapo
```

#### VAPO
```bash
python scripts/finetune.py \
    --model_path outputs/pretrain/final \
    --tokenizer_path path/to/tokenizer \
    --algorithm vapo \
    --train_data data/prompts.jsonl \
    --output_dir outputs/finetune-vapo \
    --dense_reward
```

---

## Model Configurations

| Size  | Parameters | Layers | Hidden | Heads | KV Heads |
|-------|------------|--------|--------|-------|----------|
| Small | ~125M      | 12     | 768    | 12    | 4        |
| Base  | ~360M      | 28     | 1024   | 16    | 8        |
| Large | ~760M      | 32     | 1536   | 24    | 8        |
| XL    | ~1.5B      | 36     | 2048   | 32    | 8        |

---

## Python API

```python
from src.model import AutoRegressiveLM, ModelConfig

# Create model
config = ModelConfig.base()
model = AutoRegressiveLM(config)

# Generate text
input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
)
text = tokenizer.decode(output[0])
```

---

## DAPO vs VAPO

| Feature | DAPO | VAPO |
|---------|------|------|
| Clipping | Decoupled (upper/lower) | Standard PPO |
| Sampling | Dynamic temperature | Fixed |
| Rewards | Sparse (end of sequence) | Dense (per-token) |
| Value Model | Optional | Required |
| Best For | Preventing entropy collapse | Long sequences |

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

---

## License

This project is licensed under the MIT License.

---

## Contact

- **Author:** Archit Sood @ EllanorAI
- **Email:** architsood@ellanorai.org
- **Website:** https://ellanorai.org

---

## Citation

```bibtex
@software{sparsa_lm,
  title = {SPARSA-LM: AutoRegressive Language Model with DAPO/VAPO RL Finetuning},
  author = {Sood, Archit},
  year = {2024},
  url = {https://github.com/Archit03/SPARSA-LM-Base-0.1}
}
```
