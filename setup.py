from setuptools import setup, find_packages

setup(
    name='sparsa-lm',
    fullname='SPARSA Language Model - AutoRegressive Architecture',
    version='0.2.0',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        # Core PyTorch
        'torch>=2.1.0',

        # HuggingFace Ecosystem
        'transformers>=4.36.0',
        'datasets>=2.16.0',
        'tokenizers>=0.15.0',
        'accelerate>=0.25.0',
        'safetensors>=0.4.0',

        # Distributed Training
        'deepspeed>=0.12.0',

        # Numerical
        'numpy>=1.24.0',

        # Configuration
        'pyyaml>=6.0',

        # Logging and Tracking
        'wandb>=0.16.0',
        'tqdm>=4.66.0',

        # Utilities
        'rich>=13.7.0',
    ],
    extras_require={
        'flash-attn': [
            'flash-attn>=2.4.0',
        ],
        'inference': [
            'vllm>=0.2.7',
            'triton>=2.1.0',
        ],
        'dev': [
            'pytest>=7.4.0',
            'black>=23.11.0',
            'isort>=5.12.0',
            'mypy>=1.7.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'sparsa-pretrain=scripts.pretrain:main',
            'sparsa-finetune=scripts.finetune:main',
        ],
    },
    author="Archit Sood @ EllanorAI",
    author_email="architsood@ellanorai.org",
    description="SPARSA-LM: AutoRegressive Language Model with DAPO/VAPO RL Finetuning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Archit03/SPARSA-LM-Base-0.1",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "language-model",
        "transformer",
        "autoregressive",
        "deep-learning",
        "pytorch",
        "dapo",
        "vapo",
        "reinforcement-learning",
    ],
)
