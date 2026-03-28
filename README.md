# Decoder-Only Transformer (GPT-Style) from Scratch

## Introduction
* This project implements a decoder-only Transformer (GPT-style) from scratch using PyTorch.
* All core components—including token embeddings, positional encoding, masked multi-head self-attention, and feed-forward networks—are built manually without relying on high-level abstractions.
* The model is trained for autoregressive text generation, enabling analysis of token relationships, attention patterns, and learning dynamics during training.

## Project Goals
* Implement a decoder-only Transformer from scratch to gain a deep, component-level understanding of GPT-style architectures  
* Analyze the impact of architectural choices and hyperparameters through controlled training experiments  
* Study how token representations, attention patterns, and language structure evolve during autoregressive training  
* Build a foundation for systematic experimentation on training dynamics, generalization, and overfitting behavior  

## Current Status
* Decoder-only Transformer fully implemented from scratch, including all core components  
* Model successfully trained on the dataset and supports autoregressive text generation  
* Training pipeline is stable and reproducible across runs  
* Currently in a deep experimentation phase, focusing on training dynamics, hyperparameter effects, and overfitting behavior  

## How to Run
```bash
git clone https://github.com/<your-username>/decoder-only-transformer-from-scratch.git
cd decoder-only-transformer-from-scratch

pip install -r requirements.txt

# Train the model
python main.py

# Generate text using a trained checkpoint
python generate.py

## Example Generated Text
Prompt: hello
Generated: helloains and transport the economy is now home . At the ground , the main couple 
ends along the estimated in the Mediterranean . The game again finished with Nesbid and juniora 
started the north side . By 04 , and August 3 , the former
```

**Note:** This is an early-stage model trained on a limited dataset. Generated text may appear nonsensical. The implementation demonstrates full end-to-end autoregressive generation and will improve with further training and hyperparameter tuning.

## Model Architecture

```text
Input IDs
    │
    ▼
Token Embeddings + Positional Embeddings
    │
    ▼
Dropout
    │
    ▼
┌───────────────────────────────┐
│        Decoder Layer × N      │
│  ┌─────────────────────────┐  │
│  │ LayerNorm               │  │
│  │ Multi-Head Attention    │  │
│  │ Residual Connection     │  │
│  │ LayerNorm               │  │
│  │ Feed Forward Network    │  │
│  │ Residual Connection     │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
    │
    ▼
Final Linear Layer → Logits
```