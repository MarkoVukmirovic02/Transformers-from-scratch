# Transformers From Scratch

A PyTorch implementation of the original encoder-decoder Transformer architecture from *Attention Is All You Need*.

The goal of this project is to implement the Transformer architecture from first principles, including embeddings, positional encoding, scaled dot-product attention, multi-head attention, encoder/decoder blocks, masking, and a bilingual machine-translation dataset pipeline.

---

## Project Overview

This project implements a Transformer for sequence-to-sequence machine translation.

The model follows the classical encoder-decoder design:

```text
Source Sentence
      ↓
Source Tokenizer
      ↓
Input Embedding
      ↓
Positional Encoding
      ↓
Encoder Stack
      ↓
Encoder Output
      ↓
Decoder with Masked Self-Attention + Cross-Attention
      ↓
Projection Layer
      ↓
Vocabulary Probabilities
```

The implementation is written manually in PyTorch without using high-level Transformer modules such as `nn.Transformer`.

---

## Motivation

Modern architectures such as GPT, BERT, Vision Transformers, multimodal Transformers, and RAG-based systems are all built on the same core ideas:

* embeddings
* attention
* positional information
* residual connections
* normalization
* token prediction

This project was built to understand those components from the inside by implementing them directly.

---

## Implemented Components

### Transformer Model

* [x] Input Embeddings
* [x] Sinusoidal Positional Encoding
* [x] Custom Layer Normalization
* [x] Position-Wise Feed Forward Network
* [x] Scaled Dot-Product Attention
* [x] Multi-Head Attention
* [x] Residual Connections
* [x] Encoder Block
* [x] Encoder Stack
* [x] Decoder Block
* [x] Decoder Stack
* [x] Cross-Attention
* [x] Projection Layer
* [x] Full Transformer Builder
* [x] Xavier Weight Initialization

### Dataset Pipeline

* [x] Bilingual Dataset Class
* [x] Source and Target Tokenization
* [x] `[SOS]`, `[EOS]`, `[PAD]`, `[UNK]` Tokens
* [x] Encoder Input Construction
* [x] Decoder Input Construction
* [x] Label Construction
* [x] Padding Masks
* [x] Causal Decoder Mask
* [x] OPUS Books Dataset Loading
* [x] WordLevel Tokenizer Training

---

## Architecture Details

### Input Embeddings

Each token ID is mapped into a dense vector of dimension `d_model`.

The embeddings are scaled by:

```text
sqrt(d_model)
```

This follows the original Transformer implementation.

---

### Positional Encoding

Since Transformers process all tokens in parallel, they do not naturally contain sequence-order information.

Sinusoidal positional encodings are added to token embeddings so the model can distinguish word order.

---

### Scaled Dot-Product Attention

Attention is computed using Query, Key, and Value projections:

```text
Q = XWq
K = XWk
V = XWv
```

The attention operation is:

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(dk))V
```

The scaling term prevents large dot products from saturating the softmax function.

---

### Multi-Head Attention

Instead of using one attention mechanism, the model uses multiple attention heads.

Each head learns a different representation subspace and can capture different relationships between tokens, such as:

* semantic similarity
* grammatical structure
* long-range dependency
* object relationship
* positional dependency

The outputs of all heads are concatenated and projected back into `d_model`.

---

### Encoder

Each encoder block contains:

```text
Multi-Head Self-Attention
        ↓
Residual Connection
        ↓
Layer Normalization
        ↓
Feed Forward Network
        ↓
Residual Connection
        ↓
Layer Normalization
```

The encoder receives the source sentence and produces contextual representations for all source tokens.

---

### Decoder

Each decoder block contains:

```text
Masked Multi-Head Self-Attention
        ↓
Cross-Attention over Encoder Output
        ↓
Feed Forward Network
```

The decoder uses:

* masked self-attention to prevent looking at future target tokens
* cross-attention to attend to the encoded source sentence
* feed-forward layers to transform token representations

---

## Dataset

The project is designed for bilingual machine translation using the OPUS Books dataset.

Example language pair:

```text
English → Italian
```

The dataset pipeline:

1. Loads bilingual sentence pairs.
2. Builds separate source and target tokenizers.
3. Converts text into token IDs.
4. Adds `[SOS]` and `[EOS]` tokens.
5. Pads sequences to fixed length.
6. Creates encoder and decoder masks.
7. Creates shifted labels for teacher forcing.

---

## Teacher Forcing

For the target sentence:

```text
I love apples
```

the decoder input becomes:

```text
[SOS] I love apples
```

while the label becomes:

```text
I love apples [EOS]
```

This teaches the model to predict the next token at every position.

---

## Repository Structure

```text
Transformers-from-Scratch/
│
├── model.py              # Transformer architecture
├── dataset.py            # Bilingual dataset and masking
├── train.py              # Training loop
├── config.py             # Hyperparameters and paths
│
├── tokenizer/            # Saved tokenizers
├── checkpoints/          # Model checkpoints
├── docs/                 # Theory notes
└── README.md
```

---

## Key Learning Outcomes

Through this project I implemented and studied:

* sequence-to-sequence modeling
* self-attention
* scaled dot-product attention
* multi-head attention
* encoder-decoder Transformers
* masking in Transformers
* teacher forcing
* tokenization
* machine translation pipelines
* PyTorch model design

---

## Future Work

Planned improvements:

* complete training loop
* validation loop
* greedy decoding
* beam search decoding
* BLEU score evaluation
* attention visualization
* training loss plots
* comparison with PyTorch `nn.Transformer`
* decoder-only GPT implementation
* Vision Transformer implementation

---

## Why This Project Matters

This project builds the foundation for understanding modern Transformer-based systems, including:

* GPT-style language models
* BERT-style encoders
* machine translation systems
* Vision Transformers
* multimodal Transformers
* Retrieval-Augmented Generation systems

Instead of only using existing libraries, this implementation focuses on understanding the mathematical and architectural structure of Transformers from first principles.

---

## References

* Vaswani et al., *Attention Is All You Need*, 2017
* The Annotated Transformer
* PyTorch Documentation
* Hugging Face Datasets
* Hugging Face Tokenizers

---

## Author

**Marko Vukmirović**

Applied Mathematics student focused on:

* deep learning
* computer vision
* autonomous-driving perception
* probabilistic state estimation
* Transformer architectures
