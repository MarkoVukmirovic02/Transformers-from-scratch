# Part I — Why Transformers Exist

# Chapter 1 — The Sequence Modeling Problem

## Introduction

Before studying Transformers, attention mechanisms, large language models, or modern multimodal systems, we must first understand the problem they were designed to solve.

Transformers were not invented because researchers suddenly discovered a better neural network architecture. They emerged as a response to fundamental limitations in existing sequence modeling approaches.

The central question is:

> How can a machine understand language when meaning depends on context, long-range relationships, and complex interactions between words?

To answer this question we first need to understand why natural language is a difficult learning problem.

---

# 1. Language as a Sequence

Natural language consists of ordered sequences of symbols.

Consider the sentence:

```text
The cat chased the mouse.
```

A machine does not directly understand the meaning of this sentence. Instead, it receives a sequence of tokens:

```text
x₁, x₂, x₃, x₄, x₅
```

where:

```text
x₁ = The
x₂ = cat
x₃ = chased
x₄ = the
x₅ = mouse
```

The objective of a language model is not merely to recognize individual words, but to understand the relationships between them.

Unlike many traditional machine-learning problems, language is inherently sequential.

Changing the order of words changes the meaning of the sentence:

```text
The cat chased the mouse.
```

and

```text
The mouse chased the cat.
```

contain exactly the same words but communicate completely different information.

Any successful language model must therefore understand both:

1. The words themselves.
2. Their relationships within a sequence.

---

# 2. Meaning Depends on Context

One of the most difficult properties of language is that words rarely possess a single fixed meaning.

Consider the word:

```text
Apple
```

in the following sentences:

```text
Apple released a new laptop.
```

```text
Apple juice tastes good.
```

Although the same token appears in both sentences, its meaning is entirely different.

In the first sentence:

```text
Apple → Technology Company
```

In the second sentence:

```text
Apple → Fruit
```

The meaning cannot be determined from the word alone.

Instead, meaning emerges from the surrounding context.

This observation creates a major challenge for machine learning systems:

> A word's meaning is not fixed; it depends on the words around it.

Therefore language understanding requires models capable of dynamically adapting word representations according to context.

---

# 3. Long-Range Dependencies

Many linguistic relationships occur between words that are separated by large distances.

Consider:

```text
The animal didn't cross the street because it was tired.
```

To correctly interpret the word:

```text
it
```

the model must determine what it refers to.

Humans immediately understand that:

```text
it → animal
```

However, a machine-learning model must somehow preserve information from earlier words while processing the remainder of the sentence.

The problem becomes more difficult as the distance increases.

Consider:

```text
The animal that escaped from the nearby farm and wandered through several neighborhoods during the previous night didn't cross the street because it was tired.
```

The relationship:

```text
it → animal
```

still exists, but now dozens of intermediate words separate them.

A successful language model must therefore capture dependencies regardless of distance.

---

# 4. Multiple Relationships Exist Simultaneously

Language contains many overlapping relationships.

Consider:

```text
The keys were on the table where Sarah left them.
```

Several distinct relationships appear simultaneously:

```text
keys ↔ table
```

Location relationship.

```text
keys ↔ Sarah
```

Ownership or action relationship.

```text
them ↔ keys
```

Coreference relationship.

A language model cannot focus on only one of these relationships.

Instead, it must represent all relevant dependencies simultaneously.

This requirement becomes increasingly important for long documents, articles, conversations, and books.

---

# 5. Language Understanding Requires Global Context

Many sentences cannot be understood using only local information.

Consider:

```text
She saw the man with the telescope.
```

This sentence contains ambiguity.

Possible interpretation:

```text
The man possessed the telescope.
```

Alternative interpretation:

```text
She used the telescope.
```

The correct interpretation depends on additional context.

Language understanding therefore requires access to information beyond neighboring words.

A model must be capable of examining broader sentence structure to resolve ambiguity.

---

# 6. Scalability Requirements

Modern language models are trained on enormous datasets.

Typical training corpora contain:

* Millions of documents
* Billions of words
* Trillions of tokens

Any successful architecture must therefore satisfy two requirements:

### Expressiveness

The model must capture:

* semantics
* grammar
* context
* long-range dependencies

### Computational Efficiency

The model must:

* train efficiently
* utilize modern hardware
* scale to extremely large datasets

An architecture that understands language but cannot scale is impractical.

---

# Key Observations

Natural language presents several fundamental challenges:

1. Language is sequential.
2. Word meaning depends on context.
3. Important relationships may span large distances.
4. Multiple relationships exist simultaneously.
5. Global context is often required.
6. Models must scale to massive datasets.

These challenges motivated decades of research into sequence modeling.

Before Transformers, researchers attempted to solve these problems using recurrent neural networks, Long Short-Term Memory networks (LSTMs), and Gated Recurrent Units (GRUs).

Although these approaches achieved significant success, they eventually encountered fundamental limitations.

Understanding these limitations is the subject of the next chapter.

---

# Summary

Natural language is not simply a collection of words. It is a structured system of relationships, dependencies, and contextual interactions.

A successful language model must:

* understand order,
* adapt word meanings to context,
* capture long-range dependencies,
* represent multiple simultaneous relationships,
* and scale efficiently to massive datasets.

These requirements ultimately motivated the development of Transformer architectures and attention mechanisms.


# Part I — Why Transformers Exist

# Chapter 2 — Why RNNs Were Not Enough

## Introduction

In the previous chapter we identified the fundamental challenges of language modeling:

* Context-dependent meaning
* Long-range dependencies
* Multiple simultaneous relationships
* Scalability requirements

Before Transformers, researchers attempted to solve these problems using recurrent neural networks (RNNs) and their improved variants:

* Recurrent Neural Networks (RNNs)
* Long Short-Term Memory Networks (LSTMs)
* Gated Recurrent Units (GRUs)

These architectures dominated natural language processing for many years and represented a significant improvement over traditional statistical methods.

However, despite their success, they possessed fundamental limitations that eventually motivated the development of Transformers.

To understand why Transformers were revolutionary, we must first understand how recurrent architectures process information.

---

# 1. The Recurrent Idea

Unlike feed-forward neural networks, recurrent neural networks process sequences one element at a time.

Suppose we have the sentence:

```text
The cat chased the mouse.
```

The RNN receives tokens sequentially:

```text
The
↓
cat
↓
chased
↓
the
↓
mouse
```

At every step the network maintains a hidden state:

```text
h_t
```

which acts as a summary of everything observed so far.

The recurrence relation can be written as:

h_t=f(x_t,h_{t-1})

where:

* (x_t) is the current input token
* (h_{t-1}) is the previous hidden state
* (h_t) is the updated hidden state

The hidden state functions as the model's memory.

Every new word updates this memory.

---

# 2. The Hidden State as a Compression Mechanism

The central idea of RNNs is simple:

> Compress everything observed so far into a single vector.

For example:

```text
The
↓
"The"

The cat
↓
"The cat"

The cat chased
↓
"The cat chased"

...
```

At every step the hidden state attempts to summarize all previous information.

Suppose:

```text
h_t ∈ ℝ^512
```

Regardless of whether the sequence contains:

```text
10 words
100 words
1000 words
```

the entire history must ultimately be represented inside a vector of fixed dimension.

This immediately creates a problem.

---

# 3. The Information Bottleneck

Imagine reading a book and repeatedly replacing everything you have read so far with a short summary.

After enough iterations:

```text
summary
↓
summary of summary
↓
summary of summary of summary
↓
...
```

information inevitably disappears.

The same phenomenon occurs inside recurrent networks.

Every update performs something similar to:

```text
old information
+
new information
↓
compressed representation
```

As the sequence grows longer:

* important details may be overwritten
* older information becomes diluted
* subtle relationships become difficult to preserve

The model must continuously decide:

> Which information should be remembered and which should be discarded?

This limitation exists even before discussing optimization or gradients.

It is a structural limitation of the architecture itself.

---

# 4. Long Information Paths

Consider:

```text
The animal didn't cross the street because it was tired.
```

To correctly interpret:

```text
it
```

the model must preserve information about:

```text
animal
```

across multiple intermediate words.

In a recurrent architecture information travels through a chain:

```text
animal
↓
hidden state
↓
hidden state
↓
hidden state
↓
...
↓
it
```

The dependency path length grows with sentence length.

Suppose:

```text
token₁
...
token₁₀₀₀
```

If token 1000 requires information from token 1, the information must travel through:

```text
999 intermediate steps
```

This creates an extremely long communication pathway.

Ideally we would like:

```text
token₁ ↔ token₁₀₀₀
```

direct interaction.

RNNs cannot provide this.

Every dependency must travel through the recurrent chain.

---

# 5. The Sequential Computation Bottleneck

The most important practical limitation of recurrent architectures is their sequential nature.

To compute:

```text
h₅
```

we first require:

```text
h₄
```

To compute:

```text
h₄
```

we require:

```text
h₃
```

and so on.

Therefore:

```text
h₁
→ h₂
→ h₃
→ h₄
→ ...
```

must be computed sequentially.

This prevents effective parallelization.

Modern GPUs achieve their performance through massive parallel matrix operations.

RNNs cannot fully exploit this hardware because each computation depends on the previous one.

As datasets and model sizes increased, this limitation became increasingly severe.

---

# 6. Vanishing Gradients

The sequential structure also creates optimization difficulties.

Suppose the loss function depends on information located near the beginning of a long sequence.

During training the gradient must propagate backward through many recurrent steps.

Using the chain rule:

```text
∂L/∂h₁
=
∂L/∂hₙ
·
∂hₙ/∂hₙ₋₁
·
...
·
∂h₂/∂h₁
```

This becomes a product of many derivatives.

If those derivatives are typically smaller than one:

```text
0.9 × 0.9 × 0.9 × ...
```

the resulting gradient rapidly approaches zero.

As a result:

* early layers learn slowly
* long-range dependencies become difficult to capture
* training becomes unstable

This phenomenon is known as the **vanishing gradient problem**.

---

# 7. LSTMs and GRUs

Researchers recognized these difficulties and proposed improved recurrent architectures.

The most successful were:

### Long Short-Term Memory Networks (LSTMs)

Introduced memory cells and gating mechanisms that help preserve information over longer periods.

### Gated Recurrent Units (GRUs)

A simplified alternative to LSTMs using fewer parameters.

These architectures significantly improved sequence modeling.

However, they did not eliminate the fundamental issues:

* information still flows sequentially
* long communication paths remain
* computation is still sequential
* parallelization remains limited

LSTMs and GRUs delayed the problem but did not fundamentally solve it.

---

# 8. What Researchers Really Wanted

By approximately 2016, researchers increasingly recognized that recurrent architectures were constrained by their design.

The ideal sequence model would satisfy several properties:

### Direct Communication

Any token should communicate directly with any other token.

Instead of:

```text
token₁
↓
token₂
↓
...
↓
tokenₙ
```

we would prefer:

```text
tokenᵢ ↔ tokenⱼ
```

for any pair of tokens.

---

### Context Awareness

Word meaning should adapt dynamically according to surrounding words.

Example:

```text
Apple released a laptop.
```

```text
Apple juice tastes good.
```

The representation of:

```text
Apple
```

should change depending on context.

---

### Parallel Computation

All tokens should be processed simultaneously rather than sequentially.

This would dramatically improve training efficiency.

---

### Long-Range Dependencies

Relationships should remain easy to model regardless of distance.

A dependency spanning:

```text
5 words
50 words
500 words
```

should be handled equally well.

---

# Key Insight

Researchers eventually realized that perhaps language understanding did not require recurrence at all.

Instead of forcing information through a chain of hidden states:

```text
token₁
↓
token₂
↓
...
↓
tokenₙ
```

what if every token could directly examine every other token?

This simple idea led to one of the most important innovations in modern machine learning:

> Attention.

The next chapter introduces the attention mechanism and explains how it eliminates many of the fundamental limitations of recurrent architectures.

---

# Summary

Recurrent architectures introduced the concept of sequence modeling but suffered from several fundamental limitations:

1. Hidden states compress information into fixed-size representations.
2. Long-range dependencies require information to travel through many intermediate steps.
3. Computation is inherently sequential.
4. Gradients may vanish during training.
5. Scaling to massive datasets is difficult.

Although LSTMs and GRUs improved performance, they did not fundamentally remove these limitations.

Researchers therefore began searching for architectures based on direct token-to-token interaction rather than recurrence.

This search ultimately led to the development of attention mechanisms and Transformer architectures.


# Part I — Why Transformers Exist

# Chapter 3 — The Attention Revolution

## Introduction

In the previous chapter we identified the fundamental limitations of recurrent neural networks:

* Information must travel through long chains of hidden states.
* Important details may be lost during repeated compression.
* Computation is inherently sequential.
* Long-range dependencies are difficult to learn.
* Training becomes increasingly difficult as sequences grow.

Researchers therefore began asking a fundamental question:

> Why should information be forced to travel through hundreds of intermediate states?

Suppose two words are strongly related.

For example:

```text
The animal didn't cross the street because it was tired.
```

The word:

```text
it
```

depends strongly on:

```text
animal
```

Yet an RNN requires information to pass through every intermediate word.

Researchers began searching for a mechanism that would allow direct communication between tokens.

The solution became known as **attention**.

---

# 1. The Central Idea

Instead of processing information through a chain:

```text
token₁
↓
token₂
↓
token₃
↓
...
↓
tokenₙ
```

we would like:

```text
tokenᵢ ↔ tokenⱼ
```

for any pair of tokens.

Every word should be able to directly examine every other word and determine:

* Which words are relevant?
* How relevant are they?
* How much information should be transferred?

This immediately removes the long information path problem.

Instead of:

```text
token₁
↓
999 intermediate steps
↓
token₁₀₀₀
```

we obtain:

```text
token₁ ↔ token₁₀₀₀
```

direct interaction.

---

# 2. Contextual Meaning

Consider:

```text
I love Apple products.
```

and

```text
I love apple juice.
```

The token:

```text
Apple
```

appears in both sentences.

However:

```text
Apple → technology company
```

in the first sentence,

while:

```text
apple → fruit
```

in the second.

A successful language model must therefore create representations that depend on context.

Static word embeddings cannot achieve this.

Attention provides a mechanism for dynamically modifying representations according to surrounding words.

---

# 3. Similarity as a Measure of Relevance

Suppose we represent words as vectors:

```text
v₁
v₂
v₃
...
```

How can we determine whether two words are related?

A natural mathematical choice is the **dot product**.

For two vectors:

```text
v₁
v₂
```

their similarity can be measured using:

v_1^T v_2

Geometric interpretation:

### Large Positive Value

Vectors point in similar directions.

```text
high similarity
```

### Near Zero

Vectors are approximately orthogonal.

```text
little relationship
```

### Negative Value

Vectors point in opposing directions.

```text
low similarity
```

The dot product therefore provides a natural measure of relevance.

---

# 4. The Limitation of Direct Embedding Similarity

At first glance we might simply compare embeddings directly.

However, words may participate in many different types of relationships.

Consider:

```text
The keys were on the table where Sarah left them.
```

Possible relationships:

```text
keys ↔ table
```

location

```text
keys ↔ Sarah
```

ownership

```text
them ↔ keys
```

coreference

A single embedding representation is insufficient for capturing all possible interactions.

Researchers therefore introduced learned projections.

---

# 5. Queries, Keys and Values

Every token begins with an embedding:

```text
x
```

Instead of using this embedding directly, three new representations are created:

```text
Query
Key
Value
```

using learned matrices.

Conceptually:

### Query

Represents:

> What information am I looking for?

### Key

Represents:

> What information do I offer?

### Value

Represents:

> What information should be transferred if selected?

This separation allows the model to learn complex interaction patterns.

A word can simultaneously:

* search for information,
* advertise information,
* and provide information.

---

# 6. The Information Retrieval Analogy

A useful analogy is a search engine.

Suppose you type:

```text
best neural network architecture
```

The search query contains information about what you are looking for.

Every document on the internet contains information about its contents.

The search engine compares:

```text
query
```

with

```text
document descriptions
```

and returns the most relevant results.

Attention operates similarly.

Each token creates:

```text
Query
```

and compares it against every:

```text
Key
```

in the sequence.

Relevant tokens receive larger scores.

---

# 7. Attention Scores

Suppose the token:

```text
it
```

examines the sentence:

```text
animal
street
because
tired
```

The model computes similarity scores:

```text
animal  → 12
street  → 3
because → 1
tired   → 8
```

These scores indicate relative relevance.

Higher scores correspond to stronger relationships.

However, raw similarity scores are not yet suitable as attention weights.

---

# 8. Why Softmax Is Needed

The attention scores:

```text
12
3
1
8
```

have several problems:

* They may be negative.
* They do not sum to one.
* They are difficult to interpret probabilistically.

We therefore apply the Softmax transformation.

\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}

Softmax produces:

* positive values,
* normalized outputs,
* values summing to one.

Example:

```text
animal  → 0.97
street  → 0.01
because → 0.00
tired   → 0.02
```

These values can now be interpreted as attention weights.

---

# 9. Why Attention Uses a Weighted Sum

A common question is:

> Why not simply choose the most important word?

Why not:

```text
argmax(attention)
```

and select only:

```text
animal
```

The answer is that language is rarely determined by a single word.

Consider:

```text
I love Apple products.
```

The meaning of:

```text
Apple
```

depends on:

* Apple
* products
* love
* surrounding context

Instead of making a hard decision, attention performs a weighted combination.

Example:

```text
Apple    → 0.45
products → 0.35
love     → 0.15
I        → 0.05
```

The resulting representation becomes:

```text
0.45·VApple
+
0.35·Vproducts
+
0.15·Vlove
+
0.05·VI
```

This produces a contextual representation rather than a discrete selection.

---

# 10. Contextual Embeddings

The output of attention is therefore not a word.

It is a new vector representation.

This new vector contains information gathered from all relevant tokens.

As a result:

```text
Apple
```

in:

```text
Apple released a laptop.
```

becomes different from:

```text
apple
```

in:

```text
apple juice tastes good.
```

The model dynamically adapts word representations according to context.

This is one of the most important innovations introduced by attention mechanisms.

---

# Key Insight

Attention transforms language understanding from:

```text
sequential information flow
```

into:

```text
direct token-to-token interaction
```

Every token can:

1. Examine every other token.
2. Determine relevance.
3. Gather information from important words.
4. Construct a context-dependent representation.

This solves many of the fundamental limitations of recurrent architectures.

---

# Summary

The attention mechanism was introduced to eliminate the long communication paths inherent in recurrent networks.

Its core ideas are:

1. Every token can directly interact with every other token.
2. Relevance can be measured using vector similarity.
3. Queries search for information.
4. Keys advertise information.
5. Values contain transferable information.
6. Softmax converts similarity scores into attention weights.
7. Weighted combinations of values create contextual embeddings.

The next chapter formalizes these ideas mathematically and derives the complete scaled dot-product attention mechanism used in Transformer architectures.


In the next chapter we examine why recurrent architectures struggled to satisfy these requirements and why a fundamentally different approach became necessary.


# Part II — Self Attention Mathematics

# Chapter 4 — From Similarity to Self-Attention

## Introduction

In the previous chapter we introduced the central idea behind attention:

> Every token should be able to directly inspect every other token and determine which pieces of information are relevant.

This solved one of the major limitations of recurrent neural networks. Instead of forcing information to travel through long chains of hidden states, attention allows direct token-to-token interaction.

However, an important question remains:

> How can a token determine which other tokens are relevant?

This chapter introduces the first mathematical formulation of self-attention. We begin with a simple attention mechanism based on vector similarity and gradually develop the Query-Key-Value framework used in modern Transformers.

---

# 4.1 Contextual Embeddings

Traditional word embeddings assign a fixed vector to every word.

For example:

```text
Apple
```

might correspond to some embedding:

```text
EApple
```

The problem is that the meaning of a word depends on context.

Consider:

```text
Love Apple Phones
```

and

```text
Love Apple Juice
```

The meaning of:

```text
Apple
```

is different in the two sentences.

Therefore we would like to create a new representation:

```text
EApple'
```

that depends on surrounding words.

One possible representation is:

```text
EApple'
=
0.1 ELove
+
0.5 EApple
+
0.4 EPhones
```

For the sentence:

```text
Love Apple Phones
```

while for:

```text
Love Apple Juice
```

we may obtain:

```text
EApple'
=
0.1 ELove
+
0.5 EApple
+
0.4 EJuice
```

Notice that the contextual embedding is no longer determined solely by the word itself.

Instead, it becomes a weighted combination of information gathered from surrounding words.

This is the central goal of self-attention:

> Construct context-dependent embeddings from surrounding information.

---

# 4.2 Similarity as a Measure of Relevance

The next question is:

> How should we determine the weights?

Where do the numbers:

```text
0.1
0.5
0.4
```

come from?

To answer this we need a way to measure the relationship between words.

Since embeddings are vectors, a natural choice is vector similarity.

Suppose we have two embeddings:

```text
Ea
Eb
```

A simple similarity measure is the dot product:

E_a^T E_b

Geometrically:

### Large Positive Value

The vectors point in similar directions.

```text
high similarity
```

### Near Zero

The vectors are approximately orthogonal.

```text
little relationship
```

### Negative Value

The vectors point in opposing directions.

```text
low similarity
```

The dot product therefore provides a natural measure of how strongly two words are related.

---

# 4.3 Naive Attention

Suppose we wish to construct:

```text
EApple'
```

for:

```text
Love Apple Phones
```

We can compute similarities:

```text
EApple · ELoveᵀ
EApple · EAppleᵀ
EApple · EPhonesᵀ
```

These scores indicate how relevant each word is to Apple.

For example:

```text
Love   → 2
Apple  → 5
Phones → 4
```

The larger the score, the more influence that word should have when constructing the new embedding.

However, these values are not yet suitable as weights.

---

# 4.4 Why Softmax Is Necessary

The similarity scores obtained from dot products have several problems.

They may be:

* negative
* larger than one
* difficult to interpret probabilistically

For example:

```text
2
5
4
```

do not tell us how much influence each word should contribute.

We therefore apply the Softmax transformation:

\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}

Softmax has several useful properties:

### Positivity

All outputs are positive.

### Normalization

The outputs sum to one.

### Relative Importance

Larger scores receive larger weights.

For example:

```text
2
5
4
```

might become:

```text
0.04
0.67
0.29
```

These values can now be interpreted as attention weights.

---

# 4.5 Constructing New Embeddings

After applying Softmax we obtain attention weights:

```text
x1
x2
x3
```

We then compute:

```text
EApple'
=
x1 ELove
+
x2 EApple
+
x3 EPhones
```

This operation performs a weighted average of surrounding information.

The resulting embedding contains information gathered from the entire sentence.

This process transforms static embeddings into contextual embeddings.

---

# 4.6 Limitations of Naive Attention

Although this approach is intuitive, it contains two major problems.

---

## Problem 1: No Learnable Parameters

The attention mechanism currently relies only on existing embeddings.

Nothing is being learned.

The model cannot adapt its notion of relevance during training.

As a result, its representational power is limited.

---

## Problem 2: Symmetric Relationships

The dot product satisfies:

E_a^T E_b = E_b^T E_a

Therefore:

```text
similarity(Apple, Phones)
=
similarity(Phones, Apple)
```

This symmetry is problematic because language relationships are often directional.

Consider:

```text
dog chases cat
```

The relationship:

```text
dog → cat
```

is not identical to:

```text
cat → dog
```

We therefore require a richer representation.

---

# 4.7 Query, Key and Value

To solve these limitations, Transformers introduce three learned projections.

Starting from an embedding:

```text
E
```

we compute:

Q = EW_Q

K = EW_K

V = EW_V

where:

```text
WQ
WK
WV
```

are learnable parameter matrices.

---

## Query

Represents:

> What information am I searching for?

---

## Key

Represents:

> What information do I offer?

---

## Value

Represents:

> What information should be transferred if selected?

---

# 4.8 Self-Attention with Q, K and V

Consider again:

```text
Love Apple Phones
```

We compute:

```text
QApple = EApple WQ
KApple = EApple WK
VApple = EApple WV
```

and similarly for all other words.

To determine what information Apple should receive we compute:

```text
QApple · KLoveᵀ
QApple · KAppleᵀ
QApple · KPhonesᵀ
```

These scores are transformed using Softmax:

```text
x1
x2
x3
```

We then construct:

```text
EApple'
=
x1 VLove
+
x2 VApple
+
x3 VPhones
```

The resulting embedding now depends on learned notions of relevance rather than raw embedding similarity.

This significantly increases the expressive power of the model.

---

# 4.9 Parallel Computation

One of the most important properties of self-attention is that all contextual embeddings can be computed simultaneously.

For:

```text
Love Apple Phones
```

we compute:

```text
ELove'
EApple'
EPhones'
```

independently.

There is no recurrent chain:

```text
word1
→ word2
→ word3
```

as in RNNs.

This enables massive parallelization on modern hardware.

The ability to process all tokens simultaneously is one of the primary reasons Transformers scale so effectively.

---

# Key Insight

Self-attention transforms fixed word embeddings into contextual representations.

The process consists of:

1. Measuring relevance between words.
2. Converting relevance scores into attention weights.
3. Gathering information from surrounding words.
4. Constructing new context-dependent embeddings.

Query, Key and Value projections introduce learnable parameters and directional relationships, making attention significantly more expressive than simple embedding similarity.

---

# Summary

In this chapter we developed the first mathematical version of self-attention.

We introduced:

* contextual embeddings,
* similarity-based attention,
* Softmax attention weights,
* limitations of naive attention,
* Query-Key-Value representations,
* and the construction of context-dependent embeddings.

We also observed that all attention computations can be performed in parallel, eliminating one of the major limitations of recurrent architectures.

In the next chapter we derive the complete scaled dot-product attention mechanism and explain why modern Transformers compute:

Attention(Q,K,V) = Softmax(QKᵀ / √dk)V

instead of the simpler formulation introduced here.

# Part II — Self Attention Mathematics

# Chapter 5 — Scaled Dot Product Attention

## Introduction

In the previous chapter we developed the fundamental idea of self-attention.

Starting from a token embedding, we:

1. Measured similarity between words.
2. Converted similarity scores into attention weights.
3. Used these weights to aggregate information from surrounding words.
4. Constructed contextual embeddings.

We also introduced the Query-Key-Value framework:

```text
Q = XWQ
K = XWK
V = XWV
```

However, an important question remains:

> Why do modern Transformers compute attention using:

Attention(Q,K,V) = Softmax(QKᵀ/√dk)V

instead of simply:

```text
Softmax(QKᵀ)V
```

The answer lies in the statistical properties of high-dimensional vectors and the behavior of the Softmax function.

---

# 5.1 Review of Self-Attention

Consider the sentence:

```text
Love Apple Phones
```

Suppose we wish to compute a contextual representation for:

```text
Apple
```

We first create:

```text
QApple
```

representing:

> What information am I searching for?

```text
KLove
KApple
KPhones
```

representing:

> What information do these words offer?

and

```text
VLove
VApple
VPhones
```

representing:

> What information should be transferred?

We then compute similarity scores:

```text
QApple · KLoveᵀ
QApple · KAppleᵀ
QApple · KPhonesᵀ
```

These scores determine how strongly Apple should attend to each word.

---

# 5.2 Matrix Formulation

For a sequence containing multiple tokens we collect all Query vectors into a matrix:

```text
Q
```

all Key vectors into:

```text
K
```

and all Value vectors into:

```text
V
```

Instead of computing similarities one word at a time, we compute all similarities simultaneously:

QK^T

The result is called the attention score matrix.

Each element:

```text
(i,j)
```

contains the similarity between:

```text
token i
```

and

```text
token j
```

This matrix contains all pairwise token relationships.

---

# 5.3 The Variance Problem

Suppose Query and Key vectors have dimension:

```text
dk
```

For example:

```text
dk = 64
```

or

```text
dk = 128
```

or

```text
dk = 512
```

The dot product is:

QK^T=\sum_{i=1}^{d_k}q_i k_i

Assume:

```text
E[qᵢ] = 0
E[kᵢ] = 0
```

and

```text
Var(qᵢ)=1
Var(kᵢ)=1
```

for all dimensions.

Then each term:

```text
qᵢkᵢ
```

has approximately unit variance.

Since variances add:

Var(QK^T)\approx d_k

This means that the variance of attention scores grows linearly with dimensionality.

---

# 5.4 Why Increasing Variance Is Dangerous

Consider:

```text
dk = 1
```

Attention scores may look like:

```text
[1.2, 0.8, -0.4]
```

Now suppose:

```text
dk = 512
```

The scores may become:

```text
[35, 12, -8]
```

The magnitudes become significantly larger.

At first this may seem harmless.

However, these values are passed into Softmax.

---

# 5.5 Softmax Saturation

Recall:

\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}

Suppose the attention scores are:

```text
[2,1,0]
```

Softmax produces:

```text
[0.67, 0.24, 0.09]
```

which is relatively smooth.

Now consider:

```text
[30,2,-5]
```

Softmax produces something close to:

```text
[1,0,0]
```

Almost all probability mass concentrates on a single token.

The distribution becomes extremely peaked.

This phenomenon is known as **Softmax saturation**.

---

# 5.6 Why Softmax Saturation Hurts Training

A saturated Softmax creates two major problems.

### Problem 1

Attention becomes overly confident.

One token receives nearly all attention while all others are ignored.

### Problem 2

Gradients become very small.

The optimization process receives less useful information.

The chain is:

```text
Large variance
↓
Large dot products
↓
Softmax saturation
↓
Tiny gradients
↓
Difficult optimization
```

This is the real reason scaling is necessary.

---

# 5.7 Variance Stabilization

A basic property of variance states:

Var(aX)=a^2 Var(X)

We already know:

Var(QK^T)\approx d_k

Suppose we divide the dot product by:

\sqrt{d_k}

Then:

Var\left(\frac{QK^T}{\sqrt{d_k}}\right)=\frac{1}{d_k}Var(QK^T)

Substituting:

Var(QK^T)\approx d_k

gives:

Var\left(\frac{QK^T}{\sqrt{d_k}}\right)\approx 1

The variance becomes approximately constant regardless of dimensionality.

---

# 5.8 Scaled Dot Product Attention

We can now define the complete attention mechanism.

First compute attention scores:

\frac{QK^T}{\sqrt{d_k}}

Next apply Softmax:

\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)

Finally aggregate Value vectors:

\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

This operation is called **Scaled Dot Product Attention**.

---

# 5.9 Interpretation of the Formula

Each component has a specific meaning.

### Q

Defines what information each token is searching for.

### K

Defines what information each token offers.

### QKᵀ

Measures pairwise token similarity.

### 1/√dk

Stabilizes variance and prevents Softmax saturation.

### Softmax

Converts similarities into attention weights.

### V

Contains information that will be aggregated.

The final output is a collection of contextual embeddings.

---

# Key Insight

The scaling factor:

```text
1/√dk
```

is not an arbitrary design choice.

It arises naturally from variance propagation in high-dimensional spaces.

Without scaling:

* attention scores grow with dimension,
* Softmax saturates,
* gradients become small,
* training becomes unstable.

Scaling preserves approximately constant variance and enables efficient optimization.

---

# Summary

In this chapter we derived the complete self-attention equation used in modern Transformers.

We showed that:

1. Dot-product variance grows with dimension.
2. Large attention scores cause Softmax saturation.
3. Saturated Softmax leads to small gradients.
4. Scaling by √dk stabilizes variance.
5. Stable variance improves optimization.

These observations lead directly to the modern attention equation:

Attention(Q,K,V) = Softmax(QKᵀ/√dk)V

This mechanism forms the mathematical foundation of every Transformer architecture.

The next chapter introduces Multi-Head Attention and explains why a single attention mechanism is insufficient for modeling the many different relationships that exist within natural language.




