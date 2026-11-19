---
layout: distill
title: LLMs in JAX from scratch - Part 1
# description: an example of a distill-style blog post and main elements
tags: llm
giscus_comments: true
date: 2026-11-04
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Amine Andam
    affiliations:
      name: UM6P

bibliography: 2026-11-04-tinyjxllm-p1.bib

toc:
  - name: Excellent resources
  - name: Overall
  - name: Tokenization
  - name: Custom networks with JAX
  - name: The transformer
    subsections:
      - name: Token embedding
      - name: Normalization
      - name: Multi-head attention
      - name: Rotary position embeddings (RoPE)
      - name: Feed-forward network
      - name: Transformer block
      - name: Transformer language model
      - name: Interactive parameter explorer
  - name: Training
    subsections:
      - name: Loss function
      - name: AdamW optimizer
      - name: Lazy data loading
      - name: Training step
  - name: Results of training
    subsections:
      - name: Training setup
      - name: Parameter tying
      - name: Training curves
  - name: Sampling from the model
    subsections:
      - name: Sampling strategies
      - name: Prompts and configurations
      - name: Sampling results
      - name: Generated examples
  - name: What is next

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  @media (min-width: 1025px) {
    d-article {
      overflow-x: clip;
      overflow-y: visible;
    }
    d-article > d-contents {
      position: sticky;
      top: 80px;
      height: 0;
      overflow: visible;
      border-right: 0;
    }
    d-article > d-contents > nav {
      height: max-content;
      max-height: calc(100vh - 110px);
      overflow-y: auto;
      border-right: 1px solid var(--global-divider-color);
    }
  }
  d-article d-math[block] {
    display: flex;
    justify-content: center;
  }
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

In this series of blog posts, I will build and train LLMs from scratch using JAX. By “from scratch,” I mean using only pure JAX (JAX arrays, pytrees, etc.), without using any other framework (e.g., Flax). I will create a tiny neural network library that implements all the components needed to train LLMs.

The code for this series is available on [GitHub](https://github.com/AmineAndam04/tinyjxllm).

I designed this series to reflect how I like to learn. [Read more about my approach here]({% post_url 2026-11-18-how-i-like-to-learn-math-and-cs %}).

## Excellent resources

The mathematical explanations draw mainly on the following resources:

1. _Formal Algorithms for Transformers_: [arXiv](https://arxiv.org/abs/2207.09238)

2. _Speech and Language Processing_: [Online textbook](https://web.stanford.edu/~jurafsky/slp3/)

3. _Mathematical Foundations of Modern AI Systems_: [Lecture 1](https://mathforai26.github.io/documents/scribes/lecture01.pdf) or [full course](https://mathforai26.github.io/)

## Overall

My end goal is to **pretrain** a **large language model (LLM)**. This entails training a neural network with a large number of parameters to predict the next word given the previous words using a very large corpus of text as training data.

Consider a sequence of words from this corpus, <d-math>\mathbf{x} = (x_1, x_2, \ldots, x_T)</d-math>. The goal is to use the sequence <d-math>x_{\lt t} = (x_1, \ldots, x_{t-1})</d-math>, called the _context_, to predict the next word <d-math>x_t</d-math>. The LLM takes as input the context <d-math>x_{\lt t}</d-math> and outputs a probability distribution over the possible next words <d-math>p_{\theta}(\cdot \mid x_{\lt t})</d-math>, where <d-math>\theta</d-math> denotes the model's parameters. The training data can be represented as the following context–target pairs: <d-math>\{(x_{\lt t}, x_t)\}_{t=2}^{T}</d-math>. The table below illustrates this:

| Input context                           | Target word             |
| --------------------------------------- | ----------------------- |
| <d-math>(x_1)</d-math>                  | <d-math>x_2</d-math>    |
| <d-math>(x_1, x_2)</d-math>             | <d-math>x_3</d-math>    |
| <d-math>(x_1, x_2, x_3)</d-math>        | <d-math>x_4</d-math>    |
| <d-math>\ldots</d-math>                 | <d-math>\ldots</d-math> |
| <d-math>x_{\lt t}</d-math>              | <d-math>x_t</d-math>    |
| <d-math>\ldots</d-math>                 | <d-math>\ldots</d-math> |
| <d-math>(x_1, \ldots, x_{T-1})</d-math> | <d-math>x_T</d-math>    |

I train the model by minimizing the cross-entropy loss, equivalently the average negative log-likelihood:

<d-math block>
\mathcal{L}(\theta) = -\frac{1}{T-1}\sum_{t=2}^{T}\log p_{\theta}(x_t \mid x_{\lt t})
</d-math>

This loss encourages the model to assign a higher probability to the correct next word for each context.

## Tokenization

Tokenization is a preprocessing step for training LLMs. It consists of segmenting raw text into **tokens** and assigning each token an integer ID. Tokens can represent whole words (e.g., `dog`, `cat`), parts of words (e.g., `ing`, `ed`, `tion`, `ally`), or characters (e.g., `a`, `b`, 😃). From this point on, the units of prediction are tokens rather than words.

There are different tokenization methods. Throughout this series, I use **byte-level byte pair encoding (BPE)**. This process has two stages:

**1. Training the tokenizer.** I start by splitting the raw text into smaller pieces, such as words with their leading spaces. For example, `"I have a new dog 😃"` can be split into `["I", " have", " a", " new", " dog", " 😃"]`. I then represent each piece as a sequence of bytes, which are numerical units used to store text. For example, `"dog"` becomes `[100, 111, 103]` in UTF-8, corresponding to `d`, `o`, and `g`. A character such as 😃 takes several bytes.

```python
text = "dog"
print(list(text.encode("utf-8")))

text = "😃"
print(list(text.encode("utf-8")))
```

Output:

```text
[100, 111, 103]
[240, 159, 152, 131]
```

A byte has 256 possible values. Initially, each byte is a token, so the word `"dog"` is represented by three token IDs: 100, 111, and 103. BPE then repeatedly merges the most frequent pair of adjacent tokens into a new token, without merging across the boundaries of the pieces defined above. Initially, these tokens are individual bytes; later, they can be tokens created by previous merges.

_Training_ the tokenizer produces two things:

**First**, the vocabulary:

<d-math block>
\mathcal{V} = \{v_0, v_1, \ldots, v_{V-1}\}, \qquad V = |\mathcal{V}|
</d-math>

Each <d-math>v_i</d-math> is a token represented as bytes, and <d-math>i</d-math> is its ID. In Python, this can be stored as a dictionary mapping IDs to byte sequences, for example `{97: b"a", 100: b"d", 103: b"g", 111: b"o"}`. I initialize it with individual byte values and any special tokens. I choose the target vocabulary size <d-math>V</d-math> before training, then add a new entry for each merge until I reach that size.

The following table shows selected entries from the vocabulary obtained by training BPE on the TinyStories dataset <d-cite key="eldan2023tinystories"></d-cite>:

| Token ID | Token, shown as readable text              |
| -------- | ------------------------------------------ |
| 32       | `␠`                                        |
| 97       | `a`                                        |
| 256      | <code>&lt;&#124;endoftext&#124;&gt;</code> |
| 264      | `ed`                                       |
| 300      | `ing`                                      |
| 398      | `␠time`                                    |
| 430      | `Once`                                     |
| 439      | `␠upon`                                    |

Here, `␠` represents a space. The entries are shown as readable text; internally, they store bytes.

**Second**, the merge rules, in the order they were learned. Each rule records which pair to merge and the ID of the new token. For example, these are three rules from the TinyStories tokenizer:

```text
Rule 1:  b" "  + b"t"  → b" t"    (ID 257)
Rule 2:  b"h"  + b"e"  → b"he"    (ID 258)
Rule 7:  b" t" + b"he" → b" the"  (ID 263)
```

The last merge becomes possible after the earlier two have created `b" t"` and `b"he"`.

**2. Encoding the dataset.** Once the vocabulary and merge rules are learned, I use them to encode the entire dataset for pretraining. Using the notation from the previous section, each text becomes a sequence <d-math>\mathbf{x} = (x_1, x_2, \ldots, x_T)</d-math>, where each <d-math>x_t</d-math> is now a token ID rather than a word. I convert the text to bytes and apply the learned merges to obtain these IDs.

For example, using my TinyStories tokenizer:

```text
"Once upon a time"
        ↓
["Once", " upon", " a", " time"]
        ↓
[430, 439, 259, 398]
```

These token IDs are what I feed into the model. To decode them, I look up their vocabulary entries, join the bytes, and convert them back to text.

Tokenizers trained on different data and with different vocabulary sizes can encode the same text differently. The examples below use a **10,000-entry vocabulary trained on TinyStories** and a **50,257-entry vocabulary trained on OpenWebText (OWT)** <d-cite key="gokaslan2019openwebtext"></d-cite>. `␠software` is one token in the OWT vocabulary and three in the TinyStories vocabulary.

| Input text   | TinyStories tokenizer: tokens and IDs                           | OWT tokenizer: tokens and IDs |
| ------------ | --------------------------------------------------------------- | ----------------------------- |
| `␠software`  | `["␠soft", "w", "are"]`<br>`[1049, 119, 517]`                   | `["␠software"]`<br>`[3476]`   |
| `␠algorithm` | `["␠al", "g", "or", "ith", "m"]`<br>`[554, 103, 305, 326, 109]` | `["␠algorithm"]`<br>`[11159]` |

## Custom networks with JAX

Throughout this series, I will use JAX to implement all the network layers from scratch. I register my layer classes as **pytrees** so JAX can access their parameter arrays when applying transformations such as `jax.grad` and `jax.jit`. Consider a linear layer, which has two trainable parameter arrays: a weight matrix <d-math>W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}</d-math> and a bias vector <d-math>b \in \mathbb{R}^{d_{\mathrm{out}}}</d-math>. Given an input <d-math>X \in \mathbb{R}^{\ldots \times d_{\mathrm{in}}}</d-math>, the layer outputs:

<d-math block>
XW^{T} + b \in \mathbb{R}^{\ldots \times d_{\mathrm{out}}}
</d-math>

I start by implementing the math inside a class:

```python
class Linear:
    def __init__(
        self, in_features: int, out_features: int, use_bias: bool = True, *, key: PRNGKeyArray
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        lim = 1 / math.sqrt(in_features)
        self.weight = uniform_init(key, (out_features, in_features), lim)
        self.bias = constant_init(shape=(out_features,), constant=0) if use_bias else None

    def __call__(self, x: Array) -> Array:
        x = x @ self.weight.T
        if self.use_bias:
            x = x + self.bias
        return x
```

I then use `@register_pytree_with_keys_class` to tell JAX how to flatten and unflatten a `Linear` instance. The documentation is provided [here](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_with_keys_class.html) and [here](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_with_keys.html). In general, `children` contains parameter arrays and `aux_data` contains static metadata.

```python
from jax.tree_util import GetAttrKey, register_pytree_with_keys_class

@register_pytree_with_keys_class
class Linear:
    ...
    def tree_flatten_with_keys(self):
        children = ((GetAttrKey("weight"), self.weight), (GetAttrKey("bias"), self.bias))
        aux_data = (self.in_features, self.out_features, self.use_bias)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.in_features, obj.out_features, obj.use_bias = aux_data
        obj.weight, obj.bias = children
        return obj
```

`GetAttrKey("weight")` labels a child with its attribute name. This helps with debugging and supports a custom `jax_pprint` helper that displays the layers and their parameter shapes. For example, once I have implemented `TransformerLM`, I can inspect it as follows:

```python
model = TransformerLM(
    vocab_size=10000,
    context_length=128,
    num_layers=3,
    d_model=32,
    num_heads=4,
    d_ff=54,
    key=jax.random.key(0),
)
jax_pprint(model)
```

Output:

<pre tabindex="0" role="region" aria-label="Full model output, scroll to read all layers" style="max-height: 28rem; overflow: auto;"><code>Model: TransformerLM
  token_embeddings: Embedding
    embeddings: shape=(10000, 32), dtype=float32
  blocks:
    [0]: TransformerBlock
      norm1: RMSNorm
        gain: shape=(32,), dtype=float32
      attn: MultiHeadSelfAttention
        w_query: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        w_key: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        w_value: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        w_output: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        rope: RoPE
      norm2: RMSNorm
        gain: shape=(32,), dtype=float32
      ff: FeedForward
        w1: Linear
          weight: shape=(54, 32), dtype=float32
          bias: None
        w2: Linear
          weight: shape=(32, 54), dtype=float32
          bias: None
        w3: Linear
          weight: shape=(54, 32), dtype=float32
          bias: None
    [1]: TransformerBlock
      norm1: RMSNorm
        gain: shape=(32,), dtype=float32
      attn: MultiHeadSelfAttention
        w_query: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        w_key: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        w_value: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        w_output: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        rope: RoPE
      norm2: RMSNorm
        gain: shape=(32,), dtype=float32
      ff: FeedForward
        w1: Linear
          weight: shape=(54, 32), dtype=float32
          bias: None
        w2: Linear
          weight: shape=(32, 54), dtype=float32
          bias: None
        w3: Linear
          weight: shape=(54, 32), dtype=float32
          bias: None
    [2]: TransformerBlock
      norm1: RMSNorm
        gain: shape=(32,), dtype=float32
      attn: MultiHeadSelfAttention
        w_query: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        w_key: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        w_value: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        w_output: Linear
          weight: shape=(32, 32), dtype=float32
          bias: None
        rope: RoPE
      norm2: RMSNorm
        gain: shape=(32,), dtype=float32
      ff: FeedForward
        w1: Linear
          weight: shape=(54, 32), dtype=float32
          bias: None
        w2: Linear
          weight: shape=(32, 54), dtype=float32
          bias: None
        w3: Linear
          weight: shape=(54, 32), dtype=float32
          bias: None
  ln_final: RMSNorm
    gain: shape=(32,), dtype=float32
  lm_head: Linear
    weight: shape=(10000, 32), dtype=float32
    bias: None</code></pre>

The indentation shows how the layers are nested. Each array is displayed with its shape and data type; for example, the embedding table contains 10,000 vectors of size 32.

I will implement many custom layers, so writing these two methods in every class would be repetitive. Instead, I define a decorator that creates them and registers the class automatically. Each layer only needs to specify the attribute names for its `children` and `aux_data`:

```python
def auto_pytree_with_keys(children: tuple[str, ...], aux_data: tuple[str, ...]):
    def decorator(cls):
        def tree_flatten_with_keys(obj):
            children_ = [(GetAttrKey(chld), getattr(obj, chld)) for chld in children]
            aux_data_ = tuple(getattr(obj, aux) for aux in aux_data)
            return children_, aux_data_

        def tree_unflatten(aux_data_, children_):
            obj = object.__new__(cls)
            for f, value in zip(children, children_, strict=True):
                setattr(obj, f, value)
            for f, value in zip(aux_data, aux_data_, strict=True):
                setattr(obj, f, value)
            return obj

        cls.tree_flatten_with_keys = tree_flatten_with_keys
        cls.tree_unflatten = tree_unflatten

        return register_pytree_with_keys_class(cls)

    return decorator
```

My `Linear` class then uses this decorator in place of `@register_pytree_with_keys_class` and the two explicit methods. Its initialization and forward computation stay the same:

```python
@auto_pytree_with_keys(
    children=("weight", "bias"),
    aux_data=("in_features", "out_features", "use_bias"),
)
class Linear:
    ...  # Same __init__ and __call__ as above.
```

## The transformer

I will build the transformer from four main components: token embeddings, normalization, multi-head attention, and a feed-forward network. The diagram shows how they fit together, with the transformer block repeated <d-math> L </d-math> times. Below, <d-math>X \in \{0, \ldots, |\mathcal{V}|-1\}^{B \times T}</d-math> denotes the input token IDs: a batch of <d-math>B</d-math> sequences, each of length <d-math>T</d-math>.

<figure>
  <img src="{{ 'assets/img/transformer_block_overview.svg' | relative_url }}" alt="Decoder-only transformer with final normalization and a language-model head" width="640" height="1010" style="display: block; width: 100%; max-width: 640px; height: auto; margin: 1.5rem auto;" loading="lazy">
  <figcaption>Decoder-only transformer.</figcaption>
</figure>

### Token embedding

Token embedding assigns a trainable vector in <d-math>\mathbb{R}^{d_{\mathrm{model}}}</d-math> to each token in the vocabulary. In practice, I learn an embedding matrix <d-math>W_e \in \mathbb{R}^{|\mathcal{V}| \times d_{\mathrm{model}}}</d-math>.

In JAX, I create a class that stores this matrix. Row <d-math>i</d-math> contains the vector for the token with ID <d-math>i</d-math>. Calling the layer looks up the vectors for the input token IDs:

```python
@auto_pytree_with_keys(
    children=("embeddings",),aux_data=("num_embeddings", "dim_embedding"))
class Embedding:
    def __init__(self, num_embeddings: int, dim_embedding: int, *, key: PRNGKeyArray) -> None:
        self.num_embeddings = num_embeddings
        self.dim_embedding = dim_embedding
        self.embeddings = normal_init(key, (num_embeddings, dim_embedding), std=0.02)

    def __call__(self, x: Int[ArrayLike, "..."]) -> Array:
        if not jnp.issubdtype(x.dtype, jnp.integer):
            raise TypeError(f"Inputs should be integers, got {x.dtype}")
        return self.embeddings.at[x].get(mode="fill", fill_value=jnp.nan)

    def decode(self, x: Array):
        return x @ self.embeddings.T
```

Positional information can be added alongside token embeddings to help the transformer represent token order. For example, "the dog bites the man" and "the man bites the dog" contain the same tokens but have different meanings. Token embeddings alone do not encode these differences in position.

There are different ways to represent position. One straightforward approach is to learn positional embeddings in the same way as token embeddings, using each token's position in the sequence as the input instead of its ID.

I use rotary position embeddings (RoPE), which introduce positional information inside the multi-head attention layer. RoPE is explained later in this post.

Number of trainable parameters: the embedding layer has <d-math>|\mathcal{V}| \times d_{\mathrm{model}}</d-math> parameters.

### Normalization

Normalization helps control the scale of neural network activations. I use root mean square normalization (RMSNorm). Given an activation vector <d-math>u \in \mathbb{R}^{d_{\mathrm{model}}}</d-math>, RMSNorm computes:

<d-math block>
\mathrm{RMSNorm}(u) = \gamma \odot \frac{u}{\sqrt{\frac{1}{d_{\mathrm{model}}}\lVert u\rVert_2^2 + \epsilon}}
</d-math>

Here, <d-math>\gamma \in \mathbb{R}^{d_{\mathrm{model}}}</d-math> is a trainable parameter, <d-math>\odot</d-math> denotes elementwise multiplication, and <d-math>\epsilon</d-math> is a small positive constant to prevent division by zero. In the implementation, I compute the mean square along the last axis, so each token's activation vector is normalized independently:

```python
@auto_pytree_with_keys(children=("gain",),aux_data=("d_model", "eps"))
class RMSNorm:
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.d_model = d_model
        self.eps = eps
        self.gain = jnp.ones((d_model,))

    def __call__(self, x: Array) -> Array:
        ms = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(ms + self.eps) * self.gain
        return x
```

Number of trainable parameters: the RMSNorm layer has <d-math>d_{\mathrm{model}}</d-math> parameters.

### Multi-head attention

The multi-head attention (MHA) layer has four learnable projection matrices: <d-math>W^Q</d-math> for queries, <d-math>W^K</d-math> for keys, <d-math>W^V</d-math> for values, and <d-math>W^O</d-math> for the output. All of them are in <d-math>\mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}</d-math>.

After embedding and normalization, the input to the MHA layer is <d-math>X \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}</d-math>. I start by projecting <d-math>X</d-math> to obtain the queries, keys, and values:

<d-math block class="mathjax_ignore">
\begin{aligned}
Q &= XW^Q \\
K &= XW^K \\
V &= XW^V
\end{aligned}
\qquad Q, K, V \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}
</d-math>

The next step is to compute **attention scores**. For each sequence in the batch, the score <d-math>S_{i,j}</d-math> measures how relevant the token at position <d-math>j</d-math> is to the token at position <d-math>i</d-math>. The resulting representation at position <d-math>i</d-math> is used to predict the next token, at position <d-math>i+1</d-math>. Ignoring scaling and masking for the moment, I compute dot-product scores and apply softmax across each row:

<d-math block class="mathjax_ignore">
\begin{aligned}
S &= QK^{\top} \\
A &= \mathrm{softmax}_{\mathrm{row}}(S)
\end{aligned}
\qquad S, A \in \mathbb{R}^{B \times T \times T}
</d-math>

Here, <d-math>\top</d-math> denotes a transpose of the last two axes; the batch axis is unchanged. The entries of <d-math>A</d-math> are the normalized attention weights.

A single head produces one attention weight for each token pair. However, tokens can be relevant to one another in several ways: one token might be the subject of another, or provide information about whether the subject is singular or plural. Multi-head attention allows the model to learn several attention patterns in parallel. With <d-math>h</d-math> heads, I compute <d-math>hT^2</d-math> score entries per sequence instead of <d-math>T^2</d-math>. The heads learn these patterns without being assigned specific roles.

To do this, I split the last dimension of <d-math>Q</d-math>, <d-math>K</d-math>, and <d-math>V</d-math> into <d-math>h</d-math> heads, each of size <d-math>d_h = d_{\mathrm{model}}/h</d-math>. This requires <d-math>d_{\mathrm{model}}</d-math> to be divisible by <d-math>h</d-math>. I then rearrange the axes to place the head axis before the sequence axis:

<d-math block class="mathjax_ignore">
Q_{\mathrm{heads}}, K_{\mathrm{heads}}, V_{\mathrm{heads}}
\in \mathbb{R}^{B \times h \times T \times d_h}
</d-math>

The following figure illustrates this:
<figure>
  <img src="{{ 'assets/img/mha_head_split.svg' | relative_url }}" alt="Splitting the attention projections into heads" width="640" height="1010" style="display: block; width: 100%; max-width: 640px; height: auto; margin: 1.5rem auto;" loading="lazy">
  <figcaption>Splitting the attention projections into heads.</figcaption>
</figure>

For each head <d-math>r \in \{1, \ldots, h\}</d-math>, the queries, keys, and values have shape <d-math>B \times T \times d_h</d-math>. I compute the scaled scores, attention weights, and head output:

<d-math block class="mathjax_ignore">
\begin{aligned}
S_r &= \frac{Q_r K_r^{\top}}{\sqrt{d_h}}
&&\in \mathbb{R}^{B \times T \times T} \\
A_r &= \mathrm{softmax}_{\mathrm{row}}(S_r + M)
&&\in \mathbb{R}^{B \times T \times T} \\
O_r &= A_r V_r
&&\in \mathbb{R}^{B \times T \times d_h}
\end{aligned}
</d-math>

Here, <d-math>M \in \mathbb{R}^{T \times T}</d-math> is the causal mask. It is used to prevent each position from attending to future tokens when predicting $x_{i+1}$ from tokens up to $x_i$:

<d-math block class="mathjax_ignore">
M_{i,j} =
\begin{cases}
0 & \text{if } j \le i \\
-\infty & \text{if } j > i
\end{cases}
</d-math>

Finally, I concatenate the head outputs along the last axis and apply the output projection:

<d-math block class="mathjax_ignore">
\begin{aligned}
O &= [\,O_1 \mid O_2 \mid \cdots \mid O_h\,]
&&\in \mathbb{R}^{B \times T \times d_{\mathrm{model}}} \\
Y &= OW^O
&&\in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}
\end{aligned}
</d-math>

I first implement scaled dot-product attention in JAX. The following function returns the head output <d-math>O_r = A_r V_r</d-math>:

```python
def scaled_dot_product_attention(Q: Array, K: Array, V: Array, mask: None | Array = None):
    scores = (1 / math.sqrt(K.shape[-1])) * jnp.einsum("... a b, ... c b -> ... a c", Q, K)
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)
    out = jnp.einsum("... a b, ... b c -> ... a c", jax.nn.softmax(scores, axis=-1), V)
    return out
```

I then use this function inside the multi-head attention layer. The RoPE step is explained in the next section.

```python
@auto_pytree_with_keys(
    children=("w_query", "w_key", "w_value", "w_output", "rope"),
    aux_data=("d_model", "num_heads", "use_bias"),
)
class MultiHeadSelfAttention:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_bias: bool = False,
        rope: RoPE | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        assert d_model % num_heads == 0
        if rope is not None:
            assert rope.d_h == d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.rope = rope
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.w_query = Linear(
            in_features=d_model, out_features=d_model, use_bias=use_bias, key=key1
        )
        self.w_key = Linear(in_features=d_model, out_features=d_model, use_bias=use_bias, key=key2)
        self.w_value = Linear(
            in_features=d_model, out_features=d_model, use_bias=use_bias, key=key3
        )
        self.w_output = Linear(
            in_features=d_model, out_features=d_model, use_bias=use_bias, key=key4
        )

    def __call__(self, x: Array, token_positions: Array | None = None) -> Array:
        query = rearrange(self.w_query(x), "... seq (h d) -> ... h seq d ", h=self.num_heads)
        key = rearrange(self.w_key(x), "... seq (h d) -> ... h seq d ", h=self.num_heads)
        value = rearrange(self.w_value(x), "... seq (h d) -> ... h seq d ", h=self.num_heads)
        if self.rope:
            if token_positions is None:
                token_positions = jnp.arange(x.shape[-2])
            token_positions = token_positions[..., None, :]
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)
        mask = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2]), dtype=bool))
        out = scaled_dot_product_attention(Q=query, K=key, V=value, mask=mask)
        out = rearrange(out, "... h seq d -> ... seq (h d)")
        return self.w_output(out)
```

Number of trainable parameters: the bias-free MHA layer has <d-math> 4 d_{\mathrm{model}}^2</d-math> parameters.

### Rotary position embeddings (RoPE)

Consider two tokens: one at position <d-math>i</d-math> and one at position <d-math>j</d-math>. Ignoring the scaling factor, their dot-product score in one attention head is:

<d-math block class="mathjax_ignore">
s_{i,j} = q_i k_j^{\top}
</d-math>

The goal is to make this score depend on the relative position of the tokens, <d-math>i-j</d-math>. RoPE, introduced in RoFormer <d-cite key="su2023roformer"></d-cite>, provides a way to do this:

<d-math block class="mathjax_ignore">
s_{i,j} = q_i R_{i-j} k_j^{\top}
</d-math>
Here, <d-math>R_{i-j}</d-math> is a <d-math>d_h \times d_h</d-math> rotation matrix that depends on <d-math>i-j</d-math>.

To obtain this score, I rotate the query and key as follows:

<d-math block class="mathjax_ignore">
\begin{aligned}
\widetilde{q}_i &= q_i R_i \\
\widetilde{k}_j &= k_j R_j
\end{aligned}
</d-math>

For any position <d-math>i</d-math> in the sequence, the rotation matrix is:

<d-math block class="mathjax_ignore">
R_i =
\begin{pmatrix}
\cos(i\omega_1) & \sin(i\omega_1) & 0 & 0 & \cdots & 0 & 0 \\
-\sin(i\omega_1) & \cos(i\omega_1) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos(i\omega_2) & \sin(i\omega_2) & \cdots & 0 & 0 \\
0 & 0 & -\sin(i\omega_2) & \cos(i\omega_2) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos(i\omega_p) & \sin(i\omega_p) \\
0 & 0 & 0 & 0 & \cdots & -\sin(i\omega_p) & \cos(i\omega_p)
\end{pmatrix}
</d-math>

where <d-math>\omega_r=\theta^{-2(r-1)/d_h}</d-math> for <d-math>r=1,\ldots,\frac{d_h}{2}</d-math>. The base <d-math>\theta</d-math> is a fixed hyperparameter.

These rotations satisfy <d-math>R_i R_j^{\top}=R_{i-j}</d-math>. Therefore:

<d-math block class="mathjax_ignore">
\begin{aligned}
s_{i,j} &= \widetilde{q}_i\widetilde{k}_j^{\top} \\
&= q_i R_i R_j^{\top} k_j^{\top} \\
&= q_i R_{i-j} k_j^{\top}
\end{aligned}
</d-math>

I apply RoPE to queries and keys; the values stay unchanged.

For efficiency, I precompute and cache the cosine and sine values instead of constructing the full matrix <d-math>R_i</d-math>. For a row vector <d-math>x</d-math>, the same rotation can be computed using elementwise operations, following [Section 3.4.2 of the paper](https://arxiv.org/html/2104.09864v5#S3.SS4.SSS2):

<d-math block class="mathjax_ignore">
(xR_i)^{\top} =
\begin{pmatrix}
x_1 \\ x_2 \\ \vdots \\ x_{d_h-1} \\ x_{d_h}
\end{pmatrix}
\odot
\begin{pmatrix}
\cos(i\omega_1) \\ \cos(i\omega_1) \\ \vdots \\ \cos(i\omega_p) \\ \cos(i\omega_p)
\end{pmatrix}
+
\begin{pmatrix}
-x_2 \\ x_1 \\ \vdots \\ -x_{d_h} \\ x_{d_h-1}
\end{pmatrix}
\odot
\begin{pmatrix}
\sin(i\omega_1) \\ \sin(i\omega_1) \\ \vdots \\ \sin(i\omega_p) \\ \sin(i\omega_p)
\end{pmatrix}
</d-math>

This is how I implement RoPE. I cache the cosine and sine values, then rotate each pair <d-math>(x_1,x_2)</d-math> into <d-math>(-x_2,x_1)</d-math> to form the second term:

```python
from functools import cache
@cache
def rope_rotations(max_seq_len: int, d_h: int, theta: float):
    with jax.ensure_compile_time_eval():
        exponents = -jnp.arange(0, d_h, step=2) / d_h
        thetas = theta**exponents
        position = jnp.arange(max_seq_len)
        rotates = jnp.outer(position, thetas)
        rotates = jnp.repeat(rotates, 2, axis=-1)
        return jnp.cos(rotates), jnp.sin(rotates)  # of shape (max_seq_len,d_h)


@auto_pytree_with_keys(
    children=(),
    aux_data=("theta", "d_h", "max_seq_len"),
)
class RoPE:

    def __init__(self, theta: float, d_h: int, max_seq_len: int) -> None:
        self.theta = theta
        self.d_h = d_h
        self.max_seq_len = max_seq_len

    def __call__(self, x: Array, token_positions: Array) -> Array:
        coses, sines = rope_rotations(max_seq_len=self.max_seq_len, d_h=self.d_h, theta=self.theta)
        coses = coses[token_positions]
        sines = sines[token_positions]
        frst, scnd = jnp.unstack(rearrange(x, " ... (i j) -> ... i j", j=2), axis=-1)
        x_sins = rearrange(jnp.stack([-scnd, frst], axis=-1), "... i j -> ... (i j)")
        return x * coses + x_sins * sines
```

The `RoPE` class has no trainable parameters, so its `children` tuple is empty.

### Feed-forward network

The feed-forward network applies the same transformation independently to each token. I use **SwiGLU**, which combines two projections through an elementwise product, then projects back to the model dimension:

<d-math block class="mathjax_ignore">
\begin{aligned}
\mathrm{FFN}(X) &= \bigl(\mathrm{SiLU}(XW_1^{\top}) \odot XW_3^{\top}\bigr)W_2^{\top} \\
\mathrm{SiLU}(z) &= z\,\sigma(z)
\end{aligned}
</d-math>

Here, <d-math>\sigma</d-math> is the sigmoid function. The input and output both have shape <d-math>B \times T \times d_{\mathrm{model}}</d-math>, while the two intermediate projections have shape <d-math>B \times T \times d_{\mathrm{ff}}</d-math>. The weight matrices have dimensions:

<d-math block class="mathjax_ignore">
\begin{aligned}
W_1, W_3 &\in \mathbb{R}^{d_{\mathrm{ff}} \times d_{\mathrm{model}}} \\
W_2 &\in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}
\end{aligned}
</d-math>

I set <d-math> d_{\mathrm{ff}} \approx \frac{8}{3}d_{\mathrm{model}}</d-math>, rounded to a nearby multiple of 64.

I implement this in JAX as follows:

```python
def find_d_ff_(d_model: int, multiple_of: int = 64) -> int:
    d_ff = 8 * d_model / 3
    return multiple_of * round(d_ff / multiple_of)

@auto_pytree_with_keys(
    children=("w1", "w2", "w3"),
    aux_data=("d_model", "d_ff", "act_fun"),
)
class FeedForward:

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        act_fun: Callable = jax.nn.silu,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.d_model = d_model
        self.act_fun = act_fun
        if d_ff is None:
            d_ff = find_d_ff_(d_model=d_model)
        self.d_ff = d_ff
        key1, key2, key3 = jax.random.split(key, 3)
        self.w1 = Linear(in_features=d_model, out_features=d_ff, use_bias=False, key=key1)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, use_bias=False, key=key2)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, use_bias=False, key=key3)

    def __call__(self, x: Array) -> Array:
        return self.w2(self.act_fun(self.w1(x)) * self.w3(x))
```

Number of trainable parameters: the feed-forward network has <d-math>3d_{\mathrm{ff}} \times d_{\mathrm{model}}</d-math> parameters.

### Transformer block

I now assemble the normalization, multi-head attention, and feed-forward layers into a transformer block:

```python
@auto_pytree_with_keys(
    children=("norm1", "attn", "norm2", "ff"),
    aux_data=(),
)
class TransformerBlock:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        *,
        use_bias: bool = False,
        act_fun: Callable = jax.nn.silu,
        eps: float = 1e-5,
        rope: RoPE | None = None,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2 = jax.random.split(key)
        self.norm1 = RMSNorm(d_model=d_model, eps=eps)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, use_bias=use_bias, rope=rope, key=key1
        )

        self.norm2 = RMSNorm(d_model=d_model, eps=eps)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, act_fun=act_fun, key=key2)

    def __call__(self, x: Array, token_positions: None | Array = None) -> Array:
        x = x + self.attn(x=self.norm1(x), token_positions=token_positions)
        x = x + self.ff(self.norm2(x))
        return x
```

Number of trainable parameters: one transformer block with bias-free projections has <d-math>4d_{\mathrm{model}}^2 + 3d_{\mathrm{ff}}d_{\mathrm{model}} + 2d_{\mathrm{model}}</d-math> parameters.

### Transformer language model

I stack <d-math>L</d-math> transformer blocks and apply a final RMSNorm. The `lm_head` then projects the final normalized activations to logits of shape <d-math>B \times T \times |\mathcal{V}|</d-math>: one score per vocabulary token at each position.

```python
@auto_pytree_with_keys(
    children=("token_embeddings", "blocks", "ln_final", "lm_head"),
    aux_data=("context_length",),
)
class TransformerLM:
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        *,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        act_fun: Callable = jax.nn.silu,
        eps: float = 1e-5,
        key: PRNGKeyArray,
    ) -> None:
        assert d_model % num_heads == 0
        keys = jax.random.split(key, num_layers + 2)
        self.context_length = context_length

        rope = RoPE(theta=rope_theta, d_h=d_model // num_heads, max_seq_len=context_length)
        self.token_embeddings = Embedding(vocab_size, d_model, key=keys[0])
        self.blocks = tuple(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_bias=use_bias,
                act_fun=act_fun,
                eps=eps,
                rope=rope,
                key=keys[i + 1],
            )
            for i in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model=d_model, eps=eps)
        self.lm_head = Linear(d_model, vocab_size, use_bias=False, key=keys[-1])

    def __call__(self, token_ids: Array, token_positions: Array | None = None) -> Array:
        x = self.token_embeddings(token_ids)
        for block in self.blocks:
            x = block(x, token_positions)
        return self.lm_head(self.ln_final(x))
```

Number of trainable parameters: the final RMSNorm has <d-math>d_{\mathrm{model}}</d-math> parameters, and the `lm_head` has <d-math>|\mathcal{V}|d_{\mathrm{model}}</d-math> parameters.

The following table summarizes the trainable parameter counts:

| Component                         | Trainable parameters                                                                                                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Token embeddings                  | <d-math>\lvert\mathcal{V}\rvert d_{\mathrm{model}}</d-math>                                                                                                             |
| One transformer block             | <d-math>4d_{\mathrm{model}}^2 + 3d_{\mathrm{ff}}d_{\mathrm{model}} + 2d_{\mathrm{model}}</d-math>                                                                       |
| Final RMSNorm                     | <d-math>d_{\mathrm{model}}</d-math>                                                                                                                                     |
| LM head                           | <d-math>\lvert\mathcal{V}\rvert d_{\mathrm{model}}</d-math>                                                                                                             |
| Total (<d-math>L</d-math> blocks) | <d-math>2\lvert\mathcal{V}\rvert d_{\mathrm{model}} + L(4d_{\mathrm{model}}^2 + 3d_{\mathrm{ff}}d_{\mathrm{model}} + 2d_{\mathrm{model}}) + d_{\mathrm{model}}</d-math> |

### Interactive parameter explorer

Change the dimensions below to see how the parameter counts change throughout the transformer. Select a layer to see its calculation.

<iframe
  src="{{ '/assets/html/transformer-parameter-explorer.html' | relative_url }}"
  title="Interactive transformer diagram with live parameter counts"
  width="100%"
  height="1000"
  style="display: block; width: 100%; border: 0;"
  loading="lazy"
></iframe>

## Training

I now train the model to predict the next token. For each batch, `x` contains the input token IDs and `y` contains their next-token targets. Both have shape <d-math>B \times T</d-math>, while the model returns logits of shape <d-math>B \times T \times |\mathcal{V}|</d-math>.

### Loss function

I use the cross-entropy loss introduced at the beginning of this post. `log_softmax` computes the log probabilities directly for numerical stability. I then select the log probability of each target token and average its negative value over the batch and sequence positions:

```python
def cross_entropy_loss(logits: Array, targets: Array) -> Array:
    loss = -jnp.take_along_axis(
        jax.nn.log_softmax(x=logits, axis=-1), indices=targets[..., None], axis=-1
    ).mean()
    return loss
```

### AdamW optimizer

I implement AdamW from scratch. Given the gradient <d-math>g_t</d-math>, I update the moving averages of gradients and squared gradients, starting from <d-math>m_0=v_0=0</d-math>:

<d-math block class="mathjax_ignore">
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\end{aligned}
</d-math>

<d-math block class="mathjax_ignore">
\begin{aligned}
\alpha_t &= \eta_t\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
\theta'_t &= (1-\eta_t\lambda)\theta_{t-1} \\
\theta_t &= \theta'_t - \alpha_t\frac{m_t}{\sqrt{v_t}+\epsilon}
\end{aligned}
</d-math>

Here, <d-math>\eta_t</d-math> is the learning rate, <d-math>\lambda</d-math> is weight decay, and <d-math>\epsilon</d-math> prevents division by zero. I store the moments and step counter in `AdamWState` and use `jax.tree.map` to apply the updates across the model:

```python
class AdamWState(NamedTuple):
    lr: float
    beta_1: float
    beta_2: float
    decay: float
    eps: float
    step: int
    m: Any
    v: Any

def init_adamw(model, lr, beta_1, beta_2, decay, eps):
    return AdamWState(
        lr=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        decay=decay,
        eps=eps,
        step=1,
        m=jax.tree.map(jnp.zeros_like, model),
        v=jax.tree.map(jnp.zeros_like, model),
    )

def update_adamw(model, grad, state):
    t, lr, decay = state.step, state.lr, state.decay
    beta_1, beta_2 = state.beta_1, state.beta_2

    alpha = lr * (1 - beta_2**t) ** 0.5 / (1 - beta_1**t)

    m = jax.tree.map(lambda m, g: beta_1 * m + (1 - beta_1) * g, state.m, grad)
    v = jax.tree.map(lambda v, g: beta_2 * v + (1 - beta_2) * g * g, state.v, grad)

    model = jax.tree.map(
        lambda p, m_, v_: p - lr * decay * p - alpha * m_ / (jnp.sqrt(v_) + state.eps), model, m, v
    )
    state = state._replace(m=m, v=v, step=t + 1)
    return model, state
```

### Lazy data loading

After training the tokenizer, I tokenized the full dataset and saved it in a NumPy `.npy` file. To load it, I open it with [`mmap_mode="r"`](https://numpy.org/doc/stable/reference/generated/numpy.load.html). Memory mapping makes the required portions available on demand without loading the entire dataset into RAM.

```python
def data_loading(data: Any, batch_size: int, context_length: int, rng):
    """
    Lazy loading of data
    """
    starts = rng.integers(0, len(data) - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length + 1)
    window = np.asarray(data[idx], dtype=np.int32)
    return jnp.asarray(window[:, :-1]), jnp.asarray(window[:, 1:])
```

`window` has shape <d-math>B \times (T+1)</d-math>. I return the first <d-math>T</d-math> tokens as inputs and the last <d-math>T</d-math> as next-token targets. Only the sampled windows are materialized, and both returned JAX arrays have shape <d-math>B \times T</d-math>.

### Training step

In the training script, I define the following functions inside `train(args)`, after initializing the model and optimizer. `loss_fn` runs the model and computes the loss:

```python
@jax.jit
def loss_fn(model, x, y):
    y_hat = model(token_ids=x)
    return cross_entropy_loss(logits=y_hat, targets=y)
```

`train_step` uses `jax.value_and_grad` to compute the loss and its gradients with respect to the model parameters. I measure the gradient norm, optionally clip the gradients, and apply the optimizer update:

```python
@jax.jit(donate_argnums=(0, 1))
def train_step(model, optim_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    grad_norm = global_norm(grads)
    if args.clip_grads > 0:
        grads = clip_grads(grads, args.clip_grads, eps=args.optim_eps, norm=grad_norm)
    model, optim_state = optim_update(model=model, grad=grads, state=optim_state)
    return model, optim_state, (loss, grad_norm)
```

The function returns the updated model and optimizer state, along with the loss and gradient norm. The optimizer and full training script will be provided on GitHub.

## Results of training

I used this implementation to pretrain a language model on **TinyStories**. I rented an RTX 4090 GPU from Vast.ai for these experiments. The configurations, training curves, and run files are available in this [public W&B project](https://forge.coreweave.com/wandb/andamamine-cleanmarl/JAX-LLMs-Blog).

### Training setup

With my 10,000-token vocabulary, the TinyStories training dataset contains 529,022,087 tokens. I used a context length of <d-math>T = 256</d-math> tokens. For a batch size <d-math>B</d-math>, each optimization step processes <d-math>B \times T</d-math> tokens, so the number of steps needed to go through the dataset is:

<d-math block>
N_{\mathrm{steps}} \approx \frac{N_{\mathrm{tokens}}}{B \times T}
</d-math>

This gives approximately 64,000 steps for a batch size of 32 and 32,000 steps for a batch size of 64.

The initial model has four transformer blocks, <d-math>d_{\mathrm{model}} = 512</d-math>, <d-math>d_{\mathrm{ff}} = 1344</d-math>, and 16 attention heads. The model has 22,696,448 trainable parameters in total.

I trained with AdamW and a cosine learning-rate scheduler, with warmup over the first 10% of training steps. I used <d-math>\beta_1 = 0.9</d-math>, weight decay 0.01, and <d-math>\epsilon = 10^{-5}</d-math>. The default minimum learning rate was 0.0001, and gradient clipping is disabled. All runs use seed 42.

In my initial experiments, I tried batch sizes of 32 and 64, along with different values of <d-math>\beta_2</d-math>, weight decay, the minimum learning rate, and the RoPE base <d-math>\theta</d-math>. Learning rates around 0.003–0.004 had worked well in earlier TinyStories experiments, so I kept using them without spending much time tuning the learning rate.

The table below shows the average training time over seven experiments for each batch size.

| Batch size | Training steps | Training time (minutes)         |
| ---------- | -------------- | ------------------------------- |
| 32         | 64,000         | <d-math>38.09 \pm 0.13</d-math> |
| 64         | 32,000         | <d-math>33.85 \pm 0.10</d-math> |

The best model I found in these initial experiments used <d-math>\beta_2 = 0.99</d-math>, a batch size of 32, and a peak learning rate of 0.004, reaching a validation loss of [1.2846](https://forge.coreweave.com/wandb/andamamine-cleanmarl/JAX-LLMs-Blog/runs/qxn2col1). I kept this configuration for the following experiments.

### Parameter tying

With the dimensions of the four-block model entered in the interactive tool above, the token embedding and the LM head each have 5.12 million parameters. Together, they account for about 45% of the model: almost half of its parameters map token IDs to vectors at the input and project the final representations back to vocabulary scores at the output. I wanted to see whether I could allocate some of these parameters to the transformer blocks instead.

That is why I tried **parameter tying**. The idea is to use the same trainable matrix for the token embedding and the LM head. I keep the embedding matrix <d-math>W_e</d-math> as it is and reuse its transpose for the output projection. Given the activations <d-math>Z</d-math> after the final normalization, the logits become:

<d-math block>
\mathrm{logits}=ZW_e^{\top}
</d-math>

The `decode` method in the `Embedding` layer already provides this projection:

```python
class Embedding:
    ...

    def decode(self, x: Array):
        return x @ self.embeddings.T
```

In `TransformerLM`, I remove the separate `lm_head` from `__init__` and from the pytree’s `children` tuple. I then use `decode` after the final normalization:

```python
def __call__(self, token_ids: Array, token_positions: Array | None = None) -> Array:
    x = self.token_embeddings(token_ids)
    for block in self.blocks:
        x = block(x, token_positions)
    return self.token_embeddings.decode(self.ln_final(x))
```

This frees up 5.12 million parameters from the separate LM head. One transformer block needs about 3.11 million parameters, so I tried adding one or two blocks while keeping the total parameter count close to my original model. I also tried keeping four blocks and increasing the feed-forward width from 1,344 to 2,048. I used the best training configuration from the initial experiments for all of these architecture comparisons.

<div role="region" aria-label="Architecture comparison results" tabindex="0" style="max-width: 100%; overflow-x: auto;" markdown="1">

| Architecture           | Blocks | Feed-forward width | Parameters | Validation loss                                                                              | Training time (minutes) |
| ---------------------- | ------ | ------------------ | ---------- | -------------------------------------------------------------------------------------------- | ----------------------- |
| Untied, four blocks    | 4      | 1,344              | 22,696,448 | [1.2846](https://forge.coreweave.com/wandb/andamamine-cleanmarl/JAX-LLMs-Blog/runs/qxn2col1) | 38.13                   |
| Tied, extra block      | 5      | 1,344              | 20,690,432 | [1.2456](https://forge.coreweave.com/wandb/andamamine-cleanmarl/JAX-LLMs-Blog/runs/wzo7uhum) | 44.79                   |
| Tied, two extra blocks | 6      | 1,344              | 23,804,416 | [1.2254](https://forge.coreweave.com/wandb/andamamine-cleanmarl/JAX-LLMs-Blog/runs/x262pwfq) | 51.82                   |
| Tied, wider FFN        | 4      | 2,048              | 21,901,824 | [1.2508](https://forge.coreweave.com/wandb/andamamine-cleanmarl/JAX-LLMs-Blog/runs/qa81s2lz) | 41.90                   |

</div>

### Training curves

The training times in the table are taken from the validation curves at step 64,000.

The six-block model gives the lowest loss, 1.2254, with about 4.9% more parameters than the four-block model. The five-block model also improves the loss while using fewer parameters. The figure below shows validation loss for the four-block model at a learning rate of 0.004 and the tied six-block model. Hover to compare values, or use the slider to zoom.

```echarts
{{ site.data.tinyjxllm_p1_validation_loss | jsonify }}
```

## Sampling from the model

I now use the tied six-block model to generate text. Starting from a prompt, I predict the next token, append it to the context, and repeat. The sampling strategy determines how I choose each token from the model's logits.

### Sampling strategies

- Greedy: always choose the token with the highest logit.

```python
def greedy_sampling(logits, **kwargs):
    return jnp.argmax(logits)
```

- Random: sample from the full vocabulary using `softmax(logits / temp)`. Temperature controls how concentrated the probabilities are. Lower values favor the most likely tokens; higher values allow more variety.

```python
def random_sampling(logits, key, temp, **kwargs):
    return jax.random.categorical(key, logits / temp, axis=-1)
```

- Top-k: keep the `k` tokens with the highest logits, then sample among them.

```python
def top_k_sampling(logits, key, top_k, temp, **kwargs):
    k_logits, k_indices = jax.lax.top_k(logits, top_k)
    choice = random_sampling(k_logits, key, temp)
    return k_indices[choice]
```

- Top-p (nucleus): keep the most probable tokens whose cumulative probability reaches `p`, then sample from that set.

```python
def top_p_sampling(logits, key, top_p, temp, **kwargs):
    logits = logits / temp
    sorted_idx = jnp.argsort(-logits)
    sorted_logits = logits[sorted_idx]
    probs = jax.nn.softmax(sorted_logits)
    cum_probs = jnp.cumsum(probs)
    keep = (cum_probs - probs) < top_p
    masked = jnp.where(keep, sorted_logits, -jnp.inf)
    choice = jax.random.categorical(key, masked)
    return sorted_idx[choice]
```

### Prompts and configurations

I use the configurations in the table below. Each configuration generates up to 256 new tokens per prompt.

<div role="region" aria-label="Sampling configurations" tabindex="0" style="max-width: 100%; overflow-x: auto;" markdown="1">

| Strategy | Temperature   | k           | p                | Configurations |
| -------- | ------------- | ----------- | ---------------- | -------------- |
| Greedy   | —             | —           | —                | 1              |
| Random   | 0.7, 0.9, 1.0 | —           | —                | 3 per seed     |
| Top-k    | 0.7, 0.9, 1.0 | 40, 50, 640 | —                | 9 per seed     |
| Top-p    | 0.7, 0.9, 1.0 | —           | 0.9, 0.95, 0.975 | 9 per seed     |

</div>

I use the following prompts:

| Prompt | Text                                                                                            |
| ------ | ----------------------------------------------------------------------------------------------- |
| 1      | Tom and Jane are friends. One day, Jane goes to Tom’s house. Tom has a big pot of soup.         |
| 2      | Alice was so tired when she got back home so she went                                           |
| 3      | Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked |
| 4      | Alice and Jack walked up the street and met a girl in a red dress. The girl said to them,       |
| 5      | Once upon a time, in an ancient house, there lived a girl named Lily.                           |
| 6      | Once upon a time, there was a beautiful cat.                                                    |

### Sampling results

I used an LLM judge to rate all 132 completions: 22 configurations, six shared prompts. Each criterion was scored out of 10:

- **Grammar:** correct sentences and word usage.
- **Creativity:** distinctive ideas, rather than repetition or random incoherence.
- **Consistency:** stable characters and events that agree with the prompt.
- **Plot coherence:** connected events and a supported ending.

The table shows the highest-scoring setting per method. Overall is the equally weighted average of the four criteria.

<div role="region" aria-label="Sampling evaluation results" tabindex="0" style="max-width: 100%; overflow-x: auto;" markdown="1">

| Setting                   | Grammar | Creativity | Consistency | Plot coherence | Overall |
| ------------------------- | ------- | ---------- | ----------- | -------------- | ------- |
| Top-p, T=0.7, p=0.975     | 8.3     | 3.5        | 6.0         | 5.5            | 5.8     |
| Random, T=0.7             | 8.2     | 3.7        | 6.0         | 4.8            | 5.7     |
| Top-k, T=0.7, k=40/50/640 | 7.7     | 2.7        | 5.7         | 4.5            | 5.1     |
| Greedy                    | 8.8     | 2.2        | 4.2         | 3.8            | 4.8     |

</div>


Greedy has the highest grammar score (8.8) but the lowest creativity (2.2) among all 22 settings.

The five highest overall scores are listed in the table below. Four of the five configurations use top-p.

| Configuration | Grammar | Creativity | Consistency | Plot coherence | Overall |
| --- | --- | --- | --- | --- | --- |
| Top-p, T=0.7, p=0.975 | 8.33 | 3.50 | 6.00 | 5.50 | 5.83 |
| Top-p, T=0.9, p=0.9 | 8.33 | 3.67 | 5.67 | 5.33 | 5.75 |
| Random, T=0.7 | 8.17 | 3.67 | 6.00 | 4.83 | 5.67 |
| Top-p, T=0.7, p=0.95 | 8.67 | 2.67 | 6.50 | 4.67 | 5.62 |
| Top-p, T=0.9, p=0.95 | 8.33 | 4.17 | 5.33 | 4.67 | 5.62 |

### Generated examples

Illustrative examples from prompt P4. Generated wording is unchanged; only the terminal `<|endoftext|>` marker is removed.

**Prompt P4:**

> Alice and Jack walked up the street and met a girl in a red dress. The girl said to them,

**Top-p, T=0.7, p=0.975**

> "I am very poor and I need some money to buy some food." Alice and Jack were very kind, so they said, "We will help you. We will all buy food and you will be happy."
> The girl was very happy and thanked Alice and Jack. They all went back to the park and started to eat the food. They were having a great time and the girl was very happy too.
> At the end of the day, the girl thanked Alice and Jack for their help and said, "Thank you for the food and for helping me. You are very kind."
> Alice and Jack smiled and said, "We are happy to help. It made us feel good and strong!"
> The girl gave them a big hug and thanked them again. Alice and Jack waved goodbye, knowing that they had done something kind and compassion.



**Greedy**

> "Hi, I'm Lucy. What's your name?" Alice said, "I'm Alice. I'm Alice." Lucy said, "Do you want to play with me?" Alice said, "Yes, let's play!"
> Alice and Jack played together all day. They ran around the street and laughed. Alice was so happy to have a new friend.
> At the end of the day, Alice said goodbye to Jack and thanked him for playing with her. She said, "Bye, Jack. See you soon!"
> Alice and Jack waved goodbye and Alice went home. She was so happy to have made a new friend.



**Top-k, T=0.9, k=50**

> "I wish your luck was to see my house. Can your house come with you?"
> Alice smiled and said, "Yes, of course! I can help". She showed the little boy how to bring the items to her house.
> The old lady thanked Alice for her help and said, "Thank you for your help! I can't wait to see your house."
> Alice smiled proudly and said, "I'm so happy I'm ready to help".
> The old lady replied, "Well, why don't you come visit me again? You will always be welcome to your big house".
> Alice thanked the old lady and went home with a big smile.

## What is next

Next, I will focus on optimizing the code. We will focus on: 

- How to profile JAX code
- How to optimize the network layers and training code.
- Add a KV cache to speed up text generation.
