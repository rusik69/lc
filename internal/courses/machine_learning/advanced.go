package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          100,
			Title:       "Recurrent Neural Networks & LSTMs",
			Description: "Understand RNNs, LSTM architecture, GRUs, and sequence-to-sequence models for sequential data.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "Recurrent Neural Networks (RNNs)",
					Content: `RNNs process sequential data by maintaining hidden state across time steps.

**Key Concept:**
- Process sequences element by element
- Maintain hidden state (memory)
- Share parameters across time steps
- Can handle variable-length sequences

**Architecture:**
- Input: xₜ (current input)
- Hidden State: hₜ (current state)
- Output: yₜ (current output)
- Recurrence: hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)

**Forward Pass - Detailed:**

**Unrolled Network:**
RNN can be "unrolled" through time:
- t=0: h₀ = tanh(W·x₀ + U·h₋₁ + b), y₀ = V·h₀
- t=1: h₁ = tanh(W·x₁ + U·h₀ + b), y₁ = V·h₁
- t=2: h₂ = tanh(W·x₂ + U·h₁ + b), y₂ = V·h₂
- ...

**Key Insight:**
- Same weights W, U, V used at each time step
- Hidden state hₜ carries information from previous steps
- Can theoretically remember information from beginning

**Backpropagation Through Time (BPTT):**

**The Challenge:**
Need to backpropagate gradients through time steps.

**Process:**
1. Forward pass: Compute hₜ and yₜ for all t
2. Compute loss: L = Σ Lₜ(yₜ, ŷₜ)
3. Backward pass: Compute gradients w.r.t. all time steps
4. Update: Average gradients across time steps

**Gradient Flow:**
∂L/∂U = Σₜ (∂L/∂hₜ) × (∂hₜ/∂U)
- Gradient depends on all future time steps
- Must unroll network backward through time

**Vanishing Gradient Problem - Mathematical Derivation:**

**The Problem:**
When backpropagating through time, gradients can vanish exponentially.

**Why It Happens:**
Consider gradient w.r.t. hidden state at time t-k:
∂L/∂hₜ₋ₖ = (∂L/∂hₜ) × (∂hₜ/∂hₜ₋₁) × ... × (∂hₜ₋ₖ₊₁/∂hₜ₋ₖ)

Each term: ∂hₜ/∂hₜ₋₁ = Uᵀ × diag(tanh'(zₜ))
- If |tanh'(z)| < 1 (which it is for most values)
- After k steps: gradient ≈ (small_value)ᵏ → 0

**Result:**
- Early time steps receive very small gradients
- Can't learn long-term dependencies
- Network "forgets" information from far past

**Exploding Gradient:**
- If weights U are large, gradients can explode
- Solution: Gradient clipping

**Applications:**
- Natural Language Processing
- Time Series Prediction
- Speech Recognition
- Machine Translation

**Limitations:**
- Vanishing gradient problem (can't learn long dependencies)
- Difficulty learning long-term dependencies
- Sequential processing (slow, can't parallelize)
- Limited memory (hidden state has fixed size)`,
					CodeExamples: `import torch
import torch.nn as nn

# Simple RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use last output
        return out`,
				},
				{
					Title: "LSTM Architecture",
					Content: `LSTM (Long Short-Term Memory) solves vanishing gradient problem with gated architecture.

**LSTM Components:**
- **Forget Gate**: Decides what to forget from cell state
- **Input Gate**: Decides what new information to store
- **Cell State**: Long-term memory (main information highway)
- **Output Gate**: Decides what parts of cell state to output

**Gate-by-Gate Explanation:**

**1. Forget Gate:**
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
- Output: [0, 1] for each element in cell state
- 0 = "completely forget", 1 = "completely keep"
- **Purpose**: Remove irrelevant information from cell state

**2. Input Gate:**
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)  # How much to update
C̃ₜ = tanh(WC·[hₜ₋₁, xₜ] + bC)  # New candidate values
- **Purpose**: Decide what new information to add to cell state
- iₜ: How much of candidate to add
- C̃ₜ: New values to potentially add

**3. Cell State Update:**
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
- **Key**: Additive update (not multiplicative like RNN!)
- fₜ * Cₜ₋₁: Keep relevant parts of old state
- iₜ * C̃ₜ: Add new information
- **Why this works**: Gradient flows through addition (no vanishing!)

**4. Output Gate:**
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
hₜ = oₜ * tanh(Cₜ)
- **Purpose**: Control what parts of cell state become hidden state
- oₜ: Filter for cell state
- hₜ: Output to next time step and for prediction

**How LSTM Solves Vanishing Gradient:**

**Key Insight:**
Cell state update: Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ

**Gradient Flow:**
∂Cₜ/∂Cₜ₋₁ = fₜ (plus terms from gates)
- Gradient can flow through cell state with minimal attenuation
- Forget gate can be ≈ 1, allowing gradient to pass through
- Additive update preserves gradient magnitude

**Cell State Flow:**
- Cell state Cₜ is the "information highway"
- Information can flow unchanged if forget gate ≈ 1
- Allows learning long-term dependencies

**Forget Gate Importance:**
- If fₜ ≈ 0: Forget everything (reset)
- If fₜ ≈ 1: Keep everything (remember)
- Learns when to remember vs forget

**Advantages:**
- Handles long-term dependencies (cell state preserves information)
- Prevents vanishing gradients (additive updates)
- Better than vanilla RNN (gates control information flow)
- Can learn when to remember/forget (adaptive memory)

**GRU (Gated Recurrent Unit) - Simplified LSTM:**

**Differences from LSTM:**
- Combines forget and input gates into update gate
- No separate cell state (hidden state serves both purposes)
- Fewer parameters (faster training)
- Often performs similarly to LSTM

**GRU Gates:**
- Update gate: zₜ = σ(Wz·[hₜ₋₁, xₜ])
- Reset gate: rₜ = σ(Wr·[hₜ₋₁, xₜ])
- Candidate: h̃ₜ = tanh(W·[rₜ * hₜ₋₁, xₜ])
- Hidden: hₜ = (1-zₜ) * hₜ₋₁ + zₜ * h̃ₜ`,
					CodeExamples: `import torch
import torch.nn as nn

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          101,
			Title:       "Transformers Architecture",
			Description: "Deep dive into attention mechanism, self-attention, multi-head attention, positional encoding, and transformer architecture.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Attention Mechanism",
					Content: `Attention allows models to focus on relevant parts of input when making predictions.

**Key Idea:**
- Not all input tokens are equally important
- Learn which tokens to attend to
- Weighted combination of values based on relevance

**Attention Formula - Step-by-Step:**

**Scaled Dot-Product Attention:**
Attention(Q, K, V) = softmax(QKᵀ/√dₖ) · V

**Step 1: Compute Attention Scores**
Scores = QKᵀ
- Dot product between query and key
- Measures similarity/relevance
- Higher score = more relevant

**Step 2: Scale Scores**
Scores_scaled = Scores / √dₖ
- Prevents dot products from becoming too large
- Large values → softmax saturates → small gradients
- √dₖ: Normalization factor

**Step 3: Apply Softmax**
Attention_weights = softmax(Scores_scaled)
- Converts scores to probabilities
- Sum to 1 (probability distribution)
- Higher scores → higher weights

**Step 4: Weighted Sum**
Output = Attention_weights · V
- Weighted combination of values
- Positions with high attention weights contribute more

**Detailed Example:**

**Input**: "The cat sat on the mat"
**Query**: "What did the cat do?" (position 3: "sat")

**Step-by-Step:**
1. Compute QKᵀ for "sat" with all positions:
   - "The": score = 0.1
   - "cat": score = 0.8 (high - subject)
   - "sat": score = 1.0 (self-attention)
   - "on": score = 0.6 (preposition)
   - "the": score = 0.2
   - "mat": score = 0.7 (object)

2. Scale: Divide by √dₖ (e.g., √64 = 8)

3. Softmax: [0.05, 0.25, 0.35, 0.15, 0.06, 0.14]
   - "sat" gets highest weight (0.35)
   - "cat" gets high weight (0.25)

4. Weighted sum: 0.35×V("sat") + 0.25×V("cat") + ...

**Where:**
- **Q (Query)**: What we're looking for (e.g., "What is the action?")
- **K (Key)**: What each position offers (e.g., "I am a noun", "I am a verb")
- **V (Value)**: Actual content at each position (embedding vectors)
- **dₖ**: Dimension of keys (scaling factor, typically 64)

**Why Attention?**
- Captures long-range dependencies (can attend to any position)
- Parallelizable (unlike RNNs, all positions processed simultaneously)
- Interpretable (attention weights show what model focuses on)
- Flexible (can attend to any position, not just previous ones)`,
					CodeExamples: `import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(query, key, value, mask=None):
    """Scaled Dot-Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights`,
				},
				{
					Title: "Self-Attention",
					Content: `Self-attention computes attention within the same sequence, allowing each position to attend to all positions.

**Self-Attention Process:**
1. Create Q, K, V from same input (via linear projections)
2. Compute attention scores between all positions
3. Weighted sum of values based on attention

**Benefits:**
- Captures relationships between all positions
- Parallel computation
- No sequential dependency

**Multi-Head Attention:**
- Multiple attention heads in parallel
- Each head learns different relationships
- Concatenate and project outputs
- Captures diverse patterns`,
					CodeExamples: `import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        return self.W_o(context)`,
				},
				{
					Title: "Positional Encoding",
					Content: `Transformers need positional information since they process all tokens in parallel (unlike RNNs).

**Why Positional Encoding?**
- RNNs: Process sequentially (position is implicit)
- Transformers: Process in parallel (no inherent position)
- Need to inject positional information

**Sinusoidal Positional Encoding - Derivation:**

**Formula:**
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos: Position in sequence (0, 1, 2, ...)
- i: Dimension index (0, 1, 2, ..., d_model/2)
- d_model: Embedding dimension

**Properties:**
- **Unique**: Each position has unique encoding
- **Relative**: Can compute relative positions via addition
- **Extrapolation**: Can handle sequences longer than training
- **Deterministic**: Not learned, fixed function

**Intuition:**
- Different frequencies (2i/d_model) capture different scales
- Low frequencies: Long-range patterns
- High frequencies: Short-range patterns
- Sin/cos: Allows model to learn relative positions

**Example:**
For pos=0, d_model=512:
- PE(0, 0) = sin(0 / 10000^0) = sin(0) = 0
- PE(0, 1) = cos(0 / 10000^0) = cos(0) = 1
- PE(0, 2) = sin(0 / 10000^(2/512)) = sin(0) = 0
- ...

**Relative Position Property:**
PE(pos+k) can be expressed as linear function of PE(pos)
- Allows model to learn "k positions apart"
- Important for understanding relationships

**Alternative: Learned Positional Embeddings**
- Trainable parameters: E[pos] (learned embedding for position pos)
- Often works better in practice
- But can't extrapolate beyond training length
- Used in many modern models (BERT, GPT)`,
					CodeExamples: `import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]`,
				},
				{
					Title: "Transformer Architecture",
					Content: `Transformer architecture combines attention with feed-forward networks and residual connections.

**Transformer Block - Detailed Flow:**

**Input**: X (sequence of embeddings)

**1. Multi-Head Self-Attention:**
- Compute Q, K, V from X
- Apply attention: Attention(Q, K, V)
- Multiple heads in parallel
- Output: X_attn

**2. Add & Norm (First):**
- Residual: X_attn + X
- Layer Norm: LayerNorm(X_attn + X)
- Output: X_norm1

**3. Feed-Forward Network:**
- Two linear layers with ReLU: FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
- Expands then contracts dimension
- Output: X_ffn

**4. Add & Norm (Second):**
- Residual: X_ffn + X_norm1
- Layer Norm: LayerNorm(X_ffn + X_norm1)
- Output: Final output

**Layer Normalization:**

**Purpose:**
- Normalize activations within each sample
- Stabilizes training
- Allows higher learning rates

**Formula:**
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
Where:
- μ: Mean of x
- σ²: Variance of x
- γ, β: Learnable parameters
- ε: Small constant (numerical stability)

**Why Layer Norm?**
- Normalizes across features (not batch)
- Works well with variable-length sequences
- Better than batch norm for sequences

**Residual Connections:**

**Purpose:**
- Helps gradient flow (solves vanishing gradient)
- Enables deeper networks
- Allows identity mapping (if needed)

**Formula:**
Output = LayerNorm(Sublayer(x) + x)

**Why They Work:**
- Gradient can flow directly through addition
- Network can learn residual (difference) rather than full mapping
- Easier optimization

**Feed-Forward Network:**

**Architecture:**
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

**Typical Dimensions:**
- Input: d_model (e.g., 512)
- Hidden: d_ff (e.g., 2048) - 4× expansion
- Output: d_model

**Purpose:**
- Adds non-linearity
- Processes information independently per position
- "Thinking" step after attention

**Full Transformer:**
- **Encoder**: Processes input sequence (bidirectional attention)
- **Decoder**: Generates output sequence (masked self-attention + encoder-decoder attention)
- **Stacked Layers**: Multiple transformer blocks (typically 6-12 layers)

**Key Components:**
- **Layer Normalization**: Stabilizes training (normalizes activations)
- **Residual Connections**: Helps with gradient flow (enables deep networks)
- **Feed-Forward**: Two linear layers with ReLU (adds non-linearity)
- **Masking**: Prevents attention to future tokens (decoder, for autoregressive generation)`,
					CodeExamples: `import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          102,
			Title:       "Large Language Models (LLMs) Fundamentals",
			Description: "Understand what LLMs are, pre-training vs fine-tuning, tokenization, context windows, and major model families.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "What are Large Language Models?",
					Content: `Large Language Models (LLMs) are transformer-based models trained on massive text corpora to understand and generate human-like text.

**Key Characteristics:**
- **Scale**: Billions of parameters (GPT-3: 175B, GPT-4: ~1T+)
- **Pre-training**: Trained on vast internet text
- **Emergent Abilities**: Capabilities emerge at scale
- **Few-Shot Learning**: Can perform tasks with few examples

**How LLMs Work:**
1. **Pre-training**: Learn language patterns from text
2. **Fine-tuning**: Adapt to specific tasks
3. **Inference**: Generate text autoregressively

**Emergent Abilities:**
- **In-Context Learning**: Learn from examples in prompt
- **Chain-of-Thought**: Step-by-step reasoning
- **Code Generation**: Write and debug code
- **Multilingual**: Handle multiple languages

**Model Families:**
- **GPT (Generative Pre-trained Transformer)**: OpenAI's autoregressive models
- **BERT (Bidirectional Encoder Representations)**: Google's encoder model
- **T5 (Text-to-Text Transfer Transformer)**: Google's encoder-decoder
- **LLaMA**: Meta's open-source models
- **Claude**: Anthropic's models`,
					CodeExamples: `# Using Hugging Face Transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
text = "The future of AI is"
inputs = tokenizer.encode(text, return_tensors='pt')
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)`,
				},
				{
					Title: "Pre-training vs Fine-tuning",
					Content: `Understanding the difference between pre-training and fine-tuning is crucial for working with LLMs.

**Pre-training - Detailed Pipeline:**

**Objective**: Learn general language patterns from massive text data

**Data Pipeline:**
1. **Data Collection**: Web crawl, books, articles (terabytes of text)
2. **Cleaning**: Remove duplicates, filter low-quality content
3. **Tokenization**: Convert text to tokens (subwords/words)
4. **Shuffling**: Randomize order for better learning
5. **Batching**: Create batches for training

**Training Objective:**
- **Autoregressive (GPT-style)**: Predict next token given previous tokens
  - Loss: L = -Σ log P(xₜ | x₁, ..., xₜ₋₁)
  - Learns to generate text
- **Masked Language Modeling (BERT-style)**: Predict masked tokens
  - Loss: L = -Σ log P(x_masked | context)
  - Learns bidirectional understanding

**Distributed Training:**

**Why Distributed?**
- Models too large for single GPU (175B+ parameters)
- Need parallelization across many GPUs

**Data Parallelism:**
- Split batch across GPUs
- Each GPU computes gradients
- Average gradients across GPUs
- All GPUs update same model

**Model Parallelism:**
- Split model across GPUs
- Each GPU holds part of model
- Forward/backward pass across GPUs
- For very large models

**Pipeline Parallelism:**
- Split model into stages
- Process different batches in parallel
- Like assembly line

**Mixed Precision Training:**

**Purpose:**
- Reduce memory usage
- Speed up training

**How It Works:**
- **FP32**: Full precision (32-bit floats) - for gradients
- **FP16/BF16**: Half precision (16-bit) - for forward pass
- **Automatic**: Framework converts between precisions

**Benefits:**
- 2× memory reduction
- 1.5-2× speedup
- Minimal accuracy loss

**Challenges:**
- Gradient underflow (too small)
- Solution: Gradient scaling (scale up, then scale down)

**Fine-tuning - Detailed:**

**Objective**: Adapt pre-trained model to specific task/domain

**Data Requirements:**
- Smaller dataset (thousands to millions of examples)
- Task-specific labels
- Can be domain-specific (medical, legal, etc.)

**Fine-tuning Approaches:**

**1. Full Fine-tuning:**
- Update all model parameters
- **Pros**: Best performance, full adaptation
- **Cons**: Expensive, risk of catastrophic forgetting
- **Use**: When you have sufficient compute and data

**2. Parameter-Efficient Fine-tuning (PEFT):**

**LoRA (Low-Rank Adaptation):**
- Add low-rank matrices to attention layers
- Freeze original weights, train only new matrices
- **Pros**: Much fewer parameters, faster training
- **Cons**: Slightly lower performance
- **Use**: When compute/memory limited

**Adapters:**
- Add small adapter layers between transformer blocks
- Train only adapters
- **Pros**: Modular, can stack adapters
- **Cons**: Adds inference latency

**3. Prompt Tuning:**
- Learn soft prompts (trainable embeddings)
- Prepend to input, freeze model
- **Pros**: Very efficient, no model changes
- **Cons**: Limited expressiveness

**4. Instruction Tuning:**
- Fine-tune on instruction-following format
- Format: "Instruction: ... Input: ... Output: ..."
- **Purpose**: Make model follow instructions better
- **Result**: Better zero-shot performance

**Fine-tuning Process:**
1. Load pre-trained weights
2. (Optional) Add task-specific head
3. Train on task-specific data
4. Use lower learning rate (1e-5 to 1e-3)
5. Monitor validation performance

**Catastrophic Forgetting:**
- Model forgets pre-training knowledge
- **Prevention**: Lower learning rate, gradual unfreezing, regularization`,
					CodeExamples: `# Full Fine-tuning Example
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

dataset = Dataset.from_dict({'text': ['Your training texts here...']})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()`,
				},
				{
					Title: "Tokenization",
					Content: `Tokenization converts text into tokens (subword units) that models can process.

**Tokenization Methods:**

**1. Word-Level:**
- Split on whitespace
- Problem: Large vocabulary, OOV (out-of-vocabulary) words

**2. Character-Level:**
- Each character is a token
- Problem: Very long sequences, loses word meaning

**3. Subword Tokenization (Most Common):**
- **BPE (Byte Pair Encoding)**: GPT models
- **WordPiece**: BERT models
- **SentencePiece**: T5, LLaMA models

**BPE Algorithm:**
1. Start with character vocabulary
2. Find most frequent pair of tokens
3. Merge into new token
4. Repeat until desired vocabulary size

**Why Subword?**
- Handles unknown words (decompose into subwords)
- Smaller vocabulary than word-level
- Better than character-level (preserves some meaning)
- Language-agnostic`,
					CodeExamples: `from transformers import GPT2Tokenizer, BertTokenizer

# GPT-2 uses BPE
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Hello, how are you?"
tokens = gpt_tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# Output: ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']

token_ids = gpt_tokenizer.encode(text)
print(f"Token IDs: {token_ids}")

# BERT uses WordPiece
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokens = bert_tokenizer.tokenize(text)
print(f"BERT Tokens: {bert_tokens}")
# Output: ['hello', ',', 'how', 'are', 'you', '?']`,
				},
				{
					Title: "Context Windows and Model Limits",
					Content: `Context window is the maximum number of tokens a model can process in one go.

**Context Window Sizes:**
- **GPT-3**: 2,048 tokens (~1,500 words)
- **GPT-4**: 8,192 tokens (standard), 32,768 (extended)
- **Claude**: Up to 200,000 tokens
- **LLaMA 2**: 4,096 tokens
- **Recent Models**: 128K+ tokens

**Why Context Matters:**
- **Long Documents**: Need large context for full document understanding
- **Conversations**: Maintain conversation history
- **Code**: Large codebases require long context
- **RAG**: Include retrieved documents in context

**Context Window Limitations:**
- **Computational Cost**: O(n²) attention complexity
- **Memory**: Quadratic memory with sequence length
- **Solutions**: Sparse attention, sliding windows, retrieval

**Managing Context:**
- **Truncation**: Cut off excess tokens
- **Summarization**: Summarize old context
- **Retrieval**: Retrieve relevant parts (RAG)
- **Chunking**: Process in chunks`,
					CodeExamples: `# Managing context window
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Long text
long_text = "Your very long text here..." * 1000

# Tokenize
tokens = tokenizer.encode(long_text)
print(f"Total tokens: {len(tokens)}")
print(f"Model max length: {model.config.max_position_embeddings}")

# Truncate if too long
max_length = 1024
if len(tokens) > max_length:
    tokens = tokens[:max_length]
    print(f"Truncated to {len(tokens)} tokens")

# Or use tokenizer's truncation
encoded = tokenizer(long_text, max_length=max_length, truncation=True, return_tensors='pt')`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          103,
			Title:       "LLM Training & Optimization",
			Description: "Deep dive into pre-training process, data preparation, training infrastructure, distributed training, model parallelism, and optimization techniques.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "Pre-training Process",
					Content: `Pre-training is the foundation of LLM capabilities, requiring massive compute and careful engineering.

**Pre-training Pipeline:**

**1. Data Collection:**
- Web scraping (Common Crawl)
- Books, articles, code repositories
- Filtering and deduplication
- Quality filtering (removing low-quality text)

**2. Data Preprocessing:**
- Cleaning (remove HTML, normalize text)
- Deduplication (exact and fuzzy)
- Language detection
- Toxicity filtering
- Quality scoring

**3. Tokenization:**
- Train tokenizer on corpus
- Create vocabulary
- Encode entire dataset

**4. Training:**
- **Objective**: Next token prediction (autoregressive)
- **Loss**: Cross-entropy loss
- **Optimizer**: AdamW with learning rate scheduling
- **Batch Size**: Very large (millions of tokens per batch)
- **Gradient Accumulation**: Simulate larger batches

**5. Evaluation:**
- Perplexity on held-out data
- Downstream task evaluation
- Safety evaluations

**Key Challenges:**
- **Scale**: Requires thousands of GPUs
- **Stability**: Training instability at scale
- **Data Quality**: Garbage in, garbage out
- **Cost**: Millions of dollars in compute`,
					CodeExamples: `# Conceptual pre-training setup
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

# Model configuration
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
)

model = GPT2LMHeadModel(config)

# Training loop (simplified)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
criterion = nn.CrossEntropyLoss()

# Simulated training step
def train_step(batch):
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs.logits.view(-1, config.vocab_size), 
                     targets.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()`,
				},
				{
					Title: "Distributed Training",
					Content: `Training LLMs requires distributing computation across many GPUs/machines.

**Data Parallelism:**
- Split batch across GPUs
- Each GPU has full model copy
- Synchronize gradients
- **Limitation**: Model must fit on single GPU

**Model Parallelism:**
- Split model across GPUs
- Each GPU holds part of model
- **Pipeline Parallelism**: Layers on different GPUs
- **Tensor Parallelism**: Split tensors across GPUs

**Hybrid Approaches:**
- **3D Parallelism**: Data + Pipeline + Tensor
- **ZeRO**: Zero Redundancy Optimizer (Microsoft)
- **FSDP**: Fully Sharded Data Parallel (PyTorch)

**Communication:**
- **All-Reduce**: Synchronize gradients (data parallel)
- **Point-to-Point**: Between pipeline stages
- **Ring All-Reduce**: Efficient gradient sync`,
					CodeExamples: `# Data Parallelism with PyTorch
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

model = GPT2LMHeadModel(config)
if torch.cuda.device_count() > 1:
    model = DataParallel(model)

# Model Parallelism (conceptual)
class ModelParallelGPT2(nn.Module):
    def __init__(self, config, device_ids):
        super().__init__()
        self.device_ids = device_ids
        # Split layers across devices
        self.layers_part1 = nn.ModuleList([...]).to(device_ids[0])
        self.layers_part2 = nn.ModuleList([...]).to(device_ids[1])
    
    def forward(self, x):
        x = self.layers_part1(x)
        x = x.to(self.device_ids[1])
        x = self.layers_part2(x)
        return x`,
				},
				{
					Title: "Optimization Techniques",
					Content: `Training large models requires sophisticated optimization techniques.

**Mixed Precision Training:**
- Use FP16/BF16 for forward/backward
- FP32 for master weights
- 2x speedup, 2x memory savings
- Gradient scaling to prevent underflow

**Gradient Checkpointing:**
- Trade compute for memory
- Recompute activations during backward
- Saves memory at cost of ~33% slower

**Learning Rate Scheduling:**
- **Warmup**: Gradually increase LR
- **Cosine Decay**: Smooth decrease
- **Linear Decay**: Simple decrease

**Optimizers:**
- **AdamW**: Weight decay separate from gradient
- **Lion**: Memory-efficient alternative
- **8-bit Adam**: Quantized optimizer states`,
					CodeExamples: `from torch.cuda.amp import autocast, GradScaler

# Mixed Precision Training
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch['input_ids'])
        loss = criterion(outputs.logits, batch['labels'])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          104,
			Title:       "Fine-tuning LLMs",
			Description: "Learn fine-tuning strategies, LoRA, QLoRA, parameter-efficient fine-tuning, and fine-tuning datasets.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Fine-tuning Strategies",
					Content: `Fine-tuning adapts pre-trained LLMs to specific tasks or domains.

**Full Fine-tuning:**
- Update all model parameters
- Best performance but expensive
- Risk of catastrophic forgetting

**Parameter-Efficient Fine-Tuning (PEFT):**
- Update only subset of parameters
- Much cheaper and faster
- Often matches full fine-tuning performance

**Instruction Tuning:**
- Fine-tune on instruction-following format
- Improves zero-shot and few-shot performance
- Format: Instruction + Input + Output

**Domain Adaptation:**
- Fine-tune on domain-specific data
- Medical, legal, code, etc.
- Improves performance in target domain`,
					CodeExamples: `# Full Fine-tuning
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./fine-tuned-model',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()`,
				},
				{
					Title: "LoRA (Low-Rank Adaptation)",
					Content: `LoRA is a parameter-efficient fine-tuning method that adds trainable low-rank matrices.

**LoRA Concept:**
- Freeze original weights
- Add low-rank decomposition: ΔW = BA
- Train only A and B matrices
- Much fewer parameters (0.1-1% of original)

**Why LoRA Works:**
- Low-rank assumption: weight updates are low-rank
- Reduces memory and compute
- Can combine multiple LoRA adapters

**LoRA Configuration:**
- **rank (r)**: Dimension of low-rank matrices (typically 4-64)
- **alpha**: Scaling factor
- **target_modules**: Which layers to apply LoRA`,
					CodeExamples: `from peft import LoraConfig, get_peft_model

# LoRA Configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.1,
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4M || all params: 7B || trainable%: 0.06

# Train as normal
trainer = Trainer(model=model, ...)
trainer.train()`,
				},
				{
					Title: "QLoRA",
					Content: `QLoRA combines quantization with LoRA for even more efficient fine-tuning.

**QLoRA Components:**
1. **4-bit Quantization**: Reduce model to 4-bit (NF4)
2. **LoRA**: Parameter-efficient fine-tuning
3. **Paged Optimizers**: Handle memory spikes

**Benefits:**
- Fine-tune 65B model on single 48GB GPU
- ~4x memory reduction vs full fine-tuning
- Performance close to full fine-tuning

**Quantization:**
- **NF4 (NormalFloat4)**: 4-bit quantization
- **Double Quantization**: Quantize quantization constants
- **Paged Optimizers**: Use CPU offloading`,
					CodeExamples: `from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load quantized model
model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(...)
model = get_peft_model(model, lora_config)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          105,
			Title:       "Prompt Engineering",
			Description: "Master prompt design principles, few-shot learning, chain-of-thought, prompt templates, and best practices.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Prompt Design Principles",
					Content: `Effective prompts are crucial for getting good results from LLMs.

**Key Principles:**
1. **Be Specific**: Clear, unambiguous instructions
2. **Provide Context**: Include relevant background
3. **Show Examples**: Demonstrate desired format
4. **Specify Format**: Request specific output format
5. **Iterate**: Refine prompts based on outputs

**Prompt Structure:**
- **Role**: Define model's role/persona
- **Context**: Background information
- **Task**: What to do
- **Examples**: Few-shot examples
- **Format**: Output format specification`,
					CodeExamples: `# Good Prompt
prompt = """
You are an expert Python programmer. 
Write a function that calculates the factorial of a number.

Requirements:
- Use recursion
- Handle edge cases (0, negative numbers)
- Include docstring

Example:
Input: 5
Output: 120

Write the function:
"""

# Bad Prompt
bad_prompt = "factorial function"  # Too vague`,
				},
				{
					Title: "Few-Shot Learning",
					Content: `Few-shot learning provides examples in the prompt to guide model behavior.

**Zero-Shot:**
- No examples, just instruction
- Relies on pre-training knowledge

**One-Shot:**
- Single example
- Shows format/pattern

**Few-Shot:**
- Multiple examples (typically 2-5)
- Demonstrates task pattern
- More examples often help, but diminishing returns

**In-Context Learning:**
- Model learns from examples in prompt
- No weight updates
- Emergent ability of large models`,
					CodeExamples: `# Few-Shot Example
prompt = """
Classify the sentiment of these movie reviews:

Review: "This movie was amazing! Best film I've seen."
Sentiment: Positive

Review: "Terrible acting and boring plot."
Sentiment: Negative

Review: "It was okay, nothing special."
Sentiment: Neutral

Review: "Absolutely loved it! Highly recommend."
Sentiment: Positive
"""`,
				},
				{
					Title: "Chain-of-Thought",
					Content: `Chain-of-Thought (CoT) prompts model to show reasoning steps.

**Why CoT Works:**
- Breaks complex problems into steps
- Model reasons through problem
- Often improves accuracy on reasoning tasks

**CoT Prompting:**
- Ask model to "think step by step"
- Show example with reasoning
- Model generates reasoning then answer

**Zero-Shot CoT:**
- Just add "Let's think step by step"
- No examples needed
- Works surprisingly well`,
					CodeExamples: `# Chain-of-Thought Prompt
prompt = """
Solve this math problem step by step:

Problem: A store has 15 apples. They sell 6 in the morning and 4 in the afternoon. How many are left?

Let's think step by step:
1. Start with 15 apples
2. Sell 6 in morning: 15 - 6 = 9 apples left
3. Sell 4 in afternoon: 9 - 4 = 5 apples left
Answer: 5 apples

Now solve this:
Problem: Sarah has 20 books. She gives away 8 and buys 5 more. How many does she have?

Let's think step by step:
"""`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          106,
			Title:       "Retrieval-Augmented Generation (RAG)",
			Description: "Learn RAG architecture, vector databases, embeddings, document chunking, retrieval strategies, and RAG implementation.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "RAG Architecture",
					Content: `RAG combines retrieval with generation to ground LLM responses in external knowledge.

**RAG Pipeline:**
1. **Query**: User question
2. **Retrieval**: Find relevant documents/chunks
3. **Augmentation**: Add retrieved context to prompt
4. **Generation**: LLM generates answer using context

**Why RAG?**
- **Knowledge Cutoff**: LLMs have training cutoff date
- **Domain Knowledge**: Access to private/domain docs
- **Factual Accuracy**: Ground answers in sources
- **Reduced Hallucination**: Context reduces made-up facts

**Components:**
- **Vector Database**: Store document embeddings
- **Embedding Model**: Convert text to vectors
- **Retrieval**: Find similar chunks
- **LLM**: Generate answer`,
					CodeExamples: `from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Initialize components
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
llm = OpenAI()

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Query
answer = qa_chain.run("What is machine learning?")`,
				},
				{
					Title: "Vector Databases and Embeddings",
					Content: `Vector databases store and search high-dimensional embeddings efficiently.

**Embeddings:**
- Dense vector representations of text
- Similar texts have similar vectors
- Capture semantic meaning

**Vector Databases:**
- **Pinecone**: Managed vector DB
- **Weaviate**: Open-source vector DB
- **Chroma**: Lightweight, embedded
- **FAISS**: Facebook's similarity search
- **Qdrant**: Fast vector search

**Similarity Search:**
- **Cosine Similarity**: Most common
- **Euclidean Distance**: Alternative
- **Dot Product**: Fast approximation`,
					CodeExamples: `import numpy as np
from sentence_transformers import SentenceTransformer

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["Machine learning is AI", "Deep learning uses neural networks"]
embeddings = model.encode(texts)

# Similarity search
query = "What is AI?"
query_embedding = model.encode([query])
similarities = np.dot(embeddings, query_embedding.T)
most_similar_idx = np.argmax(similarities)`,
				},
				{
					Title: "Document Chunking Strategies",
					Content: `Chunking splits documents into smaller pieces for retrieval.

**Chunking Methods:**
- **Fixed Size**: Simple, but may split sentences
- **Sentence-Based**: Split on sentences
- **Semantic Chunking**: Group by meaning
- **Recursive**: Try different sizes, pick best

**Chunk Size:**
- Too small: Lose context
- Too large: Retrieve irrelevant info
- Typical: 200-500 tokens

**Overlap:**
- Overlap chunks to preserve context
- Typical: 50-100 tokens`,
					CodeExamples: `from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)

chunks = text_splitter.split_text(long_document)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          107,
			Title:       "LLM Deployment & Production",
			Description: "Learn model serving, quantization, pruning, inference optimization, API design, and monitoring LLM applications.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Model Serving",
					Content: `Serving LLMs in production requires efficient inference infrastructure.

**Serving Options:**
- **vLLM**: Fast inference with PagedAttention
- **Text Generation Inference (TGI)**: Hugging Face's server
- **TensorRT-LLM**: NVIDIA's optimized server
- **Triton Inference Server**: Flexible serving

**Key Features:**
- **Batching**: Process multiple requests together
- **Continuous Batching**: Add requests to batch dynamically
- **KV Cache**: Cache attention keys/values
- **Quantization**: Reduce model size`,
					CodeExamples: `# vLLM serving
from vllm import LLM, SamplingParams

llm = LLM(model="gpt-2")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)`,
				},
				{
					Title: "Quantization",
					Content: `Quantization reduces model precision to save memory and speed up inference.

**Quantization Types:**
- **INT8**: 8-bit integers (4x smaller)
- **INT4**: 4-bit integers (8x smaller)
- **FP16/BF16**: Half precision (2x smaller)

**Quantization Methods:**
- **Post-Training Quantization**: Quantize after training
- **Quantization-Aware Training**: Train with quantization

**Tools:**
- **GPTQ**: Post-training quantization
- **AWQ**: Activation-aware quantization
- **BitsAndBytes**: 8-bit and 4-bit quantization`,
					CodeExamples: `from transformers import BitsAndBytesConfig

# 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config=quantization_config
)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          108,
			Title:       "Advanced LLM Topics",
			Description: "Explore multi-modal models, vision-language models, RLHF, model alignment, safety, and bias.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Multi-Modal Models",
					Content: `Multi-modal models process multiple input types (text, images, audio).

**Vision-Language Models:**
- **CLIP**: Contrastive learning for image-text
- **GPT-4V**: Vision-capable GPT-4
- **LLaVA**: Large Language and Vision Assistant
- **Flamingo**: Few-shot vision-language learning

**Applications:**
- Image captioning
- Visual question answering
- Image generation from text
- Document understanding`,
					CodeExamples: `from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process image and text
image = Image.open("image.jpg")
text = ["a photo of a cat", "a photo of a dog"]
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)`,
				},
				{
					Title: "Reinforcement Learning from Human Feedback (RLHF)",
					Content: `RLHF aligns models with human preferences using reinforcement learning.

**RLHF Pipeline:**
1. **Supervised Fine-Tuning**: Fine-tune on human demonstrations
2. **Reward Modeling**: Train reward model on human preferences
3. **RL Fine-Tuning**: Optimize policy using reward model (PPO)

**Why RLHF?**
- Aligns model with human values
- Reduces harmful outputs
- Improves helpfulness
- Used in ChatGPT, Claude`,
					CodeExamples: `# Conceptual RLHF
# Step 1: Collect human preferences
preferences = [
    (response_a, response_b, preferred=0),  # Human prefers A
    ...
]

# Step 2: Train reward model
reward_model = train_reward_model(preferences)

# Step 3: RL fine-tuning with PPO
policy = fine_tune_with_ppo(model, reward_model)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          109,
			Title:       "ML Engineering & MLOps",
			Description: "Learn model versioning, experiment tracking, model deployment, monitoring, CI/CD for ML, and model governance.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Experiment Tracking",
					Content: `Track ML experiments to compare models and reproduce results.

**Tools:**
- **MLflow**: Open-source platform
- **Weights & Biases**: Popular commercial tool
- **TensorBoard**: TensorFlow's visualization
- **Neptune**: Experiment management

**What to Track:**
- Hyperparameters
- Metrics (loss, accuracy, etc.)
- Code version
- Data version
- Model artifacts`,
					CodeExamples: `import mlflow

mlflow.set_experiment("llm-fine-tuning")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("batch_size", 16)
    mlflow.log_metric("loss", 0.5)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.pytorch.log_model(model, "model")`,
				},
				{
					Title: "Model Deployment",
					Content: `Deploying models to production requires careful planning.

**Deployment Patterns:**
- **Batch Inference**: Process in batches
- **Real-time API**: On-demand predictions
- **Edge Deployment**: On-device inference

**Considerations:**
- **Latency**: Response time requirements
- **Throughput**: Requests per second
- **Cost**: Infrastructure costs
- **Monitoring**: Track performance`,
					CodeExamples: `# FastAPI deployment
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
classifier = pipeline("text-classification", model="model-name")

@app.post("/predict")
def predict(text: str):
    return classifier(text)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          110,
			Title:       "Time Series Analysis",
			Description: "Learn time series fundamentals, ARIMA models, forecasting, and deep learning approaches for sequential data.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Time Series Fundamentals",
					Content: `Time series data consists of observations collected over time, where the order and timing of observations matter.

**Key Characteristics:**
- **Temporal Ordering**: Data points are ordered by time
- **Dependencies**: Current values depend on past values
- **Trend**: Long-term increase or decrease
- **Seasonality**: Repeating patterns (daily, weekly, yearly)
- **Cyclical**: Non-fixed period patterns
- **Noise**: Random fluctuations

**Components of Time Series:**
- **Trend (T)**: Long-term direction
- **Seasonality (S)**: Regular patterns
- **Cyclical (C)**: Irregular cycles
- **Irregular/Noise (I)**: Random variation

**Decomposition:**
Additive: Y(t) = T(t) + S(t) + C(t) + I(t)
Multiplicative: Y(t) = T(t) × S(t) × C(t) × I(t)

**Stationarity:**
A time series is stationary if:
- Mean is constant over time
- Variance is constant over time
- Autocorrelation doesn't depend on time

**Why Stationarity Matters:**
- Most time series models assume stationarity
- Non-stationary series can be differenced to become stationary
- Easier to model and forecast

**Checking Stationarity:**
- **Visual Inspection**: Plot shows constant mean/variance
- **Augmented Dickey-Fuller (ADF) Test**: Statistical test for stationarity
- **KPSS Test**: Alternative stationarity test

**Differencing:**
- First difference: Δy(t) = y(t) - y(t-1)
- Removes trend
- May need multiple differences

**Applications:**
- Stock price prediction
- Weather forecasting
- Sales forecasting
- Energy demand prediction
- Economic indicators`,
					CodeExamples: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample time series
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(100, 200, 365)
seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25)
noise = np.random.normal(0, 5, 365)
ts = pd.Series(trend + seasonal + noise, index=dates)

# Plot time series
plt.figure(figsize=(12, 6))
plt.plot(ts)
plt.title('Time Series with Trend and Seasonality')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Decomposition
decomposition = seasonal_decompose(ts, model='additive', period=365)
decomposition.plot()
plt.show()

# Check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value}')
    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is non-stationary")

check_stationarity(ts)

# Make stationary through differencing
ts_diff = ts.diff().dropna()
check_stationarity(ts_diff)`,
				},
				{
					Title: "ARIMA Models",
					Content: `ARIMA (AutoRegressive Integrated Moving Average) is a powerful class of time series models.

**ARIMA Components:**
- **AR (p)**: AutoRegressive - uses p lagged values
- **I (d)**: Integrated - differencing order d
- **MA (q)**: Moving Average - uses q lagged forecast errors

**ARIMA(p, d, q) Notation:**
- p: Number of AR terms
- d: Number of differences
- q: Number of MA terms

**AR Model:**
y(t) = c + φ₁y(t-1) + φ₂y(t-2) + ... + φₚy(t-p) + ε(t)
- Predicts current value using past values
- φ: AR coefficients

**MA Model:**
y(t) = μ + ε(t) + θ₁ε(t-1) + θ₂ε(t-2) + ... + θₚε(t-q)
- Predicts using past forecast errors
- θ: MA coefficients

**ARIMA Model:**
Combines AR, differencing, and MA:
- First difference d times to make stationary
- Apply ARMA(p, q) to differenced series

**SARIMA (Seasonal ARIMA):**
ARIMA(p, d, q)(P, D, Q)ₛ
- Handles seasonality
- s: Seasonal period (e.g., 12 for monthly)
- (P, D, Q): Seasonal components

**Model Selection:**
- **ACF/PACF Plots**: Identify p and q
- **AIC/BIC**: Compare models (lower is better)
- **Auto ARIMA**: Automatically select best parameters

**ACF (Autocorrelation Function):**
- Correlation between series and lagged versions
- Helps identify MA order (q)

**PACF (Partial Autocorrelation Function):**
- Correlation controlling for intermediate lags
- Helps identify AR order (p)

**Forecasting with ARIMA:**
1. Check stationarity
2. Identify p, d, q using ACF/PACF
3. Fit model
4. Validate residuals
5. Forecast future values`,
					CodeExamples: `from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load or generate data
data = ts  # from previous example

# Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data, lags=40, ax=axes[0])
plot_pacf(data, lags=40, ax=axes[1])
plt.show()

# Fit ARIMA model
model = ARIMA(data, order=(2, 1, 2))  # ARIMA(2,1,2)
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast
forecast = fitted_model.forecast(steps=30)
print(f"Forecast for next 30 periods:\n{forecast}")

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(data, label='Historical')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()

# Auto ARIMA (requires pmdarima)
try:
    from pmdarima import auto_arima
    auto_model = auto_arima(data, seasonal=True, m=12, 
                           stepwise=True, suppress_warnings=True)
    print(f"Best model: {auto_model.order}")
    print(f"Best seasonal: {auto_model.seasonal_order}")
except ImportError:
    print("Install pmdarima: pip install pmdarima")`,
				},
				{
					Title: "Exponential Smoothing",
					Content: `Exponential smoothing methods use weighted averages of past observations, with more weight on recent data.

**Simple Exponential Smoothing:**
- For data with no trend or seasonality
- Formula: ŷ(t+1) = αy(t) + (1-α)ŷ(t)
- α: Smoothing parameter (0 < α < 1)
- Higher α: More weight on recent observations

**Holt's Linear Trend:**
- Handles trend
- Two smoothing parameters:
  - α: Level smoothing
  - β: Trend smoothing
- Forecast: Level + Trend

**Holt-Winters (Triple Exponential Smoothing):**
- Handles trend and seasonality
- Three smoothing parameters:
  - α: Level
  - β: Trend
  - γ: Seasonality
- Additive: Seasonality constant magnitude
- Multiplicative: Seasonality proportional to level

**Why Exponential Smoothing:**
- Simple and interpretable
- Good for short-term forecasting
- Handles trend and seasonality
- No assumptions about data distribution

**Parameter Selection:**
- Grid search over α, β, γ
- Minimize forecast error (MSE, MAE)
- Cross-validation

**Advantages:**
- Fast computation
- Easy to understand
- Good baseline method
- Handles missing values

**Limitations:**
- Assumes patterns continue
- May not capture structural breaks
- Less flexible than ARIMA`,
					CodeExamples: `from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Simple Exponential Smoothing
ses_model = ExponentialSmoothing(data, trend=None, seasonal=None)
ses_fitted = ses_model.fit(smoothing_level=0.3)
ses_forecast = ses_fitted.forecast(steps=30)

# Holt's Linear Trend
holt_model = ExponentialSmoothing(data, trend='add', seasonal=None)
holt_fitted = holt_model.fit()
holt_forecast = holt_fitted.forecast(steps=30)

# Holt-Winters (Additive)
hw_additive = ExponentialSmoothing(data, trend='add', seasonal='add', 
                                   seasonal_periods=12)
hw_add_fitted = hw_additive.fit()
hw_add_forecast = hw_add_fitted.forecast(steps=30)

# Holt-Winters (Multiplicative)
hw_multiplicative = ExponentialSmoothing(data, trend='mul', seasonal='mul',
                                        seasonal_periods=12)
hw_mul_fitted = hw_multiplicative.fit()
hw_mul_forecast = hw_mul_fitted.forecast(steps=30)

# Compare models
print("Simple ES Forecast:", ses_forecast[:5])
print("Holt Forecast:", holt_forecast[:5])
print("HW Additive Forecast:", hw_add_forecast[:5])

# Plot comparisons
plt.figure(figsize=(14, 8))
plt.plot(data[-100:], label='Historical')
plt.plot(ses_forecast, label='Simple ES')
plt.plot(holt_forecast, label='Holt')
plt.plot(hw_add_forecast, label='Holt-Winters Additive')
plt.legend()
plt.title('Exponential Smoothing Forecasts')
plt.show()`,
				},
				{
					Title: "LSTM for Time Series",
					Content: `LSTM (Long Short-Term Memory) networks can capture complex patterns in time series data.

**Why LSTM for Time Series:**
- Handles long-term dependencies
- Can learn non-linear patterns
- Doesn't require stationarity assumptions
- Can handle multiple input features

**Architecture:**
- Input: Sequence of past values
- Hidden state: Maintains memory
- Output: Next value prediction

**Data Preparation:**
- Create sequences: [x(t-n), ..., x(t-1)] → x(t)
- Normalize data (important for neural networks)
- Split into train/validation/test

**Sequence Length:**
- How many past values to use
- Too short: May miss patterns
- Too long: May include irrelevant data
- Typical: 7-60 time steps

**LSTM Architecture for Forecasting:**
- Input layer: Sequence input
- LSTM layers: 1-3 layers typically
- Dense layers: Output prediction
- Can be univariate or multivariate

**Univariate Forecasting:**
- Single time series
- Predict next value(s) from past values

**Multivariate Forecasting:**
- Multiple related time series
- Can include external features
- More complex but potentially more accurate

**Training Considerations:**
- Loss function: MSE or MAE
- Optimizer: Adam typically works well
- Early stopping: Prevent overfitting
- Learning rate: Usually 0.001-0.01

**Advantages:**
- Captures complex patterns
- Handles non-linear relationships
- Can use multiple features
- No statistical assumptions

**Limitations:**
- Requires more data
- Longer training time
- Less interpretable
- Hyperparameter tuning needed`,
					CodeExamples: `import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Prepare data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(data_scaled, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to tensors
X_train = torch.FloatTensor(X_train).unsqueeze(-1)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test)

# Define LSTM model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

# Initialize model
model = LSTMForecaster(input_size=1, hidden_size=50, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = scaler.inverse_transform(predictions.numpy())
    y_test_actual = scaler.inverse_transform(y_test.numpy())

# Evaluate
mse = mean_squared_error(y_test_actual, predictions)
print(f'Test MSE: {mse:.2f}')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('LSTM Time Series Forecast')
plt.show()`,
				},
				{
					Title: "Time Series Forecasting",
					Content: `Forecasting involves predicting future values of a time series.

**Forecasting Approaches:**
- **Statistical Methods**: ARIMA, Exponential Smoothing
- **Machine Learning**: Random Forest, XGBoost
- **Deep Learning**: LSTM, GRU, Transformers
- **Hybrid**: Combine multiple methods

**Forecast Horizon:**
- **Short-term**: 1-7 periods ahead
- **Medium-term**: 1-12 months
- **Long-term**: 1+ years

**Forecast Types:**
- **Point Forecast**: Single value prediction
- **Interval Forecast**: Range with confidence
- **Probabilistic Forecast**: Full distribution

**Evaluation Metrics:**
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error

**Cross-Validation for Time Series:**
- **Time Series Split**: Respect temporal order
- **Walk-Forward Validation**: Train on past, test on future
- **Expanding Window**: Growing training set
- **Rolling Window**: Fixed-size sliding window

**Feature Engineering:**
- **Lag Features**: Past values
- **Rolling Statistics**: Mean, std over window
- **Time Features**: Day of week, month, etc.
- **Difference Features**: Changes between periods

**Challenges:**
- **Non-stationarity**: Trends and seasonality
- **Structural Breaks**: Sudden changes
- **External Factors**: Events affecting series
- **Uncertainty**: Future is inherently uncertain`,
					CodeExamples: `from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Feature engineering
def create_features(data, max_lag=7):
    df = pd.DataFrame(data)
    for lag in range(1, max_lag + 1):
        df[f'lag_{lag}'] = df[0].shift(lag)
    df['rolling_mean_7'] = df[0].rolling(window=7).mean()
    df['rolling_std_7'] = df[0].rolling(window=7).std()
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df.dropna()

# Create features
feature_df = create_features(data)
X = feature_df.drop(columns=[0])
y = feature_df[0]

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          111,
			Title:       "Reinforcement Learning",
			Description: "Learn reinforcement learning fundamentals, MDPs, Q-learning, policy gradients, and deep RL algorithms.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "RL Fundamentals",
					Content: `Reinforcement Learning (RL) is learning through interaction with an environment to maximize cumulative reward.

**Key Components:**
- **Agent**: Learner/decision maker
- **Environment**: World agent interacts with
- **State (s)**: Current situation
- **Action (a)**: What agent can do
- **Reward (r)**: Feedback signal
- **Policy (π)**: Strategy for selecting actions

**RL Process:**
1. Agent observes state s(t)
2. Agent selects action a(t) using policy π
3. Environment transitions to s(t+1)
4. Agent receives reward r(t+1)
5. Repeat

**Goal:**
Maximize expected cumulative reward:
R = Σ γᵗ r(t+1)
Where γ (gamma) is discount factor (0 < γ ≤ 1)

**Key Concepts:**
- **Exploration vs Exploitation**: Balance trying new actions vs using known good actions
- **Reward Shaping**: Design reward function carefully
- **Credit Assignment**: Which actions led to reward?
- **Temporal Credit Assignment**: Delayed rewards

**Types of RL:**
- **Model-Based**: Learn environment model, plan
- **Model-Free**: Learn policy/value directly
- **On-Policy**: Learn about policy being followed
- **Off-Policy**: Learn about different policy

**Applications:**
- Game playing (Chess, Go, Atari)
- Robotics (control, manipulation)
- Autonomous vehicles
- Recommendation systems
- Resource allocation
- Trading algorithms`,
					CodeExamples: `import numpy as np
import gym

# Simple RL environment example
class SimpleEnv:
    def __init__(self):
        self.state = 0
        self.max_steps = 100
        self.step_count = 0
    
    def reset(self):
        self.state = 0
        self.step_count = 0
        return self.state
    
    def step(self, action):
        self.step_count += 1
        # Simple environment: move left (-1) or right (+1)
        self.state += action
        
        # Reward: closer to target (5) is better
        reward = -abs(self.state - 5)
        
        # Done if reached target or max steps
        done = (self.state == 5) or (self.step_count >= self.max_steps)
        
        return self.state, reward, done, {}
    
    def render(self):
        print(f"State: {self.state}, Steps: {self.step_count}")

# Using OpenAI Gym
env = gym.make('CartPole-v1')
state = env.reset()

# Random policy
for _ in range(100):
    action = env.action_space.sample()  # Random action
    state, reward, done, info = env.step(action)
    if done:
        state = env.reset()
env.close()`,
				},
				{
					Title: "Markov Decision Processes",
					Content: `MDPs provide the mathematical framework for RL problems.

**MDP Components:**
- **States (S)**: Set of possible states
- **Actions (A)**: Set of possible actions
- **Transition Probabilities P(s'|s,a)**: Probability of next state given current state and action
- **Reward Function R(s,a,s')**: Expected reward
- **Discount Factor γ**: Future reward importance

**Markov Property:**
Future depends only on current state, not history:
P(s(t+1)|s(t), a(t), s(t-1), ...) = P(s(t+1)|s(t), a(t))

**Value Functions:**
- **State Value V^π(s)**: Expected return from state s following policy π
- **Action Value Q^π(s,a)**: Expected return from state s, action a, then policy π

**Bellman Equation:**
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]

**Optimal Value Functions:**
- **V*(s)**: Maximum value from state s
- **Q*(s,a)**: Maximum value from state s, action a

**Optimal Policy:**
π*(s) = argmax_a Q*(s,a)
Greedy policy with respect to Q*

**Policy Evaluation:**
Iteratively update V^π until convergence

**Policy Improvement:**
Update policy to be greedy with respect to current value function

**Policy Iteration:**
Alternate between evaluation and improvement until convergence

**Value Iteration:**
Directly compute optimal value function, then extract policy`,
					CodeExamples: `import numpy as np

# Simple Grid World MDP
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.states = size * size
        self.actions = 4  # up, down, left, right
        self.gamma = 0.9
        
        # Transition probabilities (simplified: deterministic)
        self.P = self._build_transitions()
        self.R = self._build_rewards()
    
    def _build_transitions(self):
        P = np.zeros((self.states, self.actions, self.states))
        for s in range(self.states):
            row, col = s // self.size, s % self.size
            # Up
            if row > 0:
                P[s, 0, s - self.size] = 1.0
            else:
                P[s, 0, s] = 1.0
            # Down
            if row < self.size - 1:
                P[s, 1, s + self.size] = 1.0
            else:
                P[s, 1, s] = 1.0
            # Left
            if col > 0:
                P[s, 2, s - 1] = 1.0
            else:
                P[s, 2, s] = 1.0
            # Right
            if col < self.size - 1:
                P[s, 3, s + 1] = 1.0
            else:
                P[s, 3, s] = 1.0
        return P
    
    def _build_rewards(self):
        R = np.zeros((self.states, self.actions, self.states))
        # Goal at bottom-right
        goal = self.states - 1
        for s in range(self.states):
            for a in range(self.actions):
                for s_next in range(self.states):
                    if s_next == goal and s != goal:
                        R[s, a, s_next] = 1.0
                    else:
                        R[s, a, s_next] = -0.01
        return R

# Value Iteration
def value_iteration(mdp, theta=1e-6):
    V = np.zeros(mdp.states)
    while True:
        V_new = np.zeros(mdp.states)
        for s in range(mdp.states):
            q_values = []
            for a in range(mdp.actions):
                q = sum(mdp.P[s, a, s_next] * 
                       (mdp.R[s, a, s_next] + mdp.gamma * V[s_next])
                       for s_next in range(mdp.states))
                q_values.append(q)
            V_new[s] = max(q_values)
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new
    return V

# Extract policy
def extract_policy(mdp, V):
    policy = np.zeros(mdp.states, dtype=int)
    for s in range(mdp.states):
        q_values = []
        for a in range(mdp.actions):
            q = sum(mdp.P[s, a, s_next] * 
                   (mdp.R[s, a, s_next] + mdp.gamma * V[s_next])
                   for s_next in range(mdp.states))
            q_values.append(q)
        policy[s] = np.argmax(q_values)
    return policy

# Run
mdp = GridWorld()
V = value_iteration(mdp)
policy = extract_policy(mdp, V)
print("Optimal Value Function:")
print(V.reshape(4, 4))
print("\nOptimal Policy:")
print(policy.reshape(4, 4))`,
				},
				{
					Title: "Q-Learning",
					Content: `Q-Learning is a model-free, off-policy algorithm that learns action-value function Q(s,a).

**Q-Learning Algorithm:**
1. Initialize Q(s,a) arbitrarily
2. For each episode:
   - Initialize state s
   - For each step:
     - Choose action a using policy derived from Q (e.g., ε-greedy)
     - Take action a, observe r, s'
     - Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
     - s ← s'
   - Until s is terminal

**Update Rule:**
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
- α: Learning rate
- r: Immediate reward
- γ: Discount factor
- max Q(s',a'): Best future value

**ε-Greedy Policy:**
- With probability ε: random action (exploration)
- With probability 1-ε: best action (exploitation)
- ε typically decays over time

**Why Q-Learning Works:**
- Off-policy: Can learn optimal policy while following different policy
- Model-free: Doesn't need environment model
- Converges to Q* under certain conditions

**Convergence Conditions:**
- All state-action pairs visited infinitely often
- Learning rate satisfies: Σα = ∞, Σα² < ∞

**Tabular Q-Learning:**
- Store Q-table: Q[s, a] for each state-action pair
- Works for discrete, small state/action spaces

**Limitations:**
- Doesn't scale to large state spaces
- Requires discretization for continuous states
- Memory grows with state space size`,
					CodeExamples: `import numpy as np
import random

class QLearning:
    def __init__(self, states, actions, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((states, actions))
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
    
    def decay_epsilon(self, factor=0.99):
        self.epsilon *= factor

# Example: Simple environment
class SimpleEnv:
    def __init__(self):
        self.state = 0
        self.goal = 4
    
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        if action == 0:  # Left
            self.state = max(0, self.state - 1)
        else:  # Right
            self.state = min(4, self.state + 1)
        
        reward = 1.0 if self.state == self.goal else -0.1
        done = (self.state == self.goal)
        
        return self.state, reward, done

# Training
env = SimpleEnv()
agent = QLearning(states=5, actions=2, epsilon=0.2)

for episode in range(100):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    agent.decay_epsilon()
    
    if episode % 20 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

print("\nLearned Q-table:")
print(agent.Q)`,
				},
				{
					Title: "Deep Q-Networks (DQN)",
					Content: `DQN combines Q-learning with deep neural networks to handle large state spaces.

**Key Innovation:**
Use neural network Q(s,a;θ) instead of Q-table
- Input: State s
- Output: Q-values for all actions
- Parameters: θ (network weights)

**DQN Algorithm:**
1. Initialize Q-network Q(s,a;θ)
2. Initialize target network Q(s,a;θ⁻) = Q(s,a;θ)
3. For each episode:
   - For each step:
     - Choose action a using ε-greedy
     - Store transition (s,a,r,s',done) in replay buffer
     - Sample batch from replay buffer
     - Update Q-network using batch
     - Every C steps: θ⁻ ← θ

**Experience Replay:**
- Store transitions in buffer
- Sample random batches for training
- Breaks correlation between consecutive samples
- More stable learning

**Target Network:**
- Separate network for computing targets
- Updated less frequently (every C steps)
- Stabilizes training (reduces moving target problem)

**Loss Function:**
L(θ) = E[(r + γ max Q(s',a';θ⁻) - Q(s,a;θ))²]
- Mean squared error between Q and target

**DQN Improvements:**
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage
- **Prioritized Replay**: Sample important transitions more
- **Rainbow DQN**: Combines multiple improvements

**Advantages:**
- Handles high-dimensional states (images)
- Learns from raw pixels
- Generalizes across similar states

**Challenges:**
- Requires careful hyperparameter tuning
- Can be unstable
- Needs large replay buffer`,
					CodeExamples: `import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.BoolTensor(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer()
    
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Usage example (conceptual)
# agent = DQNAgent(state_size=4, action_size=2)
# for episode in range(1000):
#     state = env.reset()
#     while not done:
#         action = agent.act(state)
#         next_state, reward, done = env.step(action)
#         agent.remember(state, action, reward, next_state, done)
#         agent.replay()
#         state = next_state
#     if episode % 10 == 0:
#         agent.update_target_network()`,
				},
				{
					Title: "Policy Gradient Methods",
					Content: `Policy gradient methods directly optimize the policy π(a|s;θ) using gradient ascent.

**Policy Gradient Theorem:**
∇_θ J(θ) = E[∇_θ log π(a|s;θ) Q^π(s,a)]
- J(θ): Expected return
- Gradient points in direction of higher return

**REINFORCE Algorithm:**
1. Sample episode using policy π
2. Compute returns G_t for each step
3. Update: θ ← θ + α ∇_θ log π(a_t|s_t;θ) G_t

**Advantages:**
- Can handle continuous action spaces
- Directly optimizes what we care about (return)
- Can learn stochastic policies

**Disadvantages:**
- High variance in gradient estimates
- Slow learning (many samples needed)
- Can get stuck in local optima

**Variance Reduction:**
- **Baseline**: Subtract baseline from returns
- **Actor-Critic**: Use value function as baseline
- **Advantage Function**: A(s,a) = Q(s,a) - V(s)

**Actor-Critic:**
- **Actor**: Policy π(a|s;θ)
- **Critic**: Value function V(s;w)
- Actor improves policy
- Critic evaluates policy

**Advantage Actor-Critic (A2C):**
- Uses advantage A(s,a) = Q(s,a) - V(s)
- Lower variance than REINFORCE
- More stable learning

**Proximal Policy Optimization (PPO):**
- Clips policy updates to prevent large changes
- More stable than vanilla policy gradients
- State-of-the-art for many RL tasks`,
					CodeExamples: `import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic:
    def __init__(self, state_size, action_size, lr_actor=0.001, lr_critic=0.01):
        self.actor = PolicyNetwork(state_size, action_size)
        self.critic = ValueNetwork(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = 0.99
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, states, actions, rewards, log_probs, next_states, dones):
        # Compute returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.stack(log_probs)
        
        # Critic update
        values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        advantages = returns - values.detach()
        actor_loss = -(log_probs * advantages).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# Usage example
# agent = ActorCritic(state_size=4, action_size=2)
# states, actions, rewards, log_probs, next_states, dones = [], [], [], [], [], []
# 
# for step in range(100):
#     state = env.reset()
#     done = False
#     while not done:
#         action, log_prob = agent.select_action(state)
#         next_state, reward, done = env.step(action)
#         states.append(state)
#         actions.append(action)
#         rewards.append(reward)
#         log_probs.append(log_prob)
#         next_states.append(next_state)
#         dones.append(done)
#         state = next_state
#     
#     agent.update(states, actions, rewards, log_probs, next_states, dones)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          112,
			Title:       "Computer Vision Advanced",
			Description: "Learn advanced computer vision techniques: object detection, segmentation, transfer learning, and vision transformers.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Object Detection",
					Content: `Object detection identifies and localizes multiple objects in images with bounding boxes.

**Tasks:**
- **Classification**: What objects are present?
- **Localization**: Where are they? (bounding boxes)

**Challenges:**
- Multiple objects per image
- Varying sizes and aspect ratios
- Occlusion and clutter
- Real-time requirements

**Two-Stage Detectors:**
- **R-CNN**: Region proposals → CNN → Classification
- **Fast R-CNN**: Shared computation, faster
- **Faster R-CNN**: Region proposal network (RPN)

**One-Stage Detectors:**
- **YOLO**: You Only Look Once - single pass
- **SSD**: Single Shot MultiBox Detector
- **RetinaNet**: Focal loss for class imbalance

**YOLO Architecture:**
- Divides image into grid
- Each grid cell predicts bounding boxes
- Single forward pass
- Very fast inference

**YOLO v1-v8 Evolution:**
- v1: Original single-stage detector
- v3: Multi-scale detection, better backbone
- v5: PyTorch implementation, easy to use
- v8: Latest with improved accuracy

**Evaluation Metrics:**
- **mAP (mean Average Precision)**: Primary metric
- **IoU (Intersection over Union)**: Overlap measure
- **Precision/Recall**: Detection quality

**Applications:**
- Autonomous vehicles
- Surveillance
- Medical imaging
- Retail analytics`,
					CodeExamples: `# Using YOLOv5 (requires ultralytics)
try:
    from ultralytics import YOLO
    
    # Load pre-trained model
    model = YOLO('yolov5s.pt')  # Small model
    
    # Detect objects
    results = model('image.jpg')
    
    # Display results
    results[0].show()
    
    # Get detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            print(f"Class: {cls}, Confidence: {conf:.2f}, Box: {bbox}")
except ImportError:
    print("Install: pip install ultralytics")

# Custom object detection with PyTorch
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Prepare image
image = F.to_tensor(F.resize(F.pil_to_tensor(image), (800, 800)))

# Inference
with torch.no_grad():
    predictions = model([image])

# Process predictions
boxes = predictions[0]['boxes']
scores = predictions[0]['scores']
labels = predictions[0]['labels']

# Filter by confidence
threshold = 0.5
keep = scores > threshold
boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]

print(f"Detected {len(boxes)} objects")`,
				},
				{
					Title: "Image Segmentation",
					Content: `Image segmentation partitions image into regions, pixel-level classification.

**Types of Segmentation:**
- **Semantic Segmentation**: Classify each pixel (no instance distinction)
- **Instance Segmentation**: Separate instances of same class
- **Panoptic Segmentation**: Combines semantic + instance

**Semantic Segmentation:**
- **U-Net**: Encoder-decoder with skip connections
- **FCN**: Fully Convolutional Networks
- **DeepLab**: Atrous convolutions, CRF post-processing

**U-Net Architecture:**
- **Encoder**: Downsampling path (contracting)
- **Decoder**: Upsampling path (expanding)
- **Skip Connections**: Preserve spatial information
- Symmetric U-shaped architecture

**Instance Segmentation:**
- **Mask R-CNN**: Extends Faster R-CNN with mask branch
- **YOLACT**: Real-time instance segmentation
- **SOLO**: Segmenting Objects by Locations

**Mask R-CNN:**
- Object detection + segmentation mask
- ROI Align (better than ROI Pooling)
- Parallel branches: classification, bbox, mask

**Applications:**
- Medical image analysis
- Autonomous driving
- Satellite imagery
- Video editing

**Evaluation Metrics:**
- **mIoU**: Mean Intersection over Union
- **Pixel Accuracy**: Percentage correct pixels
- **Dice Coefficient**: Overlap measure`,
					CodeExamples: `import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50

# Semantic Segmentation with DeepLabV3
model = deeplabv3_resnet50(pretrained=True)
model.eval()

# Prepare image
image = F.to_tensor(F.resize(F.pil_to_tensor(image), (520, 520)))

# Inference
with torch.no_grad():
    output = model([image])[0]

# Get predictions
predictions = output['out'].argmax(1).squeeze().cpu().numpy()
print(f"Segmentation shape: {predictions.shape}")

# U-Net Implementation
class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)
        
        self.final = nn.Conv2d(64, n_classes, 1)
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        return self.final(d1)

# Instance Segmentation with Mask R-CNN
mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
mask_rcnn.eval()

with torch.no_grad():
    predictions = mask_rcnn([image])

masks = predictions[0]['masks']
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']`,
				},
				{
					Title: "Transfer Learning for Vision",
					Content: `Transfer learning uses pre-trained models as starting point for new tasks.

**Why Transfer Learning:**
- Pre-trained models learned useful features
- Requires less data
- Faster training
- Better performance

**Transfer Learning Strategies:**
- **Feature Extraction**: Freeze backbone, train classifier
- **Fine-tuning**: Unfreeze some layers, train end-to-end
- **Progressive Unfreezing**: Gradually unfreeze layers

**Pre-trained Models:**
- **ImageNet**: Large dataset, many classes
- **Models**: ResNet, VGG, EfficientNet, Vision Transformer

**Feature Extraction:**
- Remove final classification layer
- Add new classifier for your task
- Freeze backbone, train only classifier
- Fast, works with small datasets

**Fine-tuning:**
- Unfreeze some/all layers
- Use lower learning rate
- Train end-to-end
- Better performance, needs more data

**Learning Rate Scheduling:**
- Lower LR for pre-trained layers
- Higher LR for new layers
- Prevents destroying learned features

**Data Augmentation:**
- Crucial for small datasets
- Rotation, flipping, color jitter
- Random crops, scaling
- Mixup, CutMix`,
					CodeExamples: `import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# Feature Extraction
def create_feature_extractor(num_classes):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# Fine-tuning
def create_finetuned_model(num_classes):
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# Different learning rates for different layers
def get_optimizer(model, lr_backbone=1e-5, lr_classifier=1e-3):
    params = [
        {'params': [p for n, p in model.named_parameters() 
                   if 'fc' not in n], 'lr': lr_backbone},
        {'params': [p for n, p in model.named_parameters() 
                   if 'fc' in n], 'lr': lr_classifier}
    ]
    return torch.optim.Adam(params)

# Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])`,
				},
				{
					Title: "Vision Transformers (ViT)",
					Content: `Vision Transformers apply transformer architecture to images, achieving state-of-the-art results.

**ViT Architecture:**
- **Patch Embedding**: Split image into patches, embed
- **Position Embedding**: Add positional information
- **Transformer Encoder**: Self-attention layers
- **Classification Head**: MLP for final prediction

**Patch Embedding:**
- Divide image into N×N patches (e.g., 16×16)
- Flatten each patch
- Linear projection to embedding dimension
- Add learnable class token

**Position Embedding:**
- Learnable or fixed sinusoidal
- Encodes spatial relationships
- Similar to NLP transformers

**Self-Attention in Vision:**
- Patches attend to other patches
- Learns spatial relationships
- Can capture long-range dependencies

**ViT vs CNNs:**
- **ViT**: Global attention from start
- **CNN**: Local receptive fields, hierarchical
- **ViT**: Needs more data to train from scratch
- **ViT**: Better with pre-training

**Hybrid Approaches:**
- CNN backbone + Transformer
- Best of both worlds
- CNN extracts features, Transformer models relationships

**Efficient Variants:**
- **DeiT**: Data-efficient image transformer
- **Swin Transformer**: Hierarchical, shifted windows
- **PVT**: Pyramid vision transformer

**Applications:**
- Image classification
- Object detection
- Segmentation
- Multi-modal learning`,
					CodeExamples: `import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 
                                          in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, 
                                      dim_feedforward=embed_dim*4,
                                      dropout=0.1, batch_first=True)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)

# Using pre-trained ViT
try:
    from transformers import ViTImageProcessor, ViTForImageClassification
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    
    # Inference
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
except ImportError:
    print("Install: pip install transformers")`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
