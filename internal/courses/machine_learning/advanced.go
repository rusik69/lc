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
- Recurrence: hₜ = f(W·xₜ + U·hₜ₋₁ + b)

**Applications:**
- Natural Language Processing
- Time Series Prediction
- Speech Recognition
- Machine Translation

**Limitations:**
- Vanishing gradient problem
- Difficulty learning long-term dependencies
- Sequential processing (slow)`,
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
- **Cell State**: Long-term memory
- **Output Gate**: Decides what parts of cell state to output

**Gates:**
- fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)  # Forget gate
- iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)  # Input gate
- C̃ₜ = tanh(WC·[hₜ₋₁, xₜ] + bC)  # Candidate values
- Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ  # Cell state
- oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)  # Output gate
- hₜ = oₜ * tanh(Cₜ)  # Hidden state

**Advantages:**
- Handles long-term dependencies
- Prevents vanishing gradients
- Better than vanilla RNN`,
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

**Attention Formula:**
Attention(Q, K, V) = softmax(QKᵀ/√dₖ) · V

Where:
- **Q (Query)**: What we're looking for
- **K (Key)**: What each position offers
- **V (Value)**: Actual content at each position
- **dₖ**: Dimension of keys (scaling factor)

**Why Attention?**
- Captures long-range dependencies
- Parallelizable (unlike RNNs)
- Interpretable (attention weights show what model focuses on)
- Flexible (can attend to any position)`,
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

**Sinusoidal Positional Encoding:**
- Add positional encodings to input embeddings
- Uses sin/cos functions of different frequencies
- Allows model to learn relative positions

**Formula:**
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

**Alternative:**
- Learned positional embeddings (trainable)
- Often works better in practice`,
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

**Transformer Block:**
1. **Multi-Head Self-Attention**
2. **Add & Norm** (residual connection + layer norm)
3. **Feed-Forward Network**
4. **Add & Norm**

**Full Transformer:**
- **Encoder**: Processes input sequence
- **Decoder**: Generates output sequence (with masked attention)
- **Stacked Layers**: Multiple transformer blocks

**Key Components:**
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Helps with gradient flow
- **Feed-Forward**: Two linear layers with ReLU
- **Masking**: Prevents attention to future tokens (decoder)`,
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

**Pre-training:**
- **Objective**: Learn general language patterns
- **Data**: Massive unlabeled text corpus (web, books, etc.)
- **Task**: Next token prediction (autoregressive) or masked language modeling
- **Scale**: Requires massive compute (weeks/months on thousands of GPUs)
- **Result**: General-purpose language understanding

**Fine-tuning:**
- **Objective**: Adapt to specific task/domain
- **Data**: Smaller, task-specific labeled dataset
- **Task**: Classification, QA, summarization, etc.
- **Scale**: Much smaller compute (hours/days on single GPU)
- **Result**: Task-specific model

**Fine-tuning Approaches:**
- **Full Fine-tuning**: Update all parameters
- **Parameter-Efficient Fine-tuning (PEFT)**: Update subset (LoRA, Adapters)
- **Prompt Tuning**: Learn soft prompts
- **Instruction Tuning**: Fine-tune on instruction-following`,
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
Sentiment:""",
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
	})
}
