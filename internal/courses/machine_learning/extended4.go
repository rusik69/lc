package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          2518,
			Title:       "Generative AI and Large Language Models",
			Description: "Understand generative models, transformer-based LLMs, training techniques, fine-tuning, prompt engineering, RAG, and responsible AI practices.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "LLMs GANs Diffusion Models and RAG Architecture",
					Content: `Generative AI creates new content — text, images, audio, code — by learning patterns from training data. Understanding the architectures and techniques behind modern generative models is essential.

**Large Language Models (LLMs):**

Architecture:
  Decoder-only Transformer (GPT family)
  Autoregressive: Predict next token given previous tokens
  P(text) = P(t1) × P(t2|t1) × P(t3|t1,t2) × ...
  
  Key components:
    Token embedding: Map tokens to vectors
    Positional encoding: RoPE (Rotary Position Embedding)
    Self-attention layers with causal mask
    Feed-forward layers
    Layer normalization (pre-norm in modern models)
    Output: Vocabulary-sized logits → softmax → probabilities

Scale history:
  GPT-1 (2018): 117M parameters
  GPT-2 (2019): 1.5B parameters
  GPT-3 (2020): 175B parameters
  PaLM (2022): 540B parameters
  GPT-4 (2023): Estimated 1.7T (MoE)
  Llama 3 (2024): 8B, 70B, 405B

Scaling Laws:
  Loss ∝ N^(-α) × D^(-β) × C^(-γ)
  N = parameters, D = data tokens, C = compute
  
  Chinchilla optimal: tokens ≈ 20 × parameters
  For 10B model → 200B tokens
  Compute-optimal training balances model size and data

Tokenization:
  BPE (Byte Pair Encoding): GPT family
    Start with characters, merge frequent pairs
    Vocabulary: 50K-100K tokens
    
  SentencePiece: Llama family
    Unigram or BPE on raw text
    Language-agnostic
    
  Token types:
    Subwords: "understanding" → "under" + "standing"
    Whole words: Common words as single tokens
    Special: [BOS], [EOS], [PAD], [UNK]

**Pretraining:**

Self-supervised learning:
  Causal LM: Predict next token (GPT)
  Masked LM: Predict masked tokens (BERT)
  Prefix LM: Bidirectional prefix + causal generation (T5)
  
Training data:
  Web crawls (Common Crawl, C4)
  Books, Wikipedia, code repositories
  Deduplication, quality filtering
  Typically trillions of tokens

Training infrastructure:
  Data parallelism: Same model on multiple GPUs, different data
  Tensor parallelism: Split layers across GPUs
  Pipeline parallelism: Split layers into stages
  Mixed precision: FP16/BF16 forward, FP32 gradients
  Gradient checkpointing: Trade compute for memory
  
  Hardware:
    Hundreds to thousands of GPUs
    High-bandwidth interconnect (NVLink, InfiniBand)
    Training runs: Weeks to months
    Cost: Millions of dollars

**Fine-tuning:**

Full fine-tuning:
  Update all parameters
  Requires full training infrastructure
  Risk of catastrophic forgetting
  Best performance when data is abundant

LoRA (Low-Rank Adaptation):
  Freeze pretrained weights
  Add low-rank matrices: W' = W + BA
  B: d × r, A: r × d (r << d, typically 4-64)
  Train only A and B (0.1-1% of parameters)
  
  Benefits:
    Much less memory (no optimizer state for frozen weights)
    Fast training
    Multiple adapters for different tasks
    Can merge into base model for inference

QLoRA:
  Quantize base model to 4-bit
  Apply LoRA on quantized model
  Fine-tune 65B model on single 48GB GPU
  4-bit NormalFloat (NF4) quantization

Instruction Tuning:
  Fine-tune on instruction-response pairs
  Teaches model to follow instructions
  Datasets: FLAN, Alpaca, ShareGPT

RLHF (Reinforcement Learning from Human Feedback):
  1. Supervised fine-tuning (SFT) on demonstrations
  2. Train reward model on human preference comparisons
  3. Optimize policy (LLM) using PPO to maximize reward
  
  DPO (Direct Preference Optimization):
    Skip reward model training
    Directly optimize from preference pairs
    Simpler, often similar performance

**Inference Optimization:**

Quantization:
  INT8: 2x smaller, minimal quality loss
  INT4/GPTQ: 4x smaller, some quality loss
  AWQ (Activation-Aware): Preserve important weights
  GGUF: Format for CPU inference (llama.cpp)

KV Cache:
  Cache key-value pairs from previous tokens
  Avoid recomputing attention for all past tokens
  Memory: batch_size × seq_len × num_layers × 2 × d_model
  
  Optimizations:
    Paged attention (vLLM): Manage KV cache like virtual memory
    Sliding window: Only cache recent tokens (Mistral)
    Multi-query attention: Share KV heads

Speculative Decoding:
  Small draft model generates candidates quickly
  Large model verifies in parallel
  Accept or reject candidate tokens
  2-3x faster with same output quality

Batching:
  Continuous batching: Add/remove requests dynamically
  Dynamic batching: Group requests by similar length
  Iteration-level scheduling (vLLM, TensorRT-LLM)

**Prompt Engineering:**

Techniques:
  Zero-shot: Direct instruction without examples
  Few-shot: Provide examples before task
  Chain-of-Thought (CoT): "Let's think step by step"
  Self-consistency: Multiple reasoning paths, majority vote
  Tree-of-Thought: Explore multiple reasoning branches
  ReAct: Reasoning + Acting (tool use)

System prompts:
  Define persona, constraints, output format
  Guard rails for safety
  
Structured output:
  JSON mode
  Function calling / tool use
  Grammar-constrained generation

**RAG (Retrieval-Augmented Generation):**

Architecture:
  1. User query
  2. Embed query using embedding model
  3. Search vector database for similar documents
  4. Retrieve top-k relevant chunks
  5. Compose prompt: system + context + query
  6. LLM generates grounded response

Components:
  Document processing:
    Chunking: Split documents into chunks (500-1000 tokens)
    Overlap: 10-20% overlap between chunks
    Metadata: Source, page, section, date
    
  Embedding model:
    Sentence transformers (all-MiniLM, E5, BGE)
    OpenAI text-embedding-3
    Dimension: 384-1536
    
  Vector database:
    FAISS, Pinecone, Weaviate, Chroma, Qdrant, Milvus
    Similarity: Cosine, Euclidean, dot product
    
    Index types:
      Flat: Exact search (small datasets)
      IVF: Inverted file index (partitioned search)
      HNSW: Hierarchical navigable small world (graph-based)
      PQ: Product quantization (compressed)
    
  Retrieval:
    Dense retrieval: Embedding similarity
    Sparse retrieval: BM25, TF-IDF
    Hybrid: Combine dense + sparse with reciprocal rank fusion
    Reranking: Cross-encoder reranker on top-k results

Advanced RAG:
  Query transformation: Rephrase, decompose, expand
  Self-RAG: Model decides when to retrieve
  Corrective RAG: Verify retrieval quality
  Graph RAG: Knowledge graph-enhanced retrieval
  Multi-hop RAG: Chain multiple retrievals

**Generative Adversarial Networks (GANs):**

Architecture:
  Generator G: Random noise → fake data
  Discriminator D: Data → real/fake classification
  
  Training:
    1. Train D to distinguish real from fake
    2. Train G to fool D
    3. Repeat until equilibrium
    
  Loss:
    min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]

Variants:
  DCGAN: Deep convolutional GAN
  Conditional GAN: Generate specific types
  StyleGAN: High-quality face generation, style mixing
  CycleGAN: Unpaired image-to-image translation
  Wasserstein GAN: Earth mover's distance, stable training

Challenges:
  Mode collapse: Generator produces limited variety
  Training instability: Oscillation, divergence
  Evaluation: FID (Frechet Inception Distance), IS (Inception Score)

**Diffusion Models:**

Forward process:
  Gradually add Gaussian noise to data
  x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * epsilon
  After T steps: pure noise

Reverse process:
  Neural network learns to denoise
  Start from random noise
  Iteratively remove noise to generate data
  
  Training:
    Predict noise epsilon given noisy image x_t and timestep t
    Loss: MSE between predicted and actual noise

Architecture:
  U-Net: Encoder-decoder with skip connections
  Cross-attention for conditioning (text, class)
  Time embedding for timestep awareness

Variants:
  DDPM: Denoising Diffusion Probabilistic Models
  Stable Diffusion: Latent space diffusion (compressed)
  DALL-E: Text-to-image generation
  Imagen: Cascaded diffusion models

**Responsible AI:**

Bias and Fairness:
  Training data reflects societal biases
  Representation: Different groups represented equally?
  Performance: Equal accuracy across demographics?
  Mitigation: Data curation, debiasing, evaluation

Safety:
  Content filtering: Block harmful outputs
  Red teaming: Adversarial testing
  Guardrails: Input/output validation
  Constitutional AI: AI-assisted safety training

Hallucination:
  Model generates plausible but incorrect information
  Mitigation: RAG, citation, self-consistency, verification
  Calibration: Model should express uncertainty

Privacy:
  Training data memorization
  Differential privacy: Add noise to training
  Data deduplication
  PII detection and removal

Evaluation:
  Benchmarks: MMLU, HellaSwag, TruthfulQA, HumanEval
  Human evaluation: Quality, helpfulness, safety
  Red teaming: Adversarial prompts
  Automated: LLM-as-judge`,
					CodeExamples: `# Generative AI and LLM Implementation Examples

import math
import random
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================
# Text Generation (Simplified Autoregressive LM)
# ============================================================

class SimpleLanguageModel:
    """N-gram language model for demonstration."""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams: Dict[tuple, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        self.vocabulary: set = set()
    
    def train(self, text: str):
        tokens = text.split()
        self.vocabulary.update(tokens)
        
        for i in range(len(tokens) - self.n):
            context = tuple(tokens[i:i + self.n - 1])
            next_token = tokens[i + self.n - 1]
            self.ngrams[context][next_token] += 1
    
    def predict_next(self, context: List[str],
                     temperature: float = 1.0) -> str:
        ctx = tuple(context[-(self.n - 1):])
        candidates = self.ngrams.get(ctx, {})
        
        if not candidates:
            return random.choice(list(self.vocabulary)) if self.vocabulary else ""
        
        tokens = list(candidates.keys())
        counts = [candidates[t] for t in tokens]
        
        # Apply temperature
        if temperature != 1.0:
            log_probs = [math.log(c) / temperature for c in counts]
            max_lp = max(log_probs)
            probs = [math.exp(lp - max_lp) for lp in log_probs]
        else:
            probs = [float(c) for c in counts]
        
        total = sum(probs)
        probs = [p / total for p in probs]
        
        r = random.random()
        cumsum = 0
        for token, prob in zip(tokens, probs):
            cumsum += prob
            if r <= cumsum:
                return token
        return tokens[-1]
    
    def generate(self, seed: List[str], max_tokens: int = 50,
                 temperature: float = 1.0) -> str:
        tokens = list(seed)
        
        for _ in range(max_tokens):
            next_token = self.predict_next(tokens, temperature)
            tokens.append(next_token)
        
        return ' '.join(tokens)


# ============================================================
# Sampling Strategies
# ============================================================

def top_k_sampling(logits: List[float], k: int,
                   temperature: float = 1.0) -> int:
    """Top-k sampling."""
    indexed = list(enumerate(logits))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_k_items = indexed[:k]
    
    indices = [idx for idx, _ in top_k_items]
    values = [val / temperature for _, val in top_k_items]
    
    max_val = max(values)
    exp_values = [math.exp(v - max_val) for v in values]
    total = sum(exp_values)
    probs = [e / total for e in exp_values]
    
    r = random.random()
    cumsum = 0
    for idx, prob in zip(indices, probs):
        cumsum += prob
        if r <= cumsum:
            return idx
    return indices[-1]


def top_p_sampling(logits: List[float], p: float,
                   temperature: float = 1.0) -> int:
    """Nucleus (top-p) sampling."""
    scaled = [l / temperature for l in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(v - max_val) for v in scaled]
    total = sum(exp_vals)
    probs = [(i, e / total) for i, e in enumerate(exp_vals)]
    
    probs.sort(key=lambda x: x[1], reverse=True)
    
    cumsum = 0
    candidates = []
    for idx, prob in probs:
        cumsum += prob
        candidates.append((idx, prob))
        if cumsum >= p:
            break
    
    # Renormalize
    c_total = sum(pr for _, pr in candidates)
    
    r = random.random() * c_total
    cumsum = 0
    for idx, prob in candidates:
        cumsum += prob
        if r <= cumsum:
            return idx
    return candidates[-1][0]


def beam_search(initial_tokens: List[int],
                score_fn: Callable[[List[int]], List[float]],
                beam_width: int = 5, max_length: int = 50,
                eos_token: int = 0) -> List[int]:
    """Beam search decoding."""
    beams = [(initial_tokens, 0.0)]
    completed = []
    
    for _ in range(max_length):
        all_candidates = []
        
        for tokens, score in beams:
            if tokens[-1] == eos_token:
                completed.append((tokens, score))
                continue
            
            logits = score_fn(tokens)
            indexed = list(enumerate(logits))
            indexed.sort(key=lambda x: x[1], reverse=True)
            
            for token_id, token_score in indexed[:beam_width]:
                new_tokens = tokens + [token_id]
                new_score = score + token_score
                all_candidates.append((new_tokens, new_score))
        
        if not all_candidates:
            break
        
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]
    
    all_results = completed + beams
    all_results.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
    return all_results[0][0] if all_results else initial_tokens


# ============================================================
# Token Embedding and Vocabulary
# ============================================================

class Tokenizer:
    """Simple word-level tokenizer."""
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self._next_id = 0
        
        # Add special tokens
        for special in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]:
            self._add_token(special)
    
    def _add_token(self, token: str) -> int:
        if token not in self.word_to_id:
            self.word_to_id[token] = self._next_id
            self.id_to_word[self._next_id] = token
            self._next_id += 1
        return self.word_to_id[token]
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        freq: Dict[str, int] = defaultdict(int)
        for text in texts:
            for word in text.lower().split():
                freq[word] += 1
        
        for word, count in sorted(freq.items()):
            if count >= min_freq:
                self._add_token(word)
    
    def encode(self, text: str) -> List[int]:
        tokens = [self.word_to_id.get("[BOS]", 0)]
        for word in text.lower().split():
            tokens.append(
                self.word_to_id.get(word, self.word_to_id["[UNK]"]))
        tokens.append(self.word_to_id.get("[EOS]", 0))
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        words = []
        for token_id in ids:
            word = self.id_to_word.get(token_id, "[UNK]")
            if word not in ("[PAD]", "[BOS]", "[EOS]"):
                words.append(word)
        return ' '.join(words)
    
    @property
    def vocab_size(self) -> int:
        return self._next_id


class TokenEmbedding:
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, d_model: int):
        scale = math.sqrt(1.0 / d_model)
        self.embeddings = [[random.gauss(0, scale)
                           for _ in range(d_model)]
                          for _ in range(vocab_size)]
        self.d_model = d_model
    
    def forward(self, token_ids: List[int]) -> List[List[float]]:
        return [self.embeddings[tid][:] for tid in token_ids]


# ============================================================
# RAG (Retrieval-Augmented Generation)
# ============================================================

class VectorStore:
    """Simple vector store for RAG."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
    
    def add(self, vector: List[float], document: Dict[str, Any]):
        assert len(vector) == self.dimension
        self.vectors.append(vector)
        self.documents.append(document)
    
    def search(self, query_vector: List[float],
               top_k: int = 5) -> List[Tuple[float, Dict]]:
        scores = []
        for i, vec in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, vec)
            scores.append((similarity, self.documents[i]))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]
    
    def _cosine_similarity(self, a: List[float],
                          b: List[float]) -> float:
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    @property
    def size(self) -> int:
        return len(self.vectors)


class DocumentChunker:
    """Split documents into chunks for RAG."""
    
    def __init__(self, chunk_size: int = 500,
                 chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict]:
        words = text.split()
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            chunk_doc = {
                "text": chunk_text,
                "chunk_index": chunk_idx,
                "start_word": start,
                "end_word": end,
                **(metadata or {}),
            }
            chunks.append(chunk_doc)
            
            start = end - self.chunk_overlap
            chunk_idx += 1
            
            if end == len(words):
                break
        
        return chunks


class SimpleEmbedder:
    """Simple bag-of-words embedder for demonstration."""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
    
    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        words = text.lower().split()
        
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(self.dimension):
                idx = (h + i * 31) % self.dimension
                vector[idx] += 1.0
        
        # Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector


class RAGPipeline:
    """Complete RAG pipeline."""
    
    def __init__(self, embedder: SimpleEmbedder,
                 vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = DocumentChunker()
    
    def ingest(self, document: str, metadata: Dict[str, Any] = None):
        chunks = self.chunker.chunk(document, metadata)
        for chunk in chunks:
            vector = self.embedder.embed(chunk["text"])
            self.vector_store.add(vector, chunk)
    
    def retrieve(self, query: str,
                 top_k: int = 3) -> List[Dict]:
        query_vector = self.embedder.embed(query)
        results = self.vector_store.search(query_vector, top_k)
        return [{"score": score, **doc} for score, doc in results]
    
    def build_prompt(self, query: str,
                     context_docs: List[Dict],
                     system_prompt: str = None) -> str:
        context_text = "\n\n".join(
            doc["text"] for doc in context_docs)
        
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"
        
        prompt += f"Context:\n{context_text}\n\n"
        prompt += f"Question: {query}\n\nAnswer:"
        return prompt
    
    def query(self, question: str, top_k: int = 3,
              system_prompt: str = None) -> Dict:
        retrieved = self.retrieve(question, top_k)
        prompt = self.build_prompt(question, retrieved, system_prompt)
        
        return {
            "prompt": prompt,
            "retrieved_docs": retrieved,
            "num_chunks": len(retrieved),
        }


# ============================================================
# LoRA Adapter (Simplified)
# ============================================================

class LoRAAdapter:
    """Low-Rank Adaptation for fine-tuning."""
    
    def __init__(self, d_in: int, d_out: int, rank: int = 4,
                 alpha: float = 1.0):
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        scale_a = math.sqrt(1.0 / d_in)
        scale_b = 0.0  # Initialize B to zero
        
        self.A = [[random.gauss(0, scale_a) for _ in range(rank)]
                   for _ in range(d_in)]
        self.B = [[scale_b for _ in range(d_out)]
                   for _ in range(rank)]
    
    def forward(self, x: List[float]) -> List[float]:
        """Compute LoRA offset: scaling * x @ A @ B"""
        # x @ A
        hidden = [sum(x[i] * self.A[i][j] for i in range(len(x)))
                  for j in range(self.rank)]
        
        # hidden @ B
        output = [sum(hidden[i] * self.B[i][j] for i in range(self.rank))
                  for j in range(len(self.B[0]))]
        
        # Scale
        return [v * self.scaling for v in output]
    
    @property
    def num_parameters(self) -> int:
        return len(self.A) * self.rank + self.rank * len(self.B[0])


class LoRALinear:
    """Linear layer with LoRA adapter."""
    
    def __init__(self, weight: List[List[float]], rank: int = 4):
        self.weight = weight  # Frozen
        d_in = len(weight)
        d_out = len(weight[0])
        self.lora = LoRAAdapter(d_in, d_out, rank)
    
    def forward(self, x: List[float]) -> List[float]:
        # Original forward
        base_out = [sum(x[i] * self.weight[i][j]
                       for i in range(len(x)))
                   for j in range(len(self.weight[0]))]
        
        # LoRA offset
        lora_out = self.lora.forward(x)
        
        # Combined
        return [base_out[j] + lora_out[j]
                for j in range(len(base_out))]`,
				},
			},
		},
	})
}
