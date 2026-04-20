package machinelearning

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterMachineLearningModules([]problems.CourseModule{
		{
			ID:          2517,
			Title:       "Convolutional and Recurrent Neural Networks",
			Description: "Master CNNs for computer vision, RNNs and LSTMs for sequential data, attention mechanisms, and practical architectures for image and text processing.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "CNNs RNNs LSTMs and Attention Mechanisms",
					Content: `Convolutional and recurrent neural networks are specialized architectures for processing structured data like images and sequences.

**Convolutional Neural Networks (CNNs):**

Convolution Operation:
  Filter/kernel slides across input
  Element-wise multiplication and sum
  Produces feature map
  
  Parameters:
    Kernel size: 3×3, 5×5, 7×7 (spatial dimensions)
    Stride: Step size for sliding (1 = every position, 2 = skip one)
    Padding: Add zeros around input border
      Valid: No padding (output smaller)
      Same: Pad to keep spatial dimensions
    Dilation: Spacing between kernel elements (dilated/atrous convolution)
  
  Output size:
    out = (input + 2*padding - kernel) / stride + 1
    
  Parameter sharing:
    Same filter weights used across all spatial positions
    Translation invariance
    Much fewer parameters than fully connected

Pooling:
  Max pooling: Take maximum value in window
    Most common: 2×2 with stride 2 (halves spatial dimensions)
    Provides translation invariance
    
  Average pooling: Take mean value in window
    Smoother, preserves background info
    
  Global average pooling: Single value per feature map
    Used before final classification layer
    Reduces parameters dramatically

CNN Architectures:

LeNet-5 (1998):
  Conv → Pool → Conv → Pool → FC → FC → Output
  First successful CNN for digit recognition
  
AlexNet (2012):
  5 Conv + 3 FC layers
  ReLU activation, dropout
  GPU training
  Won ImageNet 2012

VGGNet (2014):
  Very deep (16-19 layers)
  Only 3×3 convolutions
  Two 3×3 = effective 5×5 with fewer parameters
  
GoogLeNet/Inception (2014):
  Inception module: Parallel 1×1, 3×3, 5×5 convolutions
  1×1 convolutions for dimensionality reduction
  Auxiliary classifiers for gradient flow
  
ResNet (2015):
  Residual connections: y = F(x) + x (skip connections)
  Solve vanishing gradient in very deep networks
  50, 101, 152 layers
  Bottleneck block: 1×1 → 3×3 → 1×1
  
  Why residual works:
    Identity mapping is easy to learn (F(x) = 0)
    Gradient flows directly through skip connections
    Ensemble-like behavior

DenseNet (2017):
  Every layer connected to every other layer
  Feature reuse
  Fewer parameters than ResNet

EfficientNet (2019):
  Compound scaling: Balance depth, width, resolution
  Neural Architecture Search (NAS)
  State-of-the-art efficiency

Modern CNNs:
  ConvNeXt: Modernized ResNet with Transformer techniques
  Vision Transformer (ViT): Apply Transformer directly to image patches

**1×1 Convolutions:**
  Channel-wise linear combination
  Dimensionality reduction/expansion
  Cross-channel interaction
  Adds non-linearity (with activation)
  Network-in-Network concept

**Depthwise Separable Convolution:**
  Standard: K×K×C_in×C_out parameters
  Depthwise: K×K×1 per channel (spatial)
  Pointwise: 1×1×C_in×C_out (channel mixing)
  Total: K×K×C_in + C_in×C_out
  ~8-9x fewer parameters (for 3×3)
  Used in MobileNet, EfficientNet

**Recurrent Neural Networks (RNNs):**

Basic RNN:
  h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
  y_t = W_hy * h_t + b_y
  
  Hidden state h carries information across time steps
  Same weights shared across all time steps
  
  Problems:
    Vanishing gradients: Hard to learn long-range dependencies
    Exploding gradients: Gradients grow exponentially

LSTM (Long Short-Term Memory):
  Cell state: Long-term memory highway
  Gates control information flow:
  
  Forget gate: f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
    Decides what to forget from cell state
    
  Input gate: i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
    Decides what new info to store
    
  Cell candidate: c_tilde = tanh(W_c * [h_{t-1}, x_t] + b_c)
    New candidate values
    
  Cell update: c_t = f_t * c_{t-1} + i_t * c_tilde
    Remove old + add new
    
  Output gate: o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
    Decides what to output
    
  Hidden state: h_t = o_t * tanh(c_t)
  
  Why LSTM works:
    Cell state gradient flows through additions (not multiplications)
    Gates learn to preserve important long-term information
    Forget gate can keep gradient flowing for many time steps

GRU (Gated Recurrent Unit):
  Simplified LSTM with fewer parameters
  Two gates instead of three:
  
  Reset gate: r_t = sigmoid(W_r * [h_{t-1}, x_t])
  Update gate: z_t = sigmoid(W_z * [h_{t-1}, x_t])
  Candidate: h_tilde = tanh(W * [r_t * h_{t-1}, x_t])
  Output: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
  
  Often comparable performance to LSTM
  Faster to train (fewer parameters)

Bidirectional RNN:
  Two RNNs: forward and backward through sequence
  Output is concatenation of both directions
  Captures context from both sides
  Use: Text classification, NER, machine translation

Sequence-to-Sequence:
  Encoder: Read input sequence → context vector
  Decoder: Generate output sequence from context
  Problem: Bottleneck in fixed-size context vector
  Solution: Attention mechanism

**Attention Mechanism:**

Attention concept:
  Instead of compressing to single vector, attend to all encoder states
  Query (from decoder) × Keys (from encoder) → Attention weights
  Weighted sum of Values (from encoder) → Context vector
  
  score(query, key) = alignment function:
    Dot product: q^T * k
    Scaled dot product: q^T * k / sqrt(d_k)
    Additive: v^T * tanh(W1*q + W2*k)

Self-Attention:
  Query, Key, Value all come from same sequence
  Each position attends to all positions
  Captures dependencies regardless of distance
  
  Computation:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Multi-Head Attention:
  Multiple attention heads with different projections
  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
  MultiHead = Concat(head_1, ..., head_h) * W^O
  
  Each head can learn different types of relationships:
    Head 1: Syntactic relationships
    Head 2: Semantic relationships
    Head 3: Positional relationships

**Transformer Architecture:**

Encoder:
  Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
  Repeated N times (typically 6-12)
  
Decoder:
  Masked Multi-Head Self-Attention → Add & Norm → 
  Cross-Attention (attend to encoder) → Add & Norm → FFN → Add & Norm
  Masked attention prevents looking at future tokens

Positional Encoding:
  Sinusoidal: PE(pos, 2i) = sin(pos / 10000^(2i/d))
  Learned positional embeddings
  Rotary Position Embedding (RoPE): Used in modern LLMs

Feed-Forward Network:
  FFN(x) = GELU(xW1 + b1)W2 + b2
  Hidden dimension typically 4x model dimension
  Applied position-wise

Key properties:
  Parallel processing (unlike RNN)
  O(n^2) attention complexity (sequence length)
  Constant path length between positions

Transformer variants:
  Encoder-only: BERT (classification, NER)
  Decoder-only: GPT (text generation)
  Encoder-decoder: T5, BART (translation, summarization)

**Transfer Learning for CV and NLP:**

Computer Vision:
  Pretrained on ImageNet (1M+ images, 1000 classes)
  Fine-tune: Replace last layer, train with small lr
  Feature extraction: Freeze backbone, train classifier
  Common backbones: ResNet, EfficientNet, ViT

NLP:
  Pretrained language models (BERT, GPT)
  Self-supervised pretraining on large text corpora
  Fine-tune: Add task-specific head, train end-to-end
  Few-shot: Task instruction + examples in prompt
  Zero-shot: Task instruction only`,
					CodeExamples: `# CNN, RNN, LSTM, and Attention Implementation

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================
# Convolution Operations
# ============================================================

def conv2d(input_data: List[List[float]], kernel: List[List[float]],
           stride: int = 1, padding: int = 0) -> List[List[float]]:
    """2D convolution operation."""
    h_in = len(input_data)
    w_in = len(input_data[0])
    k_h = len(kernel)
    k_w = len(kernel[0])
    
    # Pad input
    if padding > 0:
        padded = [[0.0] * (w_in + 2 * padding)
                  for _ in range(h_in + 2 * padding)]
        for i in range(h_in):
            for j in range(w_in):
                padded[i + padding][j + padding] = input_data[i][j]
    else:
        padded = input_data
    
    h_pad = len(padded)
    w_pad = len(padded[0])
    h_out = (h_pad - k_h) // stride + 1
    w_out = (w_pad - k_w) // stride + 1
    
    output = [[0.0] * w_out for _ in range(h_out)]
    
    for i in range(h_out):
        for j in range(w_out):
            total = 0.0
            for ki in range(k_h):
                for kj in range(k_w):
                    total += padded[i * stride + ki][j * stride + kj] * kernel[ki][kj]
            output[i][j] = total
    
    return output


def max_pool2d(input_data: List[List[float]], pool_size: int = 2,
               stride: int = 2) -> List[List[float]]:
    """2D max pooling."""
    h_in = len(input_data)
    w_in = len(input_data[0])
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    
    output = [[0.0] * w_out for _ in range(h_out)]
    
    for i in range(h_out):
        for j in range(w_out):
            max_val = float('-inf')
            for pi in range(pool_size):
                for pj in range(pool_size):
                    val = input_data[i * stride + pi][j * stride + pj]
                    max_val = max(max_val, val)
            output[i][j] = max_val
    
    return output


def avg_pool2d(input_data: List[List[float]], pool_size: int = 2,
               stride: int = 2) -> List[List[float]]:
    """2D average pooling."""
    h_in = len(input_data)
    w_in = len(input_data[0])
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    
    output = [[0.0] * w_out for _ in range(h_out)]
    
    for i in range(h_out):
        for j in range(w_out):
            total = 0.0
            count = pool_size * pool_size
            for pi in range(pool_size):
                for pj in range(pool_size):
                    total += input_data[i * stride + pi][j * stride + pj]
            output[i][j] = total / count
    
    return output


def global_avg_pool(input_data: List[List[float]]) -> float:
    """Global average pooling."""
    total = sum(sum(row) for row in input_data)
    count = len(input_data) * len(input_data[0])
    return total / count


# ============================================================
# CNN Layer
# ============================================================

class Conv2DLayer:
    """Convolutional layer with multiple filters."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize filters with He initialization
        scale = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.filters = [
            [[[random.gauss(0, scale) for _ in range(kernel_size)]
              for _ in range(kernel_size)]
             for _ in range(in_channels)]
            for _ in range(out_channels)
        ]
        self.biases = [0.0] * out_channels
    
    def forward(self, input_channels: List[List[List[float]]]) -> List[List[List[float]]]:
        output = []
        for f in range(self.out_channels):
            feature_map = None
            for c in range(self.in_channels):
                conv_result = conv2d(
                    input_channels[c], self.filters[f][c],
                    self.stride, self.padding)
                if feature_map is None:
                    feature_map = conv_result
                else:
                    for i in range(len(feature_map)):
                        for j in range(len(feature_map[0])):
                            feature_map[i][j] += conv_result[i][j]
            
            # Add bias and apply ReLU
            for i in range(len(feature_map)):
                for j in range(len(feature_map[0])):
                    feature_map[i][j] = max(0, feature_map[i][j] + self.biases[f])
            
            output.append(feature_map)
        return output


# ============================================================
# RNN Implementation
# ============================================================

class RNNCell:
    """Basic RNN cell."""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        scale = math.sqrt(1.0 / hidden_size)
        self.W_xh = [[random.gauss(0, scale) for _ in range(hidden_size)]
                      for _ in range(input_size)]
        self.W_hh = [[random.gauss(0, scale) for _ in range(hidden_size)]
                      for _ in range(hidden_size)]
        self.b_h = [0.0] * hidden_size
    
    def forward(self, x: List[float],
                h_prev: List[float]) -> List[float]:
        h_new = [0.0] * self.hidden_size
        
        for j in range(self.hidden_size):
            val = self.b_h[j]
            for i in range(self.input_size):
                val += x[i] * self.W_xh[i][j]
            for i in range(self.hidden_size):
                val += h_prev[i] * self.W_hh[i][j]
            h_new[j] = math.tanh(val)
        
        return h_new
    
    def forward_sequence(self, sequence: List[List[float]]) -> Tuple[
            List[List[float]], List[float]]:
        h = [0.0] * self.hidden_size
        outputs = []
        
        for x in sequence:
            h = self.forward(x, h)
            outputs.append(h[:])
        
        return outputs, h


# ============================================================
# LSTM Implementation
# ============================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


class LSTMCell:
    """LSTM cell with forget, input, and output gates."""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        scale = math.sqrt(1.0 / hidden_size)
        
        # Forget gate weights
        self.W_f = self._init_weights(input_size + hidden_size, hidden_size, scale)
        self.b_f = [1.0] * hidden_size  # Initialize forget gate bias to 1
        
        # Input gate weights
        self.W_i = self._init_weights(input_size + hidden_size, hidden_size, scale)
        self.b_i = [0.0] * hidden_size
        
        # Cell candidate weights
        self.W_c = self._init_weights(input_size + hidden_size, hidden_size, scale)
        self.b_c = [0.0] * hidden_size
        
        # Output gate weights
        self.W_o = self._init_weights(input_size + hidden_size, hidden_size, scale)
        self.b_o = [0.0] * hidden_size
    
    def _init_weights(self, rows, cols, scale):
        return [[random.gauss(0, scale) for _ in range(cols)]
                for _ in range(rows)]
    
    def _gate(self, combined, W, b, activation):
        result = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            val = b[j]
            for i in range(len(combined)):
                val += combined[i] * W[i][j]
            result[j] = activation(val)
        return result
    
    def forward(self, x: List[float], h_prev: List[float],
                c_prev: List[float]) -> Tuple[List[float], List[float]]:
        combined = x + h_prev
        
        f_gate = self._gate(combined, self.W_f, self.b_f, sigmoid)
        i_gate = self._gate(combined, self.W_i, self.b_i, sigmoid)
        c_cand = self._gate(combined, self.W_c, self.b_c, math.tanh)
        o_gate = self._gate(combined, self.W_o, self.b_o, sigmoid)
        
        c_new = [f_gate[j] * c_prev[j] + i_gate[j] * c_cand[j]
                 for j in range(self.hidden_size)]
        h_new = [o_gate[j] * math.tanh(c_new[j])
                 for j in range(self.hidden_size)]
        
        return h_new, c_new
    
    def forward_sequence(self, sequence: List[List[float]]) -> Tuple[
            List[List[float]], List[float], List[float]]:
        h = [0.0] * self.hidden_size
        c = [0.0] * self.hidden_size
        outputs = []
        
        for x in sequence:
            h, c = self.forward(x, h, c)
            outputs.append(h[:])
        
        return outputs, h, c


# ============================================================
# GRU Implementation
# ============================================================

class GRUCell:
    """Gated Recurrent Unit cell."""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        scale = math.sqrt(1.0 / hidden_size)
        
        self.W_z = self._init(input_size + hidden_size, hidden_size, scale)
        self.b_z = [0.0] * hidden_size
        
        self.W_r = self._init(input_size + hidden_size, hidden_size, scale)
        self.b_r = [0.0] * hidden_size
        
        self.W_h = self._init(input_size + hidden_size, hidden_size, scale)
        self.b_h = [0.0] * hidden_size
    
    def _init(self, rows, cols, scale):
        return [[random.gauss(0, scale) for _ in range(cols)]
                for _ in range(rows)]
    
    def _gate(self, inp, W, b, act):
        result = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            val = b[j]
            for i in range(len(inp)):
                val += inp[i] * W[i][j]
            result[j] = act(val)
        return result
    
    def forward(self, x: List[float],
                h_prev: List[float]) -> List[float]:
        combined = x + h_prev
        
        z = self._gate(combined, self.W_z, self.b_z, sigmoid)
        r = self._gate(combined, self.W_r, self.b_r, sigmoid)
        
        reset_h = [r[j] * h_prev[j] for j in range(self.hidden_size)]
        combined_r = x + reset_h
        h_cand = self._gate(combined_r, self.W_h, self.b_h, math.tanh)
        
        h_new = [(1 - z[j]) * h_prev[j] + z[j] * h_cand[j]
                 for j in range(self.hidden_size)]
        
        return h_new


# ============================================================
# Attention Mechanism
# ============================================================

def softmax(scores: List[float]) -> List[float]:
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def dot_product_attention(query: List[float], keys: List[List[float]],
                          values: List[List[float]]) -> List[float]:
    """Scaled dot-product attention."""
    d_k = len(query)
    scale = math.sqrt(d_k)
    
    scores = [sum(query[i] * key[i] for i in range(d_k)) / scale
              for key in keys]
    
    weights = softmax(scores)
    
    d_v = len(values[0])
    context = [0.0] * d_v
    for i, weight in enumerate(weights):
        for j in range(d_v):
            context[j] += weight * values[i][j]
    
    return context


class MultiHeadAttention:
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        scale = math.sqrt(1.0 / d_model)
        self.W_q = [[random.gauss(0, scale) for _ in range(d_model)]
                     for _ in range(d_model)]
        self.W_k = [[random.gauss(0, scale) for _ in range(d_model)]
                     for _ in range(d_model)]
        self.W_v = [[random.gauss(0, scale) for _ in range(d_model)]
                     for _ in range(d_model)]
        self.W_o = [[random.gauss(0, scale) for _ in range(d_model)]
                     for _ in range(d_model)]
    
    def _linear(self, x: List[float], W: List[List[float]]) -> List[float]:
        return [sum(x[i] * W[i][j] for i in range(len(x)))
                for j in range(len(W[0]))]
    
    def forward(self, queries: List[List[float]],
                keys: List[List[float]],
                values: List[List[float]],
                mask: List[List[bool]] = None) -> List[List[float]]:
        batch_q = [self._linear(q, self.W_q) for q in queries]
        batch_k = [self._linear(k, self.W_k) for k in keys]
        batch_v = [self._linear(v, self.W_v) for v in values]
        
        all_head_outputs = []
        for q_idx, q in enumerate(batch_q):
            head_outputs = []
            
            for h in range(self.num_heads):
                start = h * self.d_k
                end = start + self.d_k
                
                q_h = q[start:end]
                k_h = [k[start:end] for k in batch_k]
                v_h = [v[start:end] for v in batch_v]
                
                context = dot_product_attention(q_h, k_h, v_h)
                head_outputs.extend(context)
            
            output = self._linear(head_outputs, self.W_o)
            all_head_outputs.append(output)
        
        return all_head_outputs


# ============================================================
# Positional Encoding
# ============================================================

def sinusoidal_encoding(position: int, d_model: int) -> List[float]:
    """Sinusoidal positional encoding."""
    encoding = [0.0] * d_model
    for i in range(0, d_model, 2):
        denominator = math.pow(10000, 2 * i / d_model)
        encoding[i] = math.sin(position / denominator)
        if i + 1 < d_model:
            encoding[i + 1] = math.cos(position / denominator)
    return encoding


def get_positional_encodings(max_len: int, d_model: int) -> List[List[float]]:
    """Generate positional encodings for all positions."""
    return [sinusoidal_encoding(pos, d_model) for pos in range(max_len)]


# ============================================================
# Transformer Encoder Block
# ============================================================

class TransformerEncoderBlock:
    """Single transformer encoder block."""
    
    def __init__(self, d_model: int, num_heads: int,
                 d_ff: int, dropout_rate: float = 0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.d_model = d_model
        self.d_ff = d_ff
        
        scale = math.sqrt(1.0 / d_model)
        self.W1 = [[random.gauss(0, scale) for _ in range(d_ff)]
                    for _ in range(d_model)]
        self.b1 = [0.0] * d_ff
        self.W2 = [[random.gauss(0, math.sqrt(1.0 / d_ff))
                     for _ in range(d_model)]
                    for _ in range(d_ff)]
        self.b2 = [0.0] * d_model
    
    def _ffn(self, x: List[float]) -> List[float]:
        hidden = [max(0, sum(x[i] * self.W1[i][j] for i in range(self.d_model)) + self.b1[j])
                  for j in range(self.d_ff)]
        output = [sum(hidden[i] * self.W2[i][j] for i in range(self.d_ff)) + self.b2[j]
                  for j in range(self.d_model)]
        return output
    
    def _layer_norm(self, x: List[float], eps: float = 1e-5) -> List[float]:
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)
        std = math.sqrt(var + eps)
        return [(xi - mean) / std for xi in x]
    
    def _add_and_norm(self, x: List[float],
                      sublayer_out: List[float]) -> List[float]:
        residual = [x[i] + sublayer_out[i] for i in range(len(x))]
        return self._layer_norm(residual)
    
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        attn_out = self.attention.forward(x, x, x)
        normed = [self._add_and_norm(x[i], attn_out[i])
                  for i in range(len(x))]
        
        ffn_out = [self._ffn(token) for token in normed]
        output = [self._add_and_norm(normed[i], ffn_out[i])
                  for i in range(len(normed))]
        
        return output


# ============================================================
# Sequence-to-Sequence with Attention
# ============================================================

class Seq2SeqWithAttention:
    """Sequence-to-sequence model with attention."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int):
        self.encoder = LSTMCell(input_size, hidden_size)
        self.decoder = LSTMCell(output_size + hidden_size, hidden_size)
        self.hidden_size = hidden_size
        
        scale = math.sqrt(1.0 / hidden_size)
        self.W_out = [[random.gauss(0, scale) for _ in range(output_size)]
                       for _ in range(hidden_size)]
    
    def encode(self, input_seq: List[List[float]]) -> Tuple[
            List[List[float]], List[float], List[float]]:
        return self.encoder.forward_sequence(input_seq)
    
    def attention(self, decoder_hidden: List[float],
                  encoder_outputs: List[List[float]]) -> List[float]:
        return dot_product_attention(
            decoder_hidden, encoder_outputs, encoder_outputs)
    
    def decode_step(self, input_token: List[float],
                    h: List[float], c: List[float],
                    encoder_outputs: List[List[float]]) -> Tuple[
                        List[float], List[float], List[float]]:
        context = self.attention(h, encoder_outputs)
        combined = input_token + context
        
        h_new, c_new = self.decoder.forward(combined, h, c)
        
        output = [sum(h_new[i] * self.W_out[i][j]
                      for i in range(self.hidden_size))
                  for j in range(len(self.W_out[0]))]
        
        return output, h_new, c_new`,
				},
			},
		},
	})
}
