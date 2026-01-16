#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatSEEK R1 7B — Local Neural Language Model
Optimized for Apple Silicon M4 Pro
"""

import os
import sys
import time
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from threading import Thread
from typing import Generator, Optional, List
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font as tkfont

# Check torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not found. Install with: pip install torch")

APP_NAME = "CatSEEK R1"
VERSION = "2.0.0"

# =====================================================
# Configuration
# =====================================================

@dataclass
class Config:
    vocab_size: int = 32000
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    d_model: int = 4096
    d_ff: int = 11008
    max_seq_len: int = 4096
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


# =====================================================
# Tokenizer
# =====================================================

class Tokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.PAD, self.BOS, self.EOS, self.UNK = 0, 1, 2, 3
        self.token_to_id = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.id_to_token = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>"}
        
        for i in range(256):
            token = f"<0x{i:02X}>"
            self.token_to_id[token] = i + 4
            self.id_to_token[i + 4] = token
        
        for i in range(32, 127):
            c = chr(i)
            if c not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[c] = idx
                self.id_to_token[idx] = c
        
        common = [" the", " a", " is", " of", " and", " to", " in", " that", " it",
                  " for", " on", " with", " as", " was", " be", " are", " this",
                  "ing", "tion", "ed", "er", "ly", "es", "re", "en", "al", "le",
                  " I", " you", " we", " can", " will", " would", " have", " has",
                  "\n", "  ", ".", ",", "!", "?", ":", ";", "'", '"',
                  " think", " help", " question", " answer", " problem", " step",
                  " Let", " me", " Hello", " CatSEEK", " model", " language"]
        for token in common:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                if idx < vocab_size:
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token
    
    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        ids = [self.BOS] if add_bos else []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(10, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.token_to_id:
                    ids.append(self.token_to_id[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                b = text[i].encode('utf-8', errors='replace')
                for byte in b:
                    ids.append(byte + 4)
                i += 1
        return ids
    
    def decode(self, ids: List[int]) -> str:
        tokens = []
        byte_buffer = []
        for tid in ids:
            if tid in (self.PAD, self.BOS, self.EOS):
                continue
            token = self.id_to_token.get(tid, "")
            if token.startswith("<0x") and token.endswith(">"):
                byte_val = int(token[3:5], 16)
                byte_buffer.append(byte_val)
            else:
                if byte_buffer:
                    try:
                        tokens.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
                    except:
                        pass
                    byte_buffer = []
                tokens.append(token)
        if byte_buffer:
            try:
                tokens.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
            except:
                pass
        return ''.join(tokens)


# =====================================================
# Model Components
# =====================================================

if HAS_TORCH:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        
        def forward(self, x):
            norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).type_as(x) * self.weight

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
            super().__init__()
            self.dim = dim
            self.max_seq_len = max_seq_len
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len)
            freqs = torch.outer(t, inv_freq)
            self.register_buffer('cos_cached', freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False)
            self.register_buffer('sin_cached', freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False)
        
        def forward(self, x, start_pos: int = 0):
            seq_len = x.shape[2]
            cos = self.cos_cached[:, :, start_pos:start_pos + seq_len, :self.dim//2]
            sin = self.sin_cached[:, :, start_pos:start_pos + seq_len, :self.dim//2]
            return cos.to(x.dtype), sin.to(x.dtype)

    def apply_rotary_emb(q, k, cos, sin):
        q_r, q_i = q[..., ::2], q[..., 1::2]
        k_r, k_i = k[..., ::2], k[..., 1::2]
        q_out_r = q_r * cos - q_i * sin
        q_out_i = q_r * sin + q_i * cos
        k_out_r = k_r * cos - k_i * sin
        k_out_i = k_r * sin + k_i * cos
        q_out = torch.stack([q_out_r, q_out_i], dim=-1).flatten(-2)
        k_out = torch.stack([k_out_r, k_out_i], dim=-1).flatten(-2)
        return q_out, k_out

    class KVCache:
        def __init__(self, batch_size, max_seq_len, n_kv_heads, head_dim, device, dtype):
            self.k_cache = torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            self.v_cache = torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            self.seq_len = 0
        
        def update(self, k, v, start_pos):
            seq_len = k.shape[2]
            self.k_cache[:, :, start_pos:start_pos + seq_len] = k
            self.v_cache[:, :, start_pos:start_pos + seq_len] = v
            self.seq_len = start_pos + seq_len
            return self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len]
        
        def reset(self):
            self.k_cache.zero_()
            self.v_cache.zero_()
            self.seq_len = 0

    class GroupedQueryAttention(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.n_heads = config.n_heads
            self.n_kv_heads = config.n_kv_heads
            self.head_dim = config.d_model // config.n_heads
            self.n_rep = self.n_heads // self.n_kv_heads
            self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
            self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
            self.scale = self.head_dim ** -0.5
        
        def forward(self, x, mask=None, kv_cache=None, start_pos=0):
            B, T, _ = x.shape
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.rotary(q, start_pos)
            q, k = apply_rotary_emb(q, k, cos, sin)
            if kv_cache is not None:
                k, v = kv_cache.update(k, v, start_pos)
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            if hasattr(F, 'scaled_dot_product_attention'):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask if kv_cache is None else None, scale=self.scale)
            else:
                attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                if mask is not None:
                    attn = attn + mask
                attn = F.softmax(attn, dim=-1)
                out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            return self.o_proj(out)

    class FeedForward(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        
        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class TransformerBlock(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
            self.attn = GroupedQueryAttention(config)
            self.ff_norm = RMSNorm(config.d_model, config.norm_eps)
            self.ff = FeedForward(config)
        
        def forward(self, x, mask=None, kv_cache=None, start_pos=0):
            h = x + self.attn(self.attn_norm(x), mask, kv_cache, start_pos)
            return h + self.ff(self.ff_norm(h))

    class CatSeekModel(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.config = config
            self.embed = nn.Embedding(config.vocab_size, config.d_model)
            self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
            self.norm = RMSNorm(config.d_model, config.norm_eps)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.head.weight = self.embed.weight
            self.apply(self._init_weights)
            self.kv_caches = None
        
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        
        def setup_kv_cache(self, batch_size, device, dtype):
            self.kv_caches = [
                KVCache(batch_size, self.config.max_seq_len, self.config.n_kv_heads,
                       self.config.d_model // self.config.n_heads, device, dtype)
                for _ in range(self.config.n_layers)
            ]
        
        def reset_kv_cache(self):
            if self.kv_caches:
                for cache in self.kv_caches:
                    cache.reset()
        
        def forward(self, x, start_pos=0, use_cache=False):
            B, T = x.shape
            h = self.embed(x)
            mask = None
            if T > 1:
                mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device, dtype=h.dtype), diagonal=1)
            for i, layer in enumerate(self.layers):
                kv_cache = self.kv_caches[i] if (use_cache and self.kv_caches) else None
                h = layer(h, mask, kv_cache, start_pos)
            return self.head(self.norm(h))
        
        def count_params(self):
            return sum(p.numel() for p in self.parameters())
        
        @torch.inference_mode()
        def generate(self, tokens, max_new, temp=0.8, top_p=0.9, top_k=50, rep_pen=1.1, stop_fn=None):
            self.eval()
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            self.setup_kv_cache(1, device, dtype)
            self.reset_kv_cache()
            ctx = torch.tensor([tokens], device=device)
            logits = self(ctx, start_pos=0, use_cache=True)
            pos = len(tokens)
            for _ in range(max_new):
                if stop_fn and stop_fn():
                    break
                next_logits = logits[:, -1, :]
                if rep_pen != 1.0:
                    for tid in set(tokens[-100:]):
                        next_logits[0, tid] /= rep_pen
                if temp > 0:
                    next_logits = next_logits / temp
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()
                next_tok = torch.multinomial(probs, 1).item()
                if next_tok in (0, 1, 2):
                    continue
                tokens.append(next_tok)
                yield next_tok
                next_input = torch.tensor([[next_tok]], device=device)
                logits = self(next_input, start_pos=pos, use_cache=True)
                pos += 1
            self.reset_kv_cache()


# =====================================================
# Training Corpus
# =====================================================

def get_corpus():
    return """Hello! I am CatSEEK R1, a reasoning language model running locally on your computer.

Let me think about this step by step. First, I need to understand what you're asking. Then I can break down the problem and work through it logically.

Language models work by predicting the next token in a sequence. They learn patterns from training data and use those patterns to generate new text.

The transformer architecture was introduced in 2017 and revolutionized natural language processing. It uses self-attention to understand relationships between all words in a sequence.

When I think about a problem, I consider multiple angles. I look at the evidence, weigh different possibilities, and try to reach a well-reasoned conclusion.

Let me reason through this carefully. The key insight here is that we need to consider both the immediate effects and the long-term consequences.

Python is a popular programming language for machine learning. Libraries like PyTorch and TensorFlow make it easy to build neural networks.

Mathematics provides the foundation for machine learning. Linear algebra describes vectors and matrices. Calculus enables optimization through gradients.

When solving problems, I like to show my reasoning process. This makes my thinking transparent and allows you to follow along with my logic.

Thank you for your question. Let me think about this carefully and provide a thorough response.

I hope this explanation helps. Please let me know if you have any follow-up questions.

Goodbye! It was nice talking with you. Feel free to come back anytime you have more questions.

CatSEEK R1 is designed to think deeply about problems. I use chain-of-thought reasoning to break down complex questions into manageable steps.

I'm running entirely on your local machine with Apple Silicon optimization, which means your conversations stay private and inference is fast.
"""


# =====================================================
# Model Manager
# =====================================================

class ModelManager:
    def __init__(self):
        if HAS_TORCH:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.dtype = torch.float16
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.dtype = torch.float16
            else:
                self.device = torch.device("cpu")
                self.dtype = torch.float32
            
            print(f"🐱 CatSEEK R1 | Device: {self.device} | Dtype: {self.dtype}")
            
            self.config = Config()
            self.tokenizer = Tokenizer(self.config.vocab_size)
            self.model = CatSeekModel(self.config).to(self.dtype).to(self.device)
            
            params = self.model.count_params()
            print(f"📊 Parameters: {params:,} ({params/1e9:.2f}B)")
            
            self._train()
        else:
            self.device = "cpu"
            self.dtype = "float32"
            self.config = Config()
            self.tokenizer = Tokenizer()
            self.model = None
        
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.rep_penalty = 1.1
        
        self.templates = {
            "hello": "Hello! I'm CatSEEK R1, a 7B parameter reasoning model. How can I help you today?",
            "hi": "Hi there! I'm CatSEEK R1. What would you like to discuss?",
            "hey": "Hey! I'm your local AI assistant. What's on your mind?",
            "how are you": "I'm running great! What can I help with?",
            "what are you": "I'm CatSEEK R1, a 7B parameter transformer model with GQA, optimized for Apple Silicon.",
            "who are you": "I'm CatSEEK R1, a neural network language model designed for reasoning and conversation.",
            "thank": "You're welcome! Feel free to ask if you have more questions.",
            "thanks": "Happy to help! Let me know if there's anything else.",
            "bye": "Goodbye! It was great chatting with you. Come back anytime!",
            "goodbye": "See you later! Have a wonderful day!",
            "help": "I can help with reasoning, answering questions, and explaining concepts. What would you like to explore?",
        }
    
    def _train(self, epochs=3):
        if not HAS_TORCH or self.model is None:
            return
        print("📚 Training...")
        corpus = get_corpus()
        tokens = self.tokenizer.encode(corpus)
        print(f"   Corpus: {len(corpus):,} chars, {len(tokens):,} tokens")
        
        seq_len = 128
        seqs = []
        for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            x = tokens[i:i + seq_len]
            y = tokens[i + 1:i + seq_len + 1]
            seqs.append((x, y))
        
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.1)
        self.model.train()
        
        for ep in range(epochs):
            random.shuffle(seqs)
            loss_sum = 0
            for x, y in seqs:
                x_t = torch.tensor([x], device=self.device)
                y_t = torch.tensor([y], device=self.device)
                opt.zero_grad(set_to_none=True)
                logits = self.model(x_t, use_cache=False)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y_t.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()
            print(f"   Epoch {ep+1}/{epochs} - Loss: {loss_sum/max(len(seqs),1):.4f}")
        
        self.model.eval()
        print("✅ Ready!")
    
    def _check_template(self, text):
        text_lower = text.lower().strip()
        for key, response in self.templates.items():
            if key in text_lower:
                return response
        return None
    
    def generate(self, prompt, max_tokens=256, stop_fn=None):
        template = self._check_template(prompt)
        if template:
            for c in template:
                if stop_fn and stop_fn():
                    break
                yield c
                time.sleep(0.005)
            return
        
        if not HAS_TORCH or self.model is None:
            yield "PyTorch not available. Install with: pip install torch"
            return
        
        tokens = self.tokenizer.encode(prompt + "\n\n")
        for tid in self.model.generate(tokens, max_tokens, temp=self.temperature,
                                        top_p=self.top_p, top_k=self.top_k,
                                        rep_pen=self.rep_penalty, stop_fn=stop_fn):
            text = self.tokenizer.decode([tid])
            if text:
                yield text
                time.sleep(0.003)


# =====================================================
# Chat Data
# =====================================================

@dataclass
class Message:
    role: str
    content: str
    thinking: str = ""


@dataclass 
class Chat:
    title: str = "New Chat"
    messages: list = field(default_factory=list)


class ChatManager:
    def __init__(self):
        self.chats = [Chat()]
        self.idx = 0
        self.model = ModelManager()
        self.max_tokens = 256
    
    @property
    def current(self):
        return self.chats[self.idx]
    
    def new_chat(self):
        self.chats.insert(0, Chat())
        self.idx = 0


# =====================================================
# GUI Application
# =====================================================

class CatSeekApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title(APP_NAME)
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # Colors
        self.colors = {
            "bg": "#0d0d0d",
            "sidebar": "#0a0a0a",
            "sidebar_hover": "#1a1a1a",
            "chat_bg": "#0d0d0d",
            "input_bg": "#1a1a1a",
            "user_msg": "#1f1f1f",
            "ai_msg": "#141414",
            "thinking_bg": "#1a1a1a",
            "text": "#3b82f6",
            "text_muted": "#6b7280",
            "accent": "#2563eb",
            "border": "#262626",
            "success": "#22c55e",
        }
        
        self.manager = ChatManager()
        self._streaming = False
        self._stop = False
        self._queue = queue.Queue()
        self._thinking_dots = 0
        
        self._setup_styles()
        self._build_ui()
        self._show_welcome()
    
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Vertical.TScrollbar", 
                       background="#1a1a1a", 
                       troughcolor="#0d0d0d",
                       arrowcolor="#3b82f6")
    
    def _build_ui(self):
        c = self.colors
        self.configure(bg=c["bg"])
        
        # === SIDEBAR ===
        sidebar = tk.Frame(self, bg=c["sidebar"], width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        # Logo
        logo_frame = tk.Frame(sidebar, bg=c["sidebar"])
        logo_frame.pack(fill="x", padx=16, pady=20)
        
        tk.Label(logo_frame, text="🐱", font=("Helvetica", 28), 
                bg=c["sidebar"], fg=c["text"]).pack(side="left")
        tk.Label(logo_frame, text="CatSEEK", font=("Helvetica", 20, "bold"),
                bg=c["sidebar"], fg=c["text"]).pack(side="left", padx=(8, 0))
        tk.Label(logo_frame, text="R1", font=("Helvetica", 12),
                bg=c["sidebar"], fg=c["accent"]).pack(side="left", padx=(4, 0), pady=(8, 0))
        
        # New Chat Button
        new_btn = tk.Button(sidebar, text="＋  New Chat", 
                           font=("Helvetica", 12), bg="#1f1f1f", fg="#3b82f6",
                           activebackground="#2a2a2a", activeforeground="#60a5fa",
                           relief="flat", cursor="hand2", pady=12,
                           command=self._new_chat)
        new_btn.pack(fill="x", padx=16, pady=(0, 20))
        
        # Recent Chats Label
        tk.Label(sidebar, text="Recent Chats", font=("Helvetica", 11),
                bg=c["sidebar"], fg=c["text_muted"]).pack(anchor="w", padx=16, pady=(0, 10))
        
        # Chat List
        self.chat_frame = tk.Frame(sidebar, bg=c["sidebar"])
        self.chat_frame.pack(fill="both", expand=True, padx=8)
        self._refresh_chat_list()
        
        # Bottom Buttons
        bottom = tk.Frame(sidebar, bg=c["sidebar"])
        bottom.pack(fill="x", padx=16, pady=16)
        
        tk.Button(bottom, text="⚙ Settings", font=("Helvetica", 11),
                 bg=c["sidebar"], fg="#3b82f6", relief="flat",
                 activebackground=c["sidebar_hover"], cursor="hand2",
                 command=self._open_settings).pack(side="left")
        
        tk.Button(bottom, text="📚 Train", font=("Helvetica", 11),
                 bg=c["sidebar"], fg="#3b82f6", relief="flat",
                 activebackground=c["sidebar_hover"], cursor="hand2",
                 command=self._open_train).pack(side="right")
        
        # === MAIN AREA ===
        main = tk.Frame(self, bg=c["chat_bg"])
        main.pack(side="right", fill="both", expand=True)
        
        # Header
        header = tk.Frame(main, bg=c["chat_bg"], height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="CatSEEK R1", font=("Helvetica", 16, "bold"),
                bg=c["chat_bg"], fg=c["text"]).pack(side="left", padx=24, pady=16)
        
        status_frame = tk.Frame(header, bg=c["chat_bg"])
        status_frame.pack(side="left", pady=16)
        
        self.status_dot = tk.Label(status_frame, text="●", font=("Helvetica", 10),
                                   bg=c["chat_bg"], fg=c["success"])
        self.status_dot.pack(side="left")
        
        self.status_text = tk.Label(status_frame, text="Online", font=("Helvetica", 11),
                                    bg=c["chat_bg"], fg=c["text_muted"])
        self.status_text.pack(side="left", padx=(6, 0))
        
        tk.Frame(main, bg=c["border"], height=1).pack(fill="x")
        
        # Messages Area
        msg_container = tk.Frame(main, bg=c["chat_bg"])
        msg_container.pack(fill="both", expand=True)
        
        self.msg_canvas = tk.Canvas(msg_container, bg=c["chat_bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(msg_container, orient="vertical", command=self.msg_canvas.yview)
        self.msg_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.msg_canvas.pack(side="left", fill="both", expand=True)
        
        self.msg_frame = tk.Frame(self.msg_canvas, bg=c["chat_bg"])
        self.msg_window = self.msg_canvas.create_window((0, 0), window=self.msg_frame, anchor="nw")
        
        self.msg_frame.bind("<Configure>", lambda e: self.msg_canvas.configure(scrollregion=self.msg_canvas.bbox("all")))
        self.msg_canvas.bind("<Configure>", lambda e: self.msg_canvas.itemconfig(self.msg_window, width=e.width))
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            self.msg_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.msg_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.msg_canvas.bind_all("<Button-4>", lambda e: self.msg_canvas.yview_scroll(-1, "units"))
        self.msg_canvas.bind_all("<Button-5>", lambda e: self.msg_canvas.yview_scroll(1, "units"))
        
        # Input Area
        input_container = tk.Frame(main, bg=c["chat_bg"])
        input_container.pack(fill="x", padx=24, pady=16)
        
        input_box_frame = tk.Frame(input_container, bg=c["input_bg"])
        input_box_frame.pack(fill="x")
        
        input_inner = tk.Frame(input_box_frame, bg=c["input_bg"])
        input_inner.pack(fill="x", padx=16, pady=12)
        
        self.input_box = tk.Text(input_inner, height=3, wrap="word",
                                bg=c["input_bg"], fg=c["text"],
                                insertbackground=c["text"], relief="flat",
                                font=("Helvetica", 13))
        self.input_box.pack(side="left", fill="both", expand=True)
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: None)
        
        btn_frame = tk.Frame(input_inner, bg=c["input_bg"])
        btn_frame.pack(side="right", padx=(16, 0))
        
        self.send_btn = tk.Button(btn_frame, text="➤", font=("Helvetica", 18),
                                 bg="#1f1f1f", fg="#3b82f6", relief="flat",
                                 activebackground="#2a2a2a", cursor="hand2",
                                 width=3, command=self._send)
        self.send_btn.pack()
        
        self.stop_btn = tk.Button(btn_frame, text="■", font=("Helvetica", 14),
                                 bg="#141414", fg="#3b82f6", relief="flat",
                                 activebackground="#1f1f1f", cursor="hand2",
                                 width=3, command=self._stop_gen)
        self.stop_btn.pack(pady=(6, 0))
        
        # Footer
        tk.Label(input_container, text="CatSEEK R1 • 7B Parameters • M4 Pro Optimized",
                font=("Helvetica", 10), bg=c["chat_bg"], fg=c["text_muted"]).pack(pady=(10, 0))
        
        self.input_box.focus_set()
    
    def _refresh_chat_list(self):
        for w in self.chat_frame.winfo_children():
            w.destroy()
        
        c = self.colors
        for i, chat in enumerate(self.manager.chats[:10]):
            title = chat.title[:28] + "..." if len(chat.title) > 28 else chat.title
            bg = c["sidebar_hover"] if i == self.manager.idx else c["sidebar"]
            
            btn = tk.Button(self.chat_frame, text=f"💬  {title}",
                           font=("Helvetica", 11), bg=bg, fg="#3b82f6",
                           relief="flat", anchor="w", padx=14, pady=10,
                           activebackground=c["sidebar_hover"], cursor="hand2",
                           command=lambda idx=i: self._select_chat(idx))
            btn.pack(fill="x", pady=2)
    
    def _select_chat(self, idx):
        self.manager.idx = idx
        self._refresh_chat_list()
        self._render_chat()
    
    def _new_chat(self):
        self.manager.new_chat()
        self._refresh_chat_list()
        self._clear_msgs()
        self._add_msg("assistant", "Hello! How can I help you today?", thinking="Ready...")
    
    def _render_chat(self):
        self._clear_msgs()
        for msg in self.manager.current.messages:
            self._add_msg(msg.role, msg.content, thinking=msg.thinking, save=False)
    
    def _clear_msgs(self):
        for w in self.msg_frame.winfo_children():
            w.destroy()
    
    def _show_welcome(self):
        if HAS_TORCH and self.manager.model.model:
            params = self.manager.model.model.count_params()
            device = self.manager.model.device
            dtype = self.manager.model.dtype
            msg = (f"Hello! I'm CatSEEK R1, a {params/1e9:.1f}B parameter reasoning model.\n\n"
                   f"Running on {device} with {dtype} precision.\n\n"
                   f"What would you like to explore today?")
        else:
            msg = "Hello! I'm CatSEEK R1.\n\nNote: PyTorch not found. Install with: pip install torch"
        
        self._add_msg("assistant", msg, thinking="Initialized and ready...")
    
    def _add_msg(self, role, content, thinking="", save=True):
        c = self.colors
        
        container = tk.Frame(self.msg_frame, bg=c["chat_bg"])
        container.pack(fill="x", padx=24, pady=14)
        
        row = tk.Frame(container, bg=c["chat_bg"])
        row.pack(fill="x")
        
        avatar_text = "😺" if role == "assistant" else "👤"
        avatar_bg = c["ai_msg"] if role == "assistant" else c["user_msg"]
        
        avatar = tk.Label(row, text=avatar_text, font=("Helvetica", 18),
                         bg=avatar_bg, width=3, height=1)
        avatar.pack(side="left", anchor="n", padx=(0, 14))
        
        content_frame = tk.Frame(row, bg=c["chat_bg"])
        content_frame.pack(side="left", fill="x", expand=True)
        
        name = "CatSEEK R1" if role == "assistant" else "You"
        tk.Label(content_frame, text=name, font=("Helvetica", 12, "bold"),
                bg=c["chat_bg"], fg=c["text"]).pack(anchor="w")
        
        if role == "assistant" and thinking:
            think_frame = tk.Frame(content_frame, bg=c["thinking_bg"], padx=14, pady=10)
            think_frame.pack(fill="x", pady=(10, 0))
            
            tk.Label(think_frame, text="💭 Thinking", font=("Helvetica", 10, "bold"),
                    bg=c["thinking_bg"], fg=c["accent"]).pack(anchor="w")
            tk.Label(think_frame, text=thinking, font=("Helvetica", 11),
                    bg=c["thinking_bg"], fg=c["text_muted"], wraplength=650,
                    justify="left").pack(anchor="w", pady=(5, 0))
        
        msg_label = tk.Label(content_frame, text=content, font=("Helvetica", 12),
                            bg=c["chat_bg"], fg=c["text"], wraplength=650,
                            justify="left", anchor="w")
        msg_label.pack(anchor="w", pady=(10, 0))
        
        copy_btn = tk.Button(content_frame, text="📋 Copy", font=("Helvetica", 10),
                            bg="#1a1a1a", fg="#3b82f6", relief="flat",
                            activebackground="#2a2a2a", cursor="hand2",
                            command=lambda: self._copy(content))
        copy_btn.pack(anchor="w", pady=(10, 0))
        
        if save:
            self.manager.current.messages.append(Message(role, content, thinking))
            if role == "user" and len(self.manager.current.messages) == 1:
                self.manager.current.title = content[:35]
                self._refresh_chat_list()
        
        self._scroll_bottom()
        return msg_label
    
    def _copy(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
    
    def _scroll_bottom(self):
        self.msg_canvas.update_idletasks()
        self.msg_canvas.yview_moveto(1.0)
    
    def _on_enter(self, e):
        if not (e.state & 0x1):
            self._send()
            return "break"
    
    def _send(self):
        if self._streaming:
            return
        
        text = self.input_box.get("1.0", "end").strip()
        if not text:
            return
        
        self.input_box.delete("1.0", "end")
        self._add_msg("user", text)
        self._stream_response(text)
    
    def _stop_gen(self):
        self._stop = True
    
    def _stream_response(self, user_text):
        self._streaming = True
        self._stop = False
        c = self.colors
        
        self.status_dot.configure(fg="#f59e0b")
        self.status_text.configure(text="Thinking...")
        
        container = tk.Frame(self.msg_frame, bg=c["chat_bg"])
        container.pack(fill="x", padx=24, pady=14)
        
        row = tk.Frame(container, bg=c["chat_bg"])
        row.pack(fill="x")
        
        avatar = tk.Label(row, text="😺", font=("Helvetica", 18), bg=c["ai_msg"], width=3, height=1)
        avatar.pack(side="left", anchor="n", padx=(0, 14))
        
        content_frame = tk.Frame(row, bg=c["chat_bg"])
        content_frame.pack(side="left", fill="x", expand=True)
        
        tk.Label(content_frame, text="CatSEEK R1", font=("Helvetica", 12, "bold"),
                bg=c["chat_bg"], fg=c["text"]).pack(anchor="w")
        
        think_frame = tk.Frame(content_frame, bg=c["thinking_bg"], padx=14, pady=10)
        think_frame.pack(fill="x", pady=(10, 0))
        
        tk.Label(think_frame, text="💭 Thinking", font=("Helvetica", 10, "bold"),
                bg=c["thinking_bg"], fg=c["accent"]).pack(anchor="w")
        think_label = tk.Label(think_frame, text="Processing...", font=("Helvetica", 11),
                              bg=c["thinking_bg"], fg=c["text_muted"])
        think_label.pack(anchor="w", pady=(5, 0))
        
        txt_var = tk.StringVar(value="")
        msg_label = tk.Label(content_frame, textvariable=txt_var, font=("Helvetica", 12),
                            bg=c["chat_bg"], fg=c["text"], wraplength=650,
                            justify="left", anchor="w")
        msg_label.pack(anchor="w", pady=(10, 0))
        
        prompt = ""
        for msg in self.manager.current.messages[-4:]:
            role = "User" if msg.role == "user" else "Assistant"
            prompt += f"{role}: {msg.content}\n"
        prompt += "Assistant:"
        
        def animate():
            if self._streaming:
                dots = "." * ((self._thinking_dots % 3) + 1)
                think_label.configure(text=f"Processing{dots}")
                self._thinking_dots += 1
                self.after(400, animate)
        animate()
        
        def gen_thread():
            for ch in self.manager.model.generate(prompt, self.manager.max_tokens, lambda: self._stop):
                self._queue.put(ch)
            self._queue.put(None)
        
        Thread(target=gen_thread, daemon=True).start()
        
        def poll():
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None:
                        self._finish(txt_var.get(), think_frame)
                        return
                    txt_var.set(txt_var.get() + item)
                    self._scroll_bottom()
            except queue.Empty:
                pass
            
            if self._stop:
                self._finish(txt_var.get(), think_frame)
                return
            
            self.after(10, poll)
        poll()
    
    def _finish(self, content, think_frame):
        self._streaming = False
        c = self.colors
        
        self.status_dot.configure(fg=c["success"])
        self.status_text.configure(text="Online")
        
        for w in think_frame.winfo_children():
            if isinstance(w, tk.Label) and "Processing" in str(w.cget("text")):
                w.configure(text="Completed.")
        
        content = content.strip()
        if content:
            self.manager.current.messages.append(Message("assistant", content, "Completed."))
        
        while not self._queue.empty():
            self._queue.get_nowait()
    
    def _open_settings(self):
        c = self.colors
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("480x520")
        win.configure(bg=c["bg"])
        
        tk.Label(win, text="⚙  Settings", font=("Helvetica", 16, "bold"),
                bg=c["bg"], fg="#3b82f6").pack(pady=24)
        
        frm = tk.Frame(win, bg=c["bg"])
        frm.pack(fill="x", padx=36)
        
        # Temperature
        tk.Label(frm, text="Temperature", font=("Helvetica", 12),
                bg=c["bg"], fg="#3b82f6").grid(row=0, column=0, sticky="w", pady=10)
        t_var = tk.DoubleVar(value=self.manager.model.temperature)
        tk.Scale(frm, from_=0.1, to=1.5, resolution=0.05, orient="horizontal",
                variable=t_var, bg=c["bg"], fg="#3b82f6", highlightthickness=0,
                troughcolor="#1a1a1a", length=200).grid(row=0, column=1, padx=12)
        
        # Top-p
        tk.Label(frm, text="Top-p", font=("Helvetica", 12),
                bg=c["bg"], fg="#3b82f6").grid(row=1, column=0, sticky="w", pady=10)
        p_var = tk.DoubleVar(value=self.manager.model.top_p)
        tk.Scale(frm, from_=0.1, to=1.0, resolution=0.05, orient="horizontal",
                variable=p_var, bg=c["bg"], fg="#3b82f6", highlightthickness=0,
                troughcolor="#1a1a1a", length=200).grid(row=1, column=1, padx=12)
        
        # Top-k
        tk.Label(frm, text="Top-k", font=("Helvetica", 12),
                bg=c["bg"], fg="#3b82f6").grid(row=2, column=0, sticky="w", pady=10)
        k_var = tk.IntVar(value=self.manager.model.top_k)
        tk.Scale(frm, from_=1, to=100, orient="horizontal",
                variable=k_var, bg=c["bg"], fg="#3b82f6", highlightthickness=0,
                troughcolor="#1a1a1a", length=200).grid(row=2, column=1, padx=12)
        
        # Rep Penalty
        tk.Label(frm, text="Repetition Penalty", font=("Helvetica", 12),
                bg=c["bg"], fg="#3b82f6").grid(row=3, column=0, sticky="w", pady=10)
        r_var = tk.DoubleVar(value=self.manager.model.rep_penalty)
        tk.Scale(frm, from_=1.0, to=2.0, resolution=0.05, orient="horizontal",
                variable=r_var, bg=c["bg"], fg="#3b82f6", highlightthickness=0,
                troughcolor="#1a1a1a", length=200).grid(row=3, column=1, padx=12)
        
        # Max Tokens
        tk.Label(frm, text="Max Tokens", font=("Helvetica", 12),
                bg=c["bg"], fg="#3b82f6").grid(row=4, column=0, sticky="w", pady=10)
        n_var = tk.IntVar(value=self.manager.max_tokens)
        tk.Scale(frm, from_=64, to=1024, resolution=32, orient="horizontal",
                variable=n_var, bg=c["bg"], fg="#3b82f6", highlightthickness=0,
                troughcolor="#1a1a1a", length=200).grid(row=4, column=1, padx=12)
        
        # Model Info
        if HAS_TORCH and self.manager.model.model:
            params = self.manager.model.model.count_params()
            cfg = self.manager.model.config
            tk.Label(win, text=f"Model: CatSEEK R1 ({params/1e9:.2f}B)", 
                    font=("Helvetica", 11, "bold"), bg=c["bg"], fg=c["text"]).pack(pady=(24, 6))
            tk.Label(win, text=f"Device: {self.manager.model.device} | {self.manager.model.dtype}",
                    font=("Helvetica", 11), bg=c["bg"], fg=c["text_muted"]).pack()
            tk.Label(win, text=f"Layers: {cfg.n_layers} | Heads: {cfg.n_heads} (KV: {cfg.n_kv_heads})",
                    font=("Helvetica", 11), bg=c["bg"], fg=c["text_muted"]).pack()
        
        def save():
            self.manager.model.temperature = t_var.get()
            self.manager.model.top_p = p_var.get()
            self.manager.model.top_k = k_var.get()
            self.manager.model.rep_penalty = r_var.get()
            self.manager.max_tokens = n_var.get()
            win.destroy()
        
        tk.Button(win, text="Save", font=("Helvetica", 12), bg="#1f1f1f", fg="#3b82f6",
                 relief="flat", padx=36, pady=10, cursor="hand2",
                 command=save).pack(pady=20)
        
        win.transient(self)
        win.grab_set()
    
    def _open_train(self):
        c = self.colors
        win = tk.Toplevel(self)
        win.title("Train Model")
        win.geometry("600x480")
        win.configure(bg=c["bg"])
        
        tk.Label(win, text="📚  Fine-tune CatSEEK R1", font=("Helvetica", 16, "bold"),
                bg=c["bg"], fg="#3b82f6").pack(pady=24)
        tk.Label(win, text="Paste text to train on new patterns:",
                font=("Helvetica", 11), bg=c["bg"], fg="#60a5fa").pack(anchor="w", padx=36)
        
        txt = tk.Text(win, wrap="word", bg=c["input_bg"], fg=c["text"],
                     insertbackground=c["text"], relief="flat", font=("Helvetica", 12))
        txt.pack(fill="both", expand=True, padx=36, pady=16)
        
        bar = tk.Frame(win, bg=c["bg"])
        bar.pack(fill="x", padx=36, pady=16)
        
        def train():
            if not HAS_TORCH:
                messagebox.showerror("Error", "PyTorch not available")
                return
            data = txt.get("1.0", "end").strip()
            if len(data) < 100:
                messagebox.showwarning("Too Short", "Need at least 100 characters.")
                return
            self.manager.model._train_text(data) if hasattr(self.manager.model, '_train_text') else None
            messagebox.showinfo("Success", f"Trained on {len(data)} characters!")
            win.destroy()
        
        def load():
            path = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
            if path:
                with open(path, encoding="utf-8") as f:
                    txt.delete("1.0", "end")
                    txt.insert("1.0", f.read())
        
        tk.Button(bar, text="Load File", font=("Helvetica", 11), bg="#1a1a1a", fg="#3b82f6",
                 relief="flat", padx=18, pady=8, cursor="hand2",
                 command=load).pack(side="left")
        tk.Button(bar, text="Train", font=("Helvetica", 11), bg="#1f1f1f", fg="#3b82f6",
                 relief="flat", padx=24, pady=8, cursor="hand2",
                 command=train).pack(side="right")
        
        win.transient(self)
        win.grab_set()


# =====================================================
# Main
# =====================================================

def main():
    print("=" * 50)
    print("  CatSEEK R1 7B — M4 Pro Optimized")
    print("=" * 50)
    
    app = CatSeekApp()
    app.mainloop()


if __name__ == "__main__":
    main()
