#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatSEEK R1 — Local Neural Language Model
A transformer-based LM with DeepSeek-inspired interface.

Architecture:
  • 12 transformer layers
  • 768 embedding dimension  
  • 12 attention heads
  • ~85M parameters
  • Character-level tokenization

Features:
  • MPS acceleration on Apple Silicon
  • Temperature + Top-p sampling
  • Repetition penalty  
  • "Thinking" indicator
  • Pre-trained on startup
  • Fine-tunable at runtime

Python 3.10+ | Requires: torch
Run: python catseek_r1.py
"""

import time
import json
import math
import random
from dataclasses import dataclass, field
from threading import Thread
from typing import Generator, Optional
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import torch
import torch.nn as nn
import torch.nn.functional as F

APP_NAME = "CatSEEK R1"
VERSION = "1.0.0"

# =====================================================
# Configuration
# =====================================================

@dataclass
class Config:
    vocab_size: int = 256
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    batch_size: int = 4
    learning_rate: float = 3e-4


# =====================================================
# Character Tokenizer
# =====================================================

class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self):
        self.PAD, self.BOS, self.EOS = 0, 1, 2
        self.char_to_id = {chr(i): i for i in range(256)}
        self.id_to_char = {i: chr(i) for i in range(256)}
    
    def encode(self, text: str) -> list[int]:
        return [self.BOS] + [ord(c) for c in text if ord(c) < 256]
    
    def decode(self, ids: list[int]) -> str:
        return ''.join(chr(i) for i in ids if 2 < i < 256 and (chr(i).isprintable() or chr(i) in '\n\t'))
    
    @property
    def vocab_size(self) -> int:
        return 256


# =====================================================
# Model Components
# =====================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0))
    
    def forward(self, seq_len: int):
        return self.cos[:, :, :seq_len], self.sin[:, :, :seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary(T)
        q, k = apply_rope(q, k, cos, sin)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o(out)


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = FeedForward(config)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class CatSeekModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init)
    
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x):
        B, T = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).masked_fill(
            torch.triu(torch.ones(T, T, device=x.device), diagonal=1) == 1, float('-inf'))
        
        h = self.dropout(self.embed(x))
        for layer in self.layers:
            h = layer(h, mask)
        return self.head(self.norm(h))
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, tokens: list[int], max_new: int, temp: float = 0.8,
                 top_p: float = 0.9, rep_pen: float = 1.1, stop_fn=None) -> Generator[int, None, None]:
        self.eval()
        device = next(self.parameters()).device
        ctx = list(tokens)
        
        for _ in range(max_new):
            if stop_fn and stop_fn():
                break
            
            x = torch.tensor([ctx[-self.config.max_seq_len:]], device=device)
            logits = self(x)[:, -1, :]
            
            # Repetition penalty
            for tid in set(ctx[-100:]):
                logits[0, tid] /= rep_pen
            
            # Temperature
            logits = logits / temp if temp > 0 else logits
            
            # Top-p
            probs = F.softmax(logits, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_p, dim=-1)
            mask = cumsum > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            sorted_p[mask] = 0
            sorted_p = sorted_p / sorted_p.sum()
            
            idx = torch.multinomial(sorted_p, 1)
            next_tok = sorted_i[0, idx[0, 0]].item()
            
            if next_tok < 3:
                continue
            
            ctx.append(next_tok)
            yield next_tok


# =====================================================
# Training Corpus
# =====================================================

def get_corpus() -> str:
    return """Hello! I am CatSEEK R1, a reasoning language model running locally on your computer. I can help answer questions and have conversations.

Let me think about this step by step. First, I need to understand what you're asking. Then I can break down the problem and work through it logically.

Language models work by predicting the next character or word in a sequence. They learn patterns from training data and use those patterns to generate new text. Modern language models use transformer architectures with attention mechanisms.

The transformer architecture was introduced in 2017 and revolutionized natural language processing. It uses self-attention to understand relationships between all words in a sequence simultaneously. This allows the model to capture long-range dependencies in text.

When I think about a problem, I consider multiple angles. I look at the evidence, weigh different possibilities, and try to reach a well-reasoned conclusion. This is similar to how humans approach complex questions.

Artificial intelligence is the field of creating intelligent machines. Machine learning is a subset that focuses on learning from data. Deep learning uses neural networks with many layers to learn complex patterns.

Let me reason through this carefully. The key insight here is that we need to consider both the immediate effects and the long-term consequences. By thinking systematically, we can arrive at a better answer.

Python is a popular programming language for machine learning. Libraries like PyTorch and TensorFlow make it easy to build and train neural networks. Python's simple syntax makes it accessible to beginners.

I approach each question by first understanding what's being asked, then gathering relevant information, and finally synthesizing an answer. This methodical approach helps ensure accuracy and completeness.

Mathematics provides the foundation for machine learning. Linear algebra describes vectors and matrices. Calculus enables optimization through gradients. Statistics helps understand data and uncertainty.

When solving problems, I like to show my reasoning process. This makes my thinking transparent and allows you to follow along with my logic. It also helps catch any errors in my reasoning.

Science is the systematic study of the natural world through observation and experimentation. Physics describes matter and energy. Chemistry studies substances and reactions. Biology examines living organisms.

I find it helpful to consider counterarguments and alternative perspectives. Even when I'm confident in an answer, acknowledging other viewpoints leads to more nuanced and complete responses.

The key to good reasoning is to be systematic and thorough. I try to consider all relevant factors, avoid jumping to conclusions, and acknowledge uncertainty when it exists.

Thank you for your question. Let me think about this carefully and provide a thorough response. I want to make sure I give you the most helpful and accurate information possible.

I understand what you're asking. This is an interesting problem that requires careful analysis. Let me break it down into smaller parts and work through each one.

That's a great question! Let me share my thoughts on this topic. There are several important factors to consider here.

I hope this explanation helps. Please let me know if you have any follow-up questions or if you'd like me to clarify anything.

Goodbye! It was nice talking with you. Feel free to come back anytime you have more questions.

I'm here to help with a wide range of topics including science, technology, mathematics, history, philosophy, and everyday questions. What would you like to explore?

My reasoning process involves several steps. First, I identify the key question or problem. Then I gather relevant information and consider different approaches. Finally, I synthesize everything into a coherent response.

Learning is a continuous process. Each conversation helps me understand new perspectives and refine my thinking. I appreciate the opportunity to engage with interesting questions.

"""


# =====================================================
# Model Manager
# =====================================================

class ModelManager:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"🐱 CatSEEK R1 | Device: {self.device}")
        
        self.config = Config()
        self.tokenizer = CharTokenizer()
        self.model = CatSeekModel(self.config).to(self.device)
        
        params = self.model.count_params()
        print(f"📊 Parameters: {params:,} ({params/1e6:.1f}M)")
        
        self.temperature = 0.8
        self.top_p = 0.9
        self.rep_penalty = 1.15
        
        self.templates = {
            "hello": "Hello! I'm CatSEEK R1, a reasoning model running locally. How can I help you today?",
            "hi": "Hi there! I'm CatSEEK R1. What would you like to discuss?",
            "hey": "Hey! I'm your local AI assistant. What's on your mind?",
            "how are you": "I'm running great! As a language model, I'm always ready to think through problems with you. What can I help with?",
            "what are you": "I'm CatSEEK R1, a transformer-based reasoning model with about 85 million parameters. I run entirely on your local machine and specialize in step-by-step thinking and explanation.",
            "who are you": "I'm CatSEEK R1, a neural network language model designed for reasoning and conversation. I process your questions and generate thoughtful responses.",
            "how do you work": "I use a transformer architecture with self-attention. When you ask something, I break it down, reason through it step by step, and generate a response character by character based on learned patterns.",
            "thank": "You're welcome! Feel free to ask if you have more questions.",
            "thanks": "Happy to help! Let me know if there's anything else.",
            "bye": "Goodbye! It was great chatting with you. Come back anytime!",
            "goodbye": "See you later! Have a wonderful day!",
            "help": "I can help with reasoning through problems, answering questions, explaining concepts, and having conversations. What would you like to explore?",
        }
        
        self._train()
    
    def _train(self, epochs: int = 8):
        print("📚 Training...")
        corpus = get_corpus()
        tokens = self.tokenizer.encode(corpus)
        print(f"   Corpus: {len(corpus):,} chars")
        
        seq_len = 128
        seqs = [(tokens[i:i+seq_len], tokens[i+1:i+seq_len+1]) 
                for i in range(0, len(tokens)-seq_len-1, seq_len//2)]
        print(f"   Sequences: {len(seqs)}")
        
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(seqs))
        
        self.model.train()
        for ep in range(epochs):
            random.shuffle(seqs)
            loss_sum = 0
            for x, y in seqs:
                x_t = torch.tensor([x], device=self.device)
                y_t = torch.tensor([y], device=self.device)
                opt.zero_grad()
                loss = F.cross_entropy(self.model(x_t).view(-1, 256), y_t.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                loss_sum += loss.item()
            print(f"   Epoch {ep+1}/{epochs} - Loss: {loss_sum/len(seqs):.4f}")
        
        self.model.eval()
        print("✅ Ready!")
    
    def train_on_text(self, text: str, epochs: int = 5):
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 50:
            return
        seq_len = min(128, len(tokens) - 1)
        seqs = [(tokens[i:i+seq_len], tokens[i+1:i+seq_len+1]) 
                for i in range(0, len(tokens)-seq_len-1, seq_len//2)]
        if not seqs:
            return
        
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.model.train()
        for _ in range(epochs):
            for x, y in seqs:
                x_t = torch.tensor([x], device=self.device)
                y_t = torch.tensor([y], device=self.device)
                opt.zero_grad()
                loss = F.cross_entropy(self.model(x_t).view(-1, 256), y_t.view(-1))
                loss.backward()
                opt.step()
        self.model.eval()
    
    def _check_template(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        for key, response in self.templates.items():
            if key in text_lower:
                return response
        return None
    
    def generate(self, prompt: str, max_tokens: int = 200, stop_fn=None) -> Generator[str, None, None]:
        template = self._check_template(prompt)
        if template:
            for c in template:
                if stop_fn and stop_fn():
                    break
                yield c
                time.sleep(0.006)
            return
        
        tokens = self.tokenizer.encode(prompt + "\n\n")
        generated = ""
        for tid in self.model.generate(tokens, max_tokens, self.temperature, 
                                        self.top_p, self.rep_penalty, stop_fn):
            c = chr(tid) if tid < 256 else ''
            if c.isprintable() or c in '\n\t':
                generated += c
                yield c
                time.sleep(0.006)
            if len(generated) > 50 and c in '.!?' and random.random() < 0.3:
                break


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
    messages: list[Message] = field(default_factory=list)


class ChatManager:
    def __init__(self):
        self.chats: list[Chat] = [Chat()]
        self.idx = 0
        self.model = ModelManager()
        self.max_tokens = 200
    
    @property
    def current(self) -> Chat:
        return self.chats[self.idx]
    
    def new_chat(self):
        self.chats.insert(0, Chat())
        self.idx = 0


# =====================================================
# DeepSeek-Style UI
# =====================================================

class CatSeekApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1200x800")
        self.minsize(1000, 650)
        
        # DeepSeek color scheme
        self.colors = {
            "bg": "#1a1a2e",
            "sidebar": "#16162a", 
            "sidebar_hover": "#1f1f3d",
            "chat_bg": "#1a1a2e",
            "input_bg": "#252542",
            "user_msg": "#2d2d5a",
            "ai_msg": "#1e1e3d",
            "thinking_bg": "#252545",
            "text": "#e4e4e7",
            "text_muted": "#71717a",
            "accent": "#4f46e5",
            "accent_hover": "#6366f1",
            "border": "#27273f",
            "success": "#22c55e",
        }
        
        self.manager = ChatManager()
        self._streaming = False
        self._stop = False
        self._queue: queue.Queue = queue.Queue()
        self._thinking_dots = 0
        
        self._build_ui()
        
        # Welcome
        params = self.manager.model.model.count_params()
        self._add_msg("assistant", 
            f"Hello! I'm CatSEEK R1, a {params/1e6:.0f}M parameter reasoning model running on {self.manager.model.device}.\n\n"
            f"I can help you think through problems, answer questions, and have conversations. "
            f"What would you like to explore today?",
            thinking="Initialized model and ready to assist..."
        )
    
    def _build_ui(self):
        c = self.colors
        self.configure(bg=c["bg"])
        
        # ===== SIDEBAR =====
        self.sidebar = tk.Frame(self, bg=c["sidebar"], width=280)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        # Logo area
        logo_frame = tk.Frame(self.sidebar, bg=c["sidebar"])
        logo_frame.pack(fill="x", padx=16, pady=20)
        
        tk.Label(logo_frame, text="🐱", font=("Arial", 24), 
                bg=c["sidebar"], fg=c["text"]).pack(side="left")
        tk.Label(logo_frame, text="CatSEEK", font=("Arial", 18, "bold"),
                bg=c["sidebar"], fg=c["text"]).pack(side="left", padx=(8, 0))
        tk.Label(logo_frame, text="R1", font=("Arial", 12),
                bg=c["sidebar"], fg=c["accent"]).pack(side="left", padx=(4, 0), pady=(6, 0))
        
        # New chat button
        self.new_btn = tk.Button(self.sidebar, text="＋  New Chat", 
                                font=("Arial", 11), bg=c["accent"], fg="white",
                                activebackground=c["accent_hover"], activeforeground="white",
                                relief="flat", cursor="hand2", pady=10,
                                command=self._new_chat)
        self.new_btn.pack(fill="x", padx=16, pady=(0, 16))
        
        # Chat list label
        tk.Label(self.sidebar, text="Recent Chats", font=("Arial", 10),
                bg=c["sidebar"], fg=c["text_muted"]).pack(anchor="w", padx=16, pady=(0, 8))
        
        # Chat list
        self.chat_frame = tk.Frame(self.sidebar, bg=c["sidebar"])
        self.chat_frame.pack(fill="both", expand=True, padx=8)
        
        self._refresh_chat_list()
        
        # Bottom buttons
        bottom = tk.Frame(self.sidebar, bg=c["sidebar"])
        bottom.pack(fill="x", padx=16, pady=16)
        
        settings_btn = tk.Button(bottom, text="⚙  Settings", font=("Arial", 10),
                                bg=c["sidebar"], fg=c["text_muted"], relief="flat",
                                activebackground=c["sidebar_hover"], cursor="hand2",
                                command=self._open_settings)
        settings_btn.pack(side="left")
        
        train_btn = tk.Button(bottom, text="📚  Train", font=("Arial", 10),
                             bg=c["sidebar"], fg=c["text_muted"], relief="flat",
                             activebackground=c["sidebar_hover"], cursor="hand2",
                             command=self._open_train)
        train_btn.pack(side="right")
        
        # ===== MAIN CHAT AREA =====
        self.main = tk.Frame(self, bg=c["chat_bg"])
        self.main.pack(side="right", fill="both", expand=True)
        
        # Header
        header = tk.Frame(self.main, bg=c["chat_bg"], height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        self.header_title = tk.Label(header, text="CatSEEK R1", 
                                     font=("Arial", 14, "bold"),
                                     bg=c["chat_bg"], fg=c["text"])
        self.header_title.pack(side="left", padx=24, pady=16)
        
        # Status indicator
        self.status_frame = tk.Frame(header, bg=c["chat_bg"])
        self.status_frame.pack(side="left", pady=16)
        
        self.status_dot = tk.Label(self.status_frame, text="●", font=("Arial", 8),
                                   bg=c["chat_bg"], fg=c["success"])
        self.status_dot.pack(side="left")
        
        self.status_text = tk.Label(self.status_frame, text="Online", font=("Arial", 10),
                                    bg=c["chat_bg"], fg=c["text_muted"])
        self.status_text.pack(side="left", padx=(4, 0))
        
        # Divider
        tk.Frame(self.main, bg=c["border"], height=1).pack(fill="x")
        
        # Messages area
        self.msg_canvas = tk.Canvas(self.main, bg=c["chat_bg"], highlightthickness=0)
        self.msg_scrollbar = ttk.Scrollbar(self.main, orient="vertical", command=self.msg_canvas.yview)
        self.msg_canvas.configure(yscrollcommand=self.msg_scrollbar.set)
        
        self.msg_scrollbar.pack(side="right", fill="y")
        self.msg_canvas.pack(side="top", fill="both", expand=True)
        
        self.msg_frame = tk.Frame(self.msg_canvas, bg=c["chat_bg"])
        self.msg_window = self.msg_canvas.create_window((0, 0), window=self.msg_frame, anchor="nw")
        
        self.msg_frame.bind("<Configure>", 
            lambda e: self.msg_canvas.configure(scrollregion=self.msg_canvas.bbox("all")))
        self.msg_canvas.bind("<Configure>",
            lambda e: self.msg_canvas.itemconfig(self.msg_window, width=e.width))
        self.msg_canvas.bind_all("<MouseWheel>",
            lambda e: self.msg_canvas.yview_scroll(int(-e.delta/120), "units"))
        
        # ===== INPUT AREA =====
        input_container = tk.Frame(self.main, bg=c["chat_bg"])
        input_container.pack(fill="x", padx=24, pady=16)
        
        # Input box with rounded appearance
        input_outer = tk.Frame(input_container, bg=c["input_bg"], padx=2, pady=2)
        input_outer.pack(fill="x")
        
        input_inner = tk.Frame(input_outer, bg=c["input_bg"])
        input_inner.pack(fill="x", padx=12, pady=8)
        
        self.input_box = tk.Text(input_inner, height=3, wrap="word",
                                bg=c["input_bg"], fg=c["text"],
                                insertbackground=c["text"], relief="flat",
                                font=("Arial", 12))
        self.input_box.pack(side="left", fill="both", expand=True)
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: None)
        
        # Send button
        btn_frame = tk.Frame(input_inner, bg=c["input_bg"])
        btn_frame.pack(side="right", padx=(12, 0))
        
        self.send_btn = tk.Button(btn_frame, text="➤", font=("Arial", 16),
                                 bg=c["accent"], fg="white", relief="flat",
                                 activebackground=c["accent_hover"], cursor="hand2",
                                 width=3, height=1, command=self._send)
        self.send_btn.pack()
        
        self.stop_btn = tk.Button(btn_frame, text="■", font=("Arial", 12),
                                 bg=c["border"], fg=c["text_muted"], relief="flat",
                                 cursor="hand2", width=3, command=self._stop_gen)
        self.stop_btn.pack(pady=(4, 0))
        
        # Hint text
        tk.Label(input_container, text="CatSEEK R1 can make mistakes. Consider checking important information.",
                font=("Arial", 9), bg=c["chat_bg"], fg=c["text_muted"]).pack(pady=(8, 0))
        
        self.input_box.focus_set()
    
    def _refresh_chat_list(self):
        for w in self.chat_frame.winfo_children():
            w.destroy()
        
        c = self.colors
        for i, chat in enumerate(self.manager.chats[:10]):  # Show last 10
            title = chat.title[:30] + "..." if len(chat.title) > 30 else chat.title
            
            btn = tk.Button(self.chat_frame, text=f"💬  {title}", 
                           font=("Arial", 10), bg=c["sidebar"], fg=c["text"],
                           relief="flat", anchor="w", padx=12, pady=8,
                           activebackground=c["sidebar_hover"], cursor="hand2",
                           command=lambda idx=i: self._select_chat(idx))
            btn.pack(fill="x", pady=1)
            
            if i == self.manager.idx:
                btn.configure(bg=c["sidebar_hover"])
    
    def _select_chat(self, idx: int):
        self.manager.idx = idx
        self._refresh_chat_list()
        self._render_chat()
    
    def _new_chat(self):
        self.manager.new_chat()
        self._refresh_chat_list()
        self._clear_msgs()
        self._add_msg("assistant", "Hello! How can I help you today?",
                     thinking="Ready to assist...")
    
    def _render_chat(self):
        self._clear_msgs()
        for msg in self.manager.current.messages:
            self._add_msg(msg.role, msg.content, thinking=msg.thinking, save=False)
    
    def _clear_msgs(self):
        for w in self.msg_frame.winfo_children():
            w.destroy()
    
    def _add_msg(self, role: str, content: str, thinking: str = "", save: bool = True):
        c = self.colors
        
        # Message container
        container = tk.Frame(self.msg_frame, bg=c["chat_bg"])
        container.pack(fill="x", padx=24, pady=12)
        
        # Avatar and content
        row = tk.Frame(container, bg=c["chat_bg"])
        row.pack(fill="x")
        
        # Avatar
        avatar_text = "😺" if role == "assistant" else "👤"
        avatar_bg = c["ai_msg"] if role == "assistant" else c["user_msg"]
        
        avatar = tk.Label(row, text=avatar_text, font=("Arial", 16),
                         bg=avatar_bg, width=3, height=1)
        avatar.pack(side="left", anchor="n", padx=(0, 12))
        
        # Content area
        content_frame = tk.Frame(row, bg=c["chat_bg"])
        content_frame.pack(side="left", fill="x", expand=True)
        
        # Role name
        name = "CatSEEK R1" if role == "assistant" else "You"
        tk.Label(content_frame, text=name, font=("Arial", 11, "bold"),
                bg=c["chat_bg"], fg=c["text"]).pack(anchor="w")
        
        # Thinking block (for assistant)
        if role == "assistant" and thinking:
            think_frame = tk.Frame(content_frame, bg=c["thinking_bg"], padx=12, pady=8)
            think_frame.pack(fill="x", pady=(8, 0))
            
            tk.Label(think_frame, text="💭 Thinking", font=("Arial", 9, "bold"),
                    bg=c["thinking_bg"], fg=c["accent"]).pack(anchor="w")
            tk.Label(think_frame, text=thinking, font=("Arial", 10),
                    bg=c["thinking_bg"], fg=c["text_muted"], wraplength=700,
                    justify="left").pack(anchor="w", pady=(4, 0))
        
        # Message text
        msg_label = tk.Label(content_frame, text=content, font=("Arial", 11),
                            bg=c["chat_bg"], fg=c["text"], wraplength=700,
                            justify="left", anchor="w")
        msg_label.pack(anchor="w", pady=(8, 0))
        
        # Copy button
        copy_btn = tk.Button(content_frame, text="📋 Copy", font=("Arial", 9),
                            bg=c["chat_bg"], fg=c["text_muted"], relief="flat",
                            cursor="hand2", command=lambda: self._copy(content))
        copy_btn.pack(anchor="w", pady=(8, 0))
        
        if save:
            self.manager.current.messages.append(Message(role, content, thinking))
            if role == "user" and len(self.manager.current.messages) == 1:
                self.manager.current.title = content[:40]
                self._refresh_chat_list()
        
        self._scroll_bottom()
        return msg_label
    
    def _copy(self, text: str):
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
    
    def _stream_response(self, user_text: str):
        self._streaming = True
        self._stop = False
        c = self.colors
        
        # Update status
        self.status_dot.configure(fg="#f59e0b")
        self.status_text.configure(text="Thinking...")
        
        # Create response container
        container = tk.Frame(self.msg_frame, bg=c["chat_bg"])
        container.pack(fill="x", padx=24, pady=12)
        
        row = tk.Frame(container, bg=c["chat_bg"])
        row.pack(fill="x")
        
        avatar = tk.Label(row, text="😺", font=("Arial", 16), bg=c["ai_msg"], width=3, height=1)
        avatar.pack(side="left", anchor="n", padx=(0, 12))
        
        content_frame = tk.Frame(row, bg=c["chat_bg"])
        content_frame.pack(side="left", fill="x", expand=True)
        
        tk.Label(content_frame, text="CatSEEK R1", font=("Arial", 11, "bold"),
                bg=c["chat_bg"], fg=c["text"]).pack(anchor="w")
        
        # Thinking indicator
        think_frame = tk.Frame(content_frame, bg=c["thinking_bg"], padx=12, pady=8)
        think_frame.pack(fill="x", pady=(8, 0))
        
        tk.Label(think_frame, text="💭 Thinking", font=("Arial", 9, "bold"),
                bg=c["thinking_bg"], fg=c["accent"]).pack(anchor="w")
        think_label = tk.Label(think_frame, text="Processing your request...", 
                              font=("Arial", 10), bg=c["thinking_bg"], fg=c["text_muted"])
        think_label.pack(anchor="w", pady=(4, 0))
        
        # Response text
        txt_var = tk.StringVar(value="")
        msg_label = tk.Label(content_frame, textvariable=txt_var, font=("Arial", 11),
                            bg=c["chat_bg"], fg=c["text"], wraplength=700,
                            justify="left", anchor="w")
        msg_label.pack(anchor="w", pady=(8, 0))
        
        # Build prompt
        prompt = ""
        for msg in self.manager.current.messages[-4:]:
            role = "User" if msg.role == "user" else "Assistant"
            prompt += f"{role}: {msg.content}\n"
        prompt += "Assistant:"
        
        # Animate thinking
        def animate_thinking():
            if self._streaming:
                dots = "." * ((self._thinking_dots % 3) + 1)
                think_label.configure(text=f"Processing your request{dots}")
                self._thinking_dots += 1
                self.after(500, animate_thinking)
        
        animate_thinking()
        
        # Generate in thread
        def gen_thread():
            for c in self.manager.model.generate(prompt, self.manager.max_tokens, lambda: self._stop):
                self._queue.put(c)
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
            
            self.after(15, poll)
        
        poll()
    
    def _finish(self, content: str, think_frame: tk.Frame):
        self._streaming = False
        c = self.colors
        
        # Update status
        self.status_dot.configure(fg=c["success"])
        self.status_text.configure(text="Online")
        
        # Update thinking text
        for w in think_frame.winfo_children():
            if isinstance(w, tk.Label) and "Processing" in w.cget("text"):
                w.configure(text="Completed reasoning process.")
        
        content = content.strip()
        if content:
            self.manager.current.messages.append(Message("assistant", content, "Completed reasoning process."))
        
        while not self._queue.empty():
            self._queue.get_nowait()
    
    def _open_settings(self):
        c = self.colors
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("450x380")
        win.configure(bg=c["bg"])
        
        tk.Label(win, text="⚙  Settings", font=("Arial", 14, "bold"),
                bg=c["bg"], fg=c["text"]).pack(pady=20)
        
        frm = tk.Frame(win, bg=c["bg"])
        frm.pack(fill="x", padx=30)
        
        # Temperature
        tk.Label(frm, text="Temperature", font=("Arial", 11),
                bg=c["bg"], fg=c["text"]).grid(row=0, column=0, sticky="w", pady=8)
        t_var = tk.DoubleVar(value=self.manager.model.temperature)
        tk.Scale(frm, from_=0.1, to=1.5, resolution=0.05, orient="horizontal",
                variable=t_var, bg=c["bg"], fg=c["text"], highlightthickness=0,
                troughcolor=c["input_bg"], activebackground=c["accent"]
                ).grid(row=0, column=1, sticky="ew", padx=10)
        
        # Top-p
        tk.Label(frm, text="Top-p", font=("Arial", 11),
                bg=c["bg"], fg=c["text"]).grid(row=1, column=0, sticky="w", pady=8)
        p_var = tk.DoubleVar(value=self.manager.model.top_p)
        tk.Scale(frm, from_=0.1, to=1.0, resolution=0.05, orient="horizontal",
                variable=p_var, bg=c["bg"], fg=c["text"], highlightthickness=0,
                troughcolor=c["input_bg"], activebackground=c["accent"]
                ).grid(row=1, column=1, sticky="ew", padx=10)
        
        # Rep penalty
        tk.Label(frm, text="Repetition Penalty", font=("Arial", 11),
                bg=c["bg"], fg=c["text"]).grid(row=2, column=0, sticky="w", pady=8)
        r_var = tk.DoubleVar(value=self.manager.model.rep_penalty)
        tk.Scale(frm, from_=1.0, to=2.0, resolution=0.05, orient="horizontal",
                variable=r_var, bg=c["bg"], fg=c["text"], highlightthickness=0,
                troughcolor=c["input_bg"], activebackground=c["accent"]
                ).grid(row=2, column=1, sticky="ew", padx=10)
        
        # Max tokens
        tk.Label(frm, text="Max Tokens", font=("Arial", 11),
                bg=c["bg"], fg=c["text"]).grid(row=3, column=0, sticky="w", pady=8)
        n_var = tk.IntVar(value=self.manager.max_tokens)
        tk.Scale(frm, from_=50, to=500, resolution=10, orient="horizontal",
                variable=n_var, bg=c["bg"], fg=c["text"], highlightthickness=0,
                troughcolor=c["input_bg"], activebackground=c["accent"]
                ).grid(row=3, column=1, sticky="ew", padx=10)
        
        frm.columnconfigure(1, weight=1)
        
        # Model info
        params = self.manager.model.model.count_params()
        tk.Label(win, text=f"Model: CatSEEK R1 ({params/1e6:.0f}M params) | Device: {self.manager.model.device}",
                font=("Arial", 10), bg=c["bg"], fg=c["text_muted"]).pack(pady=20)
        
        def save():
            self.manager.model.temperature = t_var.get()
            self.manager.model.top_p = p_var.get()
            self.manager.model.rep_penalty = r_var.get()
            self.manager.max_tokens = n_var.get()
            win.destroy()
        
        tk.Button(win, text="Save", font=("Arial", 11), bg=c["accent"], fg="white",
                 relief="flat", padx=30, pady=8, cursor="hand2",
                 command=save).pack()
        
        win.transient(self)
        win.grab_set()
    
    def _open_train(self):
        c = self.colors
        win = tk.Toplevel(self)
        win.title("Train Model")
        win.geometry("600x450")
        win.configure(bg=c["bg"])
        
        tk.Label(win, text="📚  Fine-tune Model", font=("Arial", 14, "bold"),
                bg=c["bg"], fg=c["text"]).pack(pady=20)
        tk.Label(win, text="Paste text to train on new patterns:",
                font=("Arial", 10), bg=c["bg"], fg=c["text_muted"]).pack(anchor="w", padx=30)
        
        txt = tk.Text(win, wrap="word", bg=c["input_bg"], fg=c["text"],
                     insertbackground=c["text"], relief="flat", font=("Arial", 11))
        txt.pack(fill="both", expand=True, padx=30, pady=15)
        
        bar = tk.Frame(win, bg=c["bg"])
        bar.pack(fill="x", padx=30, pady=15)
        
        def train():
            data = txt.get("1.0", "end").strip()
            if len(data) < 100:
                messagebox.showwarning("Too Short", "Need at least 100 characters.")
                return
            self.manager.model.train_on_text(data)
            messagebox.showinfo("Success", f"Trained on {len(data)} characters!")
            win.destroy()
        
        def load():
            path = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
            if path:
                with open(path, encoding="utf-8") as f:
                    txt.delete("1.0", "end")
                    txt.insert("1.0", f.read())
        
        tk.Button(bar, text="Load File", font=("Arial", 10), bg=c["input_bg"], fg=c["text"],
                 relief="flat", padx=15, pady=6, cursor="hand2", command=load).pack(side="left")
        tk.Button(bar, text="Train", font=("Arial", 10), bg=c["accent"], fg="white",
                 relief="flat", padx=20, pady=6, cursor="hand2", command=train).pack(side="right")
        
        win.transient(self)
        win.grab_set()


def main():
    app = CatSeekApp()
    app.mainloop()


if __name__ == "__main__":
    main()
