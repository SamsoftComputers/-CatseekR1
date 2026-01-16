#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatGPT o1 — Advanced Reasoning Model Interface
A ChatGPT + DeepSeek fusion with o1-style reasoning.

Features:
  • o1-style "Thinking" process visualization
  • Code Interpreter (Python execution sandbox)
  • File Sandbox (upload, view, manage files)
  • Canvas (drawing, diagrams, image editing)
  • Markov chain text generation
  • Dark mode UI (ChatGPT + DeepSeek fusion)

Python 3.10+ | Pure tkinter + stdlib
Run: python catgpt_o1.py
"""

import os
import sys
import io
import re
import time
import json
import math
import random
import base64
import hashlib
import tempfile
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from threading import Thread
from typing import Generator, Optional, Any
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser

APP_NAME = "CatGPT o1"
VERSION = "1.0.0"

# =====================================================
# Markov Chain Language Model
# =====================================================

class MarkovLM:
    """
    Variable-order Markov chain language model.
    Trained on corpus, generates o1-style reasoning text.
    """
    
    def __init__(self, order: int = 4):
        self.order = order
        self.chains: dict[int, dict[tuple, dict[str, int]]] = {
            i: defaultdict(lambda: defaultdict(int)) for i in range(1, order + 1)
        }
        self.starts: list[tuple] = []
        self.vocab: set[str] = set()
        self.temperature = 0.8
        self.rep_penalty = 1.2
        
        # Train on embedded corpus
        self._train_corpus()
    
    def _tokenize(self, text: str) -> list[str]:
        """Word-level tokenization."""
        tokens = re.findall(r"[\w']+|[.,!?;:\-\n]", text.lower())
        return tokens
    
    def _detokenize(self, tokens: list[str]) -> str:
        """Convert tokens back to text."""
        if not tokens:
            return ""
        
        result = [tokens[0].capitalize()]
        no_space = {",", ".", "!", "?", ";", ":", "'", "\n"}
        
        for i in range(1, len(tokens)):
            tok = tokens[i]
            if tok in no_space:
                result.append(tok)
            elif tok == "\n":
                result.append("\n")
            else:
                # Capitalize after sentence end
                if result[-1] in ".!?\n":
                    tok = tok.capitalize()
                result.append(" " + tok)
        
        text = "".join(result)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n ', '\n', text)
        return text.strip()
    
    def train(self, text: str):
        """Train on text."""
        tokens = self._tokenize(text)
        self.vocab.update(tokens)
        
        if len(tokens) < self.order + 1:
            return
        
        # Record starts
        self.starts.append(tuple(tokens[:self.order]))
        
        # Build chains for each order
        for n in range(1, self.order + 1):
            for i in range(len(tokens) - n):
                ctx = tuple(tokens[i:i + n])
                nxt = tokens[i + n]
                self.chains[n][ctx][nxt] += 1
    
    def _sample(self, dist: dict[str, int], recent: list[str]) -> Optional[str]:
        """Sample with temperature and repetition penalty."""
        if not dist:
            return None
        
        items = list(dist.items())
        words, counts = zip(*items)
        
        # Apply repetition penalty
        weights = []
        recent_set = set(recent[-30:])
        for w, c in zip(words, counts):
            weight = float(c)
            if w in recent_set:
                weight /= self.rep_penalty
            weights.append(weight)
        
        # Temperature
        if self.temperature != 1.0:
            weights = [w ** (1.0 / self.temperature) for w in weights]
        
        total = sum(weights)
        if total == 0:
            return random.choice(words)
        
        probs = [w / total for w in weights]
        return random.choices(words, weights=probs, k=1)[0]
    
    def generate(self, prompt: str = "", max_tokens: int = 200,
                 stop_fn: Optional[callable] = None) -> Generator[str, None, None]:
        """Generate text token by token."""
        if prompt:
            tokens = self._tokenize(prompt)
        elif self.starts:
            tokens = list(random.choice(self.starts))
        else:
            tokens = []
        
        generated = []
        
        for _ in range(max_tokens):
            if stop_fn and stop_fn():
                break
            
            next_tok = None
            
            # Try highest order first, back off
            for n in range(min(self.order, len(tokens)), 0, -1):
                ctx = tuple(tokens[-n:])
                if ctx in self.chains[n]:
                    next_tok = self._sample(self.chains[n][ctx], generated)
                    if next_tok:
                        break
            
            if not next_tok:
                # Random from vocab
                if self.vocab:
                    next_tok = random.choice(list(self.vocab - {",", ".", "!", "?", ";", ":"}))
                else:
                    break
            
            tokens.append(next_tok)
            generated.append(next_tok)
            
            # Yield detokenized chunk
            yield self._detokenize([next_tok])
            
            # Stop at sentence end sometimes
            if len(generated) > 20 and next_tok in ".!?" and random.random() < 0.25:
                break
    
    def _train_corpus(self):
        """Train on embedded o1-style corpus."""
        corpus = """
Let me think through this step by step.

First, I need to understand what the question is asking. The key insight here is that we need to break down the problem into smaller parts.

Thinking about this carefully, there are several factors to consider. Let me analyze each one.

Step 1: Identify the main components. We have the input, the process, and the expected output.

Step 2: Consider the constraints. What are the limitations we need to work within?

Step 3: Develop a solution approach. Based on my analysis, the best approach would be to start with the fundamentals.

Now, let me reason through the implications. If we follow this logic, then the conclusion becomes clear.

I should verify my reasoning. Let me check if this makes sense by considering edge cases.

The answer follows from combining these observations. Therefore, we can conclude that the solution involves understanding the underlying principles.

To summarize my thinking process: I first identified the problem, then analyzed the components, considered multiple approaches, and arrived at a well-reasoned conclusion.

Hello! I am CatGPT o1, an advanced reasoning model. I can help you think through complex problems step by step.

I specialize in breaking down difficult questions and showing my reasoning process. This helps you understand not just the answer, but how I arrived at it.

When you ask me something, I will think carefully about it. I consider multiple angles, weigh the evidence, and construct a logical argument.

Let me help you with your question. I will approach this systematically and show my work along the way.

Programming involves writing instructions for computers. Python is a popular language known for its readability. Functions encapsulate reusable logic. Variables store data. Loops repeat operations.

Mathematics provides tools for reasoning. Algebra manipulates symbols. Calculus studies change. Statistics analyzes data. Logic ensures valid arguments.

Science investigates the natural world. Physics describes forces and motion. Chemistry studies matter and reactions. Biology examines living things.

I can execute Python code to help solve problems. Just ask me to write and run code, and I will show you the results.

I can also help you work with files. Upload documents and I can analyze, summarize, or extract information from them.

The canvas feature lets me create diagrams, visualizations, and drawings to illustrate concepts.

Thank you for your question. Let me provide a thorough and well-reasoned response.

I appreciate your curiosity. Learning is a continuous process of asking questions and seeking understanding.

Feel free to ask follow-up questions. I am here to help you explore ideas and solve problems.

Let me reconsider this from another angle. Sometimes looking at a problem differently reveals new insights.

My reasoning leads me to believe that the solution involves several interconnected factors. Let me explain each one.

In conclusion, by thinking carefully and systematically, we can arrive at well-supported answers. The key is to show our work and verify our logic.
"""
        
        # Split into paragraphs and train
        paragraphs = corpus.strip().split('\n\n')
        for para in paragraphs:
            self.train(para)


# =====================================================
# Code Interpreter (Python Sandbox)
# =====================================================

class CodeInterpreter:
    """
    Safe Python code execution sandbox.
    Executes code and captures output.
    """
    
    def __init__(self, sandbox_dir: str = None):
        self.sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="catgpt_")
        self.globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range, "str": str,
                "int": int, "float": float, "bool": bool, "list": list,
                "dict": dict, "set": set, "tuple": tuple, "type": type,
                "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
                "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
                "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
                "open": self._safe_open, "input": lambda x="": "[input disabled]",
                "True": True, "False": False, "None": None,
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError,
                "IndexError": IndexError, "AttributeError": AttributeError,
            },
            "math": math, "random": random, "re": re, "json": json,
            "time": time, "os": None, "sys": None,  # Disabled
        }
        self.locals = {}
        self.history: list[dict] = []
    
    def _safe_open(self, filename, mode="r", *args, **kwargs):
        """Restricted file open - only in sandbox."""
        path = Path(self.sandbox_dir) / Path(filename).name
        if "w" in mode or "a" in mode:
            return open(path, mode, *args, **kwargs)
        elif path.exists():
            return open(path, mode, *args, **kwargs)
        else:
            raise FileNotFoundError(f"File not found in sandbox: {filename}")
    
    def execute(self, code: str) -> dict:
        """Execute Python code and return results."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = {
            "code": code,
            "stdout": "",
            "stderr": "",
            "result": None,
            "error": None,
            "success": False,
        }
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try exec first (statements)
                try:
                    exec(code, self.globals, self.locals)
                    result["success"] = True
                except SyntaxError:
                    # Try eval (expression)
                    result["result"] = eval(code, self.globals, self.locals)
                    result["success"] = True
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"
            result["stderr"] = traceback.format_exc()
        
        result["stdout"] = stdout_capture.getvalue()
        result["stderr"] = stderr_capture.getvalue() if not result["error"] else result["stderr"]
        
        self.history.append(result)
        return result
    
    def reset(self):
        """Reset the interpreter state."""
        self.locals = {}
        self.history = []


# =====================================================
# File Sandbox
# =====================================================

class FileSandbox:
    """
    File management sandbox.
    Upload, view, and manage files.
    """
    
    def __init__(self, sandbox_dir: str = None):
        self.sandbox_dir = Path(sandbox_dir or tempfile.mkdtemp(prefix="catgpt_files_"))
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, dict] = {}
    
    def add_file(self, filepath: str) -> dict:
        """Add file to sandbox."""
        src = Path(filepath)
        if not src.exists():
            return {"error": "File not found"}
        
        dst = self.sandbox_dir / src.name
        
        # Copy file
        with open(src, "rb") as f:
            content = f.read()
        with open(dst, "wb") as f:
            f.write(content)
        
        info = {
            "name": src.name,
            "path": str(dst),
            "size": len(content),
            "type": self._get_type(src.name),
        }
        
        self.files[src.name] = info
        return info
    
    def _get_type(self, filename: str) -> str:
        """Determine file type."""
        ext = Path(filename).suffix.lower()
        types = {
            ".py": "python", ".js": "javascript", ".html": "html",
            ".css": "css", ".json": "json", ".txt": "text",
            ".md": "markdown", ".csv": "csv", ".xml": "xml",
            ".png": "image", ".jpg": "image", ".jpeg": "image",
            ".gif": "image", ".pdf": "pdf", ".doc": "document",
            ".docx": "document", ".xls": "spreadsheet", ".xlsx": "spreadsheet",
        }
        return types.get(ext, "unknown")
    
    def read_file(self, filename: str) -> str:
        """Read file contents."""
        path = self.sandbox_dir / filename
        if not path.exists():
            return f"[File not found: {filename}]"
        
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"[Error reading file: {e}]"
    
    def write_file(self, filename: str, content: str) -> bool:
        """Write content to file."""
        path = self.sandbox_dir / filename
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            self.files[filename] = {
                "name": filename,
                "path": str(path),
                "size": len(content),
                "type": self._get_type(filename),
            }
            return True
        except Exception:
            return False
    
    def list_files(self) -> list[dict]:
        """List all files in sandbox."""
        return list(self.files.values())
    
    def delete_file(self, filename: str) -> bool:
        """Delete file from sandbox."""
        path = self.sandbox_dir / filename
        try:
            if path.exists():
                path.unlink()
            if filename in self.files:
                del self.files[filename]
            return True
        except Exception:
            return False


# =====================================================
# Canvas (Drawing Surface)
# =====================================================

class CanvasWidget(tk.Canvas):
    """
    Drawing canvas with tools.
    Supports pen, shapes, text, colors.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.tool = "pen"  # pen, line, rect, oval, text, eraser
        self.color = "#3b82f6"
        self.brush_size = 3
        self.fill = False
        
        self._start_x = None
        self._start_y = None
        self._current_item = None
        self._history: list[int] = []
        
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Button-3>", self._on_right_click)
    
    def _on_press(self, event):
        self._start_x = event.x
        self._start_y = event.y
        
        if self.tool == "pen":
            self._current_item = self.create_line(
                event.x, event.y, event.x, event.y,
                fill=self.color, width=self.brush_size,
                capstyle=tk.ROUND, smooth=True
            )
        elif self.tool == "eraser":
            self._erase(event.x, event.y)
        elif self.tool == "text":
            text = tk.simpledialog.askstring("Text", "Enter text:")
            if text:
                item = self.create_text(
                    event.x, event.y, text=text,
                    fill=self.color, font=("Arial", 12)
                )
                self._history.append(item)
    
    def _on_drag(self, event):
        if self.tool == "pen":
            if self._current_item:
                coords = self.coords(self._current_item)
                coords.extend([event.x, event.y])
                self.coords(self._current_item, *coords)
        elif self.tool == "eraser":
            self._erase(event.x, event.y)
        elif self.tool in ("line", "rect", "oval"):
            if self._current_item:
                self.delete(self._current_item)
            
            if self.tool == "line":
                self._current_item = self.create_line(
                    self._start_x, self._start_y, event.x, event.y,
                    fill=self.color, width=self.brush_size
                )
            elif self.tool == "rect":
                fill = self.color if self.fill else ""
                outline = self.color
                self._current_item = self.create_rectangle(
                    self._start_x, self._start_y, event.x, event.y,
                    fill=fill, outline=outline, width=self.brush_size
                )
            elif self.tool == "oval":
                fill = self.color if self.fill else ""
                outline = self.color
                self._current_item = self.create_oval(
                    self._start_x, self._start_y, event.x, event.y,
                    fill=fill, outline=outline, width=self.brush_size
                )
    
    def _on_release(self, event):
        if self._current_item:
            self._history.append(self._current_item)
        self._current_item = None
    
    def _on_right_click(self, event):
        """Delete item under cursor."""
        items = self.find_overlapping(event.x - 5, event.y - 5, event.x + 5, event.y + 5)
        for item in items:
            self.delete(item)
    
    def _erase(self, x, y):
        """Erase items near point."""
        items = self.find_overlapping(x - 10, y - 10, x + 10, y + 10)
        for item in items:
            self.delete(item)
    
    def undo(self):
        """Undo last action."""
        if self._history:
            item = self._history.pop()
            self.delete(item)
    
    def clear_all(self):
        """Clear canvas."""
        self.delete("all")
        self._history.clear()
    
    def export_ps(self, filename: str):
        """Export canvas to PostScript."""
        self.postscript(file=filename, colormode="color")


# =====================================================
# Chat Data
# =====================================================

@dataclass
class Message:
    role: str  # user, assistant, system, code, result
    content: str
    thinking: str = ""
    code_result: Optional[dict] = None


@dataclass
class Conversation:
    title: str = "New Chat"
    messages: list[Message] = field(default_factory=list)
    model: str = "catgpt-o1"


# =====================================================
# Main Application
# =====================================================

class CatGPTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1400x900")
        self.minsize(1200, 700)
        
        # Theme colors (ChatGPT + DeepSeek fusion, gray/black + blue)
        self.colors = {
            "bg": "#0d0d0d",
            "sidebar": "#0a0a0a",
            "sidebar_hover": "#1a1a1a",
            "chat_bg": "#0d0d0d",
            "input_bg": "#1a1a1a",
            "user_msg": "#1f1f1f",
            "ai_msg": "#141414",
            "thinking_bg": "#1a1a1a",
            "code_bg": "#0f0f0f",
            "text": "#e4e4e7",
            "text_blue": "#3b82f6",
            "text_muted": "#6b7280",
            "accent": "#2563eb",
            "accent_light": "#60a5fa",
            "border": "#262626",
            "success": "#22c55e",
            "error": "#ef4444",
            "warning": "#f59e0b",
            "btn_bg": "#1f1f1f",
            "btn_hover": "#2a2a2a",
        }
        
        # State
        self.conversations: list[Conversation] = [Conversation()]
        self.current_idx = 0
        self.model = MarkovLM(order=4)
        self.interpreter = CodeInterpreter()
        self.file_sandbox = FileSandbox()
        
        self._streaming = False
        self._stop = False
        self._queue: queue.Queue = queue.Queue()
        self._thinking_anim = 0
        
        # Current view: chat, canvas, files
        self.current_view = "chat"
        
        self._build_ui()
        
        # Welcome message
        self._add_message("assistant",
            "Hello! I'm CatGPT o1, an advanced reasoning model.\n\n"
            "I can help you:\n"
            "• Think through complex problems step-by-step\n"
            "• Execute Python code (use code blocks)\n"
            "• Work with files (upload in the Files panel)\n"
            "• Create diagrams on the Canvas\n\n"
            "What would you like to explore today?",
            thinking="Initialized and ready to assist..."
        )
    
    @property
    def current(self) -> Conversation:
        return self.conversations[self.current_idx]
    
    def _c(self, key: str) -> str:
        return self.colors.get(key, "#ffffff")
    
    # ===== UI Building =====
    def _build_ui(self):
        self.configure(bg=self._c("bg"))
        
        # Main container
        self.main_container = tk.Frame(self, bg=self._c("bg"))
        self.main_container.pack(fill="both", expand=True)
        
        # Sidebar
        self._build_sidebar()
        
        # Main area (chat/canvas/files)
        self._build_main_area()
        
        # Right panel (tools)
        self._build_right_panel()
    
    def _build_sidebar(self):
        """Build left sidebar."""
        sidebar = tk.Frame(self.main_container, bg=self._c("sidebar"), width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        # Logo
        logo_frame = tk.Frame(sidebar, bg=self._c("sidebar"))
        logo_frame.pack(fill="x", padx=16, pady=16)
        
        tk.Label(logo_frame, text="🐱", font=("Arial", 22),
                bg=self._c("sidebar"), fg=self._c("text")).pack(side="left")
        tk.Label(logo_frame, text="CatGPT", font=("Arial", 16, "bold"),
                bg=self._c("sidebar"), fg=self._c("text_blue")).pack(side="left", padx=(6, 0))
        tk.Label(logo_frame, text="o1", font=("Arial", 11),
                bg=self._c("sidebar"), fg=self._c("accent_light")).pack(side="left", padx=(4, 0), pady=(4, 0))
        
        # New chat button
        new_btn = tk.Button(sidebar, text="＋  New Chat", font=("Arial", 11),
                           bg=self._c("btn_bg"), fg=self._c("text_blue"),
                           activebackground=self._c("btn_hover"), activeforeground=self._c("accent_light"),
                           relief="flat", cursor="hand2", pady=10,
                           command=self._new_chat)
        new_btn.pack(fill="x", padx=12, pady=(0, 12))
        
        # View buttons
        view_frame = tk.Frame(sidebar, bg=self._c("sidebar"))
        view_frame.pack(fill="x", padx=12, pady=(0, 12))
        
        for view, icon in [("chat", "💬"), ("canvas", "🎨"), ("files", "📁")]:
            btn = tk.Button(view_frame, text=f"{icon}", font=("Arial", 14),
                           bg=self._c("btn_bg"), fg=self._c("text_blue"),
                           activebackground=self._c("btn_hover"), relief="flat",
                           width=4, cursor="hand2",
                           command=lambda v=view: self._switch_view(v))
            btn.pack(side="left", expand=True, padx=2)
        
        # Chat list label
        tk.Label(sidebar, text="History", font=("Arial", 10),
                bg=self._c("sidebar"), fg=self._c("text_muted")).pack(anchor="w", padx=16, pady=(8, 4))
        
        # Chat list
        self.chat_list_frame = tk.Frame(sidebar, bg=self._c("sidebar"))
        self.chat_list_frame.pack(fill="both", expand=True, padx=8)
        
        self._refresh_chat_list()
        
        # Bottom buttons
        bottom = tk.Frame(sidebar, bg=self._c("sidebar"))
        bottom.pack(fill="x", padx=12, pady=12)
        
        tk.Button(bottom, text="⚙ Settings", font=("Arial", 10),
                 bg=self._c("sidebar"), fg=self._c("text_blue"),
                 activebackground=self._c("sidebar_hover"), relief="flat",
                 cursor="hand2", command=self._open_settings).pack(side="left")
        
        tk.Button(bottom, text="🗑 Clear", font=("Arial", 10),
                 bg=self._c("sidebar"), fg=self._c("text_muted"),
                 activebackground=self._c("sidebar_hover"), relief="flat",
                 cursor="hand2", command=self._clear_chat).pack(side="right")
    
    def _build_main_area(self):
        """Build main content area."""
        self.main_area = tk.Frame(self.main_container, bg=self._c("chat_bg"))
        self.main_area.pack(side="left", fill="both", expand=True)
        
        # Header
        header = tk.Frame(self.main_area, bg=self._c("chat_bg"), height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        self.view_title = tk.Label(header, text="💬 Chat", font=("Arial", 14, "bold"),
                                   bg=self._c("chat_bg"), fg=self._c("text_blue"))
        self.view_title.pack(side="left", padx=20, pady=12)
        
        # Status
        self.status_frame = tk.Frame(header, bg=self._c("chat_bg"))
        self.status_frame.pack(side="left", pady=12)
        
        self.status_dot = tk.Label(self.status_frame, text="●", font=("Arial", 8),
                                   bg=self._c("chat_bg"), fg=self._c("success"))
        self.status_dot.pack(side="left")
        
        self.status_text = tk.Label(self.status_frame, text="Ready", font=("Arial", 10),
                                    bg=self._c("chat_bg"), fg=self._c("text_muted"))
        self.status_text.pack(side="left", padx=(4, 0))
        
        # Model selector
        self.model_var = tk.StringVar(value="catgpt-o1")
        model_menu = ttk.Combobox(header, textvariable=self.model_var, width=12,
                                  values=["catgpt-o1", "catgpt-4", "catgpt-mini"],
                                  state="readonly")
        model_menu.pack(side="right", padx=20, pady=12)
        
        tk.Frame(self.main_area, bg=self._c("border"), height=1).pack(fill="x")
        
        # Content frames (chat, canvas, files)
        self._build_chat_view()
        self._build_canvas_view()
        self._build_files_view()
        
        # Show chat by default
        self.chat_frame.pack(fill="both", expand=True)
    
    def _build_chat_view(self):
        """Build chat view."""
        self.chat_frame = tk.Frame(self.main_area, bg=self._c("chat_bg"))
        
        # Messages area
        msg_container = tk.Frame(self.chat_frame, bg=self._c("chat_bg"))
        msg_container.pack(fill="both", expand=True)
        
        self.msg_canvas = tk.Canvas(msg_container, bg=self._c("chat_bg"), highlightthickness=0)
        self.msg_scrollbar = ttk.Scrollbar(msg_container, orient="vertical", command=self.msg_canvas.yview)
        self.msg_canvas.configure(yscrollcommand=self.msg_scrollbar.set)
        
        self.msg_scrollbar.pack(side="right", fill="y")
        self.msg_canvas.pack(side="left", fill="both", expand=True)
        
        self.msg_frame = tk.Frame(self.msg_canvas, bg=self._c("chat_bg"))
        self.msg_window = self.msg_canvas.create_window((0, 0), window=self.msg_frame, anchor="nw")
        
        self.msg_frame.bind("<Configure>",
            lambda e: self.msg_canvas.configure(scrollregion=self.msg_canvas.bbox("all")))
        self.msg_canvas.bind("<Configure>",
            lambda e: self.msg_canvas.itemconfig(self.msg_window, width=e.width))
        self.msg_canvas.bind_all("<MouseWheel>",
            lambda e: self.msg_canvas.yview_scroll(int(-e.delta/120), "units"))
        
        # Input area
        input_container = tk.Frame(self.chat_frame, bg=self._c("chat_bg"))
        input_container.pack(fill="x", padx=20, pady=16)
        
        # Input box
        input_outer = tk.Frame(input_container, bg=self._c("input_bg"))
        input_outer.pack(fill="x")
        
        input_inner = tk.Frame(input_outer, bg=self._c("input_bg"))
        input_inner.pack(fill="x", padx=12, pady=8)
        
        self.input_box = tk.Text(input_inner, height=3, wrap="word",
                                bg=self._c("input_bg"), fg=self._c("text"),
                                insertbackground=self._c("text"), relief="flat",
                                font=("Arial", 12))
        self.input_box.pack(side="left", fill="both", expand=True)
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: None)
        
        # Buttons
        btn_frame = tk.Frame(input_inner, bg=self._c("input_bg"))
        btn_frame.pack(side="right", padx=(10, 0))
        
        self.send_btn = tk.Button(btn_frame, text="➤", font=("Arial", 16),
                                 bg=self._c("btn_bg"), fg=self._c("text_blue"),
                                 activebackground=self._c("btn_hover"), relief="flat",
                                 cursor="hand2", width=3, command=self._send)
        self.send_btn.pack()
        
        self.stop_btn = tk.Button(btn_frame, text="■", font=("Arial", 12),
                                 bg=self._c("btn_bg"), fg=self._c("error"),
                                 activebackground=self._c("btn_hover"), relief="flat",
                                 cursor="hand2", width=3, command=self._stop_gen)
        self.stop_btn.pack(pady=(4, 0))
        
        # Hint
        tk.Label(input_container, 
                text="Tip: Use ```python code``` for executable code blocks",
                font=("Arial", 9), bg=self._c("chat_bg"), fg=self._c("text_muted")).pack(pady=(8, 0))
    
    def _build_canvas_view(self):
        """Build canvas view."""
        self.canvas_frame = tk.Frame(self.main_area, bg=self._c("chat_bg"))
        
        # Toolbar
        toolbar = tk.Frame(self.canvas_frame, bg=self._c("input_bg"))
        toolbar.pack(fill="x", padx=20, pady=10)
        
        tools = [("✏️ Pen", "pen"), ("📏 Line", "line"), ("⬜ Rect", "rect"),
                 ("⭕ Oval", "oval"), ("🔤 Text", "text"), ("🧹 Eraser", "eraser")]
        
        self.tool_var = tk.StringVar(value="pen")
        
        for label, tool in tools:
            btn = tk.Radiobutton(toolbar, text=label, variable=self.tool_var, value=tool,
                                bg=self._c("input_bg"), fg=self._c("text_blue"),
                                selectcolor=self._c("btn_bg"), activebackground=self._c("input_bg"),
                                indicatoron=False, padx=10, pady=5, cursor="hand2",
                                command=self._update_canvas_tool)
            btn.pack(side="left", padx=2)
        
        # Color picker
        self.color_btn = tk.Button(toolbar, text="🎨", font=("Arial", 12),
                                  bg=self._c("btn_bg"), fg="#3b82f6", relief="flat",
                                  cursor="hand2", command=self._pick_color)
        self.color_btn.pack(side="left", padx=(10, 2))
        
        # Size slider
        tk.Label(toolbar, text="Size:", bg=self._c("input_bg"), 
                fg=self._c("text_muted")).pack(side="left", padx=(10, 4))
        self.size_var = tk.IntVar(value=3)
        size_scale = tk.Scale(toolbar, from_=1, to=20, orient="horizontal",
                             variable=self.size_var, bg=self._c("input_bg"),
                             fg=self._c("text_blue"), highlightthickness=0,
                             length=100, command=lambda v: self._update_canvas_size())
        size_scale.pack(side="left")
        
        # Actions
        tk.Button(toolbar, text="↩ Undo", font=("Arial", 10),
                 bg=self._c("btn_bg"), fg=self._c("text_blue"), relief="flat",
                 cursor="hand2", command=self._canvas_undo).pack(side="right", padx=2)
        tk.Button(toolbar, text="🗑 Clear", font=("Arial", 10),
                 bg=self._c("btn_bg"), fg=self._c("error"), relief="flat",
                 cursor="hand2", command=self._canvas_clear).pack(side="right", padx=2)
        tk.Button(toolbar, text="💾 Save", font=("Arial", 10),
                 bg=self._c("btn_bg"), fg=self._c("success"), relief="flat",
                 cursor="hand2", command=self._canvas_save).pack(side="right", padx=2)
        
        # Canvas
        canvas_container = tk.Frame(self.canvas_frame, bg=self._c("border"))
        canvas_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        self.drawing_canvas = CanvasWidget(canvas_container, bg="white",
                                           highlightthickness=1, highlightbackground=self._c("border"))
        self.drawing_canvas.pack(fill="both", expand=True, padx=1, pady=1)
    
    def _build_files_view(self):
        """Build files view."""
        self.files_frame = tk.Frame(self.main_area, bg=self._c("chat_bg"))
        
        # Toolbar
        toolbar = tk.Frame(self.files_frame, bg=self._c("input_bg"))
        toolbar.pack(fill="x", padx=20, pady=10)
        
        tk.Button(toolbar, text="📤 Upload File", font=("Arial", 11),
                 bg=self._c("btn_bg"), fg=self._c("text_blue"), relief="flat",
                 cursor="hand2", padx=15, pady=5,
                 command=self._upload_file).pack(side="left", padx=4)
        
        tk.Button(toolbar, text="📝 New File", font=("Arial", 11),
                 bg=self._c("btn_bg"), fg=self._c("text_blue"), relief="flat",
                 cursor="hand2", padx=15, pady=5,
                 command=self._new_file).pack(side="left", padx=4)
        
        tk.Button(toolbar, text="🔄 Refresh", font=("Arial", 11),
                 bg=self._c("btn_bg"), fg=self._c("text_muted"), relief="flat",
                 cursor="hand2", command=self._refresh_files).pack(side="right", padx=4)
        
        # File list
        list_frame = tk.Frame(self.files_frame, bg=self._c("chat_bg"))
        list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Columns
        columns = ("name", "type", "size")
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        self.file_tree.heading("name", text="Name")
        self.file_tree.heading("type", text="Type")
        self.file_tree.heading("size", text="Size")
        
        self.file_tree.column("name", width=300)
        self.file_tree.column("type", width=100)
        self.file_tree.column("size", width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        self.file_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.file_tree.bind("<Double-1>", self._open_file)
        self.file_tree.bind("<Delete>", self._delete_file)
        
        # Preview area
        preview_frame = tk.Frame(self.files_frame, bg=self._c("input_bg"))
        preview_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        tk.Label(preview_frame, text="Preview", font=("Arial", 10, "bold"),
                bg=self._c("input_bg"), fg=self._c("text_blue")).pack(anchor="w", padx=10, pady=5)
        
        self.preview_text = tk.Text(preview_frame, height=8, wrap="word",
                                   bg=self._c("code_bg"), fg=self._c("text"),
                                   font=("Consolas", 10), relief="flat")
        self.preview_text.pack(fill="x", padx=10, pady=(0, 10))
    
    def _build_right_panel(self):
        """Build right tool panel."""
        right_panel = tk.Frame(self.main_container, bg=self._c("sidebar"), width=280)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)
        
        # Title
        tk.Label(right_panel, text="🛠 Tools", font=("Arial", 12, "bold"),
                bg=self._c("sidebar"), fg=self._c("text_blue")).pack(pady=16)
        
        tk.Frame(right_panel, bg=self._c("border"), height=1).pack(fill="x", padx=12)
        
        # Code interpreter section
        tk.Label(right_panel, text="Code Interpreter", font=("Arial", 10, "bold"),
                bg=self._c("sidebar"), fg=self._c("text")).pack(anchor="w", padx=16, pady=(12, 4))
        
        code_frame = tk.Frame(right_panel, bg=self._c("sidebar"))
        code_frame.pack(fill="x", padx=12)
        
        self.code_input = tk.Text(code_frame, height=6, wrap="word",
                                 bg=self._c("code_bg"), fg=self._c("text"),
                                 font=("Consolas", 10), relief="flat")
        self.code_input.pack(fill="x", pady=4)
        
        tk.Button(code_frame, text="▶ Run Code", font=("Arial", 10),
                 bg=self._c("btn_bg"), fg=self._c("success"),
                 activebackground=self._c("btn_hover"), relief="flat",
                 cursor="hand2", command=self._run_code).pack(fill="x", pady=4)
        
        # Output
        tk.Label(right_panel, text="Output", font=("Arial", 10),
                bg=self._c("sidebar"), fg=self._c("text_muted")).pack(anchor="w", padx=16, pady=(8, 4))
        
        self.code_output = tk.Text(right_panel, height=8, wrap="word",
                                  bg=self._c("code_bg"), fg=self._c("text"),
                                  font=("Consolas", 9), relief="flat")
        self.code_output.pack(fill="x", padx=12, pady=4)
        
        tk.Frame(right_panel, bg=self._c("border"), height=1).pack(fill="x", padx=12, pady=8)
        
        # Quick actions
        tk.Label(right_panel, text="Quick Actions", font=("Arial", 10, "bold"),
                bg=self._c("sidebar"), fg=self._c("text")).pack(anchor="w", padx=16, pady=(4, 8))
        
        actions_frame = tk.Frame(right_panel, bg=self._c("sidebar"))
        actions_frame.pack(fill="x", padx=12)
        
        actions = [
            ("💡 Explain", "Explain this concept: "),
            ("🔍 Analyze", "Analyze the following: "),
            ("📝 Summarize", "Summarize this text: "),
            ("🐛 Debug", "Debug this code: "),
        ]
        
        for label, prefix in actions:
            tk.Button(actions_frame, text=label, font=("Arial", 10),
                     bg=self._c("btn_bg"), fg=self._c("text_blue"),
                     activebackground=self._c("btn_hover"), relief="flat",
                     cursor="hand2", anchor="w", padx=10,
                     command=lambda p=prefix: self._quick_action(p)).pack(fill="x", pady=2)
    
    # ===== View Switching =====
    def _switch_view(self, view: str):
        self.current_view = view
        
        # Hide all
        self.chat_frame.pack_forget()
        self.canvas_frame.pack_forget()
        self.files_frame.pack_forget()
        
        # Show selected
        titles = {"chat": "💬 Chat", "canvas": "🎨 Canvas", "files": "📁 Files"}
        self.view_title.configure(text=titles.get(view, "Chat"))
        
        if view == "chat":
            self.chat_frame.pack(fill="both", expand=True)
        elif view == "canvas":
            self.canvas_frame.pack(fill="both", expand=True)
        elif view == "files":
            self.files_frame.pack(fill="both", expand=True)
            self._refresh_files()
    
    # ===== Chat Functions =====
    def _refresh_chat_list(self):
        for w in self.chat_list_frame.winfo_children():
            w.destroy()
        
        for i, conv in enumerate(self.conversations[:15]):
            title = conv.title[:25] + "..." if len(conv.title) > 25 else conv.title
            bg = self._c("sidebar_hover") if i == self.current_idx else self._c("sidebar")
            
            btn = tk.Button(self.chat_list_frame, text=f"💬 {title}",
                           font=("Arial", 10), bg=bg, fg=self._c("text_blue"),
                           activebackground=self._c("sidebar_hover"), relief="flat",
                           anchor="w", padx=10, pady=6, cursor="hand2",
                           command=lambda idx=i: self._select_chat(idx))
            btn.pack(fill="x", pady=1)
    
    def _select_chat(self, idx: int):
        self.current_idx = idx
        self._refresh_chat_list()
        self._render_chat()
    
    def _new_chat(self):
        self.conversations.insert(0, Conversation())
        self.current_idx = 0
        self._refresh_chat_list()
        self._clear_messages()
        self._add_message("assistant", "New conversation started. How can I help?",
                         thinking="Ready for new task...")
    
    def _clear_chat(self):
        if messagebox.askyesno("Clear", "Clear this conversation?"):
            self.current.messages.clear()
            self._clear_messages()
    
    def _render_chat(self):
        self._clear_messages()
        for msg in self.current.messages:
            self._add_message(msg.role, msg.content, msg.thinking, msg.code_result, save=False)
    
    def _clear_messages(self):
        for w in self.msg_frame.winfo_children():
            w.destroy()
    
    def _add_message(self, role: str, content: str, thinking: str = "",
                    code_result: dict = None, save: bool = True):
        """Add message to chat."""
        container = tk.Frame(self.msg_frame, bg=self._c("chat_bg"))
        container.pack(fill="x", padx=20, pady=10)
        
        # Row with avatar
        row = tk.Frame(container, bg=self._c("chat_bg"))
        row.pack(fill="x")
        
        # Avatar
        avatar_text = "🐱" if role == "assistant" else "👤"
        avatar_bg = self._c("ai_msg") if role == "assistant" else self._c("user_msg")
        
        avatar = tk.Label(row, text=avatar_text, font=("Arial", 16),
                         bg=avatar_bg, width=3)
        avatar.pack(side="left", anchor="n", padx=(0, 12))
        
        # Content
        content_frame = tk.Frame(row, bg=self._c("chat_bg"))
        content_frame.pack(side="left", fill="x", expand=True)
        
        # Name
        name = "CatGPT o1" if role == "assistant" else "You"
        tk.Label(content_frame, text=name, font=("Arial", 11, "bold"),
                bg=self._c("chat_bg"), fg=self._c("text_blue")).pack(anchor="w")
        
        # Thinking block (o1 style)
        if role == "assistant" and thinking:
            think_frame = tk.Frame(content_frame, bg=self._c("thinking_bg"))
            think_frame.pack(fill="x", pady=(6, 0))
            
            think_header = tk.Frame(think_frame, bg=self._c("thinking_bg"))
            think_header.pack(fill="x", padx=10, pady=(8, 4))
            
            tk.Label(think_header, text="💭", font=("Arial", 10),
                    bg=self._c("thinking_bg"), fg=self._c("accent_light")).pack(side="left")
            tk.Label(think_header, text="Thought for a moment", font=("Arial", 9, "italic"),
                    bg=self._c("thinking_bg"), fg=self._c("text_muted")).pack(side="left", padx=(4, 0))
            
            tk.Label(think_frame, text=thinking, font=("Arial", 10),
                    bg=self._c("thinking_bg"), fg=self._c("text_muted"),
                    wraplength=650, justify="left").pack(anchor="w", padx=10, pady=(0, 8))
        
        # Message content
        msg_label = tk.Label(content_frame, text=content, font=("Arial", 11),
                            bg=self._c("chat_bg"), fg=self._c("text"),
                            wraplength=650, justify="left", anchor="w")
        msg_label.pack(anchor="w", pady=(6, 0))
        
        # Code result block
        if code_result:
            code_frame = tk.Frame(content_frame, bg=self._c("code_bg"))
            code_frame.pack(fill="x", pady=(8, 0))
            
            # Code header
            header = tk.Frame(code_frame, bg="#1a1a1a")
            header.pack(fill="x")
            tk.Label(header, text="python", font=("Consolas", 9),
                    bg="#1a1a1a", fg=self._c("text_muted")).pack(side="left", padx=10, pady=4)
            
            # Code
            code_text = tk.Text(code_frame, height=min(10, code_result["code"].count('\n') + 2),
                               bg=self._c("code_bg"), fg="#a5f3fc", font=("Consolas", 10),
                               relief="flat", wrap="none")
            code_text.insert("1.0", code_result["code"])
            code_text.configure(state="disabled")
            code_text.pack(fill="x", padx=10, pady=4)
            
            # Output
            if code_result.get("stdout") or code_result.get("error"):
                tk.Label(code_frame, text="Output:", font=("Consolas", 9),
                        bg=self._c("code_bg"), fg=self._c("text_muted")).pack(anchor="w", padx=10)
                
                output = code_result.get("stdout", "") or code_result.get("error", "")
                out_color = self._c("success") if code_result.get("success") else self._c("error")
                
                out_text = tk.Text(code_frame, height=min(5, output.count('\n') + 1),
                                  bg=self._c("code_bg"), fg=out_color, font=("Consolas", 10),
                                  relief="flat")
                out_text.insert("1.0", output[:500])
                out_text.configure(state="disabled")
                out_text.pack(fill="x", padx=10, pady=(0, 8))
        
        # Copy button
        tk.Button(content_frame, text="📋 Copy", font=("Arial", 9),
                 bg=self._c("btn_bg"), fg=self._c("text_muted"), relief="flat",
                 cursor="hand2", command=lambda: self._copy(content)).pack(anchor="w", pady=(8, 0))
        
        if save:
            self.current.messages.append(Message(role, content, thinking, code_result))
            if role == "user" and len(self.current.messages) == 1:
                self.current.title = content[:40]
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
        self._add_message("user", text)
        
        # Check for code blocks
        code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            self._execute_code_in_chat(code)
        else:
            self._stream_response(text)
    
    def _execute_code_in_chat(self, code: str):
        """Execute code and show result."""
        result = self.interpreter.execute(code)
        
        thinking = "Executing Python code in sandbox environment..."
        response = "Code executed "
        response += "successfully." if result["success"] else "with errors."
        
        if result.get("stdout"):
            response += f"\n\nOutput:\n{result['stdout'][:500]}"
        if result.get("error"):
            response += f"\n\nError:\n{result['error'][:500]}"
        
        self._add_message("assistant", response, thinking=thinking, code_result=result)
    
    def _stop_gen(self):
        self._stop = True
    
    def _stream_response(self, user_text: str):
        """Generate streaming response."""
        self._streaming = True
        self._stop = False
        
        self.status_dot.configure(fg=self._c("warning"))
        self.status_text.configure(text="Thinking...")
        
        # Create response container
        container = tk.Frame(self.msg_frame, bg=self._c("chat_bg"))
        container.pack(fill="x", padx=20, pady=10)
        
        row = tk.Frame(container, bg=self._c("chat_bg"))
        row.pack(fill="x")
        
        avatar = tk.Label(row, text="🐱", font=("Arial", 16), bg=self._c("ai_msg"), width=3)
        avatar.pack(side="left", anchor="n", padx=(0, 12))
        
        content_frame = tk.Frame(row, bg=self._c("chat_bg"))
        content_frame.pack(side="left", fill="x", expand=True)
        
        tk.Label(content_frame, text="CatGPT o1", font=("Arial", 11, "bold"),
                bg=self._c("chat_bg"), fg=self._c("text_blue")).pack(anchor="w")
        
        # Thinking block
        think_frame = tk.Frame(content_frame, bg=self._c("thinking_bg"))
        think_frame.pack(fill="x", pady=(6, 0))
        
        think_header = tk.Frame(think_frame, bg=self._c("thinking_bg"))
        think_header.pack(fill="x", padx=10, pady=(8, 4))
        
        tk.Label(think_header, text="💭", font=("Arial", 10),
                bg=self._c("thinking_bg"), fg=self._c("accent_light")).pack(side="left")
        think_status = tk.Label(think_header, text="Thinking...", font=("Arial", 9, "italic"),
                               bg=self._c("thinking_bg"), fg=self._c("text_muted"))
        think_status.pack(side="left", padx=(4, 0))
        
        think_content = tk.Label(think_frame, text="Analyzing your request and formulating a response...",
                                font=("Arial", 10), bg=self._c("thinking_bg"), fg=self._c("text_muted"),
                                wraplength=650, justify="left")
        think_content.pack(anchor="w", padx=10, pady=(0, 8))
        
        # Response
        txt_var = tk.StringVar(value="")
        msg_label = tk.Label(content_frame, textvariable=txt_var, font=("Arial", 11),
                            bg=self._c("chat_bg"), fg=self._c("text"),
                            wraplength=650, justify="left", anchor="w")
        msg_label.pack(anchor="w", pady=(6, 0))
        
        # Animate thinking
        def animate():
            if self._streaming:
                self._thinking_anim += 1
                dots = "." * ((self._thinking_anim % 3) + 1)
                think_status.configure(text=f"Thinking{dots}")
                self.after(400, animate)
        animate()
        
        # Build prompt
        prompt = f"User asks: {user_text}\n\nAssistant responds thoughtfully: "
        
        # Generate in thread
        def gen_thread():
            for token in self.model.generate(prompt, 200, lambda: self._stop):
                self._queue.put(token)
            self._queue.put(None)
        
        Thread(target=gen_thread, daemon=True).start()
        
        def poll():
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None:
                        self._finish_response(txt_var.get(), think_status, think_content)
                        return
                    current = txt_var.get()
                    txt_var.set(current + item)
                    self._scroll_bottom()
            except queue.Empty:
                pass
            
            if self._stop:
                self._finish_response(txt_var.get(), think_status, think_content)
                return
            
            self.after(20, poll)
        
        poll()
    
    def _finish_response(self, content: str, status_label, content_label):
        self._streaming = False
        
        self.status_dot.configure(fg=self._c("success"))
        self.status_text.configure(text="Ready")
        
        status_label.configure(text="Thought for a moment")
        content_label.configure(text="Analyzed the request and generated a response.")
        
        content = content.strip()
        if content:
            self.current.messages.append(Message("assistant", content, 
                                                "Analyzed the request and generated a response."))
        
        while not self._queue.empty():
            self._queue.get_nowait()
    
    # ===== Code Interpreter =====
    def _run_code(self):
        code = self.code_input.get("1.0", "end").strip()
        if not code:
            return
        
        result = self.interpreter.execute(code)
        
        self.code_output.configure(state="normal")
        self.code_output.delete("1.0", "end")
        
        if result["success"]:
            output = result.get("stdout", "") or str(result.get("result", ""))
            self.code_output.insert("1.0", output or "[No output]")
            self.code_output.configure(fg=self._c("success"))
        else:
            self.code_output.insert("1.0", result.get("error", "Unknown error"))
            self.code_output.configure(fg=self._c("error"))
        
        self.code_output.configure(state="disabled")
    
    def _quick_action(self, prefix: str):
        current = self.input_box.get("1.0", "end").strip()
        self.input_box.delete("1.0", "end")
        self.input_box.insert("1.0", prefix + current)
        self.input_box.focus_set()
    
    # ===== Canvas Functions =====
    def _update_canvas_tool(self):
        self.drawing_canvas.tool = self.tool_var.get()
    
    def _update_canvas_size(self):
        self.drawing_canvas.brush_size = self.size_var.get()
    
    def _pick_color(self):
        color = colorchooser.askcolor(title="Choose Color")[1]
        if color:
            self.drawing_canvas.color = color
            self.color_btn.configure(fg=color)
    
    def _canvas_undo(self):
        self.drawing_canvas.undo()
    
    def _canvas_clear(self):
        if messagebox.askyesno("Clear", "Clear the canvas?"):
            self.drawing_canvas.clear_all()
    
    def _canvas_save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".ps",
            filetypes=[("PostScript", "*.ps"), ("All Files", "*.*")]
        )
        if path:
            self.drawing_canvas.export_ps(path)
            messagebox.showinfo("Saved", f"Canvas saved to {path}")
    
    # ===== File Functions =====
    def _upload_file(self):
        path = filedialog.askopenfilename()
        if path:
            info = self.file_sandbox.add_file(path)
            if "error" not in info:
                self._refresh_files()
                messagebox.showinfo("Uploaded", f"File '{info['name']}' added to sandbox.")
            else:
                messagebox.showerror("Error", info["error"])
    
    def _new_file(self):
        name = tk.simpledialog.askstring("New File", "Enter filename:")
        if name:
            self.file_sandbox.write_file(name, "")
            self._refresh_files()
    
    def _refresh_files(self):
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        for f in self.file_sandbox.list_files():
            size = f"{f['size']} B" if f['size'] < 1024 else f"{f['size']//1024} KB"
            self.file_tree.insert("", "end", values=(f["name"], f["type"], size))
    
    def _open_file(self, event):
        selection = self.file_tree.selection()
        if selection:
            item = self.file_tree.item(selection[0])
            filename = item["values"][0]
            content = self.file_sandbox.read_file(filename)
            
            self.preview_text.configure(state="normal")
            self.preview_text.delete("1.0", "end")
            self.preview_text.insert("1.0", content[:2000])
            self.preview_text.configure(state="disabled")
    
    def _delete_file(self, event):
        selection = self.file_tree.selection()
        if selection:
            item = self.file_tree.item(selection[0])
            filename = item["values"][0]
            if messagebox.askyesno("Delete", f"Delete '{filename}'?"):
                self.file_sandbox.delete_file(filename)
                self._refresh_files()
    
    # ===== Settings =====
    def _open_settings(self):
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("450x400")
        win.configure(bg=self._c("bg"))
        
        tk.Label(win, text="⚙ Settings", font=("Arial", 14, "bold"),
                bg=self._c("bg"), fg=self._c("text_blue")).pack(pady=20)
        
        frm = tk.Frame(win, bg=self._c("bg"))
        frm.pack(fill="x", padx=30)
        
        # Temperature
        tk.Label(frm, text="Temperature", font=("Arial", 11),
                bg=self._c("bg"), fg=self._c("text_blue")).grid(row=0, column=0, sticky="w", pady=8)
        t_var = tk.DoubleVar(value=self.model.temperature)
        tk.Scale(frm, from_=0.1, to=1.5, resolution=0.05, orient="horizontal",
                variable=t_var, bg=self._c("bg"), fg=self._c("text_blue"),
                highlightthickness=0, troughcolor=self._c("input_bg"),
                length=200).grid(row=0, column=1, sticky="ew", padx=10)
        
        # Repetition penalty
        tk.Label(frm, text="Repetition Penalty", font=("Arial", 11),
                bg=self._c("bg"), fg=self._c("text_blue")).grid(row=1, column=0, sticky="w", pady=8)
        r_var = tk.DoubleVar(value=self.model.rep_penalty)
        tk.Scale(frm, from_=1.0, to=2.0, resolution=0.05, orient="horizontal",
                variable=r_var, bg=self._c("bg"), fg=self._c("text_blue"),
                highlightthickness=0, troughcolor=self._c("input_bg"),
                length=200).grid(row=1, column=1, sticky="ew", padx=10)
        
        # Markov order
        tk.Label(frm, text="Markov Order", font=("Arial", 11),
                bg=self._c("bg"), fg=self._c("text_blue")).grid(row=2, column=0, sticky="w", pady=8)
        o_var = tk.IntVar(value=self.model.order)
        tk.Scale(frm, from_=2, to=6, resolution=1, orient="horizontal",
                variable=o_var, bg=self._c("bg"), fg=self._c("text_blue"),
                highlightthickness=0, troughcolor=self._c("input_bg"),
                length=200).grid(row=2, column=1, sticky="ew", padx=10)
        
        frm.columnconfigure(1, weight=1)
        
        # Train on custom text
        tk.Label(win, text="Train on Custom Text:", font=("Arial", 10),
                bg=self._c("bg"), fg=self._c("text_muted")).pack(anchor="w", padx=30, pady=(20, 4))
        
        train_text = tk.Text(win, height=5, bg=self._c("input_bg"), fg=self._c("text"),
                            font=("Arial", 10), relief="flat")
        train_text.pack(fill="x", padx=30, pady=4)
        
        def save():
            self.model.temperature = t_var.get()
            self.model.rep_penalty = r_var.get()
            
            # Train on custom text
            custom = train_text.get("1.0", "end").strip()
            if custom:
                self.model.train(custom)
                messagebox.showinfo("Trained", f"Model trained on {len(custom)} characters.")
            
            win.destroy()
        
        tk.Button(win, text="Save", font=("Arial", 11),
                 bg=self._c("btn_bg"), fg=self._c("text_blue"),
                 activebackground=self._c("btn_hover"), relief="flat",
                 padx=30, pady=8, cursor="hand2", command=save).pack(pady=20)
        
        win.transient(self)
        win.grab_set()


# =====================================================
# Main
# =====================================================

def main():
    # Required for simpledialog
    import tkinter.simpledialog
    
    app = CatGPTApp()
    app.mainloop()


if __name__ == "__main__":
    main()
