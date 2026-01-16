#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatSEEK R1 — AI Chat Interface
A DeepSeek-style chat interface with reasoning capabilities.

Features:
  • Deep Think reasoning visualization
  • Code Interpreter (Python execution sandbox)
  • File Sandbox (upload, view, manage files)
  • Canvas (drawing, diagrams, image editing)
  • Clean text generation

Python 3.10+ | Pure tkinter + stdlib
Run: python catseek_r1.py
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

APP_NAME = "CatSEEK"
VERSION = "R1"

# =====================================================
# o1 Reasoning Engine (Exact Mimic)
# =====================================================

class O1ReasoningEngine:
    """
    Mimics OpenAI o1's reasoning style exactly.
    - Extended thinking process
    - Step-by-step chain of thought
    - Self-questioning and verification
    - Structured output
    """
    
    def __init__(self):
        self.temperature = 0.7
        self.rep_penalty = 1.1
        
        # o1 thinking phrases
        self.thinking_starters = [
            "Let me think through this carefully.",
            "Okay, let's break this down.",
            "Hmm, interesting question.",
            "Let me consider this step by step.",
            "Alright, I need to think about this.",
        ]
        
        self.thinking_continues = [
            "So first, ", "Now, ", "Next, ", "Then, ", "Also, ",
            "Additionally, ", "Furthermore, ", "Moreover, ",
            "Wait, ", "Hmm, ", "Actually, ", "Let me reconsider... ",
            "On second thought, ", "I should also consider ",
            "Another angle: ", "What about ", "Let's see... ",
        ]
        
        self.self_questions = [
            "Is that right?",
            "Does this make sense?",
            "Am I missing something?",
            "What are the edge cases?",
            "Could there be another interpretation?",
            "Let me verify this.",
            "Wait, let me double-check.",
            "Is there a simpler approach?",
        ]
        
        self.conclusions = [
            "So, putting it all together: ",
            "Therefore, the answer is: ",
            "In conclusion: ",
            "So my final answer: ",
            "Based on my reasoning: ",
        ]
        
        # Response templates for common queries
        self.templates = self._build_templates()
    
    def _build_templates(self) -> dict:
        return {
            "hello": {
                "thinking": "The user is greeting me. I should respond warmly.",
                "response": "Hi! I'm CatSEEK, your AI assistant. How can I help you today?"
            },
            "hi": {
                "thinking": "A greeting. I'll respond and offer assistance.",
                "response": "Hello! What can I help you with?"
            },
            "how are you": {
                "thinking": "The user is asking about my state.",
                "response": "I'm doing well, thank you! How can I assist you today?"
            },
            "what are you": {
                "thinking": "They want to know my identity.",
                "response": "I'm CatSEEK R1, an AI assistant created to help with reasoning, coding, writing, and analysis. I show my thinking process so you can follow my logic."
            },
            "who are you": {
                "thinking": "Identity question.",
                "response": "I'm CatSEEK, your AI assistant. I can help you search, read, write, and create. My Deep Think feature lets you see my reasoning process."
            },
            "help": {
                "thinking": "The user needs guidance.",
                "response": "I can help you with:\n\n• Search and research\n• Writing and editing\n• Code and debugging\n• Math and analysis\n• Creative tasks\n\nJust ask me anything!"
            },
            "thank": {
                "thinking": "User expressing gratitude.",
                "response": "You're welcome! Let me know if you need anything else."
            },
            "thanks": {
                "thinking": "Gratitude.",
                "response": "Happy to help!"
            },
            "bye": {
                "thinking": "User ending conversation.",
                "response": "Goodbye! Feel free to come back anytime."
            },
        }
    
    def _detect_query_type(self, text: str) -> str:
        """Detect the type of query for appropriate reasoning."""
        text_lower = text.lower()
        
        # Math detection
        if any(w in text_lower for w in ["calculate", "solve", "equation", "math", "sum", "product", "divide", "multiply", "+", "-", "*", "/", "="]):
            return "math"
        
        # Code detection
        if any(w in text_lower for w in ["code", "program", "function", "bug", "error", "python", "javascript", "debug", "implement", "algorithm"]):
            return "code"
        
        # Explanation detection
        if any(w in text_lower for w in ["explain", "what is", "what are", "how does", "why does", "describe", "define"]):
            return "explain"
        
        # Comparison detection
        if any(w in text_lower for w in ["compare", "difference", "versus", "vs", "better", "worse", "pros", "cons"]):
            return "compare"
        
        # Analysis detection
        if any(w in text_lower for w in ["analyze", "analysis", "evaluate", "assess", "review"]):
            return "analyze"
        
        return "general"
    
    def _generate_thinking(self, query: str, query_type: str) -> str:
        """Generate o1-style thinking process."""
        thinking = random.choice(self.thinking_starters) + "\n\n"
        
        if query_type == "math":
            thinking += "This is a math problem. Let me identify what we're solving for.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "I need to break down the mathematical components.\n\n"
            thinking += random.choice(self.self_questions) + "\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "Let me work through this step by step to avoid errors.\n\n"
            thinking += "I should verify my arithmetic at each step.\n\n"
            
        elif query_type == "code":
            thinking += "This is a programming question. Let me think about the approach.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "What's the core problem we're trying to solve?\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "I need to consider edge cases and error handling.\n\n"
            thinking += random.choice(self.self_questions) + "\n\n"
            thinking += "Let me think about the algorithm's time and space complexity.\n\n"
            
        elif query_type == "explain":
            thinking += "The user wants an explanation. I should be clear and thorough.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "let me identify the key concepts involved.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "I should start with the fundamentals and build up.\n\n"
            thinking += random.choice(self.self_questions) + "\n\n"
            thinking += "I'll use analogies if they help clarify.\n\n"
            
        elif query_type == "compare":
            thinking += "This is a comparison question. I need to be balanced and fair.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "let me identify the key dimensions to compare.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "I should consider the strengths and weaknesses of each.\n\n"
            thinking += random.choice(self.self_questions) + "\n\n"
            thinking += "Context matters here - different situations favor different options.\n\n"
            
        elif query_type == "analyze":
            thinking += "This requires analysis. I need to be systematic.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "let me identify the key factors at play.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "I should consider multiple perspectives.\n\n"
            thinking += random.choice(self.self_questions) + "\n\n"
            thinking += "Let me weigh the evidence carefully.\n\n"
            
        else:
            thinking += "Let me understand what's being asked here.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "I should consider the context and intent.\n\n"
            thinking += random.choice(self.thinking_continues)
            thinking += "what's the best way to approach this?\n\n"
            thinking += random.choice(self.self_questions) + "\n\n"
            thinking += "I'll provide a clear, helpful response.\n\n"
        
        thinking += random.choice(self.conclusions)
        return thinking
    
    def _generate_response(self, query: str, query_type: str) -> str:
        """Generate the final response after thinking."""
        
        if query_type == "math":
            return """Based on my analysis, here's the solution:

**Approach:**
I broke down the problem into smaller parts and worked through each step systematically.

**Solution:**
The key is to identify the mathematical relationships and apply the appropriate operations in order.

**Verification:**
I double-checked my work by reviewing each step and confirming the logic is sound.

Let me know if you'd like me to show more detailed steps!"""
        
        elif query_type == "code":
            return """Here's my solution:

**Approach:**
I identified the core problem and designed an algorithm to solve it efficiently.

**Implementation considerations:**
- Handle edge cases (empty input, invalid data)
- Consider time/space complexity
- Keep the code readable and maintainable

**Recommendation:**
Start with a simple working solution, then optimize if needed.

Would you like me to write actual code for a specific implementation?"""
        
        elif query_type == "explain":
            return """Here's my explanation:

**Core concept:**
At its heart, this involves understanding the fundamental principles at work.

**How it works:**
The key mechanism is the interaction between the main components. Each part plays a specific role in the overall system.

**Why it matters:**
Understanding this helps you see the bigger picture and apply the knowledge in related situations.

**Summary:**
The essential takeaway is that the concept builds on basic principles to achieve its purpose.

Would you like me to go deeper on any part?"""
        
        elif query_type == "compare":
            return """Here's my comparison:

**Key similarities:**
Both share fundamental characteristics that serve similar purposes.

**Key differences:**
The main distinctions are in their approach, scope, and best use cases.

**When to use each:**
- Choose the first when you need X
- Choose the second when you need Y

**My recommendation:**
It depends on your specific context and requirements. Consider your priorities and constraints.

Want me to elaborate on any aspect?"""
        
        elif query_type == "analyze":
            return """Here's my analysis:

**Key observations:**
Looking at the evidence, several important patterns emerge.

**Interpretation:**
These patterns suggest underlying factors that drive the observed outcomes.

**Implications:**
Based on this analysis, we can draw several conclusions about what this means going forward.

**Confidence level:**
I'm reasonably confident in this analysis, though additional data could refine the conclusions.

Shall I explore any aspect in more detail?"""
        
        else:
            return """Based on my reasoning, here's my response:

I've thought through your question carefully, considering multiple angles and potential interpretations.

**My answer:**
The key points to understand are:
1. The foundational concepts that apply here
2. How they interact in this specific context
3. What conclusions we can draw

**Additional thoughts:**
There may be nuances depending on your specific situation. Feel free to provide more context if you'd like a more tailored response.

Is there anything you'd like me to clarify or expand on?"""
    
    def generate(self, prompt: str, max_tokens: int = 200,
                 stop_fn: callable = None) -> Generator[str, None, None]:
        """Generate o1-style response with thinking."""
        
        # Check for simple template matches
        prompt_lower = prompt.lower().strip()
        for key, template in self.templates.items():
            if key in prompt_lower or prompt_lower.startswith(key):
                # Stream the response
                for char in template["response"]:
                    if stop_fn and stop_fn():
                        break
                    yield char
                    time.sleep(0.006)
                return
        
        # For other queries, generate reasoned response
        query_type = self._detect_query_type(prompt)
        response = self._generate_response(prompt, query_type)
        
        # Stream word by word
        words = response.split(' ')
        for i, word in enumerate(words):
            if stop_fn and stop_fn():
                break
            yield word
            if i < len(words) - 1:
                yield ' '
            time.sleep(0.015)
    
    def generate_thinking(self, prompt: str) -> str:
        """Generate the thinking portion for display."""
        prompt_lower = prompt.lower().strip()
        
        # Check templates
        for key, template in self.templates.items():
            if key in prompt_lower or prompt_lower.startswith(key):
                return template["thinking"]
        
        # Generate dynamic thinking
        query_type = self._detect_query_type(prompt)
        return self._generate_thinking(prompt, query_type)
    
    def train(self, text: str):
        """Add custom text patterns."""
        # Extract useful phrases
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 100:
                if sentence not in self.thinking_continues:
                    self.thinking_continues.append(sentence + ". ")


# Alias for compatibility
LanguageModel = O1ReasoningEngine
MarkovLM = O1ReasoningEngine


# =====================================================
# Code Interpreter (Python Sandbox)
# =====================================================

class CodeInterpreter:
    """
    Safe Python code execution sandbox.
    Executes code and captures output.
    """
    
    def __init__(self, sandbox_dir: str = None):
        self.sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="catseek_")
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
        self.sandbox_dir = Path(sandbox_dir or tempfile.mkdtemp(prefix="catseek_files_"))
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
    model: str = "CatSEEK-R1"


# =====================================================
# Main Application (CatSEEK-style UI)
# =====================================================

class CatSEEKApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1300x850")
        self.minsize(1100, 700)
        
        # CatSEEK-style colors (gray/black + blue text)
        self.colors = {
            "bg": "#0d0d0d",
            "sidebar": "#0a0a0a",
            "sidebar_hover": "#1a1a1a",
            "chat_bg": "#0d0d0d",
            "input_bg": "#1a1a1a",
            "input_border": "#2a2a2a",
            "user_msg": "#1f1f1f",
            "ai_msg": "#141414",
            "thinking_bg": "#111118",
            "thinking_border": "#1e1e2e",
            "code_bg": "#0c0c0c",
            "text": "#e4e4e7",
            "text_blue": "#3b82f6",
            "text_light_blue": "#60a5fa",
            "text_muted": "#6b7280",
            "accent": "#2563eb",
            "border": "#1e1e1e",
            "success": "#10b981",
            "error": "#ef4444",
            "warning": "#f59e0b",
            "btn_bg": "#1a1a1a",
            "btn_hover": "#252525",
        }
        
        # State
        self.conversations: list[Conversation] = [Conversation()]
        self.current_idx = 0
        self.model = O1ReasoningEngine()
        self.interpreter = CodeInterpreter()
        self.file_sandbox = FileSandbox()
        
        self._streaming = False
        self._stop = False
        self._queue: queue.Queue = queue.Queue()
        
        # Current view
        self.current_view = "chat"
        
        self._build_ui()
        
        # Welcome
        self._add_message("assistant",
            "Hi, I'm CatSEEK, your AI assistant. I can search, read, write, and create. How can I help you today?",
            thinking="Ready to assist with reasoning, coding, and analysis."
        )
    
    @property
    def current(self) -> Conversation:
        return self.conversations[self.current_idx]
    
    def _c(self, key: str) -> str:
        return self.colors.get(key, "#ffffff")
    
    def _build_ui(self):
        self.configure(bg=self._c("bg"))
        
        # ===== SIDEBAR (CatSEEK style) =====
        self.sidebar = tk.Frame(self, bg=self._c("sidebar"), width=280)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        # Logo area
        logo_frame = tk.Frame(self.sidebar, bg=self._c("sidebar"))
        logo_frame.pack(fill="x", padx=20, pady=20)
        
        tk.Label(logo_frame, text="🐱", font=("Arial", 28),
                bg=self._c("sidebar"), fg=self._c("text")).pack(side="left")
        
        logo_text = tk.Frame(logo_frame, bg=self._c("sidebar"))
        logo_text.pack(side="left", padx=(10, 0))
        
        tk.Label(logo_text, text="CatSEEK", font=("Arial", 18, "bold"),
                bg=self._c("sidebar"), fg=self._c("text_blue")).pack(anchor="w")
        tk.Label(logo_text, text="R1", font=("Arial", 10),
                bg=self._c("sidebar"), fg=self._c("text_muted")).pack(anchor="w")
        
        # New chat button
        new_btn_frame = tk.Frame(self.sidebar, bg=self._c("sidebar"))
        new_btn_frame.pack(fill="x", padx=16, pady=(0, 20))
        
        self.new_btn = tk.Button(new_btn_frame, text="＋  New Chat",
                                font=("Arial", 12), bg=self._c("btn_bg"), fg=self._c("text_blue"),
                                activebackground=self._c("btn_hover"), activeforeground=self._c("text_light_blue"),
                                relief="flat", cursor="hand2", pady=12, padx=20,
                                command=self._new_chat)
        self.new_btn.pack(fill="x")
        
        # Divider
        tk.Frame(self.sidebar, bg=self._c("border"), height=1).pack(fill="x", padx=16)
        
        # View tabs
        tabs_frame = tk.Frame(self.sidebar, bg=self._c("sidebar"))
        tabs_frame.pack(fill="x", padx=16, pady=16)
        
        self.tab_btns = {}
        for view, (icon, label) in [("chat", ("💬", "Chat")), ("canvas", ("🎨", "Canvas")), ("files", ("📁", "Files"))]:
            btn = tk.Button(tabs_frame, text=f"{icon} {label}", font=("Arial", 10),
                           bg=self._c("sidebar"), fg=self._c("text_blue"),
                           activebackground=self._c("sidebar_hover"), activeforeground=self._c("text_light_blue"),
                           relief="flat", cursor="hand2", padx=8, pady=6,
                           command=lambda v=view: self._switch_view(v))
            btn.pack(side="left", padx=2)
            self.tab_btns[view] = btn
        
        self.tab_btns["chat"].configure(bg=self._c("sidebar_hover"))
        
        # Chat history label
        tk.Label(self.sidebar, text="Recent", font=("Arial", 10, "bold"),
                bg=self._c("sidebar"), fg=self._c("text_muted")).pack(anchor="w", padx=20, pady=(10, 8))
        
        # Chat list
        self.chat_list_frame = tk.Frame(self.sidebar, bg=self._c("sidebar"))
        self.chat_list_frame.pack(fill="both", expand=True, padx=12)
        
        self._refresh_chat_list()
        
        # Bottom section
        bottom = tk.Frame(self.sidebar, bg=self._c("sidebar"))
        bottom.pack(fill="x", padx=16, pady=16)
        
        tk.Button(bottom, text="⚙ Settings", font=("Arial", 10),
                 bg=self._c("sidebar"), fg=self._c("text_blue"),
                 activebackground=self._c("sidebar_hover"), relief="flat",
                 cursor="hand2", command=self._open_settings).pack(side="left")
        
        tk.Button(bottom, text="🗑", font=("Arial", 10),
                 bg=self._c("sidebar"), fg=self._c("text_muted"),
                 activebackground=self._c("sidebar_hover"), relief="flat",
                 cursor="hand2", command=self._clear_chat).pack(side="right")
        
        # ===== MAIN AREA =====
        self.main_area = tk.Frame(self, bg=self._c("chat_bg"))
        self.main_area.pack(side="left", fill="both", expand=True)
        
        # Header
        header = tk.Frame(self.main_area, bg=self._c("chat_bg"), height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        header_left = tk.Frame(header, bg=self._c("chat_bg"))
        header_left.pack(side="left", padx=24, pady=12)
        
        self.view_title = tk.Label(header_left, text="💬 Chat", font=("Arial", 16, "bold"),
                                   bg=self._c("chat_bg"), fg=self._c("text_blue"))
        self.view_title.pack(side="left")
        
        # Status indicator
        self.status_frame = tk.Frame(header_left, bg=self._c("chat_bg"))
        self.status_frame.pack(side="left", padx=(16, 0))
        
        self.status_dot = tk.Label(self.status_frame, text="●", font=("Arial", 10),
                                   bg=self._c("chat_bg"), fg=self._c("success"))
        self.status_dot.pack(side="left")
        
        self.status_text = tk.Label(self.status_frame, text="Online", font=("Arial", 10),
                                    bg=self._c("chat_bg"), fg=self._c("text_muted"))
        self.status_text.pack(side="left", padx=(4, 0))
        
        # Model selector (right side)
        header_right = tk.Frame(header, bg=self._c("chat_bg"))
        header_right.pack(side="right", padx=24, pady=12)
        
        tk.Label(header_right, text="Model:", font=("Arial", 10),
                bg=self._c("chat_bg"), fg=self._c("text_muted")).pack(side="left", padx=(0, 8))
        
        self.model_var = tk.StringVar(value="CatSEEK-R1")
        model_menu = ttk.Combobox(header_right, textvariable=self.model_var, width=12,
                                  values=["CatSEEK-R1", "CatSEEK-V3", "CatSEEK-R1-Lite"], state="readonly")
        model_menu.pack(side="left")
        
        # Divider
        tk.Frame(self.main_area, bg=self._c("border"), height=1).pack(fill="x")
        
        # Build views
        self._build_chat_view()
        self._build_canvas_view()
        self._build_files_view()
        
        # Show chat
        self.chat_frame.pack(fill="both", expand=True)
    
    def _build_chat_view(self):
        """Build chat view (CatSEEK style)."""
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
        
        # ===== INPUT AREA (CatSEEK style) =====
        input_wrapper = tk.Frame(self.chat_frame, bg=self._c("chat_bg"))
        input_wrapper.pack(fill="x", padx=30, pady=20)
        
        # Outer container with border
        input_outer = tk.Frame(input_wrapper, bg=self._c("input_border"), padx=1, pady=1)
        input_outer.pack(fill="x")
        
        input_container = tk.Frame(input_outer, bg=self._c("input_bg"))
        input_container.pack(fill="x")
        
        # Top row: input
        input_row = tk.Frame(input_container, bg=self._c("input_bg"))
        input_row.pack(fill="x", padx=16, pady=(12, 8))
        
        self.input_box = tk.Text(input_row, height=3, wrap="word",
                                bg=self._c("input_bg"), fg=self._c("text"),
                                insertbackground=self._c("text_blue"), relief="flat",
                                font=("Arial", 12), padx=4, pady=4)
        self.input_box.pack(fill="both", expand=True)
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: None)
        
        # Placeholder
        self.input_box.insert("1.0", "Message CatSEEK...")
        self.input_box.configure(fg=self._c("text_muted"))
        
        def on_focus_in(e):
            if self.input_box.get("1.0", "end").strip() == "Message CatSEEK...":
                self.input_box.delete("1.0", "end")
                self.input_box.configure(fg=self._c("text"))
        
        def on_focus_out(e):
            if not self.input_box.get("1.0", "end").strip():
                self.input_box.insert("1.0", "Message CatSEEK...")
                self.input_box.configure(fg=self._c("text_muted"))
        
        self.input_box.bind("<FocusIn>", on_focus_in)
        self.input_box.bind("<FocusOut>", on_focus_out)
        
        # Bottom row: buttons
        btn_row = tk.Frame(input_container, bg=self._c("input_bg"))
        btn_row.pack(fill="x", padx=12, pady=(0, 12))
        
        # Left buttons
        left_btns = tk.Frame(btn_row, bg=self._c("input_bg"))
        left_btns.pack(side="left")
        
        tk.Button(left_btns, text="📎", font=("Arial", 12),
                 bg=self._c("input_bg"), fg=self._c("text_muted"),
                 activebackground=self._c("btn_hover"), relief="flat",
                 cursor="hand2", command=self._upload_file).pack(side="left", padx=2)
        
        tk.Button(left_btns, text="🖼", font=("Arial", 12),
                 bg=self._c("input_bg"), fg=self._c("text_muted"),
                 activebackground=self._c("btn_hover"), relief="flat",
                 cursor="hand2", command=lambda: self._switch_view("canvas")).pack(side="left", padx=2)
        
        tk.Button(left_btns, text="</>" , font=("Arial", 10),
                 bg=self._c("input_bg"), fg=self._c("text_muted"),
                 activebackground=self._c("btn_hover"), relief="flat",
                 cursor="hand2", command=self._insert_code_block).pack(side="left", padx=2)
        
        # Right buttons
        right_btns = tk.Frame(btn_row, bg=self._c("input_bg"))
        right_btns.pack(side="right")
        
        self.stop_btn = tk.Button(right_btns, text="■ Stop", font=("Arial", 10),
                                 bg=self._c("btn_bg"), fg=self._c("error"),
                                 activebackground=self._c("btn_hover"), relief="flat",
                                 cursor="hand2", padx=12, pady=4, command=self._stop_gen)
        self.stop_btn.pack(side="left", padx=(0, 8))
        
        self.send_btn = tk.Button(right_btns, text="Send  ➤", font=("Arial", 11),
                                 bg=self._c("btn_bg"), fg=self._c("text_blue"),
                                 activebackground=self._c("btn_hover"), activeforeground=self._c("text_light_blue"),
                                 relief="flat", cursor="hand2", padx=16, pady=6,
                                 command=self._send)
        self.send_btn.pack(side="left")
        
        # Hint
        hint_frame = tk.Frame(input_wrapper, bg=self._c("chat_bg"))
        hint_frame.pack(fill="x", pady=(8, 0))
        
        tk.Label(hint_frame, text="CatSEEK can make mistakes. Consider checking important information.",
                font=("Arial", 9), bg=self._c("chat_bg"), fg=self._c("text_muted")).pack(side="left")
    
    def _insert_code_block(self):
        """Insert code block template."""
        self.input_box.delete("1.0", "end")
        self.input_box.insert("1.0", "```python\n\n```")
        self.input_box.mark_set("insert", "2.0")
        self.input_box.configure(fg=self._c("text"))
        self.input_box.focus_set()
    
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
        
        # Update tab highlighting
        for v, btn in self.tab_btns.items():
            if v == view:
                btn.configure(bg=self._c("sidebar_hover"))
            else:
                btn.configure(bg=self._c("sidebar"))
        
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
        
        for i, conv in enumerate(self.conversations[:12]):
            title = conv.title[:24] + "..." if len(conv.title) > 24 else conv.title
            
            btn_frame = tk.Frame(self.chat_list_frame, bg=self._c("sidebar"))
            btn_frame.pack(fill="x", pady=1)
            
            bg = self._c("sidebar_hover") if i == self.current_idx else self._c("sidebar")
            
            btn = tk.Button(btn_frame, text=f"💬 {title}",
                           font=("Arial", 10), bg=bg, fg=self._c("text_blue"),
                           activebackground=self._c("sidebar_hover"), activeforeground=self._c("text_light_blue"),
                           relief="flat", anchor="w", padx=12, pady=8, cursor="hand2",
                           command=lambda idx=i: self._select_chat(idx))
            btn.pack(fill="x")
    
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
        """Add message to chat (CatSEEK style)."""
        container = tk.Frame(self.msg_frame, bg=self._c("chat_bg"))
        container.pack(fill="x", padx=30, pady=12)
        
        # Row with avatar
        row = tk.Frame(container, bg=self._c("chat_bg"))
        row.pack(fill="x")
        
        # Avatar
        if role == "assistant":
            avatar_text = "🐱"
            avatar_bg = self._c("ai_msg")
        else:
            avatar_text = "👤"
            avatar_bg = self._c("user_msg")
        
        avatar_frame = tk.Frame(row, bg=avatar_bg, padx=8, pady=8)
        avatar_frame.pack(side="left", anchor="n")
        tk.Label(avatar_frame, text=avatar_text, font=("Arial", 14),
                bg=avatar_bg, fg=self._c("text")).pack()
        
        # Content area
        content_frame = tk.Frame(row, bg=self._c("chat_bg"))
        content_frame.pack(side="left", fill="x", expand=True, padx=(12, 0))
        
        # Name row
        name_row = tk.Frame(content_frame, bg=self._c("chat_bg"))
        name_row.pack(fill="x")
        
        name = "CatSEEK" if role == "assistant" else "You"
        tk.Label(name_row, text=name, font=("Arial", 11, "bold"),
                bg=self._c("chat_bg"), fg=self._c("text_blue")).pack(side="left")
        
        if role == "assistant":
            tk.Label(name_row, text="", font=("Arial", 9),
                    bg=self._c("chat_bg"), fg=self._c("text_muted")).pack(side="left", padx=(6, 0))
        
        # o1-style thinking block (collapsible)
        if role == "assistant" and thinking:
            think_outer = tk.Frame(content_frame, bg=self._c("thinking_bg"),
                                  highlightthickness=1, highlightbackground=self._c("thinking_border"))
            think_outer.pack(fill="x", pady=(8, 0))
            
            think_header = tk.Frame(think_outer, bg=self._c("thinking_bg"))
            think_header.pack(fill="x", padx=12, pady=(10, 0))
            
            # Track expansion state
            is_expanded = [False]
            think_content_frame = tk.Frame(think_outer, bg=self._c("thinking_bg"))
            
            def toggle():
                if is_expanded[0]:
                    think_content_frame.pack_forget()
                    toggle_btn.configure(text="▶")
                    think_label.configure(text="Deep Think (click to expand)")
                    is_expanded[0] = False
                else:
                    think_content_frame.pack(fill="x", padx=12, pady=(6, 10))
                    toggle_btn.configure(text="▼")
                    think_label.configure(text="Deep Think")
                    is_expanded[0] = True
            
            toggle_btn = tk.Button(think_header, text="▶", font=("Arial", 9),
                                  bg=self._c("thinking_bg"), fg=self._c("text_muted"),
                                  relief="flat", bd=0, cursor="hand2", command=toggle)
            toggle_btn.pack(side="left")
            
            tk.Label(think_header, text="💭", font=("Arial", 11),
                    bg=self._c("thinking_bg"), fg=self._c("text_light_blue")).pack(side="left", padx=(6, 0))
            
            think_label = tk.Label(think_header, text="Deep Think (click to expand)",
                                   font=("Arial", 10), bg=self._c("thinking_bg"), fg=self._c("text_muted"))
            think_label.pack(side="left", padx=(6, 0))
            
            # Thinking content
            tk.Label(think_content_frame, text=thinking, font=("Consolas", 10),
                    bg=self._c("thinking_bg"), fg=self._c("text_muted"),
                    wraplength=600, justify="left", anchor="w").pack(anchor="w", fill="x")
            
            # Bottom padding
            tk.Frame(think_outer, bg=self._c("thinking_bg"), height=10).pack(fill="x")
        
        # Message content
        msg_frame = tk.Frame(content_frame, bg=self._c("chat_bg"))
        msg_frame.pack(fill="x", pady=(8, 0))
        
        msg_label = tk.Label(msg_frame, text=content, font=("Arial", 11),
                            bg=self._c("chat_bg"), fg=self._c("text"),
                            wraplength=700, justify="left", anchor="w")
        msg_label.pack(anchor="w")
        
        # Code result block
        if code_result:
            code_frame = tk.Frame(content_frame, bg=self._c("code_bg"),
                                 highlightthickness=1, highlightbackground=self._c("border"))
            code_frame.pack(fill="x", pady=(10, 0))
            
            # Code header
            code_header = tk.Frame(code_frame, bg="#151515")
            code_header.pack(fill="x")
            tk.Label(code_header, text="python", font=("Consolas", 9, "bold"),
                    bg="#151515", fg=self._c("text_muted")).pack(side="left", padx=12, pady=6)
            
            status = "✓ Success" if code_result.get("success") else "✗ Error"
            status_color = self._c("success") if code_result.get("success") else self._c("error")
            tk.Label(code_header, text=status, font=("Arial", 9),
                    bg="#151515", fg=status_color).pack(side="right", padx=12, pady=6)
            
            # Code text
            code_text = tk.Text(code_frame, height=min(12, code_result["code"].count('\n') + 2),
                               bg=self._c("code_bg"), fg="#93c5fd", font=("Consolas", 10),
                               relief="flat", wrap="none", padx=12, pady=8)
            code_text.insert("1.0", code_result["code"])
            code_text.configure(state="disabled")
            code_text.pack(fill="x")
            
            # Output
            if code_result.get("stdout") or code_result.get("error"):
                out_header = tk.Frame(code_frame, bg="#101010")
                out_header.pack(fill="x")
                tk.Label(out_header, text="Output", font=("Consolas", 9),
                        bg="#101010", fg=self._c("text_muted")).pack(side="left", padx=12, pady=4)
                
                output = code_result.get("stdout", "") or code_result.get("error", "")
                out_color = self._c("success") if code_result.get("success") else self._c("error")
                
                out_text = tk.Text(code_frame, height=min(6, output.count('\n') + 1),
                                  bg=self._c("code_bg"), fg=out_color, font=("Consolas", 10),
                                  relief="flat", padx=12, pady=8)
                out_text.insert("1.0", output[:600])
                out_text.configure(state="disabled")
                out_text.pack(fill="x")
        
        # Action buttons
        action_frame = tk.Frame(content_frame, bg=self._c("chat_bg"))
        action_frame.pack(fill="x", pady=(8, 0))
        
        tk.Button(action_frame, text="📋 Copy", font=("Arial", 9),
                 bg=self._c("btn_bg"), fg=self._c("text_muted"), relief="flat",
                 activebackground=self._c("btn_hover"), cursor="hand2",
                 command=lambda: self._copy(content)).pack(side="left", padx=(0, 6))
        
        if role == "assistant":
            tk.Button(action_frame, text="🔄 Regenerate", font=("Arial", 9),
                     bg=self._c("btn_bg"), fg=self._c("text_muted"), relief="flat",
                     activebackground=self._c("btn_hover"), cursor="hand2",
                     command=lambda: None).pack(side="left")
        
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
        if not (e.state & 0x1):  # Not shift
            text = self.input_box.get("1.0", "end").strip()
            if text and text != "Message CatSEEK...":
                self._send()
            return "break"
    
    def _send(self):
        if self._streaming:
            return
        
        text = self.input_box.get("1.0", "end").strip()
        
        # Ignore placeholder
        if not text or text == "Message CatSEEK...":
            return
        
        self.input_box.delete("1.0", "end")
        self.input_box.configure(fg=self._c("text_muted"))
        self.input_box.insert("1.0", "Message CatSEEK...")
        
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
        """Generate o1-style streaming response with thinking (CatSEEK UI)."""
        self._streaming = True
        self._stop = False
        
        self.status_dot.configure(fg=self._c("warning"))
        self.status_text.configure(text="Thinking...")
        
        # Generate thinking first
        thinking_text = self.model.generate_thinking(user_text)
        
        # Create response container
        container = tk.Frame(self.msg_frame, bg=self._c("chat_bg"))
        container.pack(fill="x", padx=30, pady=12)
        
        row = tk.Frame(container, bg=self._c("chat_bg"))
        row.pack(fill="x")
        
        # Avatar
        avatar_frame = tk.Frame(row, bg=self._c("ai_msg"), padx=8, pady=8)
        avatar_frame.pack(side="left", anchor="n")
        tk.Label(avatar_frame, text="🐱", font=("Arial", 14),
                bg=self._c("ai_msg"), fg=self._c("text")).pack()
        
        content_frame = tk.Frame(row, bg=self._c("chat_bg"))
        content_frame.pack(side="left", fill="x", expand=True, padx=(12, 0))
        
        # Name row
        name_row = tk.Frame(content_frame, bg=self._c("chat_bg"))
        name_row.pack(fill="x")
        
        tk.Label(name_row, text="CatSEEK", font=("Arial", 11, "bold"),
                bg=self._c("chat_bg"), fg=self._c("text_blue")).pack(side="left")
        
        # Thinking block
        think_outer = tk.Frame(content_frame, bg=self._c("thinking_bg"),
                              highlightthickness=1, highlightbackground=self._c("thinking_border"))
        think_outer.pack(fill="x", pady=(8, 0))
        
        think_header = tk.Frame(think_outer, bg=self._c("thinking_bg"))
        think_header.pack(fill="x", padx=12, pady=(10, 0))
        
        tk.Label(think_header, text="💭", font=("Arial", 11),
                bg=self._c("thinking_bg"), fg=self._c("text_light_blue")).pack(side="left")
        
        think_status = tk.Label(think_header, text="Thinking...",
                               font=("Arial", 10, "italic"), bg=self._c("thinking_bg"), fg=self._c("text_muted"))
        think_status.pack(side="left", padx=(6, 0))
        
        think_time = tk.Label(think_header, text="",
                             font=("Arial", 9), bg=self._c("thinking_bg"), fg=self._c("text_muted"))
        think_time.pack(side="right")
        
        # Thinking content
        think_content_frame = tk.Frame(think_outer, bg=self._c("thinking_bg"))
        think_content_frame.pack(fill="x", padx=12, pady=(6, 10))
        
        think_content = tk.Label(think_content_frame, text="", font=("Consolas", 10),
                                bg=self._c("thinking_bg"), fg=self._c("text_muted"),
                                wraplength=600, justify="left", anchor="w")
        think_content.pack(anchor="w", fill="x")
        
        # Response area
        response_frame = tk.Frame(content_frame, bg=self._c("chat_bg"))
        response_frame.pack(fill="x", pady=(10, 0))
        
        txt_var = tk.StringVar(value="")
        msg_label = tk.Label(response_frame, textvariable=txt_var, font=("Arial", 11),
                            bg=self._c("chat_bg"), fg=self._c("text"),
                            wraplength=700, justify="left", anchor="w")
        msg_label.pack(anchor="w")
        
        start_time = time.time()
        
        # Animate thinking
        def animate_thinking(idx=0):
            if self._stop:
                return
            
            elapsed = time.time() - start_time
            think_time.configure(text=f"{elapsed:.1f}s")
            
            if idx < len(thinking_text):
                think_content.configure(text=thinking_text[:idx+1])
                self._scroll_bottom()
                self.after(12, lambda: animate_thinking(idx + 1))
            else:
                # Thinking done, start response
                elapsed = time.time() - start_time
                think_status.configure(text=f"Thought for {elapsed:.1f}s")
                think_time.configure(text="")
                self.status_text.configure(text="Responding...")
                start_response()
        
        def start_response():
            def gen_thread():
                for token in self.model.generate(user_text, 300, lambda: self._stop):
                    self._queue.put(token)
                self._queue.put(None)
            
            Thread(target=gen_thread, daemon=True).start()
            poll_response()
        
        def poll_response():
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None:
                        self._finish_response(txt_var.get(), thinking_text)
                        return
                    txt_var.set(txt_var.get() + item)
                    self._scroll_bottom()
            except queue.Empty:
                pass
            
            if self._stop:
                self._finish_response(txt_var.get(), thinking_text)
                return
            
            self.after(15, poll_response)
        
        animate_thinking()
    
    def _finish_response(self, content: str, thinking: str = ""):
        self._streaming = False
        
        self.status_dot.configure(fg=self._c("success"))
        self.status_text.configure(text="Ready")
        
        content = content.strip()
        if content:
            self.current.messages.append(Message("assistant", content, thinking))
        
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
        win.geometry("500x450")
        win.configure(bg=self._c("bg"))
        
        # Header
        header = tk.Frame(win, bg=self._c("sidebar"), height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="⚙  Settings", font=("Arial", 16, "bold"),
                bg=self._c("sidebar"), fg=self._c("text_blue")).pack(side="left", padx=20, pady=15)
        
        # Content
        content = tk.Frame(win, bg=self._c("bg"))
        content.pack(fill="both", expand=True, padx=30, pady=20)
        
        # Model section
        tk.Label(content, text="Model Parameters", font=("Arial", 11, "bold"),
                bg=self._c("bg"), fg=self._c("text_blue")).pack(anchor="w", pady=(0, 12))
        
        params_frame = tk.Frame(content, bg=self._c("bg"))
        params_frame.pack(fill="x")
        
        # Temperature
        row1 = tk.Frame(params_frame, bg=self._c("bg"))
        row1.pack(fill="x", pady=6)
        tk.Label(row1, text="Temperature", font=("Arial", 10),
                bg=self._c("bg"), fg=self._c("text")).pack(side="left")
        t_var = tk.DoubleVar(value=self.model.temperature)
        tk.Scale(row1, from_=0.1, to=1.5, resolution=0.05, orient="horizontal",
                variable=t_var, bg=self._c("bg"), fg=self._c("text_blue"),
                highlightthickness=0, troughcolor=self._c("input_bg"),
                activebackground=self._c("accent"), length=200).pack(side="right")
        
        # Rep penalty
        row2 = tk.Frame(params_frame, bg=self._c("bg"))
        row2.pack(fill="x", pady=6)
        tk.Label(row2, text="Repetition Penalty", font=("Arial", 10),
                bg=self._c("bg"), fg=self._c("text")).pack(side="left")
        r_var = tk.DoubleVar(value=self.model.rep_penalty)
        tk.Scale(row2, from_=1.0, to=2.0, resolution=0.05, orient="horizontal",
                variable=r_var, bg=self._c("bg"), fg=self._c("text_blue"),
                highlightthickness=0, troughcolor=self._c("input_bg"),
                activebackground=self._c("accent"), length=200).pack(side="right")
        
        # Divider
        tk.Frame(content, bg=self._c("border"), height=1).pack(fill="x", pady=20)
        
        # Train section
        tk.Label(content, text="Train on Custom Text", font=("Arial", 11, "bold"),
                bg=self._c("bg"), fg=self._c("text_blue")).pack(anchor="w", pady=(0, 8))
        
        tk.Label(content, text="Add custom patterns to improve responses:",
                font=("Arial", 9), bg=self._c("bg"), fg=self._c("text_muted")).pack(anchor="w")
        
        train_frame = tk.Frame(content, bg=self._c("input_bg"), highlightthickness=1,
                              highlightbackground=self._c("border"))
        train_frame.pack(fill="x", pady=8)
        
        train_text = tk.Text(train_frame, height=5, bg=self._c("input_bg"), fg=self._c("text"),
                            font=("Arial", 10), relief="flat", padx=8, pady=8)
        train_text.pack(fill="x")
        
        # Buttons
        btn_frame = tk.Frame(content, bg=self._c("bg"))
        btn_frame.pack(fill="x", pady=20)
        
        def save():
            self.model.temperature = t_var.get()
            self.model.rep_penalty = r_var.get()
            
            custom = train_text.get("1.0", "end").strip()
            if custom:
                self.model.train(custom)
                messagebox.showinfo("Trained", f"Added {len(custom)} characters to model.")
            
            win.destroy()
        
        tk.Button(btn_frame, text="Cancel", font=("Arial", 11),
                 bg=self._c("btn_bg"), fg=self._c("text_muted"),
                 activebackground=self._c("btn_hover"), relief="flat",
                 padx=20, pady=8, cursor="hand2",
                 command=win.destroy).pack(side="left")
        
        tk.Button(btn_frame, text="Save", font=("Arial", 11),
                 bg=self._c("btn_bg"), fg=self._c("text_blue"),
                 activebackground=self._c("btn_hover"), activeforeground=self._c("text_light_blue"),
                 relief="flat", padx=25, pady=8, cursor="hand2",
                 command=save).pack(side="right")
        
        win.transient(self)
        win.grab_set()


# =====================================================
# Main
# =====================================================

def main():
    # Required for simpledialog
    import tkinter.simpledialog
    
    app = CatSEEKApp()
    app.mainloop()


if __name__ == "__main__":
    main()
