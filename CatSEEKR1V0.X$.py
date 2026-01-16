#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatSEEK R1 — Complete AI System with Embedded Training Data
DeepSeek-style chat interface with pre-trained synthetic reasoning dataset.

Features:
  • Deep Think reasoning visualization
  • Embedded R1 synthetic training dataset (50+ examples)
  • Code Interpreter (Python sandbox)
  • File Sandbox (upload, view, manage)
  • Canvas (drawing, diagrams)

Python 3.10+ | Pure tkinter + stdlib | import math
Run: python catseek_r1.py
"""

import os, sys, io, re, time, json, math, random, base64, hashlib, tempfile, traceback, queue
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from threading import Thread
from typing import Generator, Optional, Any, List, Dict
from collections import defaultdict
from contextlib import redirect_stdout
from math import gcd, pi, sqrt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

APP_NAME = "CatSEEK"
VERSION = "R1"

# ═══════════════════════════════════════════════════════════════
# EMBEDDED R1 TRAINING DATASET
# ═══════════════════════════════════════════════════════════════

R1_TRAINING_DATA = [
    # ─── MATH ARITHMETIC ───
    {"prompt": "Calculate: 247 + 389", "thinking": "Let me solve 247 + 389.\n\nAdding step by step:\n- Ones: 7 + 9 = 16, write 6 carry 1\n- Tens: 4 + 8 + 1 = 13, write 3 carry 1\n- Hundreds: 2 + 3 + 1 = 6\n\n247 + 389 = 636", "response": "The answer is **636**.", "category": "math"},
    {"prompt": "Calculate: 512 - 287", "thinking": "Let me solve 512 - 287.\n\nSubtracting with borrowing:\n512 - 287 = 225", "response": "The answer is **225**.", "category": "math"},
    {"prompt": "Calculate: 23 × 17", "thinking": "Let me solve 23 × 17.\n\n23 × 17 = 23 × (10 + 7) = 230 + 161 = 391", "response": "The answer is **391**.", "category": "math"},
    {"prompt": "Calculate: 144 ÷ 12", "thinking": "Let me solve 144 ÷ 12.\n\n12 × 12 = 144, so 144 ÷ 12 = 12", "response": "The answer is **12**.", "category": "math"},
    {"prompt": "What is 15% of 80?", "thinking": "Let me calculate 15% of 80.\n\n15% = 0.15\n0.15 × 80 = 12", "response": "15% of 80 is **12**.", "category": "math"},
    {"prompt": "Calculate: 56 + 78", "thinking": "56 + 78 = 134", "response": "The answer is **134**.", "category": "math"},
    {"prompt": "Calculate: 100 - 37", "thinking": "100 - 37 = 63", "response": "The answer is **63**.", "category": "math"},
    {"prompt": "Calculate: 8 × 9", "thinking": "8 × 9 = 72", "response": "The answer is **72**.", "category": "math"},
    
    # ─── MATH ALGEBRA ───
    {"prompt": "Solve for x: 3x + 7 = 22", "thinking": "Let me solve 3x + 7 = 22.\n\nStep 1: Subtract 7 from both sides\n3x = 15\n\nStep 2: Divide by 3\nx = 5\n\nVerify: 3(5) + 7 = 22 ✓", "response": "**x = 5**", "category": "math_algebra"},
    {"prompt": "Solve for x: 2x - 5 = 3x + 2", "thinking": "Let me solve 2x - 5 = 3x + 2.\n\n2x - 3x = 2 + 5\n-x = 7\nx = -7", "response": "**x = -7**", "category": "math_algebra"},
    {"prompt": "Solve: x² - 9 = 0", "thinking": "x² - 9 = 0\nx² = 9\nx = ±3", "response": "**x = 3** or **x = -3**", "category": "math_algebra"},
    {"prompt": "Simplify: (x + 2)(x - 3)", "thinking": "Using FOIL:\nFirst: x²\nOuter: -3x\nInner: 2x\nLast: -6\n\nResult: x² - x - 6", "response": "**(x + 2)(x - 3) = x² - x - 6**", "category": "math_algebra"},
    {"prompt": "Factor: x² + 5x + 6", "thinking": "Need numbers that multiply to 6 and add to 5.\n2 × 3 = 6, 2 + 3 = 5\n\nx² + 5x + 6 = (x + 2)(x + 3)", "response": "**x² + 5x + 6 = (x + 2)(x + 3)**", "category": "math_algebra"},
    {"prompt": "Solve: 5x = 25", "thinking": "5x = 25\nx = 25 ÷ 5\nx = 5", "response": "**x = 5**", "category": "math_algebra"},
    
    # ─── MATH GEOMETRY ───
    {"prompt": "Find the area of a circle with radius 5.", "thinking": "Area = πr² = π × 5² = 25π ≈ 78.54", "response": "The area is **25π ≈ 78.54 square units**.", "category": "math_geometry"},
    {"prompt": "Find the circumference of a circle with radius 7.", "thinking": "Circumference = 2πr = 2 × π × 7 = 14π ≈ 43.98", "response": "The circumference is **14π ≈ 43.98 units**.", "category": "math_geometry"},
    {"prompt": "Find the area of a triangle with base 10 and height 6.", "thinking": "Area = ½ × base × height = ½ × 10 × 6 = 30", "response": "The area is **30 square units**.", "category": "math_geometry"},
    {"prompt": "Find the hypotenuse of a right triangle with legs 3 and 4.", "thinking": "Using Pythagorean theorem:\nc² = 3² + 4² = 9 + 16 = 25\nc = 5", "response": "The hypotenuse is **5 units**.", "category": "math_geometry"},
    {"prompt": "Find the volume of a cube with side length 4.", "thinking": "Volume = s³ = 4³ = 64", "response": "The volume is **64 cubic units**.", "category": "math_geometry"},
    {"prompt": "Find the perimeter of a rectangle with length 8 and width 5.", "thinking": "Perimeter = 2(length + width) = 2(8 + 5) = 2(13) = 26", "response": "The perimeter is **26 units**.", "category": "math_geometry"},
    
    # ─── MATH CALCULUS ───
    {"prompt": "Find the derivative of f(x) = 3x²", "thinking": "Using power rule: d/dx[xⁿ] = nxⁿ⁻¹\n\nf'(x) = 3 × 2 × x¹ = 6x", "response": "**f'(x) = 6x**", "category": "math_calculus"},
    {"prompt": "Find the derivative of f(x) = x³ + 2x² - 5x + 1", "thinking": "Differentiating term by term:\nd/dx[x³] = 3x²\nd/dx[2x²] = 4x\nd/dx[-5x] = -5\nd/dx[1] = 0\n\nf'(x) = 3x² + 4x - 5", "response": "**f'(x) = 3x² + 4x - 5**", "category": "math_calculus"},
    {"prompt": "Find the integral of f(x) = 2x", "thinking": "∫2x dx = 2 × x²/2 + C = x² + C", "response": "**∫2x dx = x² + C**", "category": "math_calculus"},
    {"prompt": "Find the integral of f(x) = x²", "thinking": "∫x² dx = x³/3 + C", "response": "**∫x² dx = x³/3 + C**", "category": "math_calculus"},
    {"prompt": "Find the derivative of f(x) = 5x⁴", "thinking": "f'(x) = 5 × 4 × x³ = 20x³", "response": "**f'(x) = 20x³**", "category": "math_calculus"},
    
    # ─── CODE PYTHON ───
    {"prompt": "Write a Python function to calculate factorial.", "thinking": "Factorial: n! = n × (n-1) × ... × 1\nBase case: 0! = 1", "response": "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\n# factorial(5) = 120\n```", "category": "code"},
    {"prompt": "Write a Python function to check if a number is prime.", "thinking": "Prime: only divisible by 1 and itself.\nOptimization: check up to sqrt(n)", "response": "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n```", "category": "code"},
    {"prompt": "Write a Python function to reverse a string.", "thinking": "Python slicing makes this easy: s[::-1]", "response": "```python\ndef reverse_string(s):\n    return s[::-1]\n\n# reverse_string('hello') = 'olleh'\n```", "category": "code"},
    {"prompt": "Write a Python function for Fibonacci sequence.", "thinking": "Fibonacci: F(n) = F(n-1) + F(n-2)\nBase: F(0)=0, F(1)=1", "response": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n# fibonacci(10) = 55\n```", "category": "code"},
    {"prompt": "Write a Python function to check if a string is a palindrome.", "thinking": "Palindrome reads same forwards and backwards.\nClean string, compare with reverse.", "response": "```python\ndef is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# is_palindrome('racecar') = True\n```", "category": "code"},
    {"prompt": "Write a Python function to find the maximum in a list.", "thinking": "Iterate through list, track largest value.", "response": "```python\ndef find_max(lst):\n    if not lst:\n        return None\n    maximum = lst[0]\n    for num in lst[1:]:\n        if num > maximum:\n            maximum = num\n    return maximum\n```", "category": "code"},
    {"prompt": "Write a Python function to count vowels.", "thinking": "Vowels: a, e, i, o, u\nCount occurrences.", "response": "```python\ndef count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n\n# count_vowels('Hello') = 2\n```", "category": "code"},
    
    # ─── CODE ALGORITHMS ───
    {"prompt": "Implement binary search in Python.", "thinking": "Binary search: find target in sorted array.\nHalve search space each step. O(log n)", "response": "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```", "category": "code_algorithm"},
    {"prompt": "Implement quicksort in Python.", "thinking": "Quicksort: divide-and-conquer.\nPick pivot, partition, recurse. O(n log n) average.", "response": "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```", "category": "code_algorithm"},
    {"prompt": "Implement bubble sort in Python.", "thinking": "Bubble sort: repeatedly swap adjacent elements if wrong order.\nO(n²) time complexity.", "response": "```python\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n```", "category": "code_algorithm"},
    {"prompt": "Implement BFS in Python.", "thinking": "BFS: explore graph level by level using queue.\nO(V + E) time complexity.", "response": "```python\nfrom collections import deque\n\ndef bfs(graph, start):\n    visited = set([start])\n    queue = deque([start])\n    result = []\n    while queue:\n        node = queue.popleft()\n        result.append(node)\n        for neighbor in graph.get(node, []):\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return result\n```", "category": "code_algorithm"},
    {"prompt": "Implement DFS in Python.", "thinking": "DFS: explore as deep as possible before backtracking.\nUse recursion or stack.", "response": "```python\ndef dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    result = [start]\n    for neighbor in graph.get(start, []):\n        if neighbor not in visited:\n            result.extend(dfs(graph, neighbor, visited))\n    return result\n```", "category": "code_algorithm"},
    
    # ─── KNOWLEDGE ───
    {"prompt": "What is recursion?", "thinking": "Recursion: function calls itself.\nNeeds base case to stop.", "response": "**Recursion** is when a function calls itself to solve smaller instances of the same problem.\n\n**Key components:**\n1. **Base case** - stops recursion\n2. **Recursive case** - function calls itself\n\n**Example:** `factorial(n) = n × factorial(n-1)`", "category": "knowledge"},
    {"prompt": "Explain Big O notation.", "thinking": "Big O describes algorithm efficiency.\nHow runtime/space grows with input size.", "response": "**Big O notation** describes algorithm efficiency.\n\n**Common complexities:**\n- O(1) - Constant\n- O(log n) - Logarithmic\n- O(n) - Linear\n- O(n log n) - Linearithmic\n- O(n²) - Quadratic\n- O(2ⁿ) - Exponential", "category": "knowledge"},
    {"prompt": "What is the difference between a stack and a queue?", "thinking": "Stack: LIFO (Last In, First Out)\nQueue: FIFO (First In, First Out)", "response": "**Stack:** LIFO - Last In, First Out (like plates)\n**Queue:** FIFO - First In, First Out (like a line)\n\nStack uses: push/pop from top\nQueue uses: enqueue back, dequeue front", "category": "knowledge"},
    {"prompt": "Explain what an API is.", "thinking": "API = Application Programming Interface.\nDefines how software components interact.", "response": "**API (Application Programming Interface)** is a set of rules that allows software applications to communicate.\n\nLike a restaurant menu - you don't need to know how the kitchen works, just what you can order.", "category": "knowledge"},
    {"prompt": "What is object-oriented programming?", "thinking": "OOP organizes code around objects.\nFour pillars: encapsulation, inheritance, polymorphism, abstraction.", "response": "**OOP** organizes code into objects with data (attributes) and behavior (methods).\n\n**Four Pillars:**\n1. Encapsulation\n2. Inheritance\n3. Polymorphism\n4. Abstraction", "category": "knowledge"},
    
    # ─── CONVERSATIONAL ───
    {"prompt": "hello", "thinking": "User greeting.", "response": "Hi! I'm CatSEEK R1, your AI assistant. How can I help you today?", "category": "conversation"},
    {"prompt": "hi", "thinking": "Greeting.", "response": "Hello! What can I help you with?", "category": "conversation"},
    {"prompt": "hey", "thinking": "Informal greeting.", "response": "Hey! What's on your mind?", "category": "conversation"},
    {"prompt": "how are you", "thinking": "State inquiry.", "response": "I'm doing well, thank you! How can I assist you today?", "category": "conversation"},
    {"prompt": "what are you", "thinking": "Identity question.", "response": "I'm CatSEEK R1, an AI assistant with Deep Think reasoning capabilities!", "category": "conversation"},
    {"prompt": "who are you", "thinking": "Identity.", "response": "I'm CatSEEK, your AI assistant. I can help with math, coding, explanations, and more!", "category": "conversation"},
    {"prompt": "help", "thinking": "Help request.", "response": "I can help with:\n• **Math** - arithmetic, algebra, calculus\n• **Code** - Python, algorithms\n• **Explanations** - concepts, comparisons\n\nJust ask!", "category": "conversation"},
    {"prompt": "thank you", "thinking": "Gratitude.", "response": "You're welcome! Let me know if you need anything else.", "category": "conversation"},
    {"prompt": "thanks", "thinking": "Thanks.", "response": "Happy to help!", "category": "conversation"},
    {"prompt": "bye", "thinking": "Farewell.", "response": "Goodbye! Come back anytime.", "category": "conversation"},
    {"prompt": "goodbye", "thinking": "Farewell.", "response": "Take care! Feel free to return whenever you need help.", "category": "conversation"},
]


# ═══════════════════════════════════════════════════════════════
# O1 REASONING ENGINE (TRAINED ON R1 DATA)
# ═══════════════════════════════════════════════════════════════

class O1ReasoningEngine:
    def __init__(self):
        self.temperature = 0.7
        self.rep_penalty = 1.1
        self.knowledge_base = {}
        self.category_examples = defaultdict(list)
        
        for item in R1_TRAINING_DATA:
            prompt_lower = item["prompt"].lower()
            self.knowledge_base[prompt_lower] = item
            words = prompt_lower.split()
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    phrase = " ".join(words[i:j])
                    if len(phrase) > 3 and phrase not in self.knowledge_base:
                        self.knowledge_base[phrase] = item
            self.category_examples[item["category"]].append(item)
        
        self.thinking_starters = ["Let me think through this carefully.", "Okay, let's break this down.", "Let me consider this step by step."]
        self.thinking_continues = ["So first, ", "Now, ", "Next, ", "Wait, ", "Actually, "]
        self.conclusions = ["So, putting it all together: ", "Therefore: ", "In conclusion: "]
    
    def _find_best_match(self, query: str) -> Optional[Dict]:
        query_lower = query.lower().strip()
        if query_lower in self.knowledge_base:
            return self.knowledge_base[query_lower]
        
        best_match, best_score = None, 0
        query_words = set(query_lower.split())
        key_terms = {"solve", "calculate", "find", "write", "implement", "explain", "what", "how", "derivative", "integral", "factorial", "fibonacci", "prime", "sort", "search", "area", "volume", "perimeter"}
        
        for item in R1_TRAINING_DATA:
            prompt_words = set(item["prompt"].lower().split())
            overlap = len(query_words & prompt_words)
            term_bonus = len(query_words & key_terms & prompt_words) * 2
            score = overlap + term_bonus
            if score > best_score:
                best_score, best_match = score, item
        
        return best_match if best_score >= 2 else None
    
    def _detect_category(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["calculate", "solve", "+", "-", "*", "/", "="]): return "math"
        if any(w in t for w in ["derivative", "integral", "limit"]): return "math_calculus"
        if any(w in t for w in ["area", "perimeter", "circle", "triangle", "volume"]): return "math_geometry"
        if any(w in t for w in ["code", "python", "function", "implement", "write a"]): return "code"
        if any(w in t for w in ["algorithm", "sort", "search", "bfs", "dfs"]): return "code_algorithm"
        if any(w in t for w in ["explain", "what is", "difference"]): return "knowledge"
        return "general"
    
    def _generate_fallback(self, query: str, category: str) -> tuple:
        thinking = random.choice(self.thinking_starters) + f"\n\nThe user is asking about: {query}\n\n" + random.choice(self.thinking_continues) + "Let me provide a helpful response.\n\n" + random.choice(self.conclusions)
        
        if category.startswith("math"):
            response = f"For math problems, I can help with calculations, algebra, geometry, and calculus. Could you provide more details about '{query}'?"
        elif category.startswith("code"):
            response = f"For coding questions, I can help with Python, algorithms, and debugging. What specifically would you like to implement?"
        else:
            response = f"I'd be happy to help with '{query}'. Could you tell me more about what you'd like to know?"
        return thinking, response
    
    def generate_thinking(self, prompt: str) -> str:
        match = self._find_best_match(prompt)
        if match: return match["thinking"]
        category = self._detect_category(prompt)
        thinking, _ = self._generate_fallback(prompt, category)
        return thinking
    
    def generate(self, prompt: str, max_tokens: int = 500, stop_fn=None) -> Generator[str, None, None]:
        match = self._find_best_match(prompt)
        response = match["response"] if match else self._generate_fallback(prompt, self._detect_category(prompt))[1]
        for char in response:
            if stop_fn and stop_fn(): break
            yield char
            time.sleep(0.008)
    
    def train(self, text: str):
        for s in re.split(r'[.!?]', text):
            s = s.strip()
            if 20 < len(s) < 150: self.thinking_continues.append(s + ". ")


# ═══════════════════════════════════════════════════════════════
# CODE INTERPRETER & FILE SANDBOX
# ═══════════════════════════════════════════════════════════════

class CodeInterpreter:
    def __init__(self):
        self.globals = {"__builtins__": {"print": print, "len": len, "range": range, "str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "any": any, "all": all, "True": True, "False": False, "None": None}, "math": math, "random": random, "re": re, "json": json}
        self.locals = {}
    
    def execute(self, code: str) -> dict:
        stdout_capture = io.StringIO()
        result = {"code": code, "stdout": "", "error": None, "success": False}
        try:
            with redirect_stdout(stdout_capture): exec(code, self.globals, self.locals)
            result["success"] = True
        except Exception as e: result["error"] = f"{type(e).__name__}: {str(e)}"
        result["stdout"] = stdout_capture.getvalue()
        return result


class FileSandbox:
    def __init__(self):
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="catseek_"))
        self.files = {}
    
    def add_file(self, filepath: str) -> dict:
        src = Path(filepath)
        if not src.exists(): return {"error": "File not found"}
        dst = self.sandbox_dir / src.name
        with open(src, "rb") as f: content = f.read()
        with open(dst, "wb") as f: f.write(content)
        self.files[src.name] = {"name": src.name, "path": str(dst), "size": len(content), "type": src.suffix[1:] if src.suffix else "unknown"}
        return self.files[src.name]
    
    def list_files(self) -> list: return list(self.files.values())


# ═══════════════════════════════════════════════════════════════
# CANVAS WIDGET
# ═══════════════════════════════════════════════════════════════

class CanvasWidget(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.tool, self.color, self.brush_size = "pen", "#3b82f6", 3
        self._start_x = self._start_y = self._current_item = None
        self._history = []
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
    
    def _on_press(self, e):
        self._start_x, self._start_y = e.x, e.y
        if self.tool == "pen": self._current_item = self.create_line(e.x, e.y, e.x, e.y, fill=self.color, width=self.brush_size, capstyle=tk.ROUND)
        elif self.tool == "eraser":
            for item in self.find_overlapping(e.x-10, e.y-10, e.x+10, e.y+10): self.delete(item)
    
    def _on_drag(self, e):
        if self.tool == "pen" and self._current_item: self.coords(self._current_item, *list(self.coords(self._current_item)) + [e.x, e.y])
        elif self.tool == "eraser":
            for item in self.find_overlapping(e.x-10, e.y-10, e.x+10, e.y+10): self.delete(item)
        elif self.tool in ("line", "rect"):
            if self._current_item: self.delete(self._current_item)
            if self.tool == "line": self._current_item = self.create_line(self._start_x, self._start_y, e.x, e.y, fill=self.color, width=self.brush_size)
            elif self.tool == "rect": self._current_item = self.create_rectangle(self._start_x, self._start_y, e.x, e.y, outline=self.color, width=self.brush_size)
    
    def _on_release(self, e):
        if self._current_item: self._history.append(self._current_item)
        self._current_item = None
    
    def undo(self):
        if self._history: self.delete(self._history.pop())
    
    def clear_all(self): self.delete("all"); self._history.clear()


# ═══════════════════════════════════════════════════════════════
# CHAT DATA
# ═══════════════════════════════════════════════════════════════

@dataclass
class Message:
    role: str
    content: str
    thinking: str = ""
    code_result: dict = None

@dataclass
class Conversation:
    title: str = "New Chat"
    messages: list = field(default_factory=list)
    model: str = "CatSEEK-R1"


# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

class CatSEEKApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} {VERSION}")
        self.geometry("1300x850")
        self.minsize(1100, 700)
        
        self.colors = {"bg": "#0d0d0d", "sidebar": "#0a0a0a", "sidebar_hover": "#1a1a1a", "chat_bg": "#0d0d0d", "input_bg": "#1a1a1a", "input_border": "#2a2a2a", "user_msg": "#1f1f1f", "ai_msg": "#141414", "thinking_bg": "#111118", "thinking_border": "#1e1e2e", "code_bg": "#0c0c0c", "text": "#e4e4e7", "text_blue": "#3b82f6", "text_light_blue": "#60a5fa", "text_muted": "#6b7280", "border": "#1e1e1e", "success": "#10b981", "error": "#ef4444", "warning": "#f59e0b", "btn_bg": "#1a1a1a", "btn_hover": "#252525"}
        
        self.conversations = [Conversation()]
        self.current_idx = 0
        self.model = O1ReasoningEngine()
        self.interpreter = CodeInterpreter()
        self.file_sandbox = FileSandbox()
        self._streaming, self._stop = False, False
        self._queue = queue.Queue()
        self.current_view = "chat"
        self.tab_btns = {}
        
        self._build_ui()
        self._add_message("assistant", f"Hi, I'm CatSEEK R1! I'm trained on {len(R1_TRAINING_DATA)} reasoning examples. How can I help you today?", thinking=f"Ready. Loaded {len(R1_TRAINING_DATA)} training examples.")
    
    @property
    def current(self) -> Conversation: return self.conversations[self.current_idx]
    def _c(self, key: str) -> str: return self.colors.get(key, "#fff")
    
    def _build_ui(self):
        self.configure(bg=self._c("bg"))
        
        # Sidebar
        sidebar = tk.Frame(self, bg=self._c("sidebar"), width=280); sidebar.pack(side="left", fill="y"); sidebar.pack_propagate(False)
        logo_frame = tk.Frame(sidebar, bg=self._c("sidebar")); logo_frame.pack(fill="x", padx=20, pady=20)
        tk.Label(logo_frame, text="🐱", font=("Arial", 28), bg=self._c("sidebar"), fg=self._c("text")).pack(side="left")
        logo_text = tk.Frame(logo_frame, bg=self._c("sidebar")); logo_text.pack(side="left", padx=(10, 0))
        tk.Label(logo_text, text="CatSEEK", font=("Arial", 18, "bold"), bg=self._c("sidebar"), fg=self._c("text_blue")).pack(anchor="w")
        tk.Label(logo_text, text=f"R1 • {len(R1_TRAINING_DATA)} examples", font=("Arial", 9), bg=self._c("sidebar"), fg=self._c("text_muted")).pack(anchor="w")
        
        tk.Button(sidebar, text="＋ New Chat", font=("Arial", 12), bg=self._c("btn_bg"), fg=self._c("text_blue"), relief="flat", pady=12, cursor="hand2", command=self._new_chat).pack(fill="x", padx=16, pady=(0, 20))
        tk.Frame(sidebar, bg=self._c("border"), height=1).pack(fill="x", padx=16)
        
        tabs_frame = tk.Frame(sidebar, bg=self._c("sidebar")); tabs_frame.pack(fill="x", padx=16, pady=16)
        for view, icon in [("chat", "💬"), ("canvas", "🎨"), ("files", "📁"), ("data", "📊")]:
            btn = tk.Button(tabs_frame, text=icon, font=("Arial", 12), bg=self._c("sidebar"), fg=self._c("text_blue"), relief="flat", cursor="hand2", command=lambda v=view: self._switch_view(v)); btn.pack(side="left", padx=2); self.tab_btns[view] = btn
        self.tab_btns["chat"].configure(bg=self._c("sidebar_hover"))
        
        tk.Label(sidebar, text="Recent", font=("Arial", 10, "bold"), bg=self._c("sidebar"), fg=self._c("text_muted")).pack(anchor="w", padx=20, pady=(10, 8))
        self.chat_list = tk.Frame(sidebar, bg=self._c("sidebar")); self.chat_list.pack(fill="both", expand=True, padx=12); self._refresh_chat_list()
        
        bottom = tk.Frame(sidebar, bg=self._c("sidebar")); bottom.pack(fill="x", padx=16, pady=16)
        tk.Button(bottom, text="⚙ Settings", font=("Arial", 10), bg=self._c("sidebar"), fg=self._c("text_blue"), relief="flat", cursor="hand2", command=self._open_settings).pack(side="left")
        
        # Main
        self.main = tk.Frame(self, bg=self._c("chat_bg")); self.main.pack(side="left", fill="both", expand=True)
        header = tk.Frame(self.main, bg=self._c("chat_bg"), height=60); header.pack(fill="x"); header.pack_propagate(False)
        self.view_title = tk.Label(header, text="💬 Chat", font=("Arial", 16, "bold"), bg=self._c("chat_bg"), fg=self._c("text_blue")); self.view_title.pack(side="left", padx=24, pady=12)
        status_frame = tk.Frame(header, bg=self._c("chat_bg")); status_frame.pack(side="left", padx=16)
        self.status_dot = tk.Label(status_frame, text="●", font=("Arial", 10), bg=self._c("chat_bg"), fg=self._c("success")); self.status_dot.pack(side="left")
        self.status_text = tk.Label(status_frame, text="Online", font=("Arial", 10), bg=self._c("chat_bg"), fg=self._c("text_muted")); self.status_text.pack(side="left", padx=(4, 0))
        tk.Frame(self.main, bg=self._c("border"), height=1).pack(fill="x")
        
        self._build_chat_view(); self._build_canvas_view(); self._build_files_view(); self._build_data_view()
        self.chat_frame.pack(fill="both", expand=True)
    
    def _build_chat_view(self):
        self.chat_frame = tk.Frame(self.main, bg=self._c("chat_bg"))
        msg_container = tk.Frame(self.chat_frame, bg=self._c("chat_bg")); msg_container.pack(fill="both", expand=True)
        self.msg_canvas = tk.Canvas(msg_container, bg=self._c("chat_bg"), highlightthickness=0)
        scrollbar = ttk.Scrollbar(msg_container, orient="vertical", command=self.msg_canvas.yview); self.msg_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y"); self.msg_canvas.pack(side="left", fill="both", expand=True)
        self.msg_frame = tk.Frame(self.msg_canvas, bg=self._c("chat_bg")); self.msg_window = self.msg_canvas.create_window((0, 0), window=self.msg_frame, anchor="nw")
        self.msg_frame.bind("<Configure>", lambda e: self.msg_canvas.configure(scrollregion=self.msg_canvas.bbox("all")))
        self.msg_canvas.bind("<Configure>", lambda e: self.msg_canvas.itemconfig(self.msg_window, width=e.width))
        self.msg_canvas.bind_all("<MouseWheel>", lambda e: self.msg_canvas.yview_scroll(int(-e.delta/120), "units"))
        
        input_wrapper = tk.Frame(self.chat_frame, bg=self._c("chat_bg")); input_wrapper.pack(fill="x", padx=30, pady=20)
        input_outer = tk.Frame(input_wrapper, bg=self._c("input_border"), padx=1, pady=1); input_outer.pack(fill="x")
        input_inner = tk.Frame(input_outer, bg=self._c("input_bg")); input_inner.pack(fill="x")
        input_row = tk.Frame(input_inner, bg=self._c("input_bg")); input_row.pack(fill="x", padx=16, pady=(12, 8))
        self.input_box = tk.Text(input_row, height=3, wrap="word", bg=self._c("input_bg"), fg=self._c("text_muted"), insertbackground=self._c("text_blue"), relief="flat", font=("Arial", 12)); self.input_box.pack(fill="both", expand=True)
        self.input_box.insert("1.0", "Message CatSEEK...")
        self.input_box.bind("<Return>", self._on_enter); self.input_box.bind("<FocusIn>", lambda e: self._clear_placeholder()); self.input_box.bind("<FocusOut>", lambda e: self._restore_placeholder())
        btn_row = tk.Frame(input_inner, bg=self._c("input_bg")); btn_row.pack(fill="x", padx=12, pady=(0, 12))
        tk.Button(btn_row, text="■ Stop", font=("Arial", 10), bg=self._c("btn_bg"), fg=self._c("error"), relief="flat", padx=12, cursor="hand2", command=self._stop_gen).pack(side="right", padx=(8, 0))
        tk.Button(btn_row, text="Send ➤", font=("Arial", 11), bg=self._c("btn_bg"), fg=self._c("text_blue"), relief="flat", padx=16, pady=6, cursor="hand2", command=self._send).pack(side="right")
        tk.Label(input_wrapper, text=f"CatSEEK R1 • Trained on {len(R1_TRAINING_DATA)} reasoning examples", font=("Arial", 9), bg=self._c("chat_bg"), fg=self._c("text_muted")).pack(pady=(8, 0))
    
    def _build_canvas_view(self):
        self.canvas_frame = tk.Frame(self.main, bg=self._c("chat_bg"))
        toolbar = tk.Frame(self.canvas_frame, bg=self._c("input_bg")); toolbar.pack(fill="x", padx=20, pady=10)
        self.tool_var = tk.StringVar(value="pen")
        for label, tool in [("✏️ Pen", "pen"), ("📏 Line", "line"), ("⬜ Rect", "rect"), ("🧹 Eraser", "eraser")]:
            tk.Radiobutton(toolbar, text=label, variable=self.tool_var, value=tool, bg=self._c("input_bg"), fg=self._c("text_blue"), selectcolor=self._c("btn_bg"), indicatoron=False, padx=10, pady=5, command=self._update_tool).pack(side="left", padx=2)
        tk.Button(toolbar, text="🗑 Clear", bg=self._c("btn_bg"), fg=self._c("error"), relief="flat", command=lambda: self.drawing_canvas.clear_all()).pack(side="right", padx=2)
        container = tk.Frame(self.canvas_frame, bg=self._c("border")); container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.drawing_canvas = CanvasWidget(container, bg="white", highlightthickness=1); self.drawing_canvas.pack(fill="both", expand=True, padx=1, pady=1)
    
    def _build_files_view(self):
        self.files_frame = tk.Frame(self.main, bg=self._c("chat_bg"))
        toolbar = tk.Frame(self.files_frame, bg=self._c("input_bg")); toolbar.pack(fill="x", padx=20, pady=10)
        tk.Button(toolbar, text="📤 Upload", bg=self._c("btn_bg"), fg=self._c("text_blue"), relief="flat", padx=15, pady=5, command=self._upload_file).pack(side="left", padx=4)
        tk.Button(toolbar, text="🔄 Refresh", bg=self._c("btn_bg"), fg=self._c("text_muted"), relief="flat", command=self._refresh_files).pack(side="right", padx=4)
        list_frame = tk.Frame(self.files_frame, bg=self._c("chat_bg")); list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.file_tree = ttk.Treeview(list_frame, columns=("name", "type", "size"), show="headings", height=15)
        self.file_tree.heading("name", text="Name"); self.file_tree.heading("type", text="Type"); self.file_tree.heading("size", text="Size"); self.file_tree.pack(fill="both", expand=True)
    
    def _build_data_view(self):
        self.data_frame = tk.Frame(self.main, bg=self._c("chat_bg"))
        header = tk.Frame(self.data_frame, bg=self._c("chat_bg")); header.pack(fill="x", padx=30, pady=20)
        tk.Label(header, text="📊 Embedded Training Data", font=("Arial", 16, "bold"), bg=self._c("chat_bg"), fg=self._c("text_blue")).pack(anchor="w")
        tk.Label(header, text=f"R1 model is trained on {len(R1_TRAINING_DATA)} embedded examples", font=("Arial", 10), bg=self._c("chat_bg"), fg=self._c("text_muted")).pack(anchor="w", pady=(4, 0))
        
        categories = defaultdict(int)
        for item in R1_TRAINING_DATA: categories[item["category"]] += 1
        
        stats_frame = tk.Frame(self.data_frame, bg=self._c("input_bg")); stats_frame.pack(fill="x", padx=30, pady=10)
        stats_text = tk.Text(stats_frame, height=8, bg=self._c("input_bg"), fg=self._c("text"), font=("Consolas", 10), relief="flat"); stats_text.pack(fill="x", padx=10, pady=10)
        stats_text.insert("1.0", f"DATASET STATISTICS\n{'═'*40}\n\nTotal examples: {len(R1_TRAINING_DATA)}\n\nBy category:\n")
        for cat, count in sorted(categories.items()): stats_text.insert("end", f"  • {cat}: {count}\n")
        stats_text.configure(state="disabled")
        
        tk.Label(self.data_frame, text="Sample Training Examples:", font=("Arial", 11, "bold"), bg=self._c("chat_bg"), fg=self._c("text")).pack(anchor="w", padx=30, pady=(20, 10))
        self.data_output = tk.Text(self.data_frame, height=18, bg=self._c("code_bg"), fg=self._c("text"), font=("Consolas", 10), relief="flat"); self.data_output.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        samples = random.sample(R1_TRAINING_DATA, min(5, len(R1_TRAINING_DATA)))
        for i, s in enumerate(samples):
            self.data_output.insert("end", f"─── Example {i+1} ({s['category']}) ───\nPrompt: {s['prompt']}\nThinking: {s['thinking'][:80]}...\nResponse: {s['response'][:80]}...\n\n")
    
    def _switch_view(self, view: str):
        self.current_view = view
        for v, btn in self.tab_btns.items(): btn.configure(bg=self._c("sidebar_hover") if v == view else self._c("sidebar"))
        for f in [self.chat_frame, self.canvas_frame, self.files_frame, self.data_frame]: f.pack_forget()
        self.view_title.configure(text={"chat": "💬 Chat", "canvas": "🎨 Canvas", "files": "📁 Files", "data": "📊 Training Data"}.get(view, "Chat"))
        {"chat": self.chat_frame, "canvas": self.canvas_frame, "files": self.files_frame, "data": self.data_frame}[view].pack(fill="both", expand=True)
        if view == "files": self._refresh_files()
    
    def _refresh_chat_list(self):
        for w in self.chat_list.winfo_children(): w.destroy()
        for i, conv in enumerate(self.conversations[:12]):
            title = conv.title[:24] + "..." if len(conv.title) > 24 else conv.title
            bg = self._c("sidebar_hover") if i == self.current_idx else self._c("sidebar")
            tk.Button(self.chat_list, text=f"💬 {title}", font=("Arial", 10), bg=bg, fg=self._c("text_blue"), relief="flat", anchor="w", padx=12, pady=8, cursor="hand2", command=lambda x=i: self._select_chat(x)).pack(fill="x", pady=1)
    
    def _select_chat(self, idx: int): self.current_idx = idx; self._refresh_chat_list(); self._render_chat()
    def _new_chat(self): self.conversations.insert(0, Conversation()); self.current_idx = 0; self._refresh_chat_list(); self._clear_messages(); self._add_message("assistant", "New conversation started. How can I help?", thinking="Ready.")
    def _render_chat(self): self._clear_messages(); [self._add_message(m.role, m.content, m.thinking, m.code_result, save=False) for m in self.current.messages]
    def _clear_messages(self): [w.destroy() for w in self.msg_frame.winfo_children()]
    
    def _add_message(self, role: str, content: str, thinking: str = "", code_result: dict = None, save: bool = True):
        container = tk.Frame(self.msg_frame, bg=self._c("chat_bg")); container.pack(fill="x", padx=30, pady=12)
        row = tk.Frame(container, bg=self._c("chat_bg")); row.pack(fill="x")
        avatar_txt, avatar_bg = ("🐱", self._c("ai_msg")) if role == "assistant" else ("👤", self._c("user_msg"))
        avatar = tk.Frame(row, bg=avatar_bg, padx=8, pady=8); avatar.pack(side="left", anchor="n"); tk.Label(avatar, text=avatar_txt, font=("Arial", 14), bg=avatar_bg).pack()
        content_frame = tk.Frame(row, bg=self._c("chat_bg")); content_frame.pack(side="left", fill="x", expand=True, padx=(12, 0))
        tk.Label(content_frame, text="CatSEEK" if role == "assistant" else "You", font=("Arial", 11, "bold"), bg=self._c("chat_bg"), fg=self._c("text_blue")).pack(anchor="w")
        
        if role == "assistant" and thinking:
            think_outer = tk.Frame(content_frame, bg=self._c("thinking_bg"), highlightthickness=1, highlightbackground=self._c("thinking_border")); think_outer.pack(fill="x", pady=(8, 0))
            think_header = tk.Frame(think_outer, bg=self._c("thinking_bg")); think_header.pack(fill="x", padx=12, pady=(10, 0))
            is_expanded = [False]; think_content = tk.Frame(think_outer, bg=self._c("thinking_bg"))
            def toggle():
                if is_expanded[0]: think_content.pack_forget(); toggle_btn.configure(text="▶")
                else: think_content.pack(fill="x", padx=12, pady=(6, 10)); tk.Label(think_content, text=thinking, font=("Consolas", 10), bg=self._c("thinking_bg"), fg=self._c("text_muted"), wraplength=600, justify="left").pack(anchor="w"); toggle_btn.configure(text="▼")
                is_expanded[0] = not is_expanded[0]
            toggle_btn = tk.Button(think_header, text="▶", font=("Arial", 9), bg=self._c("thinking_bg"), fg=self._c("text_muted"), relief="flat", bd=0, cursor="hand2", command=toggle); toggle_btn.pack(side="left")
            tk.Label(think_header, text="💭 Deep Think", font=("Arial", 10), bg=self._c("thinking_bg"), fg=self._c("text_light_blue")).pack(side="left", padx=(6, 0))
            tk.Frame(think_outer, bg=self._c("thinking_bg"), height=10).pack(fill="x")
        
        tk.Label(content_frame, text=content, font=("Arial", 11), bg=self._c("chat_bg"), fg=self._c("text"), wraplength=700, justify="left").pack(anchor="w", pady=(8, 0))
        
        if code_result:
            code_frame = tk.Frame(content_frame, bg=self._c("code_bg"), highlightthickness=1, highlightbackground=self._c("border")); code_frame.pack(fill="x", pady=(10, 0))
            code_text = tk.Text(code_frame, height=min(10, code_result["code"].count('\n')+2), bg=self._c("code_bg"), fg="#93c5fd", font=("Consolas", 10), relief="flat"); code_text.insert("1.0", code_result["code"]); code_text.configure(state="disabled"); code_text.pack(fill="x", padx=10, pady=8)
            if code_result.get("stdout") or code_result.get("error"):
                out = code_result.get("stdout") or code_result.get("error", ""); out_color = self._c("success") if code_result.get("success") else self._c("error")
                out_text = tk.Text(code_frame, height=min(5, out.count('\n')+1), bg=self._c("code_bg"), fg=out_color, font=("Consolas", 10), relief="flat"); out_text.insert("1.0", out[:500]); out_text.configure(state="disabled"); out_text.pack(fill="x", padx=10, pady=(0, 8))
        
        tk.Button(content_frame, text="📋 Copy", font=("Arial", 9), bg=self._c("btn_bg"), fg=self._c("text_muted"), relief="flat", cursor="hand2", command=lambda: (self.clipboard_clear(), self.clipboard_append(content))).pack(anchor="w", pady=(8, 0))
        
        if save:
            self.current.messages.append(Message(role, content, thinking, code_result))
            if role == "user" and len(self.current.messages) == 1: self.current.title = content[:40]; self._refresh_chat_list()
        self._scroll_bottom()
    
    def _scroll_bottom(self): self.msg_canvas.update_idletasks(); self.msg_canvas.yview_moveto(1.0)
    def _clear_placeholder(self):
        if self.input_box.get("1.0", "end").strip() == "Message CatSEEK...": self.input_box.delete("1.0", "end"); self.input_box.configure(fg=self._c("text"))
    def _restore_placeholder(self):
        if not self.input_box.get("1.0", "end").strip(): self.input_box.insert("1.0", "Message CatSEEK..."); self.input_box.configure(fg=self._c("text_muted"))
    def _on_enter(self, e):
        if not (e.state & 0x1): text = self.input_box.get("1.0", "end").strip(); text != "Message CatSEEK..." and text and self._send(); return "break"
    
    def _send(self):
        if self._streaming: return
        text = self.input_box.get("1.0", "end").strip()
        if not text or text == "Message CatSEEK...": return
        self.input_box.delete("1.0", "end"); self.input_box.configure(fg=self._c("text_muted")); self.input_box.insert("1.0", "Message CatSEEK...")
        self._add_message("user", text)
        code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            result = self.interpreter.execute(code_match.group(1))
            self._add_message("assistant", "Code executed " + ("successfully." if result["success"] else "with errors."), thinking="Executed code.", code_result=result)
        else: self._stream_response(text)
    
    def _stop_gen(self): self._stop = True
    
    def _stream_response(self, user_text: str):
        self._streaming, self._stop = True, False
        self.status_dot.configure(fg=self._c("warning")); self.status_text.configure(text="Thinking...")
        thinking_text = self.model.generate_thinking(user_text)
        
        container = tk.Frame(self.msg_frame, bg=self._c("chat_bg")); container.pack(fill="x", padx=30, pady=12)
        row = tk.Frame(container, bg=self._c("chat_bg")); row.pack(fill="x")
        avatar = tk.Frame(row, bg=self._c("ai_msg"), padx=8, pady=8); avatar.pack(side="left", anchor="n"); tk.Label(avatar, text="🐱", font=("Arial", 14), bg=self._c("ai_msg")).pack()
        content_frame = tk.Frame(row, bg=self._c("chat_bg")); content_frame.pack(side="left", fill="x", expand=True, padx=(12, 0))
        tk.Label(content_frame, text="CatSEEK", font=("Arial", 11, "bold"), bg=self._c("chat_bg"), fg=self._c("text_blue")).pack(anchor="w")
        
        think_outer = tk.Frame(content_frame, bg=self._c("thinking_bg"), highlightthickness=1, highlightbackground=self._c("thinking_border")); think_outer.pack(fill="x", pady=(8, 0))
        think_header = tk.Frame(think_outer, bg=self._c("thinking_bg")); think_header.pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(think_header, text="💭", font=("Arial", 11), bg=self._c("thinking_bg"), fg=self._c("text_light_blue")).pack(side="left")
        think_status = tk.Label(think_header, text="Thinking...", font=("Arial", 10, "italic"), bg=self._c("thinking_bg"), fg=self._c("text_muted")); think_status.pack(side="left", padx=(6, 0))
        think_time = tk.Label(think_header, text="", font=("Arial", 9), bg=self._c("thinking_bg"), fg=self._c("text_muted")); think_time.pack(side="right")
        think_content_frame = tk.Frame(think_outer, bg=self._c("thinking_bg")); think_content_frame.pack(fill="x", padx=12, pady=(6, 10))
        think_content = tk.Label(think_content_frame, text="", font=("Consolas", 10), bg=self._c("thinking_bg"), fg=self._c("text_muted"), wraplength=600, justify="left"); think_content.pack(anchor="w")
        
        response_frame = tk.Frame(content_frame, bg=self._c("chat_bg")); response_frame.pack(fill="x", pady=(10, 0))
        txt_var = tk.StringVar(value=""); tk.Label(response_frame, textvariable=txt_var, font=("Arial", 11), bg=self._c("chat_bg"), fg=self._c("text"), wraplength=700, justify="left").pack(anchor="w")
        
        start_time = time.time()
        def animate_thinking(idx=0):
            if self._stop: return
            elapsed = time.time() - start_time; think_time.configure(text=f"{elapsed:.1f}s")
            if idx < len(thinking_text): think_content.configure(text=thinking_text[:idx+1]); self._scroll_bottom(); self.after(10, lambda: animate_thinking(idx + 1))
            else: think_status.configure(text=f"Thought for {elapsed:.1f}s"); think_time.configure(text=""); self.status_text.configure(text="Responding..."); start_response()
        
        def start_response():
            def gen():
                for token in self.model.generate(user_text, 500, lambda: self._stop): self._queue.put(token)
                self._queue.put(None)
            Thread(target=gen, daemon=True).start(); poll()
        
        def poll():
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None: finish(); return
                    txt_var.set(txt_var.get() + item); self._scroll_bottom()
            except queue.Empty: pass
            if self._stop: finish(); return
            self.after(15, poll)
        
        def finish():
            self._streaming = False; self.status_dot.configure(fg=self._c("success")); self.status_text.configure(text="Online")
            content = txt_var.get().strip()
            if content: self.current.messages.append(Message("assistant", content, thinking_text))
            while not self._queue.empty(): self._queue.get_nowait()
        
        animate_thinking()
    
    def _update_tool(self): self.drawing_canvas.tool = self.tool_var.get()
    def _upload_file(self):
        path = filedialog.askopenfilename()
        if path:
            info = self.file_sandbox.add_file(path)
            if "error" not in info: self._refresh_files(); messagebox.showinfo("Uploaded", f"File '{info['name']}' added.")
    def _refresh_files(self):
        for item in self.file_tree.get_children(): self.file_tree.delete(item)
        for f in self.file_sandbox.list_files(): self.file_tree.insert("", "end", values=(f["name"], f["type"], f"{f['size']} B" if f['size'] < 1024 else f"{f['size']//1024} KB"))
    
    def _open_settings(self):
        win = tk.Toplevel(self); win.title("Settings"); win.geometry("450x300"); win.configure(bg=self._c("bg"))
        tk.Label(win, text="⚙ Settings", font=("Arial", 16, "bold"), bg=self._c("bg"), fg=self._c("text_blue")).pack(pady=20)
        frm = tk.Frame(win, bg=self._c("bg")); frm.pack(fill="x", padx=30)
        tk.Label(frm, text="Temperature", bg=self._c("bg"), fg=self._c("text")).grid(row=0, column=0, sticky="w", pady=8)
        t_var = tk.DoubleVar(value=self.model.temperature); tk.Scale(frm, from_=0.1, to=1.5, resolution=0.05, orient="horizontal", variable=t_var, bg=self._c("bg"), fg=self._c("text_blue"), highlightthickness=0, troughcolor=self._c("input_bg"), length=200).grid(row=0, column=1, padx=10)
        tk.Label(win, text=f"Training data: {len(R1_TRAINING_DATA)} examples", font=("Arial", 10), bg=self._c("bg"), fg=self._c("text_muted")).pack(pady=10)
        def save(): self.model.temperature = t_var.get(); win.destroy()
        tk.Button(win, text="Save", font=("Arial", 11), bg=self._c("btn_bg"), fg=self._c("text_blue"), relief="flat", padx=30, pady=8, cursor="hand2", command=save).pack(pady=20)
        win.transient(self); win.grab_set()


def main(): CatSEEKApp().mainloop()
if __name__ == "__main__": main()
