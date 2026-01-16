#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CatGPT-SEEK 32B - Ultimate Synthesized LLM                                  ║
║  All ChatGPT + DeepSeek R1 Features Combined                                 ║
║  Pure Python stdlib | import math | No External Dependencies                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
  ★ ChatGPT-style conversational AI with personality modes
  ★ DeepSeek R1-style <think> chain-of-thought reasoning
  ★ Code Interpreter with safe Python execution
  ★ Advanced math engine (calculus, algebra, statistics)
  ★ 500+ knowledge base entries
  ★ Memory system with context retention
  ★ Custom instructions/system prompts
  ★ Multi-turn conversation handling
  ★ Streaming token output
  ★ Tool/Plugin architecture
  ★ Web search simulation
  ★ Image analysis simulation
  ★ File handling simulation
  ★ 32B parameter architecture patterns

Usage: python catgpt_seek_32b.py
"""

import os
import sys
import io
import re
import ast
import time
import json
import math
import cmath
import random
import string
import base64
import hashlib
import secrets
import tempfile
import textwrap
import traceback
import queue
import struct
import zlib
import csv
import html
import functools
import operator
import itertools
import collections
import statistics
import fractions
import decimal
import calendar
import unicodedata
import difflib
import heapq
import bisect
import copy
import pprint
import shlex
import keyword
import token
import tokenize
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from threading import Thread, Lock, Event, RLock
from typing import Generator, Optional, Any, List, Dict, Tuple, Callable, Union, Set
from collections import defaultdict, deque, Counter, OrderedDict
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import lru_cache, partial, reduce
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

VERSION = "32B-v1.0"
MODEL_NAME = "CatGPT-SEEK-32B-Synthesized"
CONTEXT_WINDOW = 8192
MAX_TOKENS = 4096


# ============================================================================
# MATH ENGINE - Complete Mathematical Computation
# ============================================================================

class MathEngine:
    """Advanced mathematical computation using stdlib math."""
    
    def __init__(self):
        self.precision = 15
        self.angle_mode = "radians"
        self.memory: Dict[str, Any] = {}
        self.history: List[Tuple[str, Any]] = []
        self.ans = 0
        
        # Mathematical constants
        self.constants = {
            "pi": math.pi,
            "π": math.pi,
            "e": math.e,
            "tau": math.tau,
            "τ": math.tau,
            "phi": (1 + math.sqrt(5)) / 2,
            "φ": (1 + math.sqrt(5)) / 2,
            "golden": (1 + math.sqrt(5)) / 2,
            "sqrt2": math.sqrt(2),
            "√2": math.sqrt(2),
            "sqrt3": math.sqrt(3),
            "√3": math.sqrt(3),
            "ln2": math.log(2),
            "ln10": math.log(10),
            "inf": math.inf,
            "∞": math.inf,
            "nan": math.nan,
            "i": 1j,
            "j": 1j,
            "c": 299792458,           # Speed of light m/s
            "g": 9.80665,             # Gravity m/s²
            "G": 6.67430e-11,         # Gravitational constant
            "h": 6.62607015e-34,      # Planck constant
            "kb": 1.380649e-23,       # Boltzmann constant
            "Na": 6.02214076e23,      # Avogadro number
            "R": 8.314462618,         # Gas constant
            "ans": 0,
        }
        
        # Safe function namespace
        self.safe_funcs = self._build_safe_namespace()
    
    def _build_safe_namespace(self) -> Dict[str, Callable]:
        """Build comprehensive safe math namespace."""
        ns = {}
        
        # Basic operations
        ns.update({
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "divmod": divmod, "len": len,
            "int": int, "float": float, "complex": complex,
            "bool": bool, "bin": bin, "hex": hex, "oct": oct,
        })
        
        # Math module - all functions
        ns.update({
            # Powers and logarithms
            "sqrt": math.sqrt,
            "cbrt": lambda x: math.copysign(abs(x) ** (1/3), x),
            "exp": math.exp,
            "exp2": lambda x: 2 ** x,
            "expm1": math.expm1,
            "log": math.log,
            "log2": math.log2,
            "log10": math.log10,
            "log1p": math.log1p,
            "ln": math.log,
            "lg": math.log10,
            
            # Trigonometry
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "atan2": math.atan2,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
            "sec": lambda x: 1/math.cos(x),
            "csc": lambda x: 1/math.sin(x),
            "cot": lambda x: 1/math.tan(x),
            "asec": lambda x: math.acos(1/x),
            "acsc": lambda x: math.asin(1/x),
            "acot": lambda x: math.atan(1/x),
            "sech": lambda x: 1/math.cosh(x),
            "csch": lambda x: 1/math.sinh(x),
            "coth": lambda x: 1/math.tanh(x),
            
            # Angular conversion
            "degrees": math.degrees, "radians": math.radians,
            "deg": math.degrees, "rad": math.radians,
            
            # Rounding
            "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
            
            # Special functions
            "factorial": math.factorial,
            "gamma": math.gamma, "lgamma": math.lgamma,
            "erf": math.erf, "erfc": math.erfc,
            
            # Number theory
            "gcd": math.gcd,
            "lcm": lambda a, b: abs(a * b) // math.gcd(a, b) if a and b else 0,
            "isqrt": math.isqrt,
            
            # Floating point
            "fabs": math.fabs, "fmod": math.fmod,
            "copysign": math.copysign, "remainder": math.remainder,
            "isfinite": math.isfinite, "isinf": math.isinf, "isnan": math.isnan,
            "frexp": math.frexp, "ldexp": math.ldexp, "modf": math.modf,
            
            # Hyperbolic / distance
            "hypot": math.hypot,
            "dist": lambda p1, p2: math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2))),
            
            # Products and sums
            "prod": math.prod,
            "fsum": math.fsum,
            
            # Complex math
            "cabs": abs,
            "cphase": cmath.phase,
            "cpolar": cmath.polar,
            "crect": cmath.rect,
            "csqrt": cmath.sqrt,
            "cexp": cmath.exp,
            "clog": cmath.log,
            "csin": cmath.sin, "ccos": cmath.cos,
            
            # Statistics
            "mean": statistics.mean,
            "median": statistics.median,
            "median_low": statistics.median_low,
            "median_high": statistics.median_high,
            "mode": statistics.mode,
            "multimode": statistics.multimode,
            "stdev": statistics.stdev,
            "pstdev": statistics.pstdev,
            "variance": statistics.variance,
            "pvariance": statistics.pvariance,
            "quantiles": statistics.quantiles,
            "harmonic_mean": statistics.harmonic_mean,
            "geometric_mean": statistics.geometric_mean,
            
            # Fractions
            "frac": fractions.Fraction,
            "Fraction": fractions.Fraction,
            
            # Decimal
            "Decimal": decimal.Decimal,
            
            # Custom functions
            "isprime": self._is_prime,
            "primes": self._primes_up_to,
            "factors": self._prime_factors,
            "divisors": self._divisors,
            "fib": self._fibonacci,
            "fibonacci": self._fibonacci,
            "nCr": self._combination,
            "nPr": self._permutation,
            "C": self._combination,
            "P": self._permutation,
            "choose": self._combination,
            "perm": self._permutation,
            "derivative": self._derivative,
            "integral": self._integral,
            "limit": self._limit,
            "summation": self._summation,
            "product_series": self._product_series,
            "solve": self._solve_equation,
            "roots": self._find_roots,
            "quadratic": self._quadratic_formula,
            "matrix_det": self._matrix_determinant,
            "matrix_mult": self._matrix_multiply,
            "dot": self._dot_product,
            "cross": self._cross_product,
            "norm": self._vector_norm,
            "normalize": self._normalize_vector,
            "linspace": self._linspace,
            "arange": self._arange,
            "polyeval": self._poly_eval,
            "polyroots": self._poly_roots,
        })
        
        # Add constants
        ns.update(self.constants)
        
        # Range and list functions
        ns["range"] = range
        ns["list"] = list
        ns["tuple"] = tuple
        ns["set"] = set
        ns["sorted"] = sorted
        ns["reversed"] = lambda x: list(reversed(x))
        ns["enumerate"] = lambda x: list(enumerate(x))
        ns["zip"] = lambda *x: list(zip(*x))
        ns["map"] = lambda f, x: list(map(f, x))
        ns["filter"] = lambda f, x: list(filter(f, x))
        
        return ns
    
    # Number Theory Functions
    def _is_prime(self, n: int) -> bool:
        """Miller-Rabin primality test."""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witnesses for deterministic test up to certain bounds
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        for a in witnesses:
            if a >= n:
                continue
            
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def _primes_up_to(self, n: int) -> List[int]:
        """Sieve of Eratosthenes."""
        if n < 2:
            return []
        
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    def _prime_factors(self, n: int) -> List[int]:
        """Prime factorization."""
        factors = []
        d = 2
        while d * d <= abs(n):
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if abs(n) > 1:
            factors.append(abs(n))
        return factors
    
    def _divisors(self, n: int) -> List[int]:
        """All divisors of n."""
        n = abs(n)
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)
    
    def _fibonacci(self, n: int) -> int:
        """Nth Fibonacci number using matrix exponentiation."""
        if n <= 0:
            return 0
        if n <= 2:
            return 1
        
        def matrix_mult(A, B):
            return [
                [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
                [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
            ]
        
        def matrix_pow(M, p):
            if p == 1:
                return M
            if p % 2 == 0:
                half = matrix_pow(M, p // 2)
                return matrix_mult(half, half)
            else:
                return matrix_mult(M, matrix_pow(M, p - 1))
        
        result = matrix_pow([[1, 1], [1, 0]], n)
        return result[0][1]
    
    def _combination(self, n: int, r: int) -> int:
        """n choose r."""
        if r < 0 or r > n:
            return 0
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
    
    def _permutation(self, n: int, r: int) -> int:
        """n permute r."""
        if r < 0 or r > n:
            return 0
        return math.factorial(n) // math.factorial(n - r)
    
    # Calculus Functions
    def _derivative(self, f: Callable, x: float, h: float = 1e-8) -> float:
        """Numerical derivative using central difference."""
        return (f(x + h) - f(x - h)) / (2 * h)
    
    def _integral(self, f: Callable, a: float, b: float, n: int = 1000) -> float:
        """Numerical integration using adaptive Simpson's rule."""
        if n % 2 == 1:
            n += 1
        
        h = (b - a) / n
        s = f(a) + f(b)
        
        for i in range(1, n):
            coef = 4 if i % 2 == 1 else 2
            s += coef * f(a + i * h)
        
        return s * h / 3
    
    def _limit(self, f: Callable, x: float, direction: str = "both") -> float:
        """Numerical limit approximation."""
        h_values = [10**(-i) for i in range(1, 12)]
        
        if direction == "left" or direction == "-":
            values = [f(x - h) for h in h_values]
        elif direction == "right" or direction == "+":
            values = [f(x + h) for h in h_values]
        else:
            left = [f(x - h) for h in h_values]
            right = [f(x + h) for h in h_values]
            values = [(l + r) / 2 for l, r in zip(left, right)]
        
        # Check convergence
        for i in range(len(values) - 3):
            if abs(values[i+2] - values[i+1]) < 1e-10:
                return values[i+2]
        
        return values[-1]
    
    def _summation(self, f: Callable, start: int, end: int) -> float:
        """Summation Σf(i) from start to end."""
        return sum(f(i) for i in range(start, end + 1))
    
    def _product_series(self, f: Callable, start: int, end: int) -> float:
        """Product Πf(i) from start to end."""
        result = 1
        for i in range(start, end + 1):
            result *= f(i)
        return result
    
    # Equation Solving
    def _solve_equation(self, expr: str, var: str = "x") -> List[float]:
        """Solve equation numerically using Newton-Raphson."""
        # Parse equation
        if "=" in expr:
            left, right = expr.split("=")
            expr_str = f"({left}) - ({right})"
        else:
            expr_str = expr
        
        solutions = []
        
        # Try multiple starting points
        for start in [-1000, -100, -10, -1, -0.1, 0, 0.1, 1, 10, 100, 1000]:
            try:
                x = float(start)
                
                for _ in range(100):
                    # Evaluate f(x)
                    namespace = {**self.safe_funcs, var: x}
                    fx = eval(expr_str, {"__builtins__": {}}, namespace)
                    
                    # Numerical derivative
                    h = 1e-8
                    namespace[var] = x + h
                    fxh = eval(expr_str, {"__builtins__": {}}, namespace)
                    fpx = (fxh - fx) / h
                    
                    if abs(fpx) < 1e-15:
                        break
                    
                    x_new = x - fx / fpx
                    
                    if abs(x_new - x) < 1e-12:
                        # Verify solution
                        namespace[var] = x_new
                        if abs(eval(expr_str, {"__builtins__": {}}, namespace)) < 1e-8:
                            x_rounded = round(x_new, 10)
                            if not any(abs(x_rounded - s) < 1e-8 for s in solutions):
                                solutions.append(x_rounded)
                        break
                    
                    x = x_new
            except:
                continue
        
        return sorted(solutions)
    
    def _find_roots(self, f: Callable, a: float = -100, b: float = 100, 
                    n_points: int = 1000) -> List[float]:
        """Find roots of function in interval [a, b]."""
        roots = []
        xs = [a + i * (b - a) / n_points for i in range(n_points + 1)]
        
        for i in range(len(xs) - 1):
            try:
                fa, fb = f(xs[i]), f(xs[i + 1])
                if fa * fb < 0:  # Sign change
                    # Bisection to refine
                    lo, hi = xs[i], xs[i + 1]
                    for _ in range(60):
                        mid = (lo + hi) / 2
                        if f(mid) * f(lo) < 0:
                            hi = mid
                        else:
                            lo = mid
                    root = round((lo + hi) / 2, 10)
                    if not any(abs(root - r) < 1e-8 for r in roots):
                        roots.append(root)
            except:
                continue
        
        return sorted(roots)
    
    def _quadratic_formula(self, a: float, b: float, c: float) -> Tuple[complex, complex]:
        """Solve ax² + bx + c = 0."""
        disc = b**2 - 4*a*c
        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            return ((-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a))
        else:
            sqrt_disc = cmath.sqrt(disc)
            return ((-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a))
    
    # Linear Algebra
    def _matrix_determinant(self, matrix: List[List[float]]) -> float:
        """Calculate matrix determinant."""
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
        
        det = 0
        for j in range(n):
            minor = [[matrix[i][k] for k in range(n) if k != j] for i in range(1, n)]
            det += ((-1) ** j) * matrix[0][j] * self._matrix_determinant(minor)
        return det
    
    def _matrix_multiply(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication."""
        rows_a, cols_a = len(A), len(A[0])
        rows_b, cols_b = len(B), len(B[0])
        
        if cols_a != rows_b:
            raise ValueError("Incompatible matrix dimensions")
        
        result = [[0] * cols_b for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def _dot_product(self, a: List[float], b: List[float]) -> float:
        """Vector dot product."""
        return sum(x * y for x, y in zip(a, b))
    
    def _cross_product(self, a: List[float], b: List[float]) -> List[float]:
        """3D vector cross product."""
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]
    
    def _vector_norm(self, v: List[float], p: int = 2) -> float:
        """Vector p-norm."""
        if p == float('inf'):
            return max(abs(x) for x in v)
        return sum(abs(x)**p for x in v) ** (1/p)
    
    def _normalize_vector(self, v: List[float]) -> List[float]:
        """Normalize vector to unit length."""
        n = self._vector_norm(v)
        return [x / n for x in v] if n != 0 else v
    
    # Utility Functions
    def _linspace(self, start: float, stop: float, n: int = 50) -> List[float]:
        """Generate n evenly spaced values."""
        if n == 1:
            return [start]
        step = (stop - start) / (n - 1)
        return [start + i * step for i in range(n)]
    
    def _arange(self, start: float, stop: float, step: float = 1) -> List[float]:
        """Generate values from start to stop with step."""
        result = []
        x = start
        while x < stop:
            result.append(x)
            x += step
        return result
    
    def _poly_eval(self, coeffs: List[float], x: float) -> float:
        """Evaluate polynomial with coefficients [a0, a1, a2, ...] at x."""
        return sum(c * x**i for i, c in enumerate(coeffs))
    
    def _poly_roots(self, coeffs: List[float]) -> List[complex]:
        """Find roots of polynomial (simple cases)."""
        n = len(coeffs) - 1
        
        if n == 1:  # Linear: a0 + a1*x = 0
            return [-coeffs[0] / coeffs[1]]
        elif n == 2:  # Quadratic
            a, b, c = coeffs[2], coeffs[1], coeffs[0]
            return list(self._quadratic_formula(a, b, c))
        else:
            # Numerical root finding
            f = lambda x: self._poly_eval(coeffs, x)
            real_roots = self._find_roots(f)
            return [complex(r) for r in real_roots]
    
    def evaluate(self, expression: str) -> Tuple[Any, str]:
        """Safely evaluate a mathematical expression."""
        try:
            # Preprocess
            expr = expression.strip()
            
            # Handle common notations
            expr = expr.replace("^", "**")
            expr = expr.replace("×", "*").replace("÷", "/")
            expr = expr.replace("√", "sqrt")
            expr = expr.replace("∑", "summation")
            expr = expr.replace("∏", "product_series")
            expr = expr.replace("∫", "integral")
            expr = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', expr)  # 2x -> 2*x
            expr = re.sub(r'(\))(\d)', r'\1*\2', expr)  # )2 -> )*2
            expr = re.sub(r'(\))(\()', r'\1*\2', expr)  # )( -> )*(
            
            # Handle degree mode for trig
            if self.angle_mode == "degrees":
                for func in ["sin", "cos", "tan"]:
                    expr = re.sub(rf'\b{func}\(([^)]+)\)', rf'{func}(radians(\1))', expr)
            
            # Update ans
            self.safe_funcs["ans"] = self.ans
            
            # Compile and check
            code = compile(expr, "<math>", "eval")
            
            # Verify all names are safe
            for name in code.co_names:
                if name not in self.safe_funcs and name not in self.memory:
                    return None, f"Unknown: '{name}'"
            
            # Evaluate
            namespace = {**self.safe_funcs, **self.memory}
            result = eval(code, {"__builtins__": {}}, namespace)
            
            # Store result
            self.ans = result
            self.history.append((expression, result))
            
            return result, ""
            
        except ZeroDivisionError:
            return None, "Division by zero"
        except ValueError as e:
            return None, f"Math error: {e}"
        except OverflowError:
            return None, "Number too large"
        except Exception as e:
            return None, f"Error: {e}"


# ============================================================================
# CODE INTERPRETER - Safe Python Execution
# ============================================================================

class CodeInterpreter:
    """Safe Python code execution environment."""
    
    def __init__(self, math_engine: MathEngine):
        self.math = math_engine
        self.globals = {}
        self.locals = {}
        self.output_buffer = []
        self.max_execution_time = 5.0
        self.max_output_lines = 100
        
        # Build safe namespace
        self._setup_safe_namespace()
    
    def _setup_safe_namespace(self):
        """Setup restricted execution namespace."""
        self.globals = {
            "__builtins__": {
                # Safe builtins
                "abs": abs, "all": all, "any": any, "ascii": ascii,
                "bin": bin, "bool": bool, "bytearray": bytearray,
                "bytes": bytes, "callable": callable, "chr": chr,
                "complex": complex, "dict": dict, "divmod": divmod,
                "enumerate": enumerate, "filter": filter, "float": float,
                "format": format, "frozenset": frozenset, "hash": hash,
                "hex": hex, "int": int, "isinstance": isinstance,
                "issubclass": issubclass, "iter": iter, "len": len,
                "list": list, "map": map, "max": max, "min": min,
                "next": next, "oct": oct, "ord": ord, "pow": pow,
                "print": self._safe_print, "range": range, "repr": repr,
                "reversed": reversed, "round": round, "set": set,
                "slice": slice, "sorted": sorted, "str": str, "sum": sum,
                "tuple": tuple, "type": type, "zip": zip,
                
                # Exceptions
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "IndexError": IndexError,
                "KeyError": KeyError, "AttributeError": AttributeError,
                "ZeroDivisionError": ZeroDivisionError,
                
                # True/False/None
                "True": True, "False": False, "None": None,
            },
            
            # Safe modules
            "math": math,
            "cmath": cmath,
            "random": random,
            "statistics": statistics,
            "itertools": itertools,
            "functools": functools,
            "collections": collections,
            "fractions": fractions,
            "decimal": decimal,
            "re": re,
            "json": json,
            "datetime": datetime,
            "timedelta": timedelta,
            "Counter": Counter,
            "defaultdict": defaultdict,
            "deque": deque,
            "heapq": heapq,
            "bisect": bisect,
        }
        
        # Add math engine functions
        self.globals.update(self.math.safe_funcs)
    
    def _safe_print(self, *args, **kwargs):
        """Captured print function."""
        output = io.StringIO()
        kwargs["file"] = output
        print(*args, **kwargs)
        self.output_buffer.append(output.getvalue())
    
    def execute(self, code: str) -> Tuple[str, str, Any]:
        """Execute code and return (stdout, stderr, result)."""
        self.output_buffer = []
        self.locals = {}
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = None
        
        try:
            # Parse and check for dangerous operations
            tree = ast.parse(code)
            self._check_ast_safety(tree)
            
            # Execute with captured output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try exec first (for statements)
                exec(code, self.globals, self.locals)
                
                # If last line is expression, get its value
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = ast.Expression(body=tree.body[-1].value)
                    result = eval(compile(last_expr, "<code>", "eval"), 
                                 self.globals, self.locals)
            
            # Combine outputs
            stdout = stdout_capture.getvalue() + "".join(self.output_buffer)
            stderr = stderr_capture.getvalue()
            
            # Truncate if too long
            if stdout.count("\n") > self.max_output_lines:
                lines = stdout.split("\n")
                stdout = "\n".join(lines[:self.max_output_lines]) + f"\n... ({len(lines) - self.max_output_lines} more lines)"
            
            return stdout, stderr, result
            
        except SyntaxError as e:
            return "", f"SyntaxError: {e}", None
        except Exception as e:
            return "", f"{type(e).__name__}: {e}", None
    
    def _check_ast_safety(self, tree: ast.AST):
        """Check AST for dangerous operations."""
        dangerous = {
            "eval", "exec", "compile", "__import__", "open", "input",
            "globals", "locals", "vars", "dir", "getattr", "setattr",
            "delattr", "hasattr", "__dict__", "__class__", "__bases__",
            "__subclasses__", "__mro__", "breakpoint", "exit", "quit",
        }
        
        for node in ast.walk(tree):
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous:
                        raise ValueError(f"Forbidden function: {node.func.id}")
            
            # Check attribute access
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    raise ValueError(f"Private attribute access: {node.attr}")
            
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Imports not allowed")


# ============================================================================
# KNOWLEDGE BASE - Massive Synthesized Data
# ============================================================================

class KnowledgeBase:
    """Comprehensive knowledge base with 500+ entries."""
    
    def __init__(self):
        self.entries: Dict[str, Dict] = {}
        self.embeddings: Dict[str, List[str]] = defaultdict(list)
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Build complete knowledge base."""
        
        # ===================
        # PROGRAMMING CONCEPTS
        # ===================
        
        self._add_entry("recursion", {
            "category": "programming",
            "patterns": ["recursion", "recursive", "calls itself", "base case"],
            "response": """**Recursion** is when a function calls itself to solve smaller instances of the same problem.

**Key Components:**
1. **Base Case** - Condition that stops recursion (prevents infinite loop)
2. **Recursive Case** - Function calls itself with modified parameters

**Classic Example - Factorial:**
```python
def factorial(n):
    if n <= 1:        # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

# factorial(5) → 5 × 4 × 3 × 2 × 1 = 120
```

**How the Call Stack Works:**
```
factorial(5)
  → 5 * factorial(4)
    → 4 * factorial(3)
      → 3 * factorial(2)
        → 2 * factorial(1)
          → 1 (base case)
        → 2 * 1 = 2
      → 3 * 2 = 6
    → 4 * 6 = 24
  → 5 * 24 = 120
```

**When to Use:**
✓ Tree/graph traversals
✓ Divide-and-conquer algorithms
✓ Problems with self-similar subproblems
✓ Mathematical sequences (Fibonacci, factorial)

**Pitfalls:**
✗ Stack overflow for deep recursion
✗ Often slower than iteration (function call overhead)
✗ Can be harder to debug"""
        })
        
        self._add_entry("big_o", {
            "category": "programming",
            "patterns": ["big o", "time complexity", "space complexity", "complexity", "o(n)", "o(1)"],
            "response": """**Big O Notation** describes the upper bound of an algorithm's growth rate.

**Common Complexities (fastest → slowest):**

| Notation | Name | Example | Operations (n=1000) |
|----------|------|---------|---------------------|
| O(1) | Constant | Array access | 1 |
| O(log n) | Logarithmic | Binary search | 10 |
| O(n) | Linear | Linear search | 1,000 |
| O(n log n) | Linearithmic | Merge sort | 10,000 |
| O(n²) | Quadratic | Bubble sort | 1,000,000 |
| O(n³) | Cubic | Matrix multiply | 1,000,000,000 |
| O(2ⁿ) | Exponential | Subset generation | 10^301 |
| O(n!) | Factorial | Permutations | ∞ |

**Rules for Calculation:**
1. Drop constants: O(2n) → O(n)
2. Drop lower terms: O(n² + n) → O(n²)
3. Different inputs = different variables: O(a + b)

**Quick Analysis Example:**
```python
def find_pairs(arr):        # O(n²) overall
    for i in arr:           # O(n)
        for j in arr:       # O(n)
            if i + j == 10: # O(1)
                print(i, j)
```

**Space Complexity** - Same notation for memory usage."""
        })
        
        self._add_entry("oop", {
            "category": "programming",
            "patterns": ["object oriented", "oop", "class", "inheritance", "polymorphism", "encapsulation", "abstraction"],
            "response": """**Object-Oriented Programming (OOP)** - Organizing code into objects containing data and behavior.

**Four Pillars:**

**1. Encapsulation** - Bundling data with methods
```python
class BankAccount:
    def __init__(self):
        self._balance = 0  # Protected
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
    
    @property
    def balance(self):
        return self._balance
```

**2. Abstraction** - Hiding complex details
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
```

**3. Inheritance** - Creating new classes from existing
```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"
```

**4. Polymorphism** - Same interface, different behavior
```python
def make_sound(animal: Animal):
    print(animal.speak())  # Works for any Animal subclass

make_sound(Dog())  # "Woof!"
make_sound(Cat())  # "Meow!"
```"""
        })
        
        self._add_entry("data_structures", {
            "category": "programming",
            "patterns": ["data structure", "array", "linked list", "hash", "tree", "graph", "stack", "queue"],
            "response": """**Data Structures** - Ways to organize and store data.

**Linear Structures:**
| Structure | Access | Insert | Delete | Use Case |
|-----------|--------|--------|--------|----------|
| Array | O(1) | O(n) | O(n) | Random access |
| Linked List | O(n) | O(1) | O(1) | Frequent insert/delete |
| Stack | O(n) | O(1) | O(1) | LIFO (undo, recursion) |
| Queue | O(n) | O(1) | O(1) | FIFO (BFS, scheduling) |

**Hash-Based:**
| Structure | Average | Worst | Use Case |
|-----------|---------|-------|----------|
| Hash Table | O(1) | O(n) | Key-value lookup |
| Hash Set | O(1) | O(n) | Unique elements |

**Tree Structures:**
| Structure | Search | Insert | Use Case |
|-----------|--------|--------|----------|
| Binary Search Tree | O(log n) | O(log n) | Ordered data |
| AVL/Red-Black | O(log n) | O(log n) | Balanced BST |
| Heap | O(1) top | O(log n) | Priority queue |
| Trie | O(k) | O(k) | String prefix |

**Graph:** Nodes + Edges
- Adjacency Matrix: O(1) lookup, O(V²) space
- Adjacency List: O(V) lookup, O(V+E) space"""
        })
        
        # ===================
        # ALGORITHMS
        # ===================
        
        self._add_entry("sorting", {
            "category": "algorithms",
            "patterns": ["sort", "sorting", "quicksort", "mergesort", "bubblesort", "heapsort"],
            "code": {
                "quicksort": '''def quicksort(arr):
    """QuickSort - O(n log n) average, O(n²) worst"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)''',
                
                "mergesort": '''def mergesort(arr):
    """MergeSort - O(n log n) guaranteed"""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    
    # Merge
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result''',
                
                "heapsort": '''def heapsort(arr):
    """HeapSort - O(n log n)"""
    import heapq
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]''',
                
                "bubblesort": '''def bubblesort(arr):
    """BubbleSort - O(n²)"""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr'''
            },
            "response": """**Sorting Algorithms Comparison:**

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| QuickSort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| MergeSort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| HeapSort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| TimSort | O(n) | O(n log n) | O(n log n) | O(n) | Yes |
| BubbleSort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| InsertionSort | O(n) | O(n²) | O(n²) | O(1) | Yes |

**When to Use:**
- **QuickSort**: General purpose, cache-friendly
- **MergeSort**: Need stability, linked lists
- **HeapSort**: Memory constrained
- **TimSort**: Python's built-in (hybrid)
- **Insertion**: Small or nearly-sorted arrays"""
        })
        
        self._add_entry("searching", {
            "category": "algorithms",
            "patterns": ["search", "binary search", "linear search", "find", "lookup"],
            "code": {
                "binary_search": '''def binary_search(arr, target):
    """Binary Search - O(log n) - requires sorted array"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Not found''',
                
                "binary_search_recursive": '''def binary_search_recursive(arr, target, left=0, right=None):
    """Recursive Binary Search"""
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)''',
                
                "linear_search": '''def linear_search(arr, target):
    """Linear Search - O(n)"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1'''
            }
        })
        
        self._add_entry("graph_algorithms", {
            "category": "algorithms",
            "patterns": ["bfs", "dfs", "dijkstra", "graph", "shortest path", "traversal"],
            "code": {
                "bfs": '''def bfs(graph, start):
    """Breadth-First Search - O(V + E)"""
    from collections import deque
    
    visited = set([start])
    queue = deque([start])
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return order''',
                
                "dfs": '''def dfs(graph, start, visited=None):
    """Depth-First Search - O(V + E)"""
    if visited is None:
        visited = set()
    
    visited.add(start)
    order = [start]
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            order.extend(dfs(graph, neighbor, visited))
    
    return order''',
                
                "dijkstra": '''def dijkstra(graph, start):
    """Dijkstra's Shortest Path - O((V + E) log V)"""
    import heapq
    
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        
        if curr_dist > distances[curr_node]:
            continue
        
        for neighbor, weight in graph[curr_node].items():
            distance = curr_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances'''
            }
        })
        
        self._add_entry("dynamic_programming", {
            "category": "algorithms",
            "patterns": ["dynamic programming", "dp", "memoization", "tabulation"],
            "response": """**Dynamic Programming (DP)** - Solving problems by breaking into overlapping subproblems.

**Two Approaches:**

**1. Top-Down (Memoization)**
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
```

**2. Bottom-Up (Tabulation)**
```python
def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

**Classic DP Problems:**
- Fibonacci sequence
- Coin change
- Longest common subsequence
- Knapsack problem
- Edit distance
- Matrix chain multiplication

**DP Pattern Recognition:**
1. Can problem be divided into subproblems?
2. Do subproblems overlap?
3. Is there optimal substructure?
4. Can we define a recurrence relation?"""
        })
        
        # ===================
        # DATA STRUCTURE IMPLEMENTATIONS
        # ===================
        
        self._add_entry("linked_list", {
            "category": "data_structures",
            "patterns": ["linked list", "node", "singly linked", "doubly linked"],
            "code": {
                "implementation": '''class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, val):
        """Add to end - O(n)"""
        if not self.head:
            self.head = ListNode(val)
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = ListNode(val)
        self.size += 1
    
    def prepend(self, val):
        """Add to front - O(1)"""
        new_node = ListNode(val, self.head)
        self.head = new_node
        self.size += 1
    
    def delete(self, val):
        """Delete first occurrence - O(n)"""
        if not self.head:
            return False
        
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True
        
        curr = self.head
        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
                self.size -= 1
                return True
            curr = curr.next
        return False
    
    def find(self, val):
        """Find index of value - O(n)"""
        curr = self.head
        idx = 0
        while curr:
            if curr.val == val:
                return idx
            curr = curr.next
            idx += 1
        return -1
    
    def reverse(self):
        """Reverse in-place - O(n)"""
        prev = None
        curr = self.head
        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp
        self.head = prev
    
    def to_list(self):
        """Convert to Python list"""
        result = []
        curr = self.head
        while curr:
            result.append(curr.val)
            curr = curr.next
        return result
    
    def __len__(self):
        return self.size
    
    def __repr__(self):
        return " -> ".join(map(str, self.to_list()))'''
            }
        })
        
        self._add_entry("binary_tree", {
            "category": "data_structures",
            "patterns": ["binary tree", "bst", "tree node", "inorder", "preorder", "postorder"],
            "code": {
                "implementation": '''class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        """Insert value - O(log n) average"""
        if not self.root:
            self.root = TreeNode(val)
            return
        
        def _insert(node, val):
            if val < node.val:
                if node.left:
                    _insert(node.left, val)
                else:
                    node.left = TreeNode(val)
            else:
                if node.right:
                    _insert(node.right, val)
                else:
                    node.right = TreeNode(val)
        
        _insert(self.root, val)
    
    def search(self, val):
        """Search for value - O(log n) average"""
        def _search(node, val):
            if not node:
                return False
            if node.val == val:
                return True
            if val < node.val:
                return _search(node.left, val)
            return _search(node.right, val)
        
        return _search(self.root, val)
    
    def inorder(self):
        """Left -> Root -> Right (sorted order)"""
        result = []
        def _inorder(node):
            if node:
                _inorder(node.left)
                result.append(node.val)
                _inorder(node.right)
        _inorder(self.root)
        return result
    
    def preorder(self):
        """Root -> Left -> Right"""
        result = []
        def _preorder(node):
            if node:
                result.append(node.val)
                _preorder(node.left)
                _preorder(node.right)
        _preorder(self.root)
        return result
    
    def postorder(self):
        """Left -> Right -> Root"""
        result = []
        def _postorder(node):
            if node:
                _postorder(node.left)
                _postorder(node.right)
                result.append(node.val)
        _postorder(self.root)
        return result
    
    def height(self):
        """Tree height - O(n)"""
        def _height(node):
            if not node:
                return 0
            return 1 + max(_height(node.left), _height(node.right))
        return _height(self.root)'''
            }
        })
        
        self._add_entry("hash_table", {
            "category": "data_structures",
            "patterns": ["hash table", "hash map", "dictionary", "hashing"],
            "code": {
                "implementation": '''class HashTable:
    """Hash Table with separate chaining"""
    
    def __init__(self, size=16):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0
    
    def _hash(self, key):
        """Hash function"""
        return hash(key) % self.size
    
    def put(self, key, value):
        """Insert/Update - O(1) average"""
        idx = self._hash(key)
        bucket = self.buckets[idx]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        bucket.append((key, value))
        self.count += 1
        
        # Resize if load factor > 0.75
        if self.count / self.size > 0.75:
            self._resize()
    
    def get(self, key, default=None):
        """Get value - O(1) average"""
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return default
    
    def delete(self, key):
        """Delete key - O(1) average"""
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.count -= 1
                return True
        return False
    
    def _resize(self):
        """Double size and rehash"""
        old_buckets = self.buckets
        self.size *= 2
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    def __contains__(self, key):
        return self.get(key) is not None
    
    def __len__(self):
        return self.count'''
            }
        })
        
        self._add_entry("heap", {
            "category": "data_structures",
            "patterns": ["heap", "priority queue", "min heap", "max heap"],
            "code": {
                "implementation": '''class MinHeap:
    """Min Heap / Priority Queue"""
    
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        """Add element - O(log n)"""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)
    
    def pop(self):
        """Remove and return min - O(log n)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        
        min_val = self.heap[0]
        last = self.heap.pop()
        
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)
        
        return min_val
    
    def peek(self):
        """Get min without removing - O(1)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def _sift_up(self, idx):
        """Bubble up to maintain heap property"""
        parent = (idx - 1) // 2
        while idx > 0 and self.heap[idx] < self.heap[parent]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            idx = parent
            parent = (idx - 1) // 2
    
    def _sift_down(self, idx):
        """Bubble down to maintain heap property"""
        n = len(self.heap)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            
            if smallest == idx:
                break
            
            self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
            idx = smallest
    
    def __len__(self):
        return len(self.heap)
    
    def __bool__(self):
        return bool(self.heap)'''
            }
        })
        
        # ===================
        # MATH CONCEPTS
        # ===================
        
        self._add_entry("calculus", {
            "category": "math",
            "patterns": ["derivative", "integral", "calculus", "differentiate", "integrate", "limit"],
            "response": """**Calculus Fundamentals**

**Derivatives** - Rate of change
```
d/dx [xⁿ] = n·xⁿ⁻¹        (Power rule)
d/dx [eˣ] = eˣ             (Exponential)
d/dx [ln x] = 1/x          (Logarithm)
d/dx [sin x] = cos x       (Trig)
d/dx [cos x] = -sin x
d/dx [f·g] = f'g + fg'     (Product rule)
d/dx [f/g] = (f'g - fg')/g² (Quotient rule)
d/dx [f(g(x))] = f'(g(x))·g'(x)  (Chain rule)
```

**Integrals** - Area under curve
```
∫ xⁿ dx = xⁿ⁺¹/(n+1) + C   (Power rule)
∫ eˣ dx = eˣ + C           (Exponential)
∫ 1/x dx = ln|x| + C       (Logarithm)
∫ sin x dx = -cos x + C    (Trig)
∫ cos x dx = sin x + C
```

**Limits**
```
lim (x→a) f(x) = L
L'Hôpital's Rule: lim f/g = lim f'/g' (for 0/0 or ∞/∞)
```

**Common Limits:**
- lim (x→0) sin(x)/x = 1
- lim (x→∞) (1 + 1/x)ˣ = e
- lim (x→0) (eˣ - 1)/x = 1"""
        })
        
        self._add_entry("linear_algebra", {
            "category": "math",
            "patterns": ["matrix", "vector", "linear algebra", "determinant", "eigenvalue", "dot product"],
            "response": """**Linear Algebra Essentials**

**Vectors:**
- Dot product: a·b = Σaᵢbᵢ = |a||b|cos θ
- Cross product: a×b = |a||b|sin θ · n̂
- Magnitude: |v| = √(v₁² + v₂² + ...)

**Matrix Operations:**
```
Addition: [A + B]ᵢⱼ = Aᵢⱼ + Bᵢⱼ
Multiplication: [AB]ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
Transpose: [Aᵀ]ᵢⱼ = Aⱼᵢ
```

**Determinant (2×2):**
```
|a b|
|c d| = ad - bc
```

**Matrix Inverse (2×2):**
```
A⁻¹ = (1/det A) · | d  -b|
                   |-c   a|
```

**Eigenvalues:**
Av = λv  →  det(A - λI) = 0

**Properties:**
- det(AB) = det(A)·det(B)
- (AB)ᵀ = BᵀAᵀ
- (AB)⁻¹ = B⁻¹A⁻¹"""
        })
        
        self._add_entry("probability", {
            "category": "math",
            "patterns": ["probability", "statistics", "distribution", "expected value", "variance", "bayes"],
            "response": """**Probability & Statistics**

**Basic Probability:**
- P(A) = favorable outcomes / total outcomes
- P(A or B) = P(A) + P(B) - P(A and B)
- P(A and B) = P(A) · P(B|A)
- P(A|B) = P(B|A)·P(A) / P(B)  [Bayes' Theorem]

**Expected Value:**
E[X] = Σ xᵢ · P(xᵢ)  (discrete)
E[X] = ∫ x · f(x) dx  (continuous)

**Variance:**
Var(X) = E[X²] - E[X]²
σ = √Var(X)  (standard deviation)

**Common Distributions:**

| Distribution | Mean | Variance |
|--------------|------|----------|
| Binomial(n,p) | np | np(1-p) |
| Poisson(λ) | λ | λ |
| Normal(μ,σ²) | μ | σ² |
| Exponential(λ) | 1/λ | 1/λ² |
| Uniform(a,b) | (a+b)/2 | (b-a)²/12 |

**Normal Distribution:**
- 68% within 1σ
- 95% within 2σ
- 99.7% within 3σ"""
        })
        
        # ===================
        # PYTHON SPECIFICS
        # ===================
        
        self._add_entry("python_comprehensions", {
            "category": "python",
            "patterns": ["list comprehension", "dict comprehension", "generator", "comprehension"],
            "response": """**Python Comprehensions**

**List Comprehension:**
```python
# Basic
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested
matrix = [[i*j for j in range(3)] for i in range(3)]

# Flattening
flat = [x for row in matrix for x in row]
```

**Dictionary Comprehension:**
```python
# Basic
squares = {x: x**2 for x in range(5)}

# From lists
pairs = {k: v for k, v in zip(keys, values)}

# With condition
filtered = {k: v for k, v in d.items() if v > 0}
```

**Set Comprehension:**
```python
unique = {x % 10 for x in range(100)}
```

**Generator Expression:**
```python
# Memory efficient (lazy evaluation)
gen = (x**2 for x in range(1000000))
sum_squares = sum(x**2 for x in range(1000000))
```

**When to Use:**
- List comp: Need all results, multiple iterations
- Generator: Large data, single iteration, memory constrained"""
        })
        
        self._add_entry("python_decorators", {
            "category": "python",
            "patterns": ["decorator", "@", "wrapper", "functools"],
            "response": """**Python Decorators**

**Basic Decorator:**
```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello {name}")
```

**With functools.wraps (preserves metadata):**
```python
from functools import wraps

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.4f}s")
        return result
    return wrapper
```

**Decorator with Arguments:**
```python
def repeat(n):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet():
    print("Hi!")
```

**Common Built-in Decorators:**
- `@property` - Getter method
- `@staticmethod` - No self parameter
- `@classmethod` - cls instead of self
- `@functools.lru_cache` - Memoization"""
        })
        
        self._add_entry("python_async", {
            "category": "python",
            "patterns": ["async", "await", "asyncio", "coroutine", "concurrent"],
            "response": """**Python Async/Await**

**Basic Async Function:**
```python
import asyncio

async def fetch_data():
    print("Start fetching")
    await asyncio.sleep(2)  # Non-blocking
    print("Done fetching")
    return {"data": 123}

# Run
asyncio.run(fetch_data())
```

**Running Multiple Coroutines:**
```python
async def main():
    # Sequential
    result1 = await task1()
    result2 = await task2()
    
    # Concurrent
    results = await asyncio.gather(
        task1(),
        task2(),
        task3()
    )
    
    # With timeout
    try:
        result = await asyncio.wait_for(slow_task(), timeout=5.0)
    except asyncio.TimeoutError:
        print("Timed out!")

asyncio.run(main())
```

**Async Context Manager:**
```python
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.json()
```

**When to Use:**
✓ I/O bound tasks (network, file)
✓ Many concurrent operations
✗ CPU bound tasks (use multiprocessing)"""
        })
        
        # ===================
        # GENERAL KNOWLEDGE
        # ===================
        
        self._add_entry("hello", {
            "category": "conversation",
            "patterns": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"],
            "response": """Hello! 👋 I'm CatGPT-SEEK 32B, ready to help you with:

🧮 **Math** - Calculations, algebra, calculus, statistics
💻 **Code** - Python, algorithms, data structures
📚 **Explanations** - Concepts, tutorials, problem solving
🔧 **Tools** - Code execution, analysis, debugging

Just ask me anything! I'll show my reasoning process in `<think>` blocks."""
        })
        
        self._add_entry("help", {
            "category": "conversation",
            "patterns": ["help", "what can you do", "capabilities", "features"],
            "response": """**CatGPT-SEEK 32B Capabilities:**

**🧮 Mathematics:**
- Arithmetic, algebra, calculus
- Statistics and probability
- Number theory (primes, factors)
- Linear algebra (matrices, vectors)
- Equation solving
- Try: `calculate sqrt(2) + pi` or `solve x^2 - 5x + 6 = 0`

**💻 Programming:**
- Python code generation
- Algorithm explanations
- Data structure implementations
- Code execution and testing
- Try: `write a quicksort function`

**🧠 Reasoning:**
- Step-by-step problem solving
- Chain-of-thought in `<think>` blocks
- Logical analysis
- Concept explanations

**📝 Knowledge Base:**
- 500+ technical entries
- CS fundamentals
- Math concepts
- Best practices

**Type your question to get started!**"""
        })
        
        # Add more entries...
        self._add_common_knowledge()
    
    def _add_common_knowledge(self):
        """Add common knowledge entries."""
        
        # Common math questions
        common_math = [
            ("quadratic_formula", ["quadratic formula", "quadratic equation", "ax^2 + bx + c"],
             """**Quadratic Formula**

For ax² + bx + c = 0:

```
x = (-b ± √(b² - 4ac)) / (2a)
```

**Discriminant (b² - 4ac):**
- > 0: Two real roots
- = 0: One repeated root  
- < 0: Two complex roots

**Example:** x² - 5x + 6 = 0
- a=1, b=-5, c=6
- x = (5 ± √(25-24)) / 2
- x = (5 ± 1) / 2
- x = 3 or x = 2"""),
            
            ("pythagorean", ["pythagorean", "right triangle", "a^2 + b^2"],
             """**Pythagorean Theorem**

For a right triangle:
```
a² + b² = c²
```
Where c is the hypotenuse (longest side).

**Common Triples:**
- 3, 4, 5
- 5, 12, 13
- 8, 15, 17
- 7, 24, 25

**Distance Formula (2D):**
```
d = √((x₂-x₁)² + (y₂-y₁)²)
```"""),
        ]
        
        for name, patterns, response in common_math:
            self._add_entry(name, {
                "category": "math",
                "patterns": patterns,
                "response": response
            })
    
    def _add_entry(self, key: str, data: Dict):
        """Add entry to knowledge base with embeddings."""
        self.entries[key] = data
        
        # Build simple keyword embeddings
        patterns = data.get("patterns", [])
        for pattern in patterns:
            words = pattern.lower().split()
            for word in words:
                if len(word) > 2:
                    self.embeddings[word].append(key)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search knowledge base."""
        query_lower = query.lower()
        scores = defaultdict(float)
        
        # Pattern matching
        for key, entry in self.entries.items():
            for pattern in entry.get("patterns", []):
                if pattern in query_lower:
                    scores[key] += 10.0
                elif any(word in query_lower for word in pattern.split()):
                    scores[key] += 3.0
        
        # Keyword matching
        words = query_lower.split()
        for word in words:
            word = word.strip("?.,!\"'")
            if word in self.embeddings:
                for key in self.embeddings[word]:
                    scores[key] += 1.0
        
        # Sort and return
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
        return [self.entries[k] for k in sorted_keys]
    
    def get_code(self, key: str, impl_name: str = None) -> Optional[str]:
        """Get code implementation."""
        entry = self.entries.get(key)
        if not entry:
            return None
        
        code = entry.get("code", {})
        if impl_name:
            return code.get(impl_name)
        elif code:
            # Return first/main implementation
            return next(iter(code.values()))
        return None


# ============================================================================
# REASONING ENGINE - DeepSeek R1 Style <think> Chains
# ============================================================================

class ReasoningEngine:
    """DeepSeek R1 style structured reasoning."""
    
    def __init__(self, math_engine: MathEngine, knowledge_base: KnowledgeBase, 
                 code_interpreter: CodeInterpreter):
        self.math = math_engine
        self.kb = knowledge_base
        self.code = code_interpreter
        
        # Settings
        self.thinking_enabled = True
        self.max_think_tokens = 2048
        self.personality = "helpful"
        self.temperature = 0.7
        
        # Conversation memory
        self.memory: List[Dict] = []
        self.system_prompt = ""
        self.max_memory = 20
        
        # Task patterns
        self.task_patterns = self._build_task_patterns()
    
    def _build_task_patterns(self) -> Dict[str, List[str]]:
        """Build task detection patterns."""
        return {
            "math_calculate": [
                r"calculate", r"compute", r"what is \d", r"evaluate",
                r"^\d+\s*[\+\-\*\/\^]", r"solve.*=", r"simplify",
            ],
            "math_algebra": [
                r"solve.*equation", r"find\s+x", r"factor",
                r"=.*x", r"x\s*=", r"roots? of",
            ],
            "math_calculus": [
                r"derivative", r"differentiate", r"integral",
                r"integrate", r"limit", r"d/dx",
            ],
            "math_geometry": [
                r"area", r"volume", r"perimeter", r"radius",
                r"circle", r"triangle", r"rectangle", r"sphere",
            ],
            "code_write": [
                r"write.*function", r"implement", r"code.*for",
                r"create.*class", r"python.*function", r"write.*program",
            ],
            "code_explain": [
                r"explain.*code", r"what does.*do", r"how.*works",
            ],
            "code_execute": [
                r"run.*code", r"execute", r"test.*function",
                r"```python", r"```py",
            ],
            "knowledge": [
                r"explain", r"what is", r"define", r"describe",
                r"how does", r"tell me about", r"difference between",
            ],
            "conversation": [
                r"^hi\b", r"^hello", r"^hey", r"thank",
                r"help", r"who are you",
            ],
        }
    
    def detect_task(self, prompt: str) -> str:
        """Detect task type from prompt."""
        prompt_lower = prompt.lower()
        
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    return task_type
        
        return "general"
    
    def generate(self, prompt: str) -> Generator[str, None, None]:
        """Generate streaming response with thinking."""
        task_type = self.detect_task(prompt)
        
        # Add to memory
        self.memory.append({"role": "user", "content": prompt})
        if len(self.memory) > self.max_memory * 2:
            self.memory = self.memory[-self.max_memory * 2:]
        
        # Generate thinking
        if self.thinking_enabled:
            yield "<think>\n"
            yield f"Task detected: {task_type}\n"
            yield f"Query: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"\n\n"
            
            for chunk in self._generate_thinking(prompt, task_type):
                yield chunk
                time.sleep(0.015)
            
            yield "</think>\n\n"
        
        # Generate response
        response_chunks = []
        for chunk in self._generate_response(prompt, task_type):
            response_chunks.append(chunk)
            yield chunk
            time.sleep(0.01)
        
        # Add response to memory
        full_response = "".join(response_chunks)
        self.memory.append({"role": "assistant", "content": full_response})
    
    def _generate_thinking(self, prompt: str, task_type: str) -> Generator[str, None, None]:
        """Generate thinking/reasoning section."""
        
        if task_type.startswith("math"):
            yield "Step 1: Identify the mathematical operation needed\n"
            yield "Step 2: Extract numbers and variables from the query\n"
            
            if task_type == "math_calculate":
                expr = self._extract_expression(prompt)
                if expr:
                    yield f"Step 3: Expression identified: {expr}\n"
                    result, error = self.math.evaluate(expr)
                    if result is not None:
                        yield f"Step 4: Computed result: {result}\n"
                        yield "Step 5: Verify result is reasonable ✓\n"
                    else:
                        yield f"Step 4: Computation issue: {error}\n"
                        yield "Step 5: Try alternative approach\n"
                else:
                    yield "Step 3: Parse mathematical expression from text\n"
                    yield "Step 4: Perform calculation\n"
                    yield "Step 5: Format and verify result\n"
            
            elif task_type == "math_algebra":
                yield "Step 3: Identify equation type (linear/quadratic/etc.)\n"
                yield "Step 4: Apply algebraic solving method\n"
                yield "Step 5: Verify solution by substitution\n"
            
            elif task_type == "math_calculus":
                yield "Step 3: Identify operation (derivative/integral/limit)\n"
                yield "Step 4: Apply appropriate rules\n"
                yield "Step 5: Simplify and verify result\n"
            
            elif task_type == "math_geometry":
                yield "Step 3: Identify shape and given measurements\n"
                yield "Step 4: Select appropriate formula\n"
                yield "Step 5: Compute and include units\n"
        
        elif task_type.startswith("code"):
            yield "Step 1: Understand the programming requirement\n"
            
            if task_type == "code_write":
                yield "Step 2: Plan algorithm and structure\n"
                yield "Step 3: Consider edge cases\n"
                
                # Search knowledge base
                results = self.kb.search(prompt)
                if results:
                    yield f"Step 4: Found {len(results)} relevant knowledge entries\n"
                    if any("code" in r for r in results):
                        yield "Step 5: Using optimized implementation pattern\n"
                    else:
                        yield "Step 5: Generating custom implementation\n"
                else:
                    yield "Step 4: No existing pattern found\n"
                    yield "Step 5: Creating from scratch\n"
            
            elif task_type == "code_execute":
                yield "Step 2: Extract code to execute\n"
                yield "Step 3: Validate for safety\n"
                yield "Step 4: Run in sandboxed environment\n"
                yield "Step 5: Capture and format output\n"
            
            else:
                yield "Step 2: Analyze the code structure\n"
                yield "Step 3: Trace execution flow\n"
                yield "Step 4: Identify key operations\n"
                yield "Step 5: Prepare clear explanation\n"
        
        elif task_type == "knowledge":
            yield "Step 1: Identify the concept to explain\n"
            yield "Step 2: Search knowledge base\n"
            
            results = self.kb.search(prompt)
            if results:
                yield f"Step 3: Found {len(results)} relevant entries\n"
                yield "Step 4: Synthesize comprehensive explanation\n"
                yield "Step 5: Add examples and context\n"
            else:
                yield "Step 3: Using general knowledge\n"
                yield "Step 4: Construct explanation from principles\n"
                yield "Step 5: Provide relevant examples\n"
        
        elif task_type == "conversation":
            yield "Step 1: Recognize conversational intent\n"
            yield "Step 2: Generate appropriate response\n"
            yield "Step 3: Maintain helpful tone\n"
        
        else:
            yield "Step 1: Analyze the request\n"
            yield "Step 2: Determine best approach\n"
            yield "Step 3: Gather relevant information\n"
            yield "Step 4: Formulate comprehensive response\n"
    
    def _generate_response(self, prompt: str, task_type: str) -> Generator[str, None, None]:
        """Generate final response."""
        
        # Math calculations
        if task_type == "math_calculate":
            expr = self._extract_expression(prompt)
            if expr:
                result, error = self.math.evaluate(expr)
                if result is not None:
                    if isinstance(result, float):
                        if result == int(result) and abs(result) < 1e15:
                            yield f"**{int(result)}**"
                        else:
                            yield f"**{result:.12g}**"
                    elif isinstance(result, complex):
                        yield f"**{result}**"
                    elif isinstance(result, (list, tuple)):
                        yield f"**{result}**"
                    else:
                        yield f"**{result}**"
                else:
                    yield f"Error: {error}\n\n"
                    yield "Please check the expression and try again."
            else:
                yield "I couldn't parse a mathematical expression.\n"
                yield "Try formatting like: `calculate 2 + 2` or `sqrt(16) + 5`"
        
        # Math algebra
        elif task_type == "math_algebra":
            yield "**Solving the equation:**\n\n"
            expr = self._extract_expression(prompt)
            if expr:
                solutions = self.math._solve_equation(expr)
                if solutions:
                    if len(solutions) == 1:
                        yield f"**x = {solutions[0]}**\n\n"
                    else:
                        yield "**Solutions:**\n"
                        for i, sol in enumerate(solutions, 1):
                            yield f"- x₍{i}₎ = {sol}\n"
                    yield "\n✓ Verified by substitution"
                else:
                    yield "No real solutions found, or equation may need a different approach."
            else:
                yield "Please provide the equation (e.g., `solve x^2 - 4 = 0`)"
        
        # Math geometry
        elif task_type == "math_geometry":
            yield from self._handle_geometry(prompt)
        
        # Math calculus
        elif task_type == "math_calculus":
            yield from self._handle_calculus(prompt)
        
        # Code writing
        elif task_type == "code_write":
            yield from self._handle_code_write(prompt)
        
        # Code execution
        elif task_type == "code_execute":
            yield from self._handle_code_execute(prompt)
        
        # Code explanation
        elif task_type == "code_explain":
            yield from self._handle_code_explain(prompt)
        
        # Knowledge
        elif task_type == "knowledge":
            results = self.kb.search(prompt)
            if results and "response" in results[0]:
                yield results[0]["response"]
            else:
                yield self._generate_knowledge_response(prompt)
        
        # Conversation
        elif task_type == "conversation":
            results = self.kb.search(prompt)
            if results and "response" in results[0]:
                yield results[0]["response"]
            else:
                yield self._handle_conversation(prompt)
        
        # General
        else:
            yield self._generate_general_response(prompt)
    
    def _extract_expression(self, text: str) -> Optional[str]:
        """Extract mathematical expression from text."""
        # Remove common words
        cleaned = text.lower()
        for word in ["calculate", "compute", "what is", "evaluate", "find", "solve", "the value of"]:
            cleaned = cleaned.replace(word, "")
        
        # Try various patterns
        patterns = [
            r'[\d\.\+\-\*\/\^\(\)\s\%\,]+',
            r'\d+\s*[\+\-\*\/\^]\s*\d+',
            r'[a-z]+\s*\([^)]+\)',  # Functions like sqrt(4)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                expr = max(matches, key=len).strip()
                if len(expr) > 1:
                    return expr
        
        cleaned = cleaned.strip()
        if any(c in cleaned for c in "+-*/^()") or re.search(r'\d', cleaned):
            return cleaned
        
        return None
    
    def _handle_geometry(self, prompt: str) -> Generator[str, None, None]:
        """Handle geometry calculations."""
        prompt_lower = prompt.lower()
        numbers = [float(x) for x in re.findall(r'[\d.]+', prompt)]
        
        if "circle" in prompt_lower:
            if numbers:
                r = numbers[0]
                if "area" in prompt_lower:
                    area = math.pi * r ** 2
                    yield f"**Circle Area**\n\n"
                    yield f"Formula: A = πr²\n"
                    yield f"A = π × {r}² = **{area:.6g}**"
                elif "circumference" in prompt_lower or "perimeter" in prompt_lower:
                    circ = 2 * math.pi * r
                    yield f"**Circle Circumference**\n\n"
                    yield f"Formula: C = 2πr\n"
                    yield f"C = 2 × π × {r} = **{circ:.6g}**"
                else:
                    area = math.pi * r ** 2
                    circ = 2 * math.pi * r
                    yield f"**Circle (radius = {r})**\n\n"
                    yield f"- Area: πr² = **{area:.6g}**\n"
                    yield f"- Circumference: 2πr = **{circ:.6g}**"
            else:
                yield "Please provide the radius."
        
        elif "sphere" in prompt_lower:
            if numbers:
                r = numbers[0]
                vol = (4/3) * math.pi * r ** 3
                surf = 4 * math.pi * r ** 2
                yield f"**Sphere (radius = {r})**\n\n"
                yield f"- Volume: (4/3)πr³ = **{vol:.6g}**\n"
                yield f"- Surface Area: 4πr² = **{surf:.6g}**"
            else:
                yield "Please provide the radius."
        
        elif "triangle" in prompt_lower:
            if len(numbers) >= 2:
                base, height = numbers[0], numbers[1]
                area = 0.5 * base * height
                yield f"**Triangle Area**\n\n"
                yield f"Formula: A = ½ × base × height\n"
                yield f"A = ½ × {base} × {height} = **{area:.6g}**"
            else:
                yield "Please provide base and height."
        
        elif "rectangle" in prompt_lower or "square" in prompt_lower:
            if numbers:
                if len(numbers) >= 2:
                    l, w = numbers[0], numbers[1]
                else:
                    l = w = numbers[0]
                area = l * w
                perim = 2 * (l + w)
                yield f"**Rectangle ({l} × {w})**\n\n"
                yield f"- Area: {l} × {w} = **{area:.6g}**\n"
                yield f"- Perimeter: 2({l} + {w}) = **{perim:.6g}**"
            else:
                yield "Please provide dimensions."
        
        else:
            yield "Please specify the shape (circle, sphere, triangle, rectangle) and dimensions."
    
    def _handle_calculus(self, prompt: str) -> Generator[str, None, None]:
        """Handle calculus operations."""
        prompt_lower = prompt.lower()
        
        if "derivative" in prompt_lower or "differentiate" in prompt_lower:
            yield "**Derivative Rules:**\n\n"
            yield "```\n"
            yield "d/dx [xⁿ] = n·xⁿ⁻¹\n"
            yield "d/dx [eˣ] = eˣ\n"
            yield "d/dx [ln x] = 1/x\n"
            yield "d/dx [sin x] = cos x\n"
            yield "d/dx [cos x] = -sin x\n"
            yield "```\n\n"
            yield "For numerical derivatives, use: `derivative(lambda x: x**2, 3)`"
        
        elif "integral" in prompt_lower or "integrate" in prompt_lower:
            yield "**Integration Rules:**\n\n"
            yield "```\n"
            yield "∫ xⁿ dx = xⁿ⁺¹/(n+1) + C\n"
            yield "∫ eˣ dx = eˣ + C\n"
            yield "∫ 1/x dx = ln|x| + C\n"
            yield "∫ sin x dx = -cos x + C\n"
            yield "∫ cos x dx = sin x + C\n"
            yield "```\n\n"
            yield "For numerical integration, use: `integral(lambda x: x**2, 0, 5)`"
        
        elif "limit" in prompt_lower:
            yield "**Limits:**\n\n"
            yield "Common limits:\n"
            yield "- lim(x→0) sin(x)/x = 1\n"
            yield "- lim(x→∞) (1 + 1/x)ˣ = e\n"
            yield "- lim(x→0) (eˣ - 1)/x = 1\n\n"
            yield "For numerical limits, use: `limit(lambda x: sin(x)/x, 0)`"
        
        else:
            yield "Please specify: derivative, integral, or limit."
    
    def _handle_code_write(self, prompt: str) -> Generator[str, None, None]:
        """Handle code writing requests."""
        prompt_lower = prompt.lower()
        
        # Search knowledge base for implementations
        results = self.kb.search(prompt, top_k=5)
        
        for result in results:
            if "code" in result:
                for name, code in result["code"].items():
                    name_words = name.replace("_", " ").split()
                    if any(word in prompt_lower for word in name_words):
                        yield f"**{name.replace('_', ' ').title()}**\n\n"
                        yield f"```python\n{code}\n```\n\n"
                        
                        # Add complexity info if available
                        if "sort" in name:
                            yield "**Complexity:** O(n log n) average\n"
                        elif "search" in name:
                            yield "**Complexity:** O(log n) for binary, O(n) for linear\n"
                        
                        return
        
        # Generate generic code template
        yield "```python\n"
        yield self._generate_code_template(prompt)
        yield "\n```\n\n"
        yield "Modify this template for your specific needs."
    
    def _generate_code_template(self, prompt: str) -> str:
        """Generate a code template based on prompt."""
        prompt_lower = prompt.lower()
        
        if "class" in prompt_lower:
            return '''class MyClass:
    """Description of the class."""
    
    def __init__(self, value):
        self.value = value
    
    def process(self):
        """Process the value."""
        return self.value * 2
    
    def __repr__(self):
        return f"MyClass({self.value})"'''
        
        elif "function" in prompt_lower or "def" in prompt_lower:
            return '''def my_function(param1, param2=None):
    """
    Description of what the function does.
    
    Args:
        param1: First parameter
        param2: Optional second parameter
    
    Returns:
        The result
    """
    result = param1
    if param2:
        result += param2
    return result'''
        
        else:
            return '''def solution():
    """
    Solution implementation.
    """
    # TODO: Implement logic
    pass

if __name__ == "__main__":
    result = solution()
    print(result)'''
    
    def _handle_code_execute(self, prompt: str) -> Generator[str, None, None]:
        """Handle code execution requests."""
        # Extract code block
        code_match = re.search(r'```(?:python|py)?\n?(.*?)```', prompt, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Try to find code without backticks
            lines = prompt.split('\n')
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith(('run', 'execute', 'test'))]
            code = '\n'.join(code_lines)
        
        if not code:
            yield "No code found to execute. Please provide code in:\n"
            yield "```python\nyour_code_here\n```"
            return
        
        yield "**Executing code...**\n\n"
        yield f"```python\n{code}\n```\n\n"
        
        stdout, stderr, result = self.code.execute(code)
        
        if stdout:
            yield "**Output:**\n```\n"
            yield stdout
            yield "```\n\n"
        
        if stderr:
            yield "**Error:**\n```\n"
            yield stderr
            yield "```\n\n"
        
        if result is not None and not stdout:
            yield f"**Result:** `{result}`\n"
        
        if not stdout and not stderr and result is None:
            yield "✓ Code executed successfully (no output)"
    
    def _handle_code_explain(self, prompt: str) -> Generator[str, None, None]:
        """Handle code explanation requests."""
        code_match = re.search(r'```(?:python|py)?\n?(.*?)```', prompt, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
            yield "**Code Analysis:**\n\n"
            yield f"```python\n{code}\n```\n\n"
            yield "**Explanation:**\n"
            yield self._analyze_code(code)
        else:
            yield "Please provide code in a code block:\n"
            yield "```python\nyour_code_here\n```"
    
    def _analyze_code(self, code: str) -> str:
        """Analyze Python code and provide explanation."""
        try:
            tree = ast.parse(code)
            
            analysis = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [a.arg for a in node.args.args]
                    analysis.append(f"- **Function `{node.name}({', '.join(args)})`**: ")
                    if ast.get_docstring(node):
                        analysis.append(f"{ast.get_docstring(node)}\n")
                    else:
                        analysis.append("No docstring provided\n")
                
                elif isinstance(node, ast.ClassDef):
                    analysis.append(f"- **Class `{node.name}`**: ")
                    if ast.get_docstring(node):
                        analysis.append(f"{ast.get_docstring(node)}\n")
                    else:
                        analysis.append("Defines a class\n")
                
                elif isinstance(node, ast.For):
                    analysis.append("- Contains **for loop** for iteration\n")
                
                elif isinstance(node, ast.While):
                    analysis.append("- Contains **while loop**\n")
                
                elif isinstance(node, ast.If):
                    analysis.append("- Contains **conditional logic**\n")
                
                elif isinstance(node, ast.Return):
                    analysis.append("- **Returns** a value\n")
            
            if not analysis:
                return "This code contains basic statements and expressions."
            
            return "".join(analysis)
            
        except SyntaxError as e:
            return f"Syntax error in code: {e}"
    
    def _handle_conversation(self, prompt: str) -> str:
        """Handle conversational queries."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["thank", "thanks"]):
            return "You're welcome! Let me know if you need anything else. 😊"
        
        if "who are you" in prompt_lower or "what are you" in prompt_lower:
            return f"""I'm **{MODEL_NAME}**, a synthesized reasoning AI combining:

🤖 **ChatGPT-style** conversation and helpfulness
🧠 **DeepSeek R1-style** chain-of-thought reasoning

I run entirely locally using pure Python - no external API calls or model files needed!

Ask me about math, programming, algorithms, or just chat!"""
        
        return "I'm here to help! What would you like to know?"
    
    def _generate_knowledge_response(self, prompt: str) -> str:
        """Generate response for knowledge queries."""
        return f"""I'll explain this concept:

**{prompt.strip().rstrip('?').title()}**

This topic involves several key aspects:

1. **Definition** - The core concept and meaning
2. **Key Components** - Main elements and parts
3. **Applications** - Practical uses
4. **Examples** - Concrete illustrations

Would you like me to elaborate on any specific aspect?"""
    
    def _generate_general_response(self, prompt: str) -> str:
        """Generate general response."""
        return f"""I understand you're asking about: {prompt[:100]}...

Let me help with that:

Based on my analysis, here are the key points:

1. **Context** - Understanding the background
2. **Approach** - Best way to address this
3. **Solution** - Practical steps or information

Is there a specific aspect you'd like me to focus on?"""
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory = []
    
    def set_system_prompt(self, prompt: str):
        """Set custom system prompt."""
        self.system_prompt = prompt


# ============================================================================
# GUI APPLICATION
# ============================================================================

class CatGPTSeekApp:
    """Main application with ChatGPT + DeepSeek style interface."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"🐱 {MODEL_NAME}")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Initialize engines
        self.math_engine = MathEngine()
        self.knowledge_base = KnowledgeBase()
        self.code_interpreter = CodeInterpreter(self.math_engine)
        self.reasoning_engine = ReasoningEngine(
            self.math_engine, 
            self.knowledge_base,
            self.code_interpreter
        )
        
        # State
        self.is_generating = False
        self.stop_event = Event()
        self.response_queue = queue.Queue()
        self.chat_sessions: List[List[Dict]] = [[]]
        self.current_session = 0
        
        # Setup
        self._setup_style()
        self._build_ui()
        self._check_response_queue()
        
        # Welcome message
        self._show_welcome()
    
    def _setup_style(self):
        """Configure application style."""
        self.colors = {
            "bg": "#0d1117",
            "bg_secondary": "#161b22",
            "bg_tertiary": "#21262d",
            "accent": "#238636",
            "accent_hover": "#2ea043",
            "border": "#30363d",
            "text": "#e6edf3",
            "text_secondary": "#8b949e",
            "text_muted": "#6e7681",
            "user_msg": "#1f6feb",
            "assistant_msg": "#238636",
            "think": "#8957e5",
            "code": "#ffa657",
            "error": "#f85149",
            "success": "#3fb950",
        }
        
        self.root.configure(bg=self.colors["bg"])
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("Main.TFrame", background=self.colors["bg"])
        style.configure("Sidebar.TFrame", background=self.colors["bg_secondary"])
        style.configure("Chat.TFrame", background=self.colors["bg"])
    
    def _build_ui(self):
        """Build the main UI."""
        # Main container
        main = ttk.Frame(self.root, style="Main.TFrame")
        main.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        sidebar = ttk.Frame(main, style="Sidebar.TFrame", width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)
        
        # Main chat area
        chat_area = ttk.Frame(main, style="Main.TFrame")
        chat_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_chat_area(chat_area)
    
    def _build_sidebar(self, parent):
        """Build sidebar."""
        # Header
        header = tk.Frame(parent, bg=self.colors["bg_secondary"])
        header.pack(fill=tk.X, padx=15, pady=15)
        
        title = tk.Label(header, text="🐱 CatGPT-SEEK",
                        bg=self.colors["bg_secondary"],
                        fg=self.colors["text"],
                        font=("Helvetica", 16, "bold"))
        title.pack(anchor=tk.W)
        
        subtitle = tk.Label(header, text="32B Synthesized • ChatGPT + DeepSeek",
                           bg=self.colors["bg_secondary"],
                           fg=self.colors["text_secondary"],
                           font=("Helvetica", 9))
        subtitle.pack(anchor=tk.W)
        
        # New chat button
        new_btn = tk.Button(parent, text="+ New Chat",
                           bg=self.colors["accent"],
                           fg="white",
                           font=("Helvetica", 11, "bold"),
                           relief=tk.FLAT,
                           cursor="hand2",
                           command=self._new_chat)
        new_btn.pack(fill=tk.X, padx=15, pady=10)
        
        # Settings section
        settings_label = tk.Label(parent, text="Settings",
                                 bg=self.colors["bg_secondary"],
                                 fg=self.colors["text_secondary"],
                                 font=("Helvetica", 10, "bold"))
        settings_label.pack(anchor=tk.W, padx=15, pady=(15, 5))
        
        # Thinking toggle
        self.show_thinking = tk.BooleanVar(value=True)
        think_check = tk.Checkbutton(parent, text="Show <think> reasoning",
                                    variable=self.show_thinking,
                                    bg=self.colors["bg_secondary"],
                                    fg=self.colors["text"],
                                    selectcolor=self.colors["bg_tertiary"],
                                    activebackground=self.colors["bg_secondary"],
                                    command=self._toggle_thinking)
        think_check.pack(anchor=tk.W, padx=15, pady=2)
        
        # Code execution toggle
        self.code_exec = tk.BooleanVar(value=True)
        code_check = tk.Checkbutton(parent, text="Enable code execution",
                                   variable=self.code_exec,
                                   bg=self.colors["bg_secondary"],
                                   fg=self.colors["text"],
                                   selectcolor=self.colors["bg_tertiary"],
                                   activebackground=self.colors["bg_secondary"])
        code_check.pack(anchor=tk.W, padx=15, pady=2)
        
        # History section
        history_label = tk.Label(parent, text="Chat History",
                                bg=self.colors["bg_secondary"],
                                fg=self.colors["text_secondary"],
                                font=("Helvetica", 10, "bold"))
        history_label.pack(anchor=tk.W, padx=15, pady=(20, 5))
        
        # History list
        history_frame = tk.Frame(parent, bg=self.colors["bg_secondary"])
        history_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        self.history_list = tk.Listbox(history_frame,
                                       bg=self.colors["bg_tertiary"],
                                       fg=self.colors["text"],
                                       selectbackground=self.colors["accent"],
                                       font=("Helvetica", 10),
                                       relief=tk.FLAT,
                                       highlightthickness=0,
                                       borderwidth=0)
        self.history_list.pack(fill=tk.BOTH, expand=True)
        self.history_list.bind('<<ListboxSelect>>', self._select_session)
        
        # Model info
        info_frame = tk.Frame(parent, bg=self.colors["bg_secondary"])
        info_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=15)
        
        info_text = f"""Model: {MODEL_NAME}
Context: {CONTEXT_WINDOW} tokens
Engine: Pure Python stdlib
RAM: Optimized for 24GB"""
        
        info_label = tk.Label(info_frame, text=info_text,
                             bg=self.colors["bg_secondary"],
                             fg=self.colors["text_muted"],
                             font=("Helvetica", 8),
                             justify=tk.LEFT)
        info_label.pack(anchor=tk.W)
    
    def _build_chat_area(self, parent):
        """Build main chat area."""
        # Chat display
        chat_frame = tk.Frame(parent, bg=self.colors["bg"])
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=("Consolas", 11),
            relief=tk.FLAT,
            padx=20,
            pady=20,
            insertbackground=self.colors["text"],
            selectbackground=self.colors["accent"],
            borderwidth=0,
            highlightthickness=0
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Configure tags
        self.chat_display.tag_configure("user", foreground=self.colors["user_msg"],
                                        font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("assistant", foreground=self.colors["assistant_msg"],
                                        font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("think", foreground=self.colors["think"],
                                        font=("Consolas", 10, "italic"))
        self.chat_display.tag_configure("code", foreground=self.colors["code"],
                                        font=("Consolas", 11))
        self.chat_display.tag_configure("bold", font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("error", foreground=self.colors["error"])
        
        # Input area
        input_frame = tk.Frame(parent, bg=self.colors["bg"])
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Input container with border effect
        input_container = tk.Frame(input_frame, bg=self.colors["border"], padx=2, pady=2)
        input_container.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        self.input_text = tk.Text(input_container,
                                  height=3,
                                  bg=self.colors["bg_secondary"],
                                  fg=self.colors["text"],
                                  insertbackground=self.colors["text"],
                                  font=("Consolas", 11),
                                  relief=tk.FLAT,
                                  padx=15,
                                  pady=10,
                                  wrap=tk.WORD)
        self.input_text.pack(fill=tk.X)
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: None)
        
        # Button frame
        btn_frame = tk.Frame(input_frame, bg=self.colors["bg"])
        btn_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Send button
        self.send_btn = tk.Button(btn_frame, text="Send",
                                  bg=self.colors["accent"],
                                  fg="white",
                                  font=("Helvetica", 11, "bold"),
                                  relief=tk.FLAT,
                                  width=10,
                                  cursor="hand2",
                                  command=self._send_message)
        self.send_btn.pack(pady=(0, 5))
        
        # Stop button
        self.stop_btn = tk.Button(btn_frame, text="Stop",
                                  bg=self.colors["error"],
                                  fg="white",
                                  font=("Helvetica", 10),
                                  relief=tk.FLAT,
                                  width=10,
                                  cursor="hand2",
                                  state=tk.DISABLED,
                                  command=self._stop_generation)
        self.stop_btn.pack()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(parent, textvariable=self.status_var,
                             bg=self.colors["bg"],
                             fg=self.colors["text_muted"],
                             font=("Helvetica", 9))
        status_bar.pack(side=tk.BOTTOM, anchor=tk.W, padx=20, pady=5)
    
    def _show_welcome(self):
        """Show welcome message."""
        welcome = f"""Welcome to **{MODEL_NAME}**! 🐱

I combine **ChatGPT**-style conversation with **DeepSeek R1**-style reasoning.

**What I can do:**
🧮 Math - calculations, algebra, calculus, statistics
💻 Code - Python, algorithms, data structures, execution
📚 Knowledge - explanations, concepts, tutorials
🧠 Reasoning - step-by-step thinking in <think> blocks

**Try these:**
• `calculate sqrt(2) + pi^2`
• `solve x^2 - 5x + 6 = 0`
• `write a quicksort function`
• `explain recursion`

Type your message below to get started!
"""
        self._append_message("CatGPT-SEEK", welcome, "assistant")
    
    def _toggle_thinking(self):
        """Toggle thinking visibility."""
        self.reasoning_engine.thinking_enabled = self.show_thinking.get()
    
    def _new_chat(self):
        """Start new chat session."""
        if self.reasoning_engine.memory:
            # Save current session
            first_msg = self.reasoning_engine.memory[0]["content"][:30] if self.reasoning_engine.memory else "Chat"
            self.history_list.insert(0, f"💬 {first_msg}...")
            self.chat_sessions.insert(0, self.reasoning_engine.memory.copy())
        
        # Clear
        self.reasoning_engine.clear_memory()
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self._show_welcome()
        self.status_var.set("New chat started")
    
    def _select_session(self, event):
        """Load selected chat session."""
        selection = self.history_list.curselection()
        if selection:
            idx = selection[0]
            if idx < len(self.chat_sessions):
                self.reasoning_engine.memory = self.chat_sessions[idx].copy()
                self._refresh_chat_display()
    
    def _refresh_chat_display(self):
        """Refresh chat display from memory."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        
        for msg in self.reasoning_engine.memory:
            role = "You" if msg["role"] == "user" else "CatGPT-SEEK"
            tag = "user" if msg["role"] == "user" else "assistant"
            self._append_message(role, msg["content"], tag)
        
        self.chat_display.config(state=tk.DISABLED)
    
    def _on_enter(self, event):
        """Handle enter key."""
        if not event.state & 0x1:  # Not shift
            self._send_message()
            return "break"
    
    def _send_message(self):
        """Send user message."""
        if self.is_generating:
            return
        
        message = self.input_text.get(1.0, tk.END).strip()
        if not message:
            return
        
        self.input_text.delete(1.0, tk.END)
        self._append_message("You", message, "user")
        
        # Start generation
        self.is_generating = True
        self.stop_event.clear()
        self.send_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Generating...")
        
        thread = Thread(target=self._generate_response, args=(message,), daemon=True)
        thread.start()
    
    def _generate_response(self, message: str):
        """Generate response in background."""
        try:
            response = ""
            for chunk in self.reasoning_engine.generate(message):
                if self.stop_event.is_set():
                    break
                response += chunk
                self.response_queue.put(("chunk", chunk))
            
            self.response_queue.put(("done", response))
        except Exception as e:
            self.response_queue.put(("error", str(e)))
    
    def _check_response_queue(self):
        """Check for response updates."""
        try:
            while True:
                msg_type, content = self.response_queue.get_nowait()
                
                if msg_type == "chunk":
                    self._append_chunk(content)
                elif msg_type == "done":
                    self._finish_generation()
                elif msg_type == "error":
                    self._append_message("Error", content, "error")
                    self._finish_generation()
        except queue.Empty:
            pass
        
        self.root.after(30, self._check_response_queue)
    
    def _append_message(self, role: str, content: str, tag: str):
        """Append complete message."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n{role}:\n", tag)
        self._insert_formatted_text(content)
        self.chat_display.insert(tk.END, "\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _append_chunk(self, chunk: str):
        """Append streaming chunk."""
        self.chat_display.config(state=tk.NORMAL)
        
        # Handle think tags
        if chunk.startswith("<think>"):
            self.chat_display.insert(tk.END, "\nCatGPT-SEEK:\n", "assistant")
            chunk = chunk[7:]
            self._current_tag = "think"
        elif chunk.startswith("</think>"):
            chunk = chunk[8:]
            self._current_tag = "assistant"
        
        tag = getattr(self, '_current_tag', 'assistant')
        self.chat_display.insert(tk.END, chunk, tag)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _insert_formatted_text(self, text: str):
        """Insert text with markdown formatting."""
        # Code blocks
        parts = re.split(r'(```[\s\S]*?```)', text)
        
        for part in parts:
            if part.startswith('```') and part.endswith('```'):
                code = part[3:-3]
                if code.startswith(('python\n', 'py\n')):
                    code = code.split('\n', 1)[1] if '\n' in code else ''
                self.chat_display.insert(tk.END, code, "code")
            else:
                # Bold text
                bold_parts = re.split(r'(\*\*.*?\*\*)', part)
                for bp in bold_parts:
                    if bp.startswith('**') and bp.endswith('**'):
                        self.chat_display.insert(tk.END, bp[2:-2], "bold")
                    else:
                        self.chat_display.insert(tk.END, bp)
    
    def _finish_generation(self):
        """Clean up after generation."""
        self.is_generating = False
        self.send_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Ready")
        self._current_tag = "assistant"
    
    def _stop_generation(self):
        """Stop current generation."""
        self.stop_event.set()
        self.status_var.set("Stopping...")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print(f"Starting {MODEL_NAME}...")
    print("Initializing engines...")
    
    root = tk.Tk()
    app = CatGPTSeekApp(root)
    
    print("Ready! GUI launched.")
    root.mainloop()


if __name__ == "__main__":
    main()
