"""
Defines the primitive functions used as activation functions in the EQL model.
Each function includes both a SymPy (for expression retrieval) and a
PyTorch (for training) implementation.
"""

import torch
import sympy as sp
import numpy as np
from typing import List

class BaseFunction:
    """Abstract class for primitive functions with 1 input."""
    def __init__(self, norm: float = 1.0):
        self.norm = norm

    def sp(self, x):
        """SymPy implementation."""
        raise NotImplementedError

    def torch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation."""
        raise NotImplementedError

    def name(self, x):
        return str(self.sp)

class BaseFunction2:
    """Abstract class for primitive functions with 2 inputs."""
    def __init__(self, norm: float = 1.0):
        self.norm = norm

    def sp(self, x, y):
        """SymPy implementation."""
        raise NotImplementedError

    def torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation."""
        raise NotImplementedError

    def name(self, x, y):
        return str(self.sp)

# --- 1-Input Functions ---

class Constant(BaseFunction):
    def torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)
    def sp(self, x):
        return 1

class Identity(BaseFunction):
    def torch(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.norm
    def sp(self, x):
        return x / self.norm

class Square(BaseFunction):
    def torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, 2) / self.norm
    def sp(self, x):
        return x**2 / self.norm

class Pow(BaseFunction):
    def __init__(self, power: float, norm: float = 1.0):
        super().__init__(norm)
        self.power = power
    def torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, self.power) / self.norm
    def sp(self, x):
        return x**self.power / self.norm

class Sin(BaseFunction):
    def __init__(self, freq: float = 1.0, norm: float = 1.0):
        super().__init__(norm)
        # Defaulting to 1.0 for a standard sin(x)
        self.freq = freq
        
    def torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x * self.freq) / self.norm
    def sp(self, x):
        return sp.sin(x * self.freq) / self.norm

class Sigmoid(BaseFunction):
    def __init__(self, scale: float = 20.0, norm: float = 1.0):
        super().__init__(norm)
        self.scale = scale
    def torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x * self.scale) / self.norm
    def sp(self, x):
        return 1 / (1 + sp.exp(-x * self.scale)) / self.norm
    def name(self, x):
        return "sigmoid(x)"

class Exp(BaseFunction):
    def __init__(self, norm: float = np.e):
        super().__init__(norm)
    def torch(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid overflow
        x_clamped = torch.clamp(x, -10, 10)
        return (torch.exp(x_clamped) - 1) / self.norm
    def sp(self, x):
        return (sp.exp(x) - 1) / self.norm

class Log(BaseFunction):
    def torch(self, x: torch.Tensor) -> torch.Tensor:
        # Add 1e-8 for numerical stability
        return torch.log(torch.abs(x) + 1e-8) / self.norm
    def sp(self, x):
        return sp.log(sp.Abs(x)) / self.norm

# --- 2-Input Functions ---

class Product(BaseFunction2):
    def __init__(self, norm: float = 0.1):
        super().__init__(norm)
    def torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x * y) / self.norm
    def sp(self, x, y):
        return x * y / self.norm


# --- Helper Functions ---

def count_inputs(funcs: List[Union[BaseFunction, BaseFunction2]]) -> int:
    """Counts the total number of inputs required for a list of functions."""
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction):
            i += 1
        elif isinstance(func, BaseFunction2):
            i += 2
    return i

def count_double(funcs: List[Union[BaseFunction, BaseFunction2]]) -> int:
    """Counts the number of 2-input functions in a list."""
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction2):
            i += 1
    return i
