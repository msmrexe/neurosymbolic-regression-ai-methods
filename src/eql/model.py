"""
Contains the symbolic regression neural network architecture (EQL).
Defines the `SymbolicLayer` and `SymbolicNet` modules.
"""

import torch
import torch.nn as nn
from typing import List, Union
from .functions import BaseFunction, BaseFunction2, count_double, count_inputs

class SymbolicLayer(nn.Module):
    """
    Neural network layer where activations are primitive functions.
    This layer supports multi-input functions (e.g., multiplication).
    """
    def __init__(
        self, 
        in_dim: int, 
        funcs: List[Union[BaseFunction, BaseFunction2]], 
        initial_weight: torch.Tensor = None, 
        init_stddev: float = 0.1
    ):
        """
        Args:
            in_dim: Number of input features to this layer.
            funcs: List of activation function objects (e.g., [Identity(), Sin()]).
            initial_weight: Optional tensor to set as initial weights.
            init_stddev: Standard deviation for random weight initialization.
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.funcs = [func.torch for func in funcs]
        self.n_funcs = len(funcs)
        
        self.n_double = count_double(funcs)
        self.n_single = self.n_funcs - self.n_double
        
        # Total number of inputs this layer's weights must connect to
        self.weight_in_dim = self.n_single * 1 + self.n_double * 2
        
        # Output dimension of this layer
        self.out_dim = self.n_funcs
        
        if initial_weight is not None:
            self.W = nn.Parameter(initial_weight.clone())
        else:
            # Initialize weights
            W_tensor = torch.empty(self.in_dim, self.weight_in_dim)
            nn.init.normal_(W_tensor, mean=0.0, std=init_stddev)
            self.W = nn.Parameter(W_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, in_dim).

        Returns:
            Output tensor of shape (batch_size, out_dim).
        """
        # 1. Linear transformation: (batch, in_dim) @ (in_dim, weight_in_dim)
        g = torch.matmul(x, self.W) # Shape: (batch, weight_in_dim)

        # 2. Apply activation functions
        outputs = []
        input_idx = 0
        
        # Apply single-input functions
        if self.n_single > 0:
            g_single = g[:, :self.n_single] # Shape: (batch, n_single)
            for i in range(self.n_single):
                outputs.append(self.funcs[i](g_single[:, i]))
            input_idx = self.n_single
        
        # Apply double-input functions
        if self.n_double > 0:
            for i in range(self.n_double):
                # Get the two inputs for this function
                g_double_1 = g[:, input_idx]
                g_double_2 = g[:, input_idx + 1]
                
                # Apply the function
                func_idx = self.n_single + i
                outputs.append(self.funcs[func_idx](g_double_1, g_double_2))
                
                input_idx += 2

        # Stack all outputs: (batch, out_dim)
        return torch.stack(outputs, dim=1)

    def get_weight(self) -> np.ndarray:
        return self.W.cpu().detach().numpy()

    def get_weight_tensor(self) -> torch.Tensor:
        return self.W.clone()


class SymbolicNet(nn.Module):
    """
    A multi-layer symbolic regression network.
    """
    def __init__(
        self,
        n_layers: int,
        in_dim: int,
        funcs: List[Union[BaseFunction, BaseFunction2]],
        initial_weights: List[torch.Tensor] = None,
        init_stddev: float = 0.1
    ):
        """
        Args:
            n_layers: Number of hidden symbolic layers.
            in_dim: Input dimension of the data (e.g., 1 for y=f(x)).
            funcs: List of primitive functions to use in each hidden layer.
            initial_weights: Optional list of tensors for all layer weights.
            init_stddev: Std deviation for weight initialization.
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_layers = nn.ModuleList()

        current_dim = in_dim
        
        for i in range(n_layers):
            # Use initial weight if provided
            init_w = initial_weights[i] if initial_weights else None
            
            layer = SymbolicLayer(
                in_dim=current_dim,
                funcs=funcs,
                initial_weight=init_w,
                init_stddev=init_stddev
            )
            self.hidden_layers.append(layer)
            current_dim = layer.out_dim

        # Final linear output layer
        # It connects the output of the last hidden layer to a single value
        if initial_weights:
            init_out_w = initial_weights[-1].clone()
        else:
            init_out_w = torch.empty(current_dim, 1)
            nn.init.normal_(init_out_w, mean=0.0, std=init_stddev)
            
        self.output_weight = nn.Parameter(init_out_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire network.

        Args:
            x: Input tensor, shape (batch_size, in_dim).

        Returns:
            Output tensor, shape (batch_size, 1).
        """
        # Pass through all hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Apply final linear layer
        # (batch, last_layer_out_dim) @ (last_layer_out_dim, 1)
        output = torch.matmul(x, self.output_weight)
        return output

    def get_weights(self) -> List[np.ndarray]:
        """Returns a list of all weight matrices as NumPy arrays."""
        weights = [layer.get_weight() for layer in self.hidden_layers]
        weights.append(self.output_weight.cpu().detach().numpy())
        return weights

    def get_weights_tensor(self) -> List[torch.Tensor]:
        """Returns a list of all weight parameters as Tensors."""
        weights = [layer.get_weight_tensor() for layer in self.hidden_layers]
        weights.append(self.output_weight.clone())
        return weights
