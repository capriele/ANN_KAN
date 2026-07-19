import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import re
# -----------------------------------------------------------------------------
# Symbolic library
# -----------------------------------------------------------------------------
# The entries follow the pykan-style convention:
# name -> (torch_function, sympy_function, complexity, fit_function)
# In this standalone implementation the edge-wise symbolic identification uses
# the torch and sympy functions directly and fits an affine output correction.

# singularity protection functions
f_inv = lambda x, y_th: ((x_th := 1/y_th), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/x) * (torch.abs(x) >= x_th))
f_inv2 = lambda x, y_th: ((x_th := 1/y_th**(1/2)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**2) * (torch.abs(x) >= x_th))
f_inv3 = lambda x, y_th: ((x_th := 1/y_th**(1/3)), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**3) * (torch.abs(x) >= x_th))
f_inv4 = lambda x, y_th: ((x_th := 1/y_th**(1/4)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**4) * (torch.abs(x) >= x_th))
f_inv5 = lambda x, y_th: ((x_th := 1/y_th**(1/5)), y_th/x_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(1/x**5) * (torch.abs(x) >= x_th))
f_sqrt = lambda x, y_th: ((x_th := 1/y_th**2), x_th/y_th*x * (torch.abs(x) < x_th) + torch.nan_to_num(torch.sqrt(torch.abs(x))*torch.sign(x)) * (torch.abs(x) >= x_th))
f_power1d5 = lambda x, y_th: torch.abs(x)**1.5
f_invsqrt = lambda x, y_th: ((x_th := 1/y_th**2), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1/torch.sqrt(torch.abs(x))) * (torch.abs(x) >= x_th))
f_log = lambda x, y_th: ((x_th := torch.e**(-y_th)), - y_th * (torch.abs(x) < x_th) + torch.nan_to_num(torch.log(torch.abs(x))) * (torch.abs(x) >= x_th))
f_tan = lambda x, y_th: ((clip := x % torch.pi), (delta := torch.pi/2-torch.arctan(y_th)), - y_th/delta * (clip - torch.pi/2) * (torch.abs(clip - torch.pi/2) < delta) + torch.nan_to_num(torch.tan(clip)) * (torch.abs(clip - torch.pi/2) >= delta))
f_arctanh = lambda x, y_th: ((delta := 1-torch.tanh(y_th) + 1e-4), y_th * torch.sign(x) * (torch.abs(x) > 1 - delta) + torch.nan_to_num(torch.arctanh(x)) * (torch.abs(x) <= 1 - delta))
f_arcsin = lambda x, y_th: ((), torch.pi/2 * torch.sign(x) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1))
f_arccos = lambda x, y_th: ((), torch.pi/2 * (1-torch.sign(x)) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1))
f_exp = lambda x, y_th: ((x_th := torch.log(y_th)), y_th * (x > x_th) + torch.exp(x) * (x <= x_th))

def no_fit(torch_fn):
    """pykan-style fallback symbolic fitting function."""
    return lambda x, y_th: ((), torch_fn(x))


SYMBOLIC_LIB = {
    # 'x': (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
    # 'x^2': (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
    # 'x^3': (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
    # 'x^4': (lambda x: x**4, lambda x: x**4, 3, lambda x, y_th: ((), x**4)),
    # 'x^5': (lambda x: x**5, lambda x: x**5, 3, lambda x, y_th: ((), x**5)),
    # '1/x': (lambda x: 1/x, lambda x: 1/x, 2, f_inv),
    # '1/x^2': (lambda x: 1/x**2, lambda x: 1/x**2, 2, f_inv2),
    # '1/x^3': (lambda x: 1/x**3, lambda x: 1/x**3, 3, f_inv3),
    # '1/x^4': (lambda x: 1/x**4, lambda x: 1/x**4, 4, f_inv4),
    # '1/x^5': (lambda x: 1/x**5, lambda x: 1/x**5, 5, f_inv5),
    # 'sqrt': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    # 'x^0.5': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    # 'x^1.5': (lambda x: torch.sqrt(x)**3, lambda x: sympy.sqrt(x)**3, 4, f_power1d5),
    # '1/sqrt(x)': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x), 2, f_invsqrt),
    # '1/x^0.5': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x), 2, f_invsqrt),
    # 'exp': (lambda x: torch.exp(x), lambda x: sympy.exp(x), 2, f_exp),
    # 'log': (lambda x: torch.log(x), lambda x: sympy.log(x), 2, f_log),
    # 'abs': (lambda x: torch.abs(x), lambda x: sympy.Abs(x), 3, lambda x, y_th: ((), torch.abs(x))),
    # 'sgn': (lambda x: torch.sign(x), lambda x: sympy.sign(x), 3, lambda x, y_th: ((), torch.sign(x))),
    # 'arcsin': (lambda x: torch.arcsin(x), lambda x: sympy.asin(x), 4, f_arcsin),
    # 'arccos': (lambda x: torch.arccos(x), lambda x: sympy.acos(x), 4, f_arccos),
    # 'arctanh': (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x), 4, f_arctanh),
    # Linear / constant
    "0": (
        lambda x: x * 0,
        lambda x: x * 0,
        0,
        lambda x, y_th: ((), x * 0),
    ),
    "x": (
        lambda x: x,
        lambda x: x,
        1,
        lambda x, y_th: ((), x),
    ),

    # Periodic bounded smooth functions
    "sin": (
        lambda x: torch.sin(x),
        lambda x: sympy.sin(x),
        2,
        no_fit(lambda x: torch.sin(x)),
    ),
    "cos": (
        lambda x: torch.cos(x),
        lambda x: sympy.cos(x),
        2,
        no_fit(lambda x: torch.cos(x)),
    ),
    "sin2x": (
        lambda x: torch.sin(2 * x),
        lambda x: sympy.sin(2 * x),
        3,
        no_fit(lambda x: torch.sin(2 * x)),
    ),
    "cos2x": (
        lambda x: torch.cos(2 * x),
        lambda x: sympy.cos(2 * x),
        3,
        no_fit(lambda x: torch.cos(2 * x)),
    ),

    # Saturating smooth nonlinearities
    "tanh": (
        lambda x: torch.tanh(x),
        lambda x: sympy.tanh(x),
        3,
        no_fit(lambda x: torch.tanh(x)),
    ),
    "sigmoid": (
        lambda x: torch.sigmoid(x),
        lambda x: 1 / (1 + sympy.exp(-x)),
        3,
        no_fit(lambda x: torch.sigmoid(x)),
    ),
    "softsign_smooth": (
        lambda x: x / torch.sqrt(1 + x**2),
        lambda x: x / sympy.sqrt(1 + x**2),
        3,
        no_fit(lambda x: x / torch.sqrt(1 + x**2)),
    ),
    "arctan": (
        lambda x: torch.arctan(x),
        lambda x: sympy.atan(x),
        3,
        no_fit(lambda x: torch.arctan(x)),
    ),

    # Smooth versions of abs-like behavior
    "logcosh": (
        lambda x: x + F.softplus(-2 * x) - math.log(2.0),
        lambda x: sympy.log(sympy.cosh(x)),
        4,
        no_fit(lambda x: x + F.softplus(-2 * x) - math.log(2.0)),
    ),
    "soft_abs": (
        lambda x: x * torch.tanh(x),
        lambda x: x * sympy.tanh(x),
        4,
        no_fit(lambda x: x * torch.tanh(x)),
    ),

    # Globally smooth bounded radial/rational functions
    "gaussian": (
        lambda x: torch.exp(-x**2),
        lambda x: sympy.exp(-x**2),
        3,
        no_fit(lambda x: torch.exp(-x**2)),
    ),
    "lorentzian": (
        lambda x: 1 / (1 + x**2),
        lambda x: 1 / (1 + x**2),
        3,
        no_fit(lambda x: 1 / (1 + x**2)),
    ),
    "rational_odd": (
        lambda x: x / (1 + x**2),
        lambda x: x / (1 + x**2),
        3,
        no_fit(lambda x: x / (1 + x**2)),
    ),
    "rational_even": (
        lambda x: x**2 / (1 + x**2),
        lambda x: x**2 / (1 + x**2),
        3,
        no_fit(lambda x: x**2 / (1 + x**2)),
    ),

    # Smooth, globally Lipschitz monotone growth functions
    "softplus": (
        lambda x: F.softplus(x),
        lambda x: sympy.log(1 + sympy.exp(x)),
        3,
        no_fit(lambda x: F.softplus(x)),
    ),

    # Error-function family
    "erf": (
        lambda x: torch.erf(x),
        lambda x: sympy.erf(x),
        4,
        no_fit(lambda x: torch.erf(x)),
    ),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _expand_orders(orders: Optional[List[int]], n_layers: int, default: int = 3) -> List[int]:
    if n_layers <= 0:
        return []
    if not orders:
        return [default] * n_layers
    if len(orders) == 1:
        return [int(orders[0])] * n_layers
    if len(orders) < n_layers:
        return [int(v) for v in orders] + [int(orders[-1])] * (n_layers - len(orders))
    return [int(v) for v in orders[:n_layers]]


def _safe_torch_eval(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> Optional[torch.Tensor]:
    try:
        y = fn(x)
        if torch.isfinite(y).all():
            return y
    except Exception:
        return None
    return None


def _fit_output_affine(feature: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
    """Fit target ~= c * feature + d and return c, d, mse."""
    feature = feature.reshape(-1, 1)
    target = target.reshape(-1, 1)
    ones = torch.ones_like(feature)
    design = torch.cat([feature, ones], dim=1)
    try:
        solution = torch.linalg.lstsq(design, target).solution.reshape(-1)
        c = float(solution[0].detach().cpu())
        d = float(solution[1].detach().cpu())
    except Exception:
        c = 1.0
        d = 0.0
    pred = c * feature.reshape(-1) + d
    mse = float(torch.mean((pred - target.reshape(-1)) ** 2).detach().cpu())
    return c, d, mse


def _float_to_sympy(value: float, digits: int = 6):
    value = float(value)
    if abs(value) < 10 ** (-digits):
        return sympy.Integer(0)
    return sympy.Float(value, digits)


# -----------------------------------------------------------------------------
# Native classic spline KAN, replacing pykan.KAN
# -----------------------------------------------------------------------------


class SplineKANLayer(nn.Module):
    """Dense KAN layer with trainable B-spline edge functions."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        noise_scale: float = 0.1,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        base_fun: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
    ):
        super().__init__()
        if grid_size < 1:
            raise ValueError("grid_size must be >= 1.")
        if spline_order < 0:
            raise ValueError("spline_order must be >= 0.")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("grid_range must be an increasing tuple.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.n_basis = self.grid_size + self.spline_order
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.base_fun = base_fun

        knots = self._make_open_uniform_knots(
            self.n_basis,
            self.spline_order,
            self.grid_range[0],
            self.grid_range[1],
        )
        self.register_buffer("knots", knots)
        self.register_buffer("edge_mask", torch.ones(self.in_features, self.out_features))

        self.base_weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.spline_weight = nn.Parameter(
            torch.empty(self.in_features, self.out_features, self.n_basis)
        )
        self.spline_scale = nn.Parameter(torch.ones(self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self.symbolic_edges: Optional[List[List[dict]]] = None
        self.reset_parameters(noise_scale=noise_scale)

    @staticmethod
    def _make_open_uniform_knots(n_basis: int, degree: int, left: float, right: float) -> torch.Tensor:
        interior = torch.linspace(left, right, n_basis - degree + 1)
        if degree == 0:
            return interior
        left_pad = torch.full((degree,), left)
        right_pad = torch.full((degree,), right)
        return torch.cat([left_pad, interior, right_pad])

    def reset_parameters(self, noise_scale: float = 0.1) -> None:
        bound = 1.0 / math.sqrt(max(1, self.in_features))
        nn.init.uniform_(self.base_weight, -bound, bound)
        nn.init.uniform_(self.spline_weight, -noise_scale * bound, noise_scale * bound)
        nn.init.ones_(self.spline_scale)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        knots = self.knots.to(device=x.device, dtype=x.dtype)
        x = x.clamp(self.grid_range[0], self.grid_range[1])
        x_exp = x.unsqueeze(-1)

        left = knots[:-1]
        right = knots[1:]
        basis = ((x_exp >= left) & (x_exp < right)).to(x.dtype)

        right_endpoint = torch.isclose(x, torch.full_like(x, self.grid_range[1]))
        if basis.shape[-1] > 0:
            basis = torch.cat(
                [basis[..., :-1], torch.where(right_endpoint, torch.ones_like(x), basis[..., -1]).unsqueeze(-1)],
                dim=-1,
            )

        eps = torch.as_tensor(1e-12, device=x.device, dtype=x.dtype)
        for degree in range(1, self.spline_order + 1):
            n_terms = basis.shape[-1] - 1
            left_den = knots[degree : degree + n_terms] - knots[:n_terms]
            right_den = (
                knots[degree + 1 : degree + n_terms + 1]
                - knots[1 : n_terms + 1]
            )
            left_valid = left_den.abs() > eps
            right_valid = right_den.abs() > eps
            safe_left_den = torch.where(left_valid, left_den, torch.ones_like(left_den))
            safe_right_den = torch.where(right_valid, right_den, torch.ones_like(right_den))

            left_term = (
                (x_exp - knots[:n_terms]) / safe_left_den * basis[..., :n_terms]
            ) * left_valid
            right_term = (
                (knots[degree + 1 : degree + n_terms + 1] - x_exp)
                / safe_right_den
                * basis[..., 1 : n_terms + 1]
            ) * right_valid
            basis = left_term + right_term
        return basis[..., : self.n_basis]

    def edge_values(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._bspline_basis(x)
        spline_part = torch.einsum("bin,ion->bio", basis, self.spline_weight)
        base_part = self.base_fun(x).unsqueeze(-1) * self.base_weight.unsqueeze(0)
        edge_values = base_part + self.spline_scale.unsqueeze(0) * spline_part
        return edge_values * self.edge_mask.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.edge_values(x).sum(dim=1)
        if self.bias is not None:
            y = y + self.bias
        return y

    def prune(self, threshold: float = 1e-2) -> None:
        with torch.no_grad():
            spline_norm = torch.linalg.vector_norm(self.spline_weight, dim=-1)
            score = self.base_weight.abs() + self.spline_scale.abs() * spline_norm
            self.edge_mask.copy_((score >= threshold).to(self.edge_mask.dtype))

    def identify_symbolic(
        self,
        symbolic_lib: Optional[Dict] = None,
        n_points: int = 201,
        input_affine_grid: Optional[Tuple[List[float], List[float]]] = None,
        complexity_weight: float = 1e-6,
    ) -> List[List[dict]]:
        symbolic_lib = symbolic_lib or SYMBOLIC_LIB
        if input_affine_grid is None:
            input_affine_grid = ([0.5, 1.0, 2.0], [-1.0, 0.0, 1.0])

        device = self.spline_weight.device
        dtype = self.spline_weight.dtype
        x = torch.linspace(
            self.grid_range[0], self.grid_range[1], n_points, device=device, dtype=dtype
        ).reshape(-1, 1)
        x_all = x.repeat(1, self.in_features)
        y_all = self.edge_values(x_all).detach()

        fitted: List[List[dict]] = []
        for input_idx in range(self.in_features):
            row = []
            x_i = x.reshape(-1)
            for output_idx in range(self.out_features):
                target = y_all[:, input_idx, output_idx].reshape(-1)
                best = {
                    "name": "0",
                    "a": 1.0,
                    "b": 0.0,
                    "c": 0.0,
                    "d": float(target.mean().detach().cpu()),
                    "mse": float(torch.mean((target - target.mean()) ** 2).detach().cpu()),
                    "score": float("inf"),
                }
                for name, entry in symbolic_lib.items():
                    torch_fn = entry[0]
                    complexity = float(entry[2]) if len(entry) > 2 else 1.0
                    for a in input_affine_grid[0]:
                        for b in input_affine_grid[1]:
                            z = float(a) * x_i + float(b)
                            feature = _safe_torch_eval(torch_fn, z)
                            if feature is None:
                                continue
                            c, d, mse = _fit_output_affine(feature, target)
                            score = mse + complexity_weight * complexity
                            if score < best["score"]:
                                best = {
                                    "name": name,
                                    "a": float(a),
                                    "b": float(b),
                                    "c": c,
                                    "d": d,
                                    "mse": mse,
                                    "score": score,
                                }
                row.append(best)
            fitted.append(row)
        self.symbolic_edges = fitted
        return fitted

    def symbolic_expressions(
        self,
        input_symbols: List[sympy.Expr],
        symbolic_lib: Optional[Dict] = None,
        auto_fit: bool = True,
        simplify: bool = False,
    ) -> List[sympy.Expr]:
        symbolic_lib = symbolic_lib or SYMBOLIC_LIB
        if self.symbolic_edges is None:
            if not auto_fit:
                raise RuntimeError("symbolic_edges are missing. Run identify_symbolic() first.")
            self.identify_symbolic(symbolic_lib=symbolic_lib)

        outputs: List[sympy.Expr] = []
        for output_idx in range(self.out_features):
            expr = sympy.Integer(0)
            for input_idx, input_symbol in enumerate(input_symbols):
                edge = self.symbolic_edges[input_idx][output_idx]
                name = edge["name"]
                sympy_fn = symbolic_lib[name][1]
                a = _float_to_sympy(edge["a"])
                b = _float_to_sympy(edge["b"])
                c = _float_to_sympy(edge["c"])
                d = _float_to_sympy(edge["d"])
                term = c * sympy_fn(a * input_symbol + b) + d
                expr += term
            if self.bias is not None:
                expr += _float_to_sympy(float(self.bias[output_idx].detach().cpu()))
            outputs.append(sympy.simplify(expr) if simplify else expr)
        return outputs


class SplineKAN(nn.Module):
    def __init__(
        self,
        width: List[int],
        grid: int = 5,
        k: int = 3,
        noise_scale: float = 0.1,
        seed: int = 0,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        symbolic_lib: Optional[Dict] = None,
        **_: object,
    ):
        super().__init__()
        if len(width) < 2:
            raise ValueError("width must contain at least input and output dimensions.")
        self.width = [int(v) for v in width]
        self.grid = int(grid)
        self.k = int(k)
        self.grid_range = grid_range
        self.symbolic_lib = symbolic_lib or SYMBOLIC_LIB
        #if seed is not None:
        #    torch.manual_seed(int(seed))
        self.layers = nn.ModuleList(
            [
                SplineKANLayer(
                    in_dim,
                    out_dim,
                    grid_size=self.grid,
                    spline_order=self.k,
                    noise_scale=noise_scale,
                    grid_range=grid_range,
                )
                for in_dim, out_dim in zip(self.width[:-1], self.width[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def prune(self, threshold: float = 1e-2) -> None:
        for layer in self.layers:
            layer.prune(threshold=threshold)

    def plot(self, *args, **kwargs) -> None:
        return None

    def speed(self, *args, **kwargs) -> None:
        return None

    def identify_symbolic(
        self,
        symbolic_lib: Optional[Dict] = None,
        n_points: int = 201,
        input_affine_grid: Optional[Tuple[List[float], List[float]]] = None,
        complexity_weight: float = 1e-6,
    ) -> List[List[List[dict]]]:
        symbolic_lib = symbolic_lib or self.symbolic_lib
        return [
            layer.identify_symbolic(
                symbolic_lib=symbolic_lib,
                n_points=n_points,
                input_affine_grid=input_affine_grid,
                complexity_weight=complexity_weight,
            )
            for layer in self.layers
        ]

    def symbolic_formula(
        self,
        var: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        symbolic_lib: Optional[Dict] = None,
        auto_fit: bool = True,
        simplify: bool = False,
    ) -> Union[str, List[str]]:
        symbolic_lib = symbolic_lib or self.symbolic_lib
        input_dim = self.width[0]
        if var is None:
            symbols = list(sympy.symbols(f"x0:{input_dim}"))
        elif isinstance(var, str):
            if input_dim == 1:
                symbols = [sympy.Symbol(var)]
            else:
                symbols = list(sympy.symbols(f"{var}0:{input_dim}"))
        else:
            if len(var) != input_dim:
                raise ValueError(f"Expected {input_dim} variable names, got {len(var)}.")
            symbols = [sympy.Symbol(v) for v in var]

        expressions: List[sympy.Expr] = symbols
        for layer in self.layers:
            expressions = layer.symbolic_expressions(
                expressions,
                symbolic_lib=symbolic_lib,
                auto_fit=auto_fit,
                simplify=simplify,
            )
        stringified = [str(sympy.simplify(expr) if simplify else expr) for expr in expressions]
        return stringified[0] if len(stringified) == 1 else stringified


# -----------------------------------------------------------------------------
# Native Chebyshev and Fractional KAN variants
# -----------------------------------------------------------------------------


class ChebyshevKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, order: int = 3, bias: bool = True):
        super().__init__()
        if order < 0:
            raise ValueError("Chebyshev order must be non-negative.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.order = int(order)
        self.coefficients = nn.Parameter(torch.empty(self.in_features, self.out_features, self.order + 1))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(1, self.in_features * (self.order + 1)))
        nn.init.uniform_(self.coefficients, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        basis = [torch.ones_like(x)]
        if self.order >= 1:
            basis.append(x)
        for _ in range(2, self.order + 1):
            basis.append(2.0 * x * basis[-1] - basis[-2])
        return torch.stack(basis, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._chebyshev_basis(x)
        y = torch.einsum("bio,ijo->bj", basis, self.coefficients)
        if self.bias is not None:
            y = y + self.bias
        return y


class FractionalKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, order: int = 3, bias: bool = True):
        super().__init__()
        if order < 0:
            raise ValueError("Fractional/Jacobi order must be non-negative.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.order = int(order)
        self.coefficients = nn.Parameter(torch.empty(self.in_features, self.out_features, self.order + 1))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(1, self.in_features * (self.order + 1)))
        nn.init.uniform_(self.coefficients, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _jacobi_basis(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.elu(self.alpha, alpha=1.0) + 1.0
        beta = F.elu(self.beta, alpha=1.0) + 1.0
        gamma = torch.sigmoid(self.gamma).clamp_min(1e-4)
        x01 = torch.sigmoid(x).clamp(1e-6, 1.0 - 1e-6)
        z = 2.0 * torch.pow(x01, gamma) - 1.0
        basis = [torch.ones_like(z)]
        if self.order >= 1:
            basis.append(0.5 * ((alpha - beta) + (alpha + beta + 2.0) * z))
        for n in range(2, self.order + 1):
            nf = float(n)
            two_n_ab = 2.0 * nf + alpha + beta
            a1 = 2.0 * nf * (nf + alpha + beta) * (two_n_ab - 2.0)
            a2 = (two_n_ab - 1.0) * ((two_n_ab * (two_n_ab - 2.0)) * z + alpha.pow(2) - beta.pow(2))
            a3 = 2.0 * (nf + alpha - 1.0) * (nf + beta - 1.0) * two_n_ab
            basis.append((a2 * basis[-1] - a3 * basis[-2]) / a1.clamp_min(1e-6))
        return torch.stack(basis, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._jacobi_basis(x)
        y = torch.einsum("bio,ijo->bj", basis, self.coefficients)
        if self.bias is not None:
            y = y + self.bias
        return y


class _PolynomialKAN(nn.Module):
    layer_cls: type[nn.Module]

    def __init__(self, layers: List[int], orders: Optional[List[int]] = None, use_layernorm: bool = True):
        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must contain at least input and output dimensions.")
        self.layers = [int(v) for v in layers]
        self.orders = _expand_orders(orders, len(self.layers) - 1)
        modules = []
        norms = []
        for idx, (in_dim, out_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            modules.append(self.layer_cls(in_dim, out_dim, self.orders[idx]))
            norms.append(nn.LayerNorm(out_dim) if use_layernorm and idx < len(self.layers) - 2 else nn.Identity())
        self.blocks = nn.ModuleList(modules)
        self.norms = nn.ModuleList(norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.blocks):
            x = layer(x)
            if idx < len(self.blocks) - 1:
                x = self.norms[idx](x)
        return x

    def prune(self, *args, **kwargs) -> None:
        return None

    def plot(self, *args, **kwargs) -> None:
        return None

    def speed(self, *args, **kwargs) -> None:
        return None

    def symbolic_formula(self, var: Optional[str] = None) -> str:
        return f"Symbolic formulas are currently implemented for the classic spline backend only, not for {self.__class__.__name__}."


class ChebyshevKAN(_PolynomialKAN):
    layer_cls = ChebyshevKANLayer


class FractionalKAN(_PolynomialKAN):
    layer_cls = FractionalKANLayer


# -----------------------------------------------------------------------------
# Model backbones and ANN model
# -----------------------------------------------------------------------------


def _normalize_kan_family(kan_family: str) -> str:
    family = (kan_family or "spline").strip().lower()
    aliases = {
        "kan": "spline",
        "classic": "spline",
        "custom": "spline",
        "spline": "spline",
        "bspline": "spline",
        "b-spline": "spline",
        "chebyshev": "chebyshev",
        "cheby": "chebyshev",
        "chebyshevkan": "chebyshev",
        "fractional": "fractional",
        "jacobi": "fractional",
        "fractionalkan": "fractional",
        "fkan": "fractional",
    }
    return aliases.get(family, family)


class KANBackbone(nn.Module):
    def __init__(
        self,
        width: List[int],
        kan_family: str = "spline",
        grid_size: int = 5,
        spline_order: int = 3,
        noise_scale: float = 0.1,
        seed: int = 0,
        polynomial_order: Optional[int] = None,
        symbolic_lib: Optional[Dict] = None,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.kan_family = _normalize_kan_family(kan_family)
        self.width = width
        self.seed = seed
        self.symbolic_lib = symbolic_lib or SYMBOLIC_LIB
        self.network = self._build_network(
            width=width,
            kan_family=self.kan_family,
            grid_size=grid_size,
            spline_order=spline_order,
            noise_scale=noise_scale,
            seed=seed,
            polynomial_order=polynomial_order,
            symbolic_lib=self.symbolic_lib,
            grid_range=grid_range,
        )

    def _build_network(
        self,
        width: List[int],
        kan_family: str,
        grid_size: int,
        spline_order: int,
        noise_scale: float,
        seed: int,
        polynomial_order: Optional[int],
        symbolic_lib: Optional[Dict],
        grid_range: Tuple[float, float],
    ) -> nn.Module:
        if len(width) < 2:
            raise ValueError("width must contain at least input and output dimensions.")

        if kan_family == "spline":
            return SplineKAN(
                width=width,
                grid=grid_size,
                k=spline_order,
                noise_scale=noise_scale,
                seed=seed,
                grid_range=grid_range,
                symbolic_lib=symbolic_lib,
            )

        order = int(polynomial_order or spline_order)
        orders = [order] * (len(width) - 1)
        if kan_family == "chebyshev":
            return ChebyshevKAN(layers=width, orders=orders)
        if kan_family == "fractional":
            return FractionalKAN(layers=width, orders=orders)
        raise ValueError(f"Unsupported KAN family '{kan_family}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def prune(self, threshold: float = 1e-2) -> None:
        prune_fn = getattr(self.network, "prune", None)
        if callable(prune_fn):
            prune_fn(threshold=threshold)

    def plot(self, **kwargs) -> None:
        plot_fn = getattr(self.network, "plot", None)
        if callable(plot_fn):
            plot_fn(**kwargs)

    def identify_symbolic(self, **kwargs):
        symbolic_fn = getattr(self.network, "identify_symbolic", None)
        if not callable(symbolic_fn):
            raise NotImplementedError(f"Symbolic identification is not supported for '{self.kan_family}'.")
        return symbolic_fn(**kwargs)

    def symbolic_formula(self, var: Optional[Union[str, List[str], Tuple[str, ...]]] = None, **kwargs):
        symbolic_fn = getattr(self.network, "symbolic_formula", None)
        if callable(symbolic_fn):
            return symbolic_fn(var=var, **kwargs)
        return f"Symbolic formulas are not supported for '{self.kan_family}' backend."

    def speed(self, compile: bool = True) -> None:
        speed_fn = getattr(self.network, "speed", None)
        if callable(speed_fn):
            speed_fn(compile=compile)


class EncoderNetwork(nn.Module):
    def __init__(
        self,
        stride_len: int,
        n_u: int,
        n_y: int,
        n_neurons: int,
        n_layer: int,
        state_size: int,
        nonlinearity: str = "relu",
        grid_size: int = 5,
        spline_order: int = 3,
        noise_scale: float = 0.1,
        seed: int = 0,
        kan_family: str = "spline",
        polynomial_order: Optional[int] = None,
        symbolic_lib: Optional[Dict] = None,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__()
        self.stride_len = stride_len
        self.n_u = n_u
        self.n_y = n_y
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.state_size = state_size
        input_dim = (stride_len * n_u) + (stride_len * n_y)
        width = [input_dim] + [n_neurons] * (n_layer - 1) + [state_size] if n_layer > 1 else [input_dim, state_size]
        self.kan_network = KANBackbone(
            width=width,
            kan_family=kan_family,
            grid_size=grid_size,
            spline_order=spline_order,
            noise_scale=noise_scale,
            seed=seed,
            polynomial_order=polynomial_order,
            symbolic_lib=symbolic_lib,
            grid_range=grid_range,
        )

    def prune(self, threshold: float = 1e-2) -> None:
        self.kan_network.prune(threshold=threshold)

    def forward(self, inputs_y: torch.Tensor, inputs_u: torch.Tensor) -> torch.Tensor:
        # OPTIMIZED: Assume tensors are appropriately cast and on device already
        x = torch.cat([inputs_y, inputs_u], dim=-1)
        return self.kan_network(x)


class DecoderNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        n_neurons: int,
        n_layer: int,
        nonlinearity: str = "relu",
        output_window_len: int = 1,
        N_Y: int = 1,
        affine_struct: bool = False,
        grid_size: int = 5,
        spline_order: int = 3,
        noise_scale: float = 0.1,
        seed: int = 0,
        kan_family: str = "spline",
        polynomial_order: Optional[int] = None,
        symbolic_lib: Optional[Dict] = None,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__()
        self.state_size = state_size
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.output_window_len = output_window_len
        self.N_Y = N_Y
        self.affine_struct = affine_struct
        width = [state_size] + [n_neurons] * (n_layer - 1)
        out_dim = output_window_len * state_size * N_Y if affine_struct else output_window_len * N_Y
        width.append(out_dim)
        self.kan_network = KANBackbone(
            width=width,
            kan_family=kan_family,
            grid_size=grid_size,
            spline_order=spline_order,
            noise_scale=noise_scale,
            seed=seed,
            polynomial_order=polynomial_order,
            symbolic_lib=symbolic_lib,
            grid_range=grid_range,
        )

    def prune(self, threshold: float = 1e-2) -> None:
        self.kan_network.prune(threshold=threshold)

    def forward(self, inputs_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # OPTIMIZED: removed hidden to(device) calls
        x = self.kan_network(inputs_state)
        if self.affine_struct:
            x = x.view(-1, self.output_window_len, self.N_Y, self.state_size)
            out = torch.sum(x * inputs_state.unsqueeze(1).unsqueeze(1), dim=-1)
            return x, out
        return x, x


class BridgeNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        N_U: int,
        n_neurons: int,
        n_layer: int,
        nonlinearity: str = "relu",
        affine_struct: bool = False,
        grid_size: int = 5,
        spline_order: int = 3,
        noise_scale: float = 0.1,
        seed: int = 0,
        kan_family: str = "spline",
        polynomial_order: Optional[int] = None,
        symbolic_lib: Optional[Dict] = None,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__()
        self.state_size = state_size
        self.N_U = N_U
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.affine_struct = affine_struct
        input_dim = state_size + N_U
        width = [input_dim] + [n_neurons] * max(1, n_layer - 1)
        self.kan_network = KANBackbone(
            width=width,
            kan_family=kan_family,
            grid_size=grid_size,
            spline_order=spline_order,
            noise_scale=noise_scale,
            seed=seed,
            polynomial_order=polynomial_order,
            symbolic_lib=symbolic_lib,
            grid_range=grid_range,
        )
        self.bridge_bias = nn.Linear(n_neurons, state_size)
        if affine_struct:
            self.bridge_f = nn.Linear(n_neurons, state_size * (state_size + N_U))

    def forward(self, inputs_novelU: torch.Tensor, inputs_state: torch.Tensor) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        # OPTIMIZED: clean concat, removed to(device) checks
        input_concat = torch.cat([inputs_state, inputs_novelU], dim=-1)
        x = self.kan_network(input_concat)
        bias = self.bridge_bias(x)
        if self.affine_struct:
            AB = self.bridge_f(x).view(-1, self.state_size, self.state_size + self.N_U)
            out = torch.bmm(AB, input_concat.unsqueeze(-1)).squeeze(-1) + bias
            return out, AB, bias
        return bias, x, bias

    def prune(self, threshold: float = 1e-2) -> None:
        self.kan_network.prune(threshold=threshold)

    def plot(self, **kwargs) -> None:
        self.kan_network.plot(**kwargs)

    def identify_symbolic(self, **kwargs):
        return self.kan_network.identify_symbolic(**kwargs)

    def symbolic_formula(self, var: Optional[Union[str, List[str], Tuple[str, ...]]] = None, **kwargs):
        return self.kan_network.symbolic_formula(var=var, **kwargs)

    def speed(self) -> None:
        self.kan_network.speed(compile=True)

    def set_mode(self, mode: str) -> None:
        self.kan_network.train() if mode == "train" else self.kan_network.eval()


class ANNModel(nn.Module):
    """Main KAN-based ANN model integrating Encoder, Decoder, and Bridge networks."""

    def __init__(
        self,
        stride_len: int,
        max_range: int,
        n_y: int,
        n_u: int,
        output_window_len: int,
        encoder_network: EncoderNetwork,
        decoder_network: DecoderNetwork,
        bridge_network: BridgeNetwork,
    ):
        super().__init__()
        self.stride_len = stride_len
        self.max_range = max_range
        self.n_y = n_y
        self.n_u = n_u
        self.output_window_len = output_window_len
        self.conv_encoder = encoder_network
        self.output_decoder = decoder_network
        self.bridge_network = bridge_network
        # Training can disable branches whose loss weight is zero. These flags
        # only affect auxiliary outputs; state propagation is always computed.
        self.compute_one_step = True
        self.compute_multi_step = True

    def prune(self, threshold: float = 1e-2) -> None:
        self.conv_encoder.prune(threshold=threshold)
        self.output_decoder.prune(threshold=threshold)
        self.bridge_network.prune(threshold=threshold)

    def forward(self, inputs_y: torch.Tensor, inputs_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Aggiungi questi due cast qui all'inizio:
        inputs_y = inputs_y.float()
        inputs_u = inputs_u.float()

        # Assume inputs_y and inputs_u are ALREADY on the correct device
        device = inputs_y.device
        B = inputs_y.shape[0]

        prediction_error_collection = []
        forward_error_collection = []
        forwarded_predicted_error_collection = []
        predicted_ok_collection = []
        state_k_collection = []
        
        # Will track batched generated states. Shape: [num_states, B, state_size]
        forwarded_states = None

        for k in range(self.max_range):
            i_yk = inputs_y[:, k : self.stride_len + k]
            i_uk = inputs_u[:, k : self.stride_len + k]
            target_start = self.stride_len + k - self.output_window_len + 1
            target_end = self.stride_len + k + 1
            i_target_k = inputs_y[:, target_start:target_end]
            novel_i_uk = inputs_u[:, self.stride_len + k : self.stride_len + k + 1]

            # 1. State extraction
            state_k = self.conv_encoder(
                i_yk.reshape(B, self.stride_len * self.n_y),
                i_uk.reshape(B, self.stride_len * self.n_u),
            )
            
            state_k_collection.append(state_k)
            if self.output_decoder.affine_struct:
                target_shape = (
                    B,
                    self.output_decoder.output_window_len,
                    self.output_decoder.N_Y,
                )
            else:
                target_shape = (
                    B,
                    self.output_decoder.output_window_len * self.output_decoder.N_Y,
                )
            i_target_k = i_target_k.reshape(target_shape)

            if self.compute_one_step:
                predicted_ok = self.output_decoder(state_k)[1]
                predicted_ok_collection.append(predicted_ok)
                prediction_error_collection.append(torch.abs(predicted_ok - i_target_k))

            # 2. Add current state to the pool of states for batched future expansion
            if forwarded_states is not None:
                # Shape becomes: [num_prev_states + 1, B, state_size]
                all_states = torch.cat([state_k.unsqueeze(0), forwarded_states], dim=0)
            else:
                all_states = state_k.unsqueeze(0)

            # 3. Vectorized Decoder & Error calculations
            if all_states.shape[0] > 1:
                prev_states = all_states[1:]  # Only states from previous steps
                
                # Forward errors
                curr_state_expanded = state_k.unsqueeze(0).expand(prev_states.shape[0], -1, -1)
                forward_err = torch.abs(curr_state_expanded - prev_states) # [num_prev, B, state_size]
                
                # We reshape to [B, num_prev * state_size] to maintain 100% shape compatibility with the original final torch.cat(..., dim=-1)
                forward_error_collection.append(forward_err.transpose(0, 1).reshape(B, -1))

                if self.compute_multi_step:
                    # Batch predictions for all previous states.
                    flat_prev_states = prev_states.reshape(-1, self.conv_encoder.state_size)
                    preds = self.output_decoder(flat_prev_states)[1]
                    preds = preds.view(prev_states.shape[0], B, *i_target_k.shape[1:])

                    target_expanded = i_target_k.unsqueeze(0).expand_as(preds)
                    pred_err = preds - target_expanded

                    # Shape compatibility reshape
                    forwarded_predicted_error_collection.append(
                        pred_err.transpose(0, 1).reshape(B, -1)
                    )

            # 4. Vectorized Bridge pass
            num_states = all_states.shape[0]
            flat_all_states = all_states.reshape(-1, self.conv_encoder.state_size)
            flat_novel_u = novel_i_uk.reshape(B, -1).repeat(num_states, 1)

            # One single call through the bridge for ALL states at this time-step
            bridge_out = self.bridge_network(flat_novel_u, flat_all_states)[0]
            
            # Unflatten back to [num_states, B, state_size]
            forwarded_states = bridge_out.view(num_states, B, self.conv_encoder.state_size)

        # Final backward compatible concatenations
        one_step_ahead_prediction_error = (
            torch.cat(prediction_error_collection, dim=-1)
            if prediction_error_collection
            else torch.empty(0, device=device)
        )

        if forwarded_predicted_error_collection:
            forwarded_predicted_error = torch.cat(forwarded_predicted_error_collection, dim=-1)
        else:
            forwarded_predicted_error = torch.empty(0, device=device)

        if forward_error_collection:
            forward_error = torch.cat(forward_error_collection, dim=-1)
        else:
            forward_error = torch.empty(0, device=device)

        return (
            predicted_ok_collection[0] if predicted_ok_collection else torch.empty(0, device=device),
            state_k_collection[0],
            one_step_ahead_prediction_error,
            forwarded_predicted_error,
            forward_error,
        )
