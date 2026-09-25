"""Latent-to-embedding alignment primitives.

The alignment equations use column vectors, while this module deliberately uses
PyTorch's batched *row-vector* convention:
``hidden`` is ``[..., d_a]``, ``W_out`` is ``[vocab, d_a]`` and ``W_in`` is
``[vocab, d_b]``.  In particular, ``S u`` is implemented as
``u @ S.T``.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


AlignMethod = Literal["identical", "linear", "kernel", "soft"]


@dataclass
class AlignmentState:
    method: AlignMethod
    target_norm: torch.Tensor
    matrix: Optional[torch.Tensor] = None
    omega: Optional[torch.Tensor] = None  # [m, d_a]
    numerator: Optional[torch.Tensor] = None  # S, [d_b, m]
    denominator: Optional[torch.Tensor] = None  # z, [m]
    output_weight: Optional[torch.Tensor] = None  # [vocab, d_a]
    input_weight: Optional[torch.Tensor] = None  # [vocab, d_b]
    output_bias: Optional[torch.Tensor] = None  # [vocab]
    temperature: float = 1.0
    query_chunk_size: int = 32
    soft_top_k: int = 30
    soft_top_p: float = 0.95
    soft_min_p: float = 0.001
    soft_max_topk: int = 10
    kernel_gate_mode: str = "soft"
    kernel_fixed_feature: int = 0
    kernel_topk: int = 8


def build_orf(feature_count: int, dimension: int, *, seed: int, device: torch.device) -> torch.Tensor:
    """Build the block orthogonal Gaussian directions Omega from section 3."""
    if feature_count <= 0 or dimension <= 0:
        raise ValueError("feature_count and dimension must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    blocks = []
    while sum(block.shape[0] for block in blocks) < feature_count:
        gaussian = torch.randn((dimension, dimension), generator=generator, device=device, dtype=torch.float32)
        q, r = torch.linalg.qr(gaussian)
        # Fix QR signs so Q has the Haar distribution induced by the Gaussian.
        signs = torch.sign(torch.diagonal(r))
        signs[signs == 0] = 1
        q = q * signs.unsqueeze(0)
        radii = gaussian.norm(dim=0)
        blocks.append(radii.unsqueeze(1) * q.T)  # rows are omega_r^T
    return torch.cat(blocks, dim=0)[:feature_count]


def positive_features(x: torch.Tensor, omega: torch.Tensor, *, stabilize: bool = False) -> torch.Tensor:
    """Return phi_orth(x), with x represented by row vectors.

    When ``stabilize`` is true, one common log-scale is removed per query.
    This does not change ``(S u) / (z^T u)`` and is therefore used online.
    Offline key features must use the unshifted definition.
    """
    if x.shape[-1] != omega.shape[-1]:
        raise ValueError(f"Feature dimension mismatch: {x.shape[-1]} vs {omega.shape[-1]}")
    original_shape = x.shape[:-1]
    flat_x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float32)
    log_features = flat_x @ omega.T - 0.5 * flat_x.square().sum(dim=-1, keepdim=True)
    if stabilize:
        log_features = log_features - log_features.max(dim=-1, keepdim=True).values
    features = torch.exp(log_features) / (omega.shape[0] ** 0.5)
    return features.reshape(*original_shape, omega.shape[0])


def build_kernel_state(
    output_weight: torch.Tensor,
    input_weight: torch.Tensor,
    output_bias: Optional[torch.Tensor],
    *,
    feature_count: int,
    temperature: float,
    seed: int,
    chunk_size: int,
    gate_mode: str = "soft",
    fixed_feature: int = 0,
    topk: int = 8,
) -> AlignmentState:
    """Pre-aggregate the vocabulary-dependent S and z statistics."""
    if output_weight.ndim != 2 or input_weight.ndim != 2:
        raise ValueError("Embedding weights must be rank-2 tensors")
    vocab, d_a = output_weight.shape
    if input_weight.shape[0] != vocab:
        raise ValueError("Kernel alignment requires equal vocabulary sizes")
    if temperature <= 0:
        raise ValueError("kernel temperature must be positive")
    if gate_mode not in {"soft", "argmax", "fixed", "topk"}:
        raise ValueError(f"Unsupported kernel gate mode: {gate_mode}")
    if not 0 <= fixed_feature < feature_count:
        raise ValueError("kernel fixed feature must be in [0, feature_count)")
    if gate_mode == "topk" and not 1 <= topk <= feature_count:
        raise ValueError("kernel top-k must be in [1, feature_count]")
    device = output_weight.device
    omega = build_orf(feature_count, d_a, seed=seed, device=device)
    d_b = input_weight.shape[1]
    s = torch.zeros((d_b, feature_count), device=device, dtype=torch.float32)
    z = torch.zeros(feature_count, device=device, dtype=torch.float32)
    bias = torch.zeros(vocab, device=device, dtype=torch.float32) if output_bias is None else output_bias.detach().to(device=device, dtype=torch.float32)
    # A common bias shift cancels from S u / z^T u and avoids needless overflow.
    bias_shift = bias.max()
    for start in range(0, vocab, max(1, chunk_size)):
        stop = min(start + max(1, chunk_size), vocab)
        keys = positive_features(output_weight[start:stop], omega)
        alpha = torch.exp((bias[start:stop] - bias_shift) / temperature).unsqueeze(1)
        weighted_keys = alpha * keys
        values = input_weight[start:stop].detach().to(device=device, dtype=torch.float32)
        s += values.T @ weighted_keys
        z += weighted_keys.sum(dim=0)
    return AlignmentState(
        method="kernel",
        target_norm=input_weight.detach().float().norm(dim=1).mean(),
        omega=omega,
        numerator=s,
        denominator=z,
        temperature=temperature,
        kernel_gate_mode=gate_mode,
        kernel_fixed_feature=fixed_feature,
        kernel_topk=topk,
    )


def build_soft_state(
    output_weight: torch.Tensor,
    input_weight: torch.Tensor,
    output_bias: Optional[torch.Tensor],
    *,
    temperature: float,
    query_chunk_size: int,
    top_k: int = 30,
    top_p: float = 0.95,
    min_p: float = 0.001,
    max_topk: int = 10,
) -> AlignmentState:
    """Build a Soft Thinking state for the main MAS path.

    The filtering order follows the reference implementation: temperature,
    top-k, top-p, min-p, then a final ``max_topk`` support cap.  Every filtering
    stage is renormalized before the next one.
    """
    if output_weight.ndim != 2 or input_weight.ndim != 2:
        raise ValueError("Embedding weights must be rank-2 tensors")
    if output_weight.shape[0] != input_weight.shape[0]:
        raise ValueError("Soft alignment requires equal vocabulary sizes")
    if output_bias is not None and output_bias.shape != (output_weight.shape[0],):
        raise ValueError("Output bias must have one value per vocabulary item")
    if temperature <= 0:
        raise ValueError("soft temperature must be positive")
    if query_chunk_size <= 0:
        raise ValueError("soft query chunk size must be positive")
    if top_k <= 0:
        raise ValueError("soft top-k must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("soft top-p must be in (0, 1]")
    if not 0 <= min_p <= 1:
        raise ValueError("soft min-p must be in [0, 1]")
    if max_topk <= 0:
        raise ValueError("soft max-topk must be positive")
    if max_topk > top_k:
        raise ValueError("soft max-topk must not exceed soft top-k")
    return AlignmentState(
        method="soft",
        # Soft alignment does not norm-rescale its expected embedding.
        target_norm=torch.zeros((), device=input_weight.device),
        output_weight=output_weight.detach(),
        input_weight=input_weight.detach(),
        output_bias=None if output_bias is None else output_bias.detach(),
        temperature=temperature,
        query_chunk_size=query_chunk_size,
        soft_top_k=top_k,
        soft_top_p=top_p,
        soft_min_p=min_p,
        soft_max_topk=max_topk,
    )


def _renormalize_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    denominator = probabilities.sum(dim=-1, keepdim=True)
    if not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
        raise FloatingPointError("Soft Thinking filtering removed all probability mass")
    return probabilities / denominator


def _soft_thinking_filter(
    probabilities: torch.Tensor,
    state: AlignmentState,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the reference top-k/top-p/min-p/max-topk filtering sequence."""
    vocab = probabilities.shape[-1]

    top_k = min(state.soft_top_k, vocab)
    if top_k < vocab:
        kept_probs, kept_indices = torch.topk(probabilities, k=top_k, dim=-1)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1, kept_indices, kept_probs
        )
    probabilities = _renormalize_probabilities(probabilities)

    if state.soft_top_p < 1:
        sorted_probs, sorted_indices = torch.sort(
            probabilities, dim=-1, descending=True
        )
        remove = sorted_probs.cumsum(dim=-1) > state.soft_top_p
        # Retain the first token that crosses the nucleus threshold.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1, sorted_indices, sorted_probs
        )
        probabilities = _renormalize_probabilities(probabilities)

    if state.soft_min_p > 0:
        threshold = probabilities.max(dim=-1, keepdim=True).values * state.soft_min_p
        probabilities = probabilities.masked_fill(probabilities < threshold, 0.0)
        probabilities = _renormalize_probabilities(probabilities)

    support_size = min(state.soft_max_topk, vocab)
    support_probs, support_indices = torch.topk(
        probabilities, k=support_size, dim=-1
    )
    support_probs = _renormalize_probabilities(support_probs)
    return support_probs, support_indices

def build_linear_state(output_weight: torch.Tensor, input_weight: torch.Tensor, *, ridge: float) -> AlignmentState:
    """Build the row-vector equivalent of the documented least-squares map."""
    if output_weight.shape[0] != input_weight.shape[0]:
        raise ValueError("Linear alignment requires equal vocabulary sizes")
    output = output_weight.detach().float()
    target = input_weight.detach().to(device=output.device, dtype=torch.float32)
    gram = output.T @ output
    matrix = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], device=output.device), output.T @ target)
    return AlignmentState("linear", target.norm(dim=1).mean(), matrix=matrix)


def compute_logits_entropy(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: Optional[torch.Tensor],
    *,
    temperature: float,
    query_chunk_size: int,
) -> torch.Tensor:
    """Compute exact full-vocabulary entropy for temperature-scaled logits."""
    if temperature <= 0:
        raise ValueError("logits entropy temperature must be positive")
    if query_chunk_size <= 0:
        raise ValueError("logits entropy query chunk size must be positive")
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("Logits entropy received non-finite hidden states")

    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).float()
    vocab = output_weight.shape[0]
    vocab_chunk_size = min(4096, vocab)
    entropy_chunks = []
    for query_start in range(0, flat_hidden.shape[0], query_chunk_size):
        query_stop = min(query_start + query_chunk_size, flat_hidden.shape[0])
        queries = flat_hidden[query_start:query_stop]
        logits = []
        for vocab_start in range(0, vocab, vocab_chunk_size):
            vocab_stop = min(vocab_start + vocab_chunk_size, vocab)
            keys = output_weight[vocab_start:vocab_stop].to(
                device=queries.device, dtype=torch.float32
            )
            chunk_logits = queries @ keys.T
            if output_bias is not None:
                chunk_logits = chunk_logits + output_bias[vocab_start:vocab_stop].to(
                    device=queries.device, dtype=torch.float32
                )
            logits.append(chunk_logits / temperature)
        probabilities = torch.softmax(torch.cat(logits, dim=-1), dim=-1)
        entropy_chunks.append(torch.special.entr(probabilities).sum(dim=-1))
    entropy = (
        torch.cat(entropy_chunks, dim=0)
        if entropy_chunks
        else torch.empty(0, device=flat_hidden.device, dtype=torch.float32)
    )
    # Passing an empty prefix via ``reshape(*())`` calls reshape with no
    # arguments for a single hidden vector.  The tuple form handles both the
    # scalar result and arbitrary leading dimensions.
    return entropy.reshape(hidden.shape[:-1])

def apply_soft_alignment_with_entropy(
    hidden: torch.Tensor, state: AlignmentState
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the filtered concept-token embedding and pre-filter entropy."""
    if state.method != "soft":
        raise ValueError("Entropy is only available for soft alignment")
    assert state.output_weight is not None and state.input_weight is not None
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("Soft alignment received non-finite hidden states")

    original_dtype = hidden.dtype
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).float()
    vocab = state.output_weight.shape[0]
    vocab_chunk_size = min(4096, vocab)
    aligned_chunks = []
    entropy_chunks = []
    for query_start in range(0, flat_hidden.shape[0], state.query_chunk_size):
        query_stop = min(query_start + state.query_chunk_size, flat_hidden.shape[0])
        queries = flat_hidden[query_start:query_stop]
        logits = []
        for vocab_start in range(0, vocab, vocab_chunk_size):
            vocab_stop = min(vocab_start + vocab_chunk_size, vocab)
            keys = state.output_weight[vocab_start:vocab_stop].to(
                device=queries.device, dtype=torch.float32
            )
            chunk_logits = queries @ keys.T
            if state.output_bias is not None:
                chunk_logits = chunk_logits + state.output_bias[
                    vocab_start:vocab_stop
                ].to(device=queries.device, dtype=torch.float32)
            logits.append(chunk_logits / state.temperature)
        probabilities = torch.softmax(torch.cat(logits, dim=-1), dim=-1)
        del logits
        # The official implementation measures Cold Stop entropy before
        # top-k/top-p/min-p filtering.
        entropy_chunks.append(torch.special.entr(probabilities).sum(dim=-1))
        support_probs, support_indices = _soft_thinking_filter(probabilities, state)
        values = state.input_weight[
            support_indices.to(state.input_weight.device)
        ].to(device=queries.device, dtype=torch.float32)
        chunk_aligned = torch.sum(support_probs.unsqueeze(-1) * values, dim=1)
        aligned_chunks.append(chunk_aligned)

    if aligned_chunks:
        aligned = torch.cat(aligned_chunks, dim=0)
        entropy = torch.cat(entropy_chunks, dim=0)
    else:
        aligned = torch.empty(
            (0, state.input_weight.shape[1]),
            device=flat_hidden.device,
            dtype=torch.float32,
        )
        entropy = torch.empty(0, device=flat_hidden.device, dtype=torch.float32)
    return (
        aligned.reshape(*hidden.shape[:-1], aligned.shape[-1]).to(original_dtype),
        entropy.reshape(*hidden.shape[:-1]),
    )


def apply_alignment(hidden: torch.Tensor, state: AlignmentState) -> torch.Tensor:
    """Apply one state to row-vector hidden states of shape ``[..., d_a]``."""
    original_dtype = hidden.dtype
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).float()
    if state.method == "identical":
        aligned = flat_hidden
        # Existing behaviour: identity mapping followed by embedding-norm scaling.
        aligned = aligned * (state.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    elif state.method == "linear":
        assert state.matrix is not None
        aligned = flat_hidden @ state.matrix
        aligned = aligned * (state.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    elif state.method == "soft":
        aligned, _ = apply_soft_alignment_with_entropy(hidden, state)
        return aligned
    elif state.method == "kernel":
        assert state.omega is not None and state.numerator is not None and state.denominator is not None
        u = positive_features(flat_hidden / state.temperature, state.omega, stabilize=True)
        if state.kernel_gate_mode == "fixed":
            index = state.kernel_fixed_feature
            prototype = state.numerator[:, index] / state.denominator[index]
            aligned = prototype.unsqueeze(0).expand(flat_hidden.shape[0], -1)
            return aligned.reshape(*hidden.shape[:-1], aligned.shape[-1]).to(original_dtype)
        feature_weights = u * state.denominator.unsqueeze(0)
        if state.kernel_gate_mode == "argmax":
            indices = feature_weights.argmax(dim=-1)
            selected_s = state.numerator.T[indices]
            aligned = selected_s / state.denominator[indices].unsqueeze(-1)
            return aligned.reshape(*hidden.shape[:-1], aligned.shape[-1]).to(original_dtype)
        if state.kernel_gate_mode == "topk":
            weights, indices = feature_weights.topk(state.kernel_topk, dim=-1)
            selected_s = state.numerator.T[indices]
            selected_z = state.denominator[indices]
            prototypes = selected_s / selected_z.unsqueeze(-1)
            aligned = (weights.unsqueeze(-1) * prototypes).sum(dim=-2) / weights.sum(dim=-1, keepdim=True)
            return aligned.reshape(*hidden.shape[:-1], aligned.shape[-1]).to(original_dtype)
        denom = u @ state.denominator
        if not torch.isfinite(denom).all() or (denom <= torch.finfo(denom.dtype).eps).any():
            raise FloatingPointError("Kernel alignment denominator is non-positive or non-finite")
        # h_B = S u / (z^T u). Here u is a row vector: u @ S.T.
        aligned = (u @ state.numerator.T) / denom.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported alignment method {state.method}")
    return aligned.reshape(*hidden.shape[:-1], aligned.shape[-1]).to(original_dtype)
