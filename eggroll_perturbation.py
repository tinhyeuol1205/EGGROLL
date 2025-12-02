"""
EGGROLL Step 2: Low-Rank Perturbation Generation

This module implements the core innovation of EGGROLL:
Instead of storing full perturbation ε of size (d × d),
we store A (d × r) and B (d × r) where ε ≈ A @ B.T

Based on: https://github.com/ESHyperscale/HyperscaleES
Paper: "Evolution Strategies at Hyperscale" (arXiv:2511.16652)
"""

import torch
import torch. nn as nn
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import math

# Import from Step 1
from eggroll_initialization import (
    EggrollConfig,
    FrozenNoiserParams,
    NoiserParams,
    ESMapType,
    RandomKeyGenerator,
)


# ============================================================================
# Perturbation Data Structures
# ============================================================================

@dataclass
class LoRAPerturbation:
    """
    Low-rank perturbation for a single parameter. 
    
    Stores A and B such that the perturbation ε = A @ B.T
    Memory: O(d × r + d × r) = O(2dr) instead of O(d²)
    """
    A: torch. Tensor  # Shape: (out_features, rank) or (d, r)
    B: torch. Tensor  # Shape: (in_features, rank) or (d, r)
    
    @property
    def rank(self) -> int:
        return self.A.shape[-1]
    
    def materialize(self) -> torch.Tensor:
        """
        Compute full perturbation matrix: ε = A @ B.T
        Warning: This defeats the memory savings!  Use only for debugging.
        """
        return self.A @ self.B.T
    
    def memory_bytes(self) -> int:
        """Calculate memory usage in bytes"""
        return self. A.numel() * self.A.element_size() + self.B. numel() * self.B.element_size()


@dataclass
class FullPerturbation:
    """Full perturbation for non-LoRA parameters (biases, small tensors)"""
    noise: torch.Tensor
    
    def memory_bytes(self) -> int:
        return self.noise. numel() * self.noise.element_size()


@dataclass
class PopulationMember:
    """
    A single member of the population (one perturbed model). 
    
    Contains perturbations for all trainable parameters.
    """
    index: int  # Population member index (0 to N-1)
    epoch: int  # Current epoch
    lora_perturbations: Dict[str, LoRAPerturbation]  # For LORA params
    full_perturbations: Dict[str, FullPerturbation]  # For FULL params
    
    def total_memory_bytes(self) -> int:
        """Calculate total memory usage"""
        lora_mem = sum(p.memory_bytes() for p in self.lora_perturbations.values())
        full_mem = sum(p.memory_bytes() for p in self. full_perturbations.values())
        return lora_mem + full_mem


# ============================================================================
# Core Perturbation Generation Functions
# ============================================================================

def fold_in_key(base_seed: int, *args: int) -> int:
    """
    Deterministically derive a new seed from base seed and additional integers.
    Similar to JAX's jax.random.fold_in
    """
    result = base_seed
    for arg in args:
        # Simple but effective mixing function
        result = ((result * 31337) + arg) % (2**31)
    return result


def generate_lora_perturbation(
    param_shape: Tuple[int, ... ],
    sigma: float,
    rank: int,
    seed: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> LoRAPerturbation:
    """
    Generate low-rank perturbation matrices A and B. 
    
    For a weight matrix W of shape (out_features, in_features):
    - A has shape (out_features, rank)
    - B has shape (in_features, rank)
    - Perturbation ε = σ/√r * A @ B. T has shape (out_features, in_features)
    
    The scaling by 1/√r ensures the perturbation magnitude is independent of rank.
    
    Args:
        param_shape: Shape of the parameter (out_features, in_features)
        sigma: Noise standard deviation
        rank: Low-rank dimension r
        seed: Random seed for reproducibility
        device: Device for tensors
        dtype: Data type for tensors
        
    Returns:
        LoRAPerturbation containing A and B matrices
    """
    assert len(param_shape) == 2, f"LoRA perturbation requires 2D tensor, got shape {param_shape}"
    
    out_features, in_features = param_shape
    
    # Create generator with specific seed
    gen_A = torch.Generator(device='cpu').manual_seed(seed)
    gen_B = torch.Generator(device='cpu'). manual_seed(fold_in_key(seed, 1))
    
    # Sample A and B from standard normal distribution
    # A: (out_features, rank), B: (in_features, rank)
    A = torch. randn(out_features, rank, generator=gen_A, dtype=dtype)
    B = torch.randn(in_features, rank, generator=gen_B, dtype=dtype)
    
    # Scale by σ/√r to normalize perturbation magnitude
    # This ensures ||A @ B. T||_F ≈ σ * √(out_features * in_features) regardless of rank
    scale = sigma / math.sqrt(rank)
    A = A * scale
    
    # Move to device
    A = A.to(device)
    B = B.to(device)
    
    return LoRAPerturbation(A=A, B=B)


def generate_full_perturbation(
    param_shape: Tuple[int, ... ],
    sigma: float,
    seed: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> FullPerturbation:
    """
    Generate full Gaussian perturbation for non-LoRA parameters.
    
    Used for biases, layer norms, or other small parameters.
    
    Args:
        param_shape: Shape of the parameter
        sigma: Noise standard deviation
        seed: Random seed for reproducibility
        device: Device for tensors
        dtype: Data type for tensors
        
    Returns:
        FullPerturbation containing the noise tensor
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(*param_shape, generator=gen, dtype=dtype) * sigma
    noise = noise.to(device)
    
    return FullPerturbation(noise=noise)


# ============================================================================
# Population Generation
# ============================================================================

class PerturbationGenerator:
    """
    Generator for creating population of perturbed models. 
    
    Implements the core EGGROLL perturbation strategy with on-the-fly
    noise generation from deterministic seeds.
    """
    
    def __init__(
        self,
        params: Dict[str, torch.Tensor],
        es_map: Dict[str, int],
        base_evo_keys: Dict[str, RandomKeyGenerator],
        frozen_noiser_params: FrozenNoiserParams,
        noiser_params: NoiserParams,
        device: str = "cuda",
    ):
        """
        Initialize perturbation generator.
        
        Args:
            params: Model parameters {name: tensor}
            es_map: Parameter classification map {name: ESMapType}
            base_evo_keys: Random keys for each parameter
            frozen_noiser_params: Frozen noiser configuration
            noiser_params: Mutable noiser state
            device: Device for computations
        """
        self.params = params
        self. es_map = es_map
        self.base_evo_keys = base_evo_keys
        self.frozen_noiser_params = frozen_noiser_params
        self. noiser_params = noiser_params
        self.device = device
        
        # Cache parameter info
        self. param_shapes = {name: p.shape for name, p in params.items()}
        self. param_dtypes = {name: p.dtype for name, p in params. items()}
        
    def get_iteration_seed(
        self,
        base_seed: int,
        epoch: int,
        member_idx: int,
    ) -> int:
        """
        Compute deterministic seed for a specific (epoch, member) combination.
        
        This allows regenerating the same perturbation later without storing it.
        """
        noise_reuse = self.frozen_noiser_params.noise_reuse
        
        if noise_reuse > 0:
            # Reuse noise across epochs within groups
            effective_epoch = epoch // noise_reuse
        else:
            effective_epoch = epoch
            
        return fold_in_key(base_seed, effective_epoch, member_idx)
    
    def generate_perturbation_for_param(
        self,
        name: str,
        epoch: int,
        member_idx: int,
    ) -> Optional[LoRAPerturbation | FullPerturbation]:
        """
        Generate perturbation for a single parameter.
        
        Args:
            name: Parameter name
            epoch: Current epoch
            member_idx: Population member index
            
        Returns:
            Perturbation object or None if parameter is frozen
        """
        map_type = self. es_map. get(name, ESMapType.FROZEN)
        
        if map_type == ESMapType. FROZEN:
            return None
            
        # Get deterministic seed for this (param, epoch, member)
        base_seed = self. base_evo_keys[name].seed
        seed = self. get_iteration_seed(base_seed, epoch, member_idx)
        
        param_shape = self. param_shapes[name]
        param_dtype = self. param_dtypes[name]
        sigma = self.noiser_params.sigma
        rank = self.frozen_noiser_params.rank
        
        if map_type == ESMapType.LORA and len(param_shape) == 2:
            # Low-rank perturbation for weight matrices
            return generate_lora_perturbation(
                param_shape=param_shape,
                sigma=sigma,
                rank=rank,
                seed=seed,
                device=self.device,
                dtype=param_dtype,
            )
        elif map_type == ESMapType.FULL:
            if self.frozen_noiser_params.freeze_nonlora:
                return None  # Skip non-LoRA params if frozen
            # Full perturbation for other parameters
            return generate_full_perturbation(
                param_shape=param_shape,
                sigma=sigma,
                seed=seed,
                device=self. device,
                dtype=param_dtype,
            )
        
        return None
    
    def generate_population_member(
        self,
        epoch: int,
        member_idx: int,
    ) -> PopulationMember:
        """
        Generate all perturbations for a single population member.
        
        Args:
            epoch: Current epoch
            member_idx: Population member index (0 to N-1)
            
        Returns:
            PopulationMember containing all perturbations
        """
        lora_perturbations = {}
        full_perturbations = {}
        
        for name in self.params. keys():
            perturbation = self. generate_perturbation_for_param(name, epoch, member_idx)
            
            if perturbation is None:
                continue
            elif isinstance(perturbation, LoRAPerturbation):
                lora_perturbations[name] = perturbation
            elif isinstance(perturbation, FullPerturbation):
                full_perturbations[name] = perturbation
                
        return PopulationMember(
            index=member_idx,
            epoch=epoch,
            lora_perturbations=lora_perturbations,
            full_perturbations=full_perturbations,
        )
    
    def generate_population(
        self,
        epoch: int,
        population_size: int,
    ) -> List[PopulationMember]:
        """
        Generate perturbations for entire population.
        
        Args:
            epoch: Current epoch
            population_size: Number of population members (N)
            
        Returns:
            List of PopulationMember objects
        """
        population = []
        
        for i in range(population_size):
            member = self.generate_population_member(epoch, i)
            population.append(member)
            
        return population


# ============================================================================
# Apply Perturbations to Model
# ============================================================================

def apply_lora_perturbation_to_param(
    param: torch.Tensor,
    perturbation: LoRAPerturbation,
) -> torch.Tensor:
    """
    Apply low-rank perturbation to a parameter WITHOUT materializing full matrix.
    
    θ_perturbed = θ + A @ B.T
    
    For memory efficiency, we return a lazy representation that computes
    the perturbation on-the-fly during forward pass.
    """
    # For simplicity, materialize here.  In production, use custom autograd. 
    return param + perturbation.A @ perturbation.B.T


def apply_full_perturbation_to_param(
    param: torch. Tensor,
    perturbation: FullPerturbation,
) -> torch.Tensor:
    """Apply full perturbation to a parameter."""
    return param + perturbation.noise


def create_perturbed_params(
    params: Dict[str, torch.Tensor],
    population_member: PopulationMember,
) -> Dict[str, torch.Tensor]:
    """
    Create a complete set of perturbed parameters.
    
    θ_i = θ + σ(A_i × B_i^T) for LoRA params
    θ_i = θ + ε_i for full params
    
    Args:
        params: Original model parameters
        population_member: Perturbations to apply
        
    Returns:
        Dictionary of perturbed parameters
    """
    perturbed = {}
    
    for name, param in params.items():
        if name in population_member. lora_perturbations:
            perturbed[name] = apply_lora_perturbation_to_param(
                param, 
                population_member.lora_perturbations[name]
            )
        elif name in population_member.full_perturbations:
            perturbed[name] = apply_full_perturbation_to_param(
                param,
                population_member.full_perturbations[name]
            )
        else:
            # Frozen parameter - use original
            perturbed[name] = param
            
    return perturbed


def load_perturbed_params_to_model(
    model: nn. Module,
    perturbed_params: Dict[str, torch.Tensor],
):
    """
    Load perturbed parameters into model for forward pass.
    
    This temporarily replaces model parameters with perturbed versions. 
    """
    state_dict = model. state_dict()
    
    for name, param in perturbed_params.items():
        if name in state_dict:
            state_dict[name] = param
            
    model.load_state_dict(state_dict, strict=False)


# ============================================================================
# Memory-Efficient Forward Pass with LoRA
# ============================================================================

class LoRALinear(nn.Module):
    """
    Memory-efficient linear layer with low-rank perturbation. 
    
    Computes: y = x @ (W + A @ B.T). T = x @ W.T + x @ B @ A. T
    
    This avoids materializing the full (W + A @ B. T) matrix.
    """
    
    def __init__(
        self,
        base_weight: torch.Tensor,
        perturbation: Optional[LoRAPerturbation] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.base_weight = base_weight  # (out, in)
        self.perturbation = perturbation
        self.bias = bias
        
    def forward(self, x: torch. Tensor) -> torch. Tensor:
        """
        Forward pass: y = x @ W.T + x @ B @ A.T + bias
        
        Memory: O(batch × in × rank + batch × rank × out)
        Instead of: O(out × in) to materialize perturbed weight
        """
        # Base computation: x @ W.T
        out = x @ self.base_weight.T
        
        # Add low-rank perturbation if present
        if self. perturbation is not None:
            # x @ B @ A. T  (efficient order of operations)
            # x: (batch, in), B: (in, r), A: (out, r)
            # Step 1: x @ B -> (batch, r)
            # Step 2: result @ A.T -> (batch, out)
            out = out + (x @ self.perturbation.B) @ self.perturbation.A.T
            
        # Add bias
        if self.bias is not None:
            out = out + self.bias
            
        return out


# ============================================================================
# Utility Functions
# ============================================================================

def compute_memory_savings(
    param_shape: Tuple[int, int],
    rank: int,
) -> Dict[str, Any]:
    """
    Compute memory savings from using low-rank perturbation.
    
    Full: O(d1 × d2)
    LoRA: O(d1 × r + d2 × r) = O((d1 + d2) × r)
    """
    d1, d2 = param_shape
    
    full_memory = d1 * d2
    lora_memory = (d1 + d2) * rank
    
    savings_ratio = full_memory / lora_memory
    savings_percent = (1 - lora_memory / full_memory) * 100
    
    return {
        "param_shape": param_shape,
        "rank": rank,
        "full_elements": full_memory,
        "lora_elements": lora_memory,
        "savings_ratio": savings_ratio,
        "savings_percent": savings_percent,
    }


def print_perturbation_stats(population: List[PopulationMember]):
    """Print statistics about generated perturbations"""
    print("\n" + "=" * 60)
    print("Perturbation Statistics")
    print("=" * 60)
    
    total_lora = sum(len(m.lora_perturbations) for m in population)
    total_full = sum(len(m. full_perturbations) for m in population)
    total_memory = sum(m. total_memory_bytes() for m in population)
    
    print(f"Population size: {len(population)}")
    print(f"LoRA perturbations per member: {len(population[0].lora_perturbations)}")
    print(f"Full perturbations per member: {len(population[0]. full_perturbations)}")
    print(f"Total memory: {total_memory / 1024 / 1024:. 2f} MB")
    print(f"Memory per member: {total_memory / len(population) / 1024 / 1024:. 2f} MB")
    
    # Sample memory savings for first LoRA param
    if population[0].lora_perturbations:
        first_name = list(population[0].lora_perturbations.keys())[0]
        first_lora = population[0].lora_perturbations[first_name]
        shape = (first_lora. A.shape[0], first_lora.B.shape[0])
        savings = compute_memory_savings(shape, first_lora.rank)
        print(f"\nExample savings for {first_name}:")
        print(f"  Shape: {savings['param_shape']}, Rank: {savings['rank']}")
        print(f"  Full: {savings['full_elements']:,} elements")
        print(f"  LoRA: {savings['lora_elements']:,} elements")
        print(f"  Savings: {savings['savings_percent']:.1f}% ({savings['savings_ratio']:.1f}x)")


# ============================================================================
# Example Usage / Test
# ============================================================================

if __name__ == "__main__":
    from eggroll_initialization import initialize_eggroll, EggrollConfig
    
    print("=" * 60)
    print("EGGROLL Step 2: Low-Rank Perturbation Generation")
    print("=" * 60)
    
    # Step 1: Initialize
    config = EggrollConfig(
        model_name="Helsinki-NLP/opus-mt-en-vi",
        sigma=1e-3,
        lr_scale=1.0,
        rank=16,  # Low-rank dimension
        population_size=8,
        freeze_nonlora=True,
        device="cuda" if torch.cuda. is_available() else "cpu",
    )
    
    (
        model,
        params,
        frozen_noiser_params,
        noiser_params,
        es_map,
        base_evo_keys,
        tokenizer,
    ) = initialize_eggroll(config)
    
    # Step 2: Create perturbation generator
    print("\n" + "-" * 60)
    print("Creating Perturbation Generator...")
    print("-" * 60)
    
    generator = PerturbationGenerator(
        params=params,
        es_map=es_map,
        base_evo_keys=base_evo_keys,
        frozen_noiser_params=frozen_noiser_params,
        noiser_params=noiser_params,
        device=config.device,
    )
    
    # Generate population for epoch 0
    print(f"\nGenerating population of {config.population_size} members...")
    population = generator.generate_population(
        epoch=0,
        population_size=config.population_size,
    )
    
    # Print statistics
    print_perturbation_stats(population)
    
    # Demonstrate reproducibility
    print("\n" + "-" * 60)
    print("Testing Reproducibility...")
    print("-" * 60)
    
    # Regenerate same perturbation
    member_0_v1 = generator.generate_population_member(epoch=0, member_idx=0)
    member_0_v2 = generator.generate_population_member(epoch=0, member_idx=0)
    
    # Check they're identical
    first_param = list(member_0_v1.lora_perturbations. keys())[0]
    A1 = member_0_v1.lora_perturbations[first_param]. A
    A2 = member_0_v2.lora_perturbations[first_param]. A
    
    is_identical = torch.allclose(A1, A2)
    print(f"Regenerated perturbation identical: {is_identical}")
    
    # Demonstrate perturbed parameters
    print("\n" + "-" * 60)
    print("Creating Perturbed Model...")
    print("-" * 60)
    
    perturbed_params = create_perturbed_params(params, population[0])
    
    # Show difference for one parameter
    sample_param = first_param
    original = params[sample_param]
    perturbed = perturbed_params[sample_param]
    diff = perturbed - original
    
    print(f"\nParameter: {sample_param}")
    print(f"  Original shape: {original.shape}")
    print(f"  Perturbed shape: {perturbed.shape}")
    print(f"  Perturbation norm: {torch.norm(diff). item():.6f}")
    print(f"  Perturbation mean: {diff.mean().item():.8f}")
    print(f"  Perturbation std: {diff.std(). item():.8f}")
    
    # Memory comparison
    print("\n" + "-" * 60)
    print("Memory Analysis")
    print("-" * 60)
    
    # Count LoRA-eligible params
    lora_params = [(n, p) for n, p in params. items() 
                   if es_map.get(n) == ESMapType. LORA and len(p.shape) == 2]
    
    total_full_memory = sum(p. numel() for _, p in lora_params) * 4  # float32
    total_lora_memory = sum(
        (p.shape[0] + p.shape[1]) * config.rank * 4 
        for _, p in lora_params
    )
    
    print(f"LoRA-eligible parameters: {len(lora_params)}")
    print(f"Full perturbation memory: {total_full_memory / 1024 / 1024:.2f} MB")
    print(f"LoRA perturbation memory: {total_lora_memory / 1024 / 1024:.2f} MB")
    print(f"Memory savings: {(1 - total_lora_memory / total_full_memory) * 100:.1f}%")
    print(f"Compression ratio: {total_full_memory / total_lora_memory:. 1f}x")
    
    print("\n" + "=" * 60)
    print("Step 2 Complete!  Ready for Step 3: Apply Perturbations")
    print("=" * 60)
    print("""
Next Steps:
-----------
For each population member i (i = 1... N):
  1. θ_i = θ + σ(A_i × B_i^T)  ← Done! 
  2. Forward pass with θ_i
  3. Compute reward R_i (e.g., BLEU score)
  
Then proceed to Step 5: Gradient Estimation
""")
