"""
EGGROLL Step 5 & 6: Gradient Estimation and Parameter Update

Step 5: Estimate gradient using Evolution Strategies formula:
    Δθ ≈ (1/Nσ) Σ R_i * ε_i
    
    For LoRA parameters:
    Δθ ≈ (1/Nσ) Σ R_i * (A_i @ B_i^T)

Step 6: Update parameters using optimizer:
    θ_new = θ + α * Optimizer(Δθ)

Based on: https://github.com/ESHyperscale/HyperscaleES
Paper: "Evolution Strategies at Hyperscale" (arXiv:2511.16652)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
from functools import partial

# Import from previous steps
from eggroll_initialization import (
    EggrollConfig,
    FrozenNoiserParams,
    NoiserParams,
    OptimizerState,
    ESMapType,
    RandomKeyGenerator,
)
from eggroll_perturbation import (
    LoRAPerturbation,
    PopulationMember,
    PerturbationGenerator,
    fold_in_key,
    generate_lora_perturbation,
    generate_full_perturbation,
)
from eggroll_evaluation import PopulationRewards


# ============================================================================
# Gradient Estimation Data Structures
# ============================================================================

@dataclass
class GradientEstimate:
    """Estimated gradient for a single parameter"""
    name: str
    gradient: torch.Tensor
    update_type: int  # ESMapType: LORA, FULL, or FROZEN
    
    # Statistics
    gradient_norm: float = 0.0
    gradient_mean: float = 0.0
    gradient_std: float = 0.0


@dataclass
class PopulationGradients:
    """All gradient estimates for the population"""
    epoch: int
    gradients: Dict[str, GradientEstimate]
    
    # Aggregated statistics
    total_gradient_norm: float = 0. 0
    lora_gradient_norm: float = 0.0
    full_gradient_norm: float = 0.0


# ============================================================================
# Core Gradient Estimation Functions
# ============================================================================

def compute_lora_gradient(
    fitnesses: np.ndarray,
    perturbations: List[LoRAPerturbation],
    sigma: float,
    rank: int,
) -> torch.Tensor:
    """
    Compute gradient estimate for a LoRA parameter. 
    
    Formula from EGGROLL paper:
    Δθ = (1/N) Σ R_i * (A_i @ B_i^T) / σ
    
    Efficient computation using einsum:
    Δθ = (1/N) * einsum('n,nir,njr->ij', R, A, B) / σ
    
    Args:
        fitnesses: Shaped fitness scores, shape (N,)
        perturbations: List of LoRAPerturbation for each population member
        sigma: Noise standard deviation
        rank: Low-rank dimension
        
    Returns:
        Gradient estimate, shape (out_features, in_features)
    """
    N = len(fitnesses)
    
    # Stack A and B matrices: (N, out_features, rank) and (N, in_features, rank)
    A_stack = torch.stack([p.A for p in perturbations], dim=0)  # (N, out, r)
    B_stack = torch.stack([p.B for p in perturbations], dim=0)  # (N, in, r)
    
    # Convert fitnesses to tensor
    R = torch.tensor(fitnesses, dtype=A_stack.dtype, device=A_stack.device)
    
    # Efficient gradient computation using einsum
    # R: (N,), A: (N, out, r), B: (N, in, r)
    # Result: (out, in)
    # 
    # Step by step:
    # 1. Weight A by fitness: R[:, None, None] * A -> (N, out, r)
    # 2.  Compute outer product sum: einsum('nir,njr->ij', weighted_A, B)
    
    # Reshape R for broadcasting: (N, 1, 1)
    R_broadcast = R. view(N, 1, 1)
    
    # Weighted A matrices
    weighted_A = R_broadcast * A_stack  # (N, out, r)
    
    # Compute gradient: sum over N and r, result is (out, in)
    # This is equivalent to: sum_n sum_r (R_n * A_n[:, r]) @ B_n[:, r]. T
    gradient = torch.einsum('nir,njr->ij', weighted_A, B_stack)
    
    # Normalize by population size
    # Note: sigma scaling is already in A (from perturbation generation)
    gradient = gradient / N
    
    return gradient


def compute_full_gradient(
    fitnesses: np.ndarray,
    noises: List[torch. Tensor],
    sigma: float,
) -> torch.Tensor:
    """
    Compute gradient estimate for a full (non-LoRA) parameter.
    
    Standard ES formula:
    Δθ = (1/Nσ) Σ R_i * ε_i
    
    Args:
        fitnesses: Shaped fitness scores, shape (N,)
        noises: List of noise tensors for each population member
        sigma: Noise standard deviation
        
    Returns:
        Gradient estimate, same shape as parameter
    """
    N = len(fitnesses)
    
    # Stack noises
    noise_stack = torch. stack(noises, dim=0)  # (N, *param_shape)
    
    # Convert fitnesses to tensor
    R = torch. tensor(fitnesses, dtype=noise_stack.dtype, device=noise_stack.device)
    
    # Reshape R for broadcasting
    R_broadcast = R.view(N, *([1] * (noise_stack.dim() - 1)))
    
    # Weighted sum of noises
    weighted_noise = R_broadcast * noise_stack
    gradient = weighted_noise.mean(dim=0)
    
    # Scale by 1/sigma (noise was already scaled by sigma during generation)
    gradient = gradient / sigma
    
    return gradient


# ============================================================================
# EGGROLL Gradient Estimator
# ============================================================================

class EggrollGradientEstimator:
    """
    Estimates gradients for EGGROLL using Evolution Strategies. 
    
    Key insight: We don't need to store all perturbations! 
    Since perturbations are generated from deterministic seeds,
    we can regenerate them during gradient computation.
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
        Initialize gradient estimator. 
        
        Args:
            params: Model parameters
            es_map: Parameter classification map
            base_evo_keys: Random keys for each parameter
            frozen_noiser_params: Frozen noiser configuration
            noiser_params: Mutable noiser state
            device: Computation device
        """
        self.params = params
        self.es_map = es_map
        self.base_evo_keys = base_evo_keys
        self.frozen_noiser_params = frozen_noiser_params
        self.noiser_params = noiser_params
        self.device = device
        
        # Cache parameter info
        self. param_shapes = {name: p.shape for name, p in params.items()}
        self.param_dtypes = {name: p.dtype for name, p in params. items()}
        
    def _get_iteration_seed(self, base_seed: int, epoch: int, member_idx: int) -> int:
        """Get deterministic seed for (epoch, member) combination."""
        noise_reuse = self.frozen_noiser_params.noise_reuse
        if noise_reuse > 0:
            effective_epoch = epoch // noise_reuse
        else:
            effective_epoch = epoch
        return fold_in_key(base_seed, effective_epoch, member_idx)
    
    def _regenerate_lora_perturbation(
        self,
        name: str,
        epoch: int,
        member_idx: int,
    ) -> LoRAPerturbation:
        """Regenerate LoRA perturbation from seed (memory efficient)."""
        base_seed = self. base_evo_keys[name].seed
        seed = self._get_iteration_seed(base_seed, epoch, member_idx)
        
        return generate_lora_perturbation(
            param_shape=self.param_shapes[name],
            sigma=self.noiser_params.sigma,
            rank=self.frozen_noiser_params.rank,
            seed=seed,
            device=self.device,
            dtype=self.param_dtypes[name],
        )
    
    def _regenerate_full_perturbation(
        self,
        name: str,
        epoch: int,
        member_idx: int,
    ) -> torch.Tensor:
        """Regenerate full perturbation from seed."""
        base_seed = self.base_evo_keys[name].seed
        seed = self._get_iteration_seed(base_seed, epoch, member_idx)
        
        gen = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(
            *self.param_shapes[name],
            generator=gen,
            dtype=self.param_dtypes[name],
        ) * self.noiser_params.sigma
        
        return noise. to(self.device)
    
    def estimate_gradients(
        self,
        population_rewards: PopulationRewards,
        epoch: int,
    ) -> PopulationGradients:
        """
        Estimate gradients for all parameters using ES. 
        
        This is the core of Step 5 in EGGROLL. 
        
        Args:
            population_rewards: Rewards from Step 4
            epoch: Current epoch
            
        Returns:
            PopulationGradients containing gradient estimates for all parameters
        """
        gradients = {}
        fitnesses = population_rewards.aggregated_rewards  # Already shaped
        population_size = len(fitnesses)
        
        total_norm_sq = 0.0
        lora_norm_sq = 0.0
        full_norm_sq = 0.0
        
        for name, param in self.params.items():
            map_type = self. es_map. get(name, ESMapType.FROZEN)
            
            if map_type == ESMapType. FROZEN:
                # No gradient for frozen parameters
                continue
                
            elif map_type == ESMapType.LORA and len(param. shape) == 2:
                # LoRA gradient estimation
                # Regenerate all perturbations for this parameter
                perturbations = [
                    self._regenerate_lora_perturbation(name, epoch, i)
                    for i in range(population_size)
                ]
                
                gradient = compute_lora_gradient(
                    fitnesses=fitnesses,
                    perturbations=perturbations,
                    sigma=self.noiser_params.sigma,
                    rank=self. frozen_noiser_params.rank,
                )
                
                grad_norm = torch. norm(gradient).item()
                lora_norm_sq += grad_norm ** 2
                
            elif map_type == ESMapType. FULL:
                if self.frozen_noiser_params.freeze_nonlora:
                    continue
                    
                # Full gradient estimation
                noises = [
                    self._regenerate_full_perturbation(name, epoch, i)
                    for i in range(population_size)
                ]
                
                gradient = compute_full_gradient(
                    fitnesses=fitnesses,
                    noises=noises,
                    sigma=self. noiser_params. sigma,
                )
                
                grad_norm = torch. norm(gradient).item()
                full_norm_sq += grad_norm ** 2
                
            else:
                continue
            
            total_norm_sq += grad_norm ** 2
            
            gradients[name] = GradientEstimate(
                name=name,
                gradient=gradient,
                update_type=map_type,
                gradient_norm=grad_norm,
                gradient_mean=gradient. mean().item(),
                gradient_std=gradient.std().item(),
            )
        
        return PopulationGradients(
            epoch=epoch,
            gradients=gradients,
            total_gradient_norm=math.sqrt(total_norm_sq),
            lora_gradient_norm=math. sqrt(lora_norm_sq),
            full_gradient_norm=math.sqrt(full_norm_sq),
        )
    
    def estimate_gradients_memory_efficient(
        self,
        population_rewards: PopulationRewards,
        epoch: int,
    ) -> PopulationGradients:
        """
        Memory-efficient gradient estimation. 
        
        Instead of regenerating all perturbations at once,
        accumulate gradient contributions one member at a time. 
        
        Memory: O(param_size) instead of O(N * param_size)
        """
        gradients = {}
        fitnesses = population_rewards.aggregated_rewards
        population_size = len(fitnesses)
        sigma = self.noiser_params.sigma
        rank = self.frozen_noiser_params. rank
        
        # Initialize gradient accumulators
        for name, param in self.params.items():
            map_type = self. es_map.get(name, ESMapType.FROZEN)
            
            if map_type == ESMapType.FROZEN:
                continue
            if map_type == ESMapType.FULL and self.frozen_noiser_params.freeze_nonlora:
                continue
                
            gradients[name] = torch.zeros_like(param)
        
        # Accumulate gradients one population member at a time
        for member_idx in range(population_size):
            R_i = fitnesses[member_idx]
            
            for name in gradients. keys():
                map_type = self. es_map.get(name, ESMapType.FROZEN)
                
                if map_type == ESMapType.LORA and len(self.param_shapes[name]) == 2:
                    # Regenerate LoRA perturbation
                    pert = self._regenerate_lora_perturbation(name, epoch, member_idx)
                    # Accumulate: gradient += R_i * (A @ B. T)
                    gradients[name] += R_i * (pert.A @ pert.B.T)
                    
                elif map_type == ESMapType. FULL:
                    # Regenerate full perturbation
                    noise = self._regenerate_full_perturbation(name, epoch, member_idx)
                    # Accumulate: gradient += R_i * noise / sigma
                    gradients[name] += R_i * noise / sigma
        
        # Normalize by population size
        for name in gradients.keys():
            gradients[name] /= population_size
        
        # Create result objects
        result_gradients = {}
        total_norm_sq = 0.0
        lora_norm_sq = 0.0
        full_norm_sq = 0.0
        
        for name, gradient in gradients.items():
            map_type = self.es_map.get(name, ESMapType.FROZEN)
            grad_norm = torch. norm(gradient).item()
            
            if map_type == ESMapType. LORA:
                lora_norm_sq += grad_norm ** 2
            else:
                full_norm_sq += grad_norm ** 2
            total_norm_sq += grad_norm ** 2
            
            result_gradients[name] = GradientEstimate(
                name=name,
                gradient=gradient,
                update_type=map_type,
                gradient_norm=grad_norm,
                gradient_mean=gradient.mean().item(),
                gradient_std=gradient.std().item(),
            )
        
        return PopulationGradients(
            epoch=epoch,
            gradients=result_gradients,
            total_gradient_norm=math.sqrt(total_norm_sq),
            lora_gradient_norm=math. sqrt(lora_norm_sq),
            full_gradient_norm=math.sqrt(full_norm_sq),
        )


# ============================================================================
# Optimizers for Parameter Update (Step 6)
# ============================================================================

class ESOptimizer(ABC):
    """Abstract base class for ES optimizers."""
    
    @abstractmethod
    def step(
        self,
        params: Dict[str, torch.Tensor],
        gradients: PopulationGradients,
        state: OptimizerState,
    ) -> Tuple[Dict[str, torch. Tensor], OptimizerState]:
        """
        Apply gradient update to parameters.
        
        Args:
            params: Current parameters
            gradients: Gradient estimates from Step 5
            state: Optimizer state
            
        Returns:
            Updated parameters and optimizer state
        """
        pass


class SGDOptimizer(ESOptimizer):
    """
    Simple SGD optimizer for ES.
    
    Update rule: θ = θ + lr * Δθ
    (Note: We ADD because ES gradient is in direction of improvement)
    """
    
    def __init__(self, lr: float = 1. 0, momentum: float = 0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            lr: Learning rate
            momentum: Momentum coefficient (0 = no momentum)
        """
        self.lr = lr
        self.momentum = momentum
        
    def step(
        self,
        params: Dict[str, torch.Tensor],
        gradients: PopulationGradients,
        state: OptimizerState,
    ) -> Tuple[Dict[str, torch.Tensor], OptimizerState]:
        """Apply SGD update."""
        new_params = {}
        new_momentum = {} if self.momentum > 0 else None
        
        for name, param in params.items():
            if name not in gradients.gradients:
                new_params[name] = param
                continue
                
            grad = gradients.gradients[name]. gradient
            
            if self.momentum > 0:
                # Apply momentum
                if state.momentum is not None and name in state.momentum:
                    m = self.momentum * state.momentum[name] + grad
                else:
                    m = grad
                new_momentum[name] = m
                update = self.lr * m
            else:
                update = self.lr * grad
            
            # Add gradient (ES maximizes reward)
            new_params[name] = param + update
        
        new_state = OptimizerState(
            step=state.step + 1,
            momentum=new_momentum,
            velocity=state.velocity,
        )
        
        return new_params, new_state


class AdamOptimizer(ESOptimizer):
    """
    Adam optimizer for ES. 
    
    Maintains exponential moving averages of gradients and squared gradients.
    """
    
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """
        Initialize Adam optimizer. 
        
        Args:
            lr: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            eps: Small constant for numerical stability
        """
        self. lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
    def step(
        self,
        params: Dict[str, torch.Tensor],
        gradients: PopulationGradients,
        state: OptimizerState,
    ) -> Tuple[Dict[str, torch. Tensor], OptimizerState]:
        """Apply Adam update."""
        new_params = {}
        new_momentum = {}
        new_velocity = {}
        
        t = state.step + 1  # Time step (1-indexed for bias correction)
        
        for name, param in params.items():
            if name not in gradients.gradients:
                new_params[name] = param
                if state.momentum is not None and name in state. momentum:
                    new_momentum[name] = state.momentum[name]
                if state.velocity is not None and name in state.velocity:
                    new_velocity[name] = state. velocity[name]
                continue
            
            grad = gradients.gradients[name].gradient
            
            # Get or initialize momentum and velocity
            if state.momentum is not None and name in state. momentum:
                m = state.momentum[name]
            else:
                m = torch.zeros_like(param)
                
            if state. velocity is not None and name in state.velocity:
                v = state.velocity[name]
            else:
                v = torch.zeros_like(param)
            
            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self. beta2) * (grad ** 2)
            
            # Compute bias-corrected estimates
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            # Compute update
            update = self.lr * m_hat / (torch. sqrt(v_hat) + self.eps)
            
            # Apply update (ADD for ES)
            new_params[name] = param + update
            new_momentum[name] = m
            new_velocity[name] = v
        
        new_state = OptimizerState(
            step=t,
            momentum=new_momentum,
            velocity=new_velocity,
        )
        
        return new_params, new_state


# ============================================================================
# Combined Update Function (matches original _do_update)
# ============================================================================

class EggrollUpdater:
    """
    Combined gradient estimation and parameter update. 
    
    Mirrors the `_do_update` function in the original JAX implementation.
    """
    
    def __init__(
        self,
        params: Dict[str, torch.Tensor],
        es_map: Dict[str, int],
        base_evo_keys: Dict[str, RandomKeyGenerator],
        frozen_noiser_params: FrozenNoiserParams,
        noiser_params: NoiserParams,
        optimizer: Optional[ESOptimizer] = None,
        device: str = "cuda",
    ):
        """
        Initialize updater. 
        
        Args:
            params: Model parameters
            es_map: Parameter classification map
            base_evo_keys: Random keys for parameters
            frozen_noiser_params: Frozen noiser config
            noiser_params: Mutable noiser state
            optimizer: Optimizer for updates (default: SGD with lr=lr_scale)
            device: Computation device
        """
        self.params = params
        self.es_map = es_map
        self.base_evo_keys = base_evo_keys
        self.frozen_noiser_params = frozen_noiser_params
        self.noiser_params = noiser_params
        self.device = device
        
        # Initialize gradient estimator
        self.gradient_estimator = EggrollGradientEstimator(
            params=params,
            es_map=es_map,
            base_evo_keys=base_evo_keys,
            frozen_noiser_params=frozen_noiser_params,
            noiser_params=noiser_params,
            device=device,
        )
        
        # Initialize optimizer
        if optimizer is None:
            self. optimizer = SGDOptimizer(
                lr=frozen_noiser_params.solver_kwargs.get('lr', 1.0)
            )
        else:
            self.optimizer = optimizer
            
        # Initialize optimizer state
        self. opt_state = OptimizerState(step=0)
        
    def convert_fitnesses(
        self,
        raw_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Convert raw scores to shaped fitnesses.
        
        Mirrors NOISER.convert_fitnesses from original implementation.
        
        Args:
            raw_scores: Raw reward scores, shape (generations_per_prompt,)
            
        Returns:
            Shaped fitness scores
        """
        group_size = self.frozen_noiser_params.group_size
        
        if group_size == 0:
            # Global normalization
            mean = np.mean(raw_scores)
            std = np.std(raw_scores) + 1e-5
            true_scores = (raw_scores - mean) / std
        else:
            # Group-wise normalization
            group_scores = raw_scores.reshape(-1, group_size)
            group_mean = np. mean(group_scores, axis=-1, keepdims=True)
            global_std = np.std(raw_scores) + 1e-5
            true_scores = (group_scores - group_mean) / global_std
            true_scores = true_scores.ravel()
            
        return true_scores
    
    def do_update(
        self,
        raw_scores: np. ndarray,
        epoch: int,
    ) -> Tuple[Dict[str, torch. Tensor], Dict[str, float]]:
        """
        Perform complete ES update (Steps 5 & 6). 
        
        Mirrors `_do_update` from original implementation. 
        
        Args:
            raw_scores: Raw aggregated scores from evaluation
            epoch: Current epoch
            
        Returns:
            new_params: Updated parameters
            stats: Update statistics
        """
        # Step 5a: Convert raw scores to shaped fitnesses
        fitnesses = self.convert_fitnesses(raw_scores)
        
        # Create PopulationRewards object for gradient estimator
        population_rewards = PopulationRewards(
            epoch=epoch,
            rewards={},
            mean_rewards={},
            aggregated_rewards=fitnesses,
            best_member_idx=int(np.argmax(fitnesses)),
            best_reward=float(np.max(raw_scores)),
            worst_member_idx=int(np.argmin(fitnesses)),
            worst_reward=float(np. min(raw_scores)),
            mean_reward=float(np.mean(raw_scores)),
            std_reward=float(np.std(raw_scores)),
        )
        
        # Step 5b: Estimate gradients
        gradients = self.gradient_estimator.estimate_gradients_memory_efficient(
            population_rewards=population_rewards,
            epoch=epoch,
        )
        
        # Step 6: Apply optimizer update
        new_params, self.opt_state = self.optimizer.step(
            params=self.params,
            gradients=gradients,
            state=self.opt_state,
        )
        
        # Compute parameter differences (for logging)
        param_diffs = {}
        lora_diff_sum = 0.0
        lora_count = 0
        full_diff_sum = 0.0
        full_count = 0
        
        for name in self.params. keys():
            if name in new_params:
                diff = torch.sqrt(torch.mean((new_params[name] - self. params[name]) ** 2)). item()
                param_diffs[name] = diff
                
                map_type = self. es_map.get(name, ESMapType.FROZEN)
                if map_type == ESMapType.LORA:
                    lora_diff_sum += diff
                    lora_count += 1
                elif map_type == ESMapType.FULL:
                    full_diff_sum += diff
                    full_count += 1
        
        # Update internal params reference
        self.params = new_params
        
        # Compile statistics
        stats = {
            "gradient_norm": gradients.total_gradient_norm,
            "lora_gradient_norm": gradients.lora_gradient_norm,
            "full_gradient_norm": gradients.full_gradient_norm,
            "lora_param_diff": lora_diff_sum / max(lora_count, 1),
            "full_param_diff": full_diff_sum / max(full_count, 1),
            "optimizer_step": self. opt_state.step,
        }
        
        return new_params, stats


# ============================================================================
# Utility Functions
# ============================================================================

def compute_parameter_differences(
    old_params: Dict[str, torch. Tensor],
    new_params: Dict[str, torch. Tensor],
    es_map: Dict[str, int],
) -> Dict[str, Any]:
    """
    Compute RMSE differences between old and new parameters.
    
    Matches the parameter difference computation in original code:
    jax.tree. map(lambda x, y: jnp.sqrt(jnp.mean((x - y) ** 2)), params, new_params)
    """
    diffs = {}
    lora_total = 0.0
    lora_count = 0
    full_total = 0.0
    full_count = 0
    
    for name in old_params.keys():
        if name in new_params:
            rmse = torch.sqrt(torch. mean((new_params[name] - old_params[name]) ** 2)).item()
            diffs[name] = rmse
            
            map_type = es_map. get(name, ESMapType.FROZEN)
            if map_type == ESMapType.LORA:
                lora_total += rmse
                lora_count += 1
            elif map_type == ESMapType. FULL:
                full_total += rmse
                full_count += 1
    
    return {
        "per_param": diffs,
        "lora_mean": lora_total / max(lora_count, 1),
        "full_mean": full_total / max(full_count, 1),
        "lora_count": lora_count,
        "full_count": full_count,
    }


def print_gradient_summary(gradients: PopulationGradients):
    """Print summary of gradient estimates."""
    print("\n" + "=" * 60)
    print(f"Gradient Summary (Epoch {gradients.epoch})")
    print("=" * 60)
    
    print(f"\nGradient Norms:")
    print(f"  Total:     {gradients. total_gradient_norm:.6f}")
    print(f"  LoRA:      {gradients.lora_gradient_norm:. 6f}")
    print(f"  Full:      {gradients.full_gradient_norm:.6f}")
    
    print(f"\nPer-Parameter Gradients:")
    for name, grad_est in sorted(gradients.gradients.items())[:10]:  # Show first 10
        type_str = "LORA" if grad_est.update_type == ESMapType.LORA else "FULL"
        print(f"  {name[:50]:50s} [{type_str}] norm={grad_est.gradient_norm:.6f}")
    
    if len(gradients. gradients) > 10:
        print(f"  ... and {len(gradients.gradients) - 10} more parameters")


def print_update_summary(
    old_params: Dict[str, torch. Tensor],
    new_params: Dict[str, torch. Tensor],
    es_map: Dict[str, int],
    stats: Dict[str, Any],
):
    """Print summary of parameter updates."""
    print("\n" + "=" * 60)
    print("Parameter Update Summary")
    print("=" * 60)
    
    diffs = compute_parameter_differences(old_params, new_params, es_map)
    
    print(f"\nUpdate Statistics:")
    print(f"  Optimizer step: {stats. get('optimizer_step', 'N/A')}")
    print(f"  Gradient norm: {stats.get('gradient_norm', 0):.6f}")
    
    print(f"\nParameter Changes (RMSE):")
    print(f"  LoRA params ({diffs['lora_count']}): {diffs['lora_mean']:.8f}")
    print(f"  Full params ({diffs['full_count']}): {diffs['full_mean']:. 8f}")
    
    print(f"\nLargest Changes:")
    sorted_diffs = sorted(diffs['per_param'].items(), key=lambda x: -x[1])[:5]
    for name, diff in sorted_diffs:
        print(f"  {name[:50]:50s}: {diff:.8f}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EGGROLL Steps 5 & 6: Gradient Estimation and Parameter Update")
    print("=" * 70)
    
    # Simulated setup (in practice, comes from Steps 1-4)
    # Create mock parameters
    params = {
        "encoder. layer.0.self_attn.q_proj.weight": torch.randn(1024, 1024),
        "encoder.layer.0.self_attn.k_proj. weight": torch.randn(1024, 1024),
        "encoder.layer.0.self_attn.v_proj.weight": torch.randn(1024, 1024),
        "encoder.layer.0. self_attn. out_proj.weight": torch.randn(1024, 1024),
        "encoder.layer. 0.fc1.weight": torch.randn(4096, 1024),
        "encoder.layer.0. fc2.weight": torch. randn(1024, 4096),
        "encoder.layer.0.self_attn.q_proj.bias": torch.randn(1024),
    }
    
    # ES map
    es_map = {
        "encoder.layer.0. self_attn. q_proj.weight": ESMapType. LORA,
        "encoder.layer.0.self_attn.k_proj. weight": ESMapType.LORA,
        "encoder. layer.0.self_attn.v_proj.weight": ESMapType. LORA,
        "encoder.layer. 0.self_attn.out_proj.weight": ESMapType.LORA,
        "encoder.layer.0. fc1.weight": ESMapType. LORA,
        "encoder.layer.0.fc2.weight": ESMapType. LORA,
        "encoder.layer.0.self_attn.q_proj.bias": ESMapType.FULL,
    }
    
    # Random keys
    base_key = RandomKeyGenerator(seed=42)
    base_evo_keys = {name: base_key. fold_in(i) for i, name in enumerate(params. keys())}
    
    # Noiser params
    frozen_noiser_params = FrozenNoiserParams(
        group_size=8,
        freeze_nonlora=True,
        noise_reuse=1,
        rank=16,
        solver_type="sgd",
        solver_kwargs={"lr": 1.0},
    )
    
    noiser_params = NoiserParams(
        sigma=1e-3,
        opt_state=OptimizerState(step=0),
    )
    
    device = "cpu"  # Use CPU for demo
    
    # Move params to device
    params = {k: v.to(device) for k, v in params.items()}
    
    print("\n" + "-" * 70)
    print("Setup Complete")
    print("-" * 70)
    print(f"Parameters: {len(params)}")
    print(f"LoRA params: {sum(1 for v in es_map.values() if v == ESMapType.LORA)}")
    print(f"Rank: {frozen_noiser_params.rank}")
    print(f"Sigma: {noiser_params.sigma}")
    
    # Initialize updater
    updater = EggrollUpdater(
        params=params,
        es_map=es_map,
        base_evo_keys=base_evo_keys,
        frozen_noiser_params=frozen_noiser_params,
        noiser_params=noiser_params,
        optimizer=SGDOptimizer(lr=1.0),
        device=device,
    )
    
    # Simulated raw scores from evaluation (Step 4)
    # These are aggregated scores per direction (generations_per_prompt)
    population_size = 8  # generations_per_prompt
    raw_scores = np. array([0.65, 0.42, 0.15, 0.38, 0.55, 0.48, 0.22, 0.35])
    
    print(f"\nRaw scores: {raw_scores}")
    print(f"Mean: {raw_scores.mean():.4f}, Std: {raw_scores.std():.4f}")
    
    # Step 5: Convert fitnesses
    print("\n" + "-" * 70)
    print("Step 5: Fitness Shaping")
    print("-" * 70)
    
    shaped_fitnesses = updater. convert_fitnesses(raw_scores)
    print(f"Shaped fitnesses: {shaped_fitnesses}")
    print(f"Sum: {shaped_fitnesses.sum():. 6f} (should be ~0 for centered)")
    
    # Step 5 & 6: Full update
    print("\n" + "-" * 70)
    print("Steps 5 & 6: Gradient Estimation and Parameter Update")
    print("-" * 70)
    
    old_params = {k: v.clone() for k, v in params.items()}
    
    new_params, stats = updater.do_update(
        raw_scores=raw_scores,
        epoch=0,
    )
    
    print_update_summary(old_params, new_params, es_map, stats)
    
    # Verify update happened
    print("\n" + "-" * 70)
    print("Verification")
    print("-" * 70)
    
    sample_param = "encoder.layer.0.self_attn.q_proj. weight"
    old = old_params[sample_param]
    new = new_params[sample_param]
    
    print(f"\nParameter: {sample_param}")
    print(f"  Old mean: {old.mean(). item():.8f}")
    print(f"  New mean: {new. mean().item():. 8f}")
    print(f"  Difference norm: {torch.norm(new - old). item():.8f}")
    print(f"  Max abs change: {torch.max(torch.abs(new - old)).item():.8f}")
    
    # Run multiple epochs
    print("\n" + "-" * 70)
    print("Multi-Epoch Training Simulation")
    print("-" * 70)
    
    for epoch in range(5):
        # Simulate different scores each epoch
        raw_scores = np.random.uniform(0. 1, 0.7, size=population_size)
        raw_scores[0] += 0.1  # Make first direction slightly better
        
        new_params, stats = updater.do_update(
            raw_scores=raw_scores,
            epoch=epoch,
        )
        
        print(f"Epoch {epoch}: grad_norm={stats['gradient_norm']:. 6f}, "
              f"lora_diff={stats['lora_param_diff']:.8f}, "
              f"step={stats['optimizer_step']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Steps 5 & 6 Complete!")
    print("=" * 70)
    print(f"""
Summary:
--------
• Gradient estimation using ES formula: Δθ = (1/N) Σ R_i * (A_i @ B_i^T)
• Memory-efficient: Regenerates perturbations on-the-fly from seeds
• Optimizer: SGD (can also use Adam)
• Fitness shaping: Centered normalization

Key Functions:
--------------
1. convert_fitnesses(raw_scores) -> shaped_fitnesses
2.  estimate_gradients(population_rewards, epoch) -> PopulationGradients
3. optimizer. step(params, gradients, state) -> new_params, new_state
4. do_update(raw_scores, epoch) -> new_params, stats (combined Steps 5 & 6)

Training Loop:
--------------
for epoch in range(num_epochs):
    # Step 2-3: Generate perturbed models and forward pass
    population = generator.generate_population(epoch, N, inputs)
    
    # Step 4: Compute rewards
    rewards = aggregator.compute_rewards(population, references)
    
    # Step 5-6: Update parameters
    new_params, stats = updater.do_update(rewards. aggregated_rewards, epoch)
""")