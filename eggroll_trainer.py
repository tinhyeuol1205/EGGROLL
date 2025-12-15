import os
import sys
import csv
import time
import math
import operator
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict, List, Any, Tuple, Callable
import warnings

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not installed. Tracking disabled.")

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    raise ImportError("transformers is required. Install with: pip install transformers")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EggrollTrainerConfig:
    """
    Complete configuration for EGGROLL training.
    Mirrors Args from the original JAX implementation.
    """
    # Random seed
    seed: int = 0
    
    # Model configuration
    model_name: str = "Helsinki-NLP/opus-mt-en-vi"
    dtype: Optional[str] = "float32"  # "float32", "float16", "bfloat16"
    
    # Output directories
    output_directory: str = "./outputs"
    save_path: str = "./checkpoints"
    load_path: Optional[str] = None
    
    # Save/Load options
    save_model: bool = True
    load_model: bool = False
    
    # Generation settings
    num_beams: int = 1
    
    # EGGROLL core hyperparameters
    sigma: float = 1e-3              # Noise standard deviation (σ)
    lr_scale: float = 1.0            # Learning rate (α)
    rank: int = 16                   # Low-rank dimension (r)
    noise_reuse: int = 1             # Reuse noise across epochs
    freeze_nonlora: bool = True      # Freeze non-LoRA parameters
    
    # Population settings
    generations_per_prompt: int = 8  # N: population size per unique prompt
    prompts_per_epoch: int = 8       # Number of unique prompts per epoch
    
    # Training settings
    num_epochs: int = 100
    validate_every: int = 10
    save_every: int = 100
    log_every: int = 1
    log_samples_every: int = 10
    
    # Validation settings
    validation_samples: int = 100
    
    # Optimizer settings
    optimizer_type: str = "sgd"      # "sgd" or "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    momentum: float = 0.0
    
    # Reward settings
    reward_metric: str = "bleu"      # "bleu", "meteor", "chrf", "composite"
    fitness_shaping: str = "centered_rank"  # "none", "standardize", "centered_rank"
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging settings
    track: bool = False
    wandb_project: str = "EGGROLL-Translation"
    wandb_name: str = "eggroll-run"
    wandb_mode: Literal["online", "offline"] = "online"
    
    @property
    def total_generations_per_epoch(self) -> int:
        return self.generations_per_prompt * self.prompts_per_epoch
    
    @property
    def experiment_name(self) -> str:
        return f"{self.wandb_name}_lr={self.lr_scale}_sigma={self.sigma:.2e}_rank={self.rank}"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrainingStats:
    """Statistics for a single epoch"""
    epoch: int
    
    # Fitness statistics
    avg_fitness: float = 0.0
    std_fitness: float = 0.0
    max_fitness: float = 0.0
    min_fitness: float = 0.0
    median_fitness: float = 0.0
    
    # Update statistics
    lora_param_diff: float = 0.0
    full_param_diff: float = 0.0
    gradient_norm: float = 0.0
    
    # Timing
    prompt_time: float = 0.0
    generation_time: float = 0.0
    fitness_time: float = 0.0
    update_time: float = 0.0
    validation_time: float = 0.0
    saving_time: float = 0.0
    total_time: float = 0.0
    
    # Validation
    validation_score: Optional[float] = None
    
    # Running averages
    true_train_avg_fitness: float = 0.0


@dataclass
class Checkpoint:
    """Checkpoint data structure"""
    epoch: int
    params: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    es_map: Dict[str, int]
    config: Dict[str, Any]
    stats: Dict[str, float]
    timestamp: str


# ============================================================================
# Random Key Generator (JAX-style)
# ============================================================================

class RandomKeyGenerator:
    """JAX-style random key generator for reproducible noise generation."""
    
    def __init__(self, seed: int):
        self.seed = seed
        
    def fold_in(self, key_id: int) -> 'RandomKeyGenerator':
        new_seed = ((self.seed * 31337) + key_id) % (2**31)
        return RandomKeyGenerator(new_seed)
    
    def split(self, num_keys: int) -> List['RandomKeyGenerator']:
        return [self.fold_in(i) for i in range(num_keys)]


# ============================================================================
# ES Map Types
# ============================================================================

class ESMapType:
    FULL = 0
    LORA = 1
    FROZEN = 2
    NOOP = 3


# ============================================================================
# Optimizer State
# ============================================================================

@dataclass
class OptimizerState:
    step: int = 0
    momentum: Optional[Dict[str, torch.Tensor]] = None
    velocity: Optional[Dict[str, torch.Tensor]] = None


# ============================================================================
# EGGROLL Trainer Class
# ============================================================================

class EggrollTrainer:
    """
    Complete EGGROLL trainer for translation model finetuning.
    
    Implements the full training loop from the paper:
    "Evolution Strategies at Hyperscale" (arXiv:2511.16652)
    """
    
    def __init__(self, config: EggrollTrainerConfig):
        """
        Initialize the EGGROLL trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = config.device
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Initialize components (will be set in setup())
        self.model = None
        self.tokenizer = None
        self.params = None
        self.es_map = None
        self.base_evo_keys = None
        self.opt_state = None
        self.reward_function = None
        
        # Training state
        self.current_epoch = 0
        self.true_train_fitness_sum = 0.0
        self.best_validation_score = -float('inf')
        
        # Timing
        self.start_time = None
        
        # Logging
        self.wandb_run = None
        
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    # ========================================================================
    # Step 1: Initialization
    # ========================================================================
    
    def setup(self):
        """
        Setup the trainer (Step 1: Initialization).
        
        This includes:
        - Loading the pre-trained model
        - Building ES parameter map
        - Initializing noiser parameters
        - Setting up reward function
        - Initializing wandb (if enabled)
        """
        print("=" * 70)
        print("EGGROLL Trainer Setup")
        print("=" * 70)
        
        # 1. Load model and tokenizer
        print("\n[1/6] Loading model and tokenizer...")
        self._load_model()
        
        # 2. Extract parameters and build ES map
        print("\n[2/6] Building ES parameter map...")
        self._build_es_map()
        
        # 3. Initialize random keys
        print("\n[3/6] Initializing random keys...")
        self._init_random_keys()
        
        # 4.Initialize optimizer state
        print("\n[4/6] Initializing optimizer...")
        self._init_optimizer()
        
        # 5.Setup reward function
        print("\n[5/6] Setting up reward function...")
        self._setup_reward_function()
        
        # 6. Initialize wandb
        if self.config.track:
            print("\n[6/6] Initializing wandb...")
            self._init_wandb()
        else:
            print("\n[6/6] Wandb tracking disabled.")
            
        # 7.Load checkpoint if specified
        if self.config.load_model and self.config.load_path:
            print(f"\nLoading checkpoint from: {self.config.load_path}")
            self._load_checkpoint(self.config.load_path)
            
        self._print_setup_summary()
        
    def _load_model(self):
        """Load pre-trained model and tokenizer."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float32)
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set model to eval mode (we don't use gradients)
        self.model.eval()
        
        # Extract parameters
        self.params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _build_es_map(self):
        """Build ES parameter classification map."""
        lora_targets = [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2",
        ]
        
        self.es_map = {}
        lora_count = 0
        full_count = 0
        frozen_count = 0
        
        for name, param in self.params.items():
            # Freeze embeddings and layer norms
            if "embed" in name.lower():
                self.es_map[name] = ESMapType.FROZEN
                frozen_count += 1
            elif "layer_norm" in name.lower() or "layernorm" in name.lower():
                self.es_map[name] = ESMapType.FROZEN
                frozen_count += 1
            # Biases get full updates (if not frozen)
            elif "bias" in name.lower():
                if self.config.freeze_nonlora:
                    self.es_map[name] = ESMapType.FROZEN
                    frozen_count += 1
                else:
                    self.es_map[name] = ESMapType.FULL
                    full_count += 1
            # Check for LoRA targets (2D weight matrices)
            elif any(target in name.lower() for target in lora_targets) and len(param.shape) == 2:
                self.es_map[name] = ESMapType.LORA
                lora_count += 1
            else:
                if self.config.freeze_nonlora:
                    self.es_map[name] = ESMapType.FROZEN
                    frozen_count += 1
                else:
                    self.es_map[name] = ESMapType.FULL
                    full_count += 1
                    
        print(f"  LoRA parameters: {lora_count}")
        print(f"  Full parameters: {full_count}")
        print(f"  Frozen parameters: {frozen_count}")
        
    def _init_random_keys(self):
        """Initialize random keys for each parameter."""
        master_key = RandomKeyGenerator(self.config.seed)
        self.base_model_key = master_key.fold_in(0)
        self.base_gen_key = master_key.fold_in(1)
        self.base_valid_key = master_key.fold_in(2)
        
        self.base_evo_keys = {
            name: self.base_model_key.fold_in(i)
            for i, name in enumerate(self.params.keys())
        }
        
    def _init_optimizer(self):
        """Initialize optimizer state."""
        self.opt_state = OptimizerState(step=0)
        
        if self.config.optimizer_type == "adam":
            self.opt_state.momentum = {
                name: torch.zeros_like(p)
                for name, p in self.params.items()
                if self.es_map.get(name, ESMapType.FROZEN) != ESMapType.FROZEN
            }
            self.opt_state.velocity = {
                name: torch.zeros_like(p)
                for name, p in self.params.items()
                if self.es_map.get(name, ESMapType.FROZEN) != ESMapType.FROZEN
            }
        elif self.config.momentum > 0:
            self.opt_state.momentum = {
                name: torch.zeros_like(p)
                for name, p in self.params.items()
                if self.es_map.get(name, ESMapType.FROZEN) != ESMapType.FROZEN
            }
            
    def _setup_reward_function(self):
        """Setup reward function for evaluation."""
        metric = self.config.reward_metric.lower()
        
        if metric == "bleu":
            try:
                import sacrebleu
                self._sacrebleu = sacrebleu
                self.reward_function = self._compute_bleu
                print(f"  Reward: BLEU (sacrebleu)")
            except ImportError:
                self.reward_function = self._compute_bleu_nltk
                print(f"  Reward: BLEU (nltk)")
        elif metric == "length":
            self.reward_function = self._compute_length_ratio
            print(f"  Reward: Length Ratio")
        else:
            self.reward_function = self._compute_bleu_nltk
            print(f"  Reward: BLEU (nltk, fallback)")


    def _init_wandb(self):
        """Initialize wandb tracking."""
        if not WANDB_AVAILABLE:
            print("  wandb not available, skipping.")
            return
            
        if self.config.wandb_mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
            
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.experiment_name,
            config=asdict(self.config),
        )
        print(f"  wandb run: {self.wandb_run.name}")
        
    def _print_setup_summary(self):
        """Print setup summary."""
        print("\n" + "=" * 70)
        print("Setup Complete!")
        print("=" * 70)
        print(f"""
Configuration:
  Model: {self.config.model_name}
  Device: {self.device}
  
EGGROLL Hyperparameters:
  σ (sigma): {self.config.sigma}
  α (learning rate): {self.config.lr_scale}
  r (rank): {self.config.rank}
  N (population per prompt): {self.config.generations_per_prompt}
  
Training:
  Epochs: {self.config.num_epochs}
  Prompts per epoch: {self.config.prompts_per_epoch}
  Total generations per epoch: {self.config.total_generations_per_epoch}
  Reward metric: {self.config.reward_metric}
""")

    # ========================================================================
    # Step 2 & 3: Perturbation and Forward Pass
    # ========================================================================
    
    def _get_perturbation_seed(self, base_seed: int, epoch: int, member_idx: int) -> int:
        """Get deterministic seed for perturbation."""
        if self.config.noise_reuse > 0:
            effective_epoch = epoch // self.config.noise_reuse
        else:
            effective_epoch = epoch
        return ((base_seed * 31337) + effective_epoch * 1000 + member_idx) % (2**31)
    
    def _generate_lora_perturbation(
        self,
        param_shape: Tuple[int, int],
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate low-rank perturbation matrices A and B."""
        out_features, in_features = param_shape
        rank = self.config.rank
        sigma = self.config.sigma
        
        gen_A = torch.Generator().manual_seed(seed)
        gen_B = torch.Generator().manual_seed(seed + 1)
        
        # Scale by σ/√r
        scale = sigma / math.sqrt(rank)
        
        A = torch.randn(out_features, rank, generator=gen_A) * scale
        B = torch.randn(in_features, rank, generator=gen_B)
        
        return A.to(self.device), B.to(self.device)
    
    @torch.no_grad()
    def _generate_with_perturbation(
        self,
        input_ids: torch.Tensor,
        epoch: int,
        member_idx: int,
    ) -> torch.Tensor:
        """
        Generate translation with perturbed model (Steps 2 & 3).
        
        Applies perturbation by directly modifying weights temporarily.
        """
        # Store original weights
        original_weights = {}
        
        # Apply perturbations
        for name, param in self.model.named_parameters():
            map_type = self.es_map.get(name, ESMapType.FROZEN)
            
            if map_type == ESMapType.FROZEN:
                continue
                
            original_weights[name] = param.data.clone()
            base_seed = self.base_evo_keys[name].seed
            seed = self._get_perturbation_seed(base_seed, epoch, member_idx)
            
            if map_type == ESMapType.LORA and len(param.shape) == 2:
                A, B = self._generate_lora_perturbation(param.shape, seed)
                param.data = param.data + A @ B.T
            elif map_type == ESMapType.FULL:
                gen = torch.Generator().manual_seed(seed)
                noise = torch.randn_like(param, generator=gen) * self.config.sigma
                param.data = param.data + noise.to(self.device)
        
        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=self.config.num_beams,
        )
        
        # Restore original weights
        for name, original in original_weights.items():
            param = dict(self.model.named_parameters())[name]
            param.data = original
            
        return output_ids
    
    # ========================================================================
    # Step 4: Reward Computation
    # ========================================================================
    
    def _compute_bleu(self, hypothesis: str, reference: str) -> float:
        """Compute BLEU score using sacrebleu."""
        if not hypothesis.strip() or not reference.strip():
            return 0.0
        try:
            bleu = self._sacrebleu.sentence_bleu(hypothesis, [reference], smooth_method='exp')
            return bleu.score / 100.0
        except:
            return 0.0
    
    def _compute_bleu_nltk(self, hypothesis: str, reference: str) -> float:
        """Compute BLEU score using nltk."""
        if not hypothesis.strip() or not reference.strip():
            return 0.0
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            hyp_tokens = hypothesis.lower().split()
            ref_tokens = reference.lower().split()
            if len(hyp_tokens) == 0:
                return 0.0
            return sentence_bleu([ref_tokens], hyp_tokens, 
                               smoothing_function=SmoothingFunction().method1)
        except:
            return 0.0
    
    def _compute_length_ratio(self, hypothesis: str, reference: str) -> float:
        """Compute length ratio reward."""
        if not reference.strip():
            return 0.0
        hyp_len = len(hypothesis.split())
        ref_len = len(reference.split())
        if ref_len == 0:
            return 0.0
        ratio = hyp_len / ref_len
        return max(0.0, 1.0 - abs(ratio - 1.0))
    
    def _compute_rewards(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> np.ndarray:
        """Compute rewards for all hypotheses."""
        rewards = np.array([
            self.reward_function(hyp, ref)
            for hyp, ref in zip(hypotheses, references)
        ])
        return rewards
    
    # ========================================================================
    # Step 5: Fitness Shaping
    # ========================================================================
    
    def _shape_fitnesses(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Apply fitness shaping for ES stability.
        
        Mirrors NOISER.convert_fitnesses from original code.
        """
        if self.config.fitness_shaping == "none":
            return raw_scores
            
        elif self.config.fitness_shaping == "standardize":
            mean = np.mean(raw_scores)
            std = np.std(raw_scores) + 1e-8
            return (raw_scores - mean) / std
            
        elif self.config.fitness_shaping == "centered_rank":
            n = len(raw_scores)
            ranks = np.argsort(np.argsort(raw_scores))
            shaped = (ranks.astype(np.float32) + 0.5) / n - 0.5
            return shaped
            
        else:
            # Group-wise normalization (for multiple prompts)
            group_size = self.config.generations_per_prompt
            if group_size > 0 and len(raw_scores) > group_size:
                group_scores = raw_scores.reshape(-1, group_size)
                group_mean = np.mean(group_scores, axis=-1, keepdims=True)
                global_std = np.std(raw_scores) + 1e-8
                shaped = (group_scores - group_mean) / global_std
                return shaped.ravel()
            else:
                mean = np.mean(raw_scores)
                std = np.std(raw_scores) + 1e-8
                return (raw_scores - mean) / std
    
    # ========================================================================
    # Steps 5 & 6: Gradient Estimation and Update
    # ========================================================================
    
    def _estimate_and_update(
        self,
        shaped_fitnesses: np.ndarray,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Estimate gradients and update parameters (Steps 5 & 6).
        
        Mirrors _do_update from original code.
        """
        population_size = len(shaped_fitnesses)
        stats = {}
        
        new_params = {}
        lora_diff_sum = 0.0
        lora_count = 0
        full_diff_sum = 0.0
        full_count = 0
        total_grad_norm_sq = 0.0
        
        for name, param in self.params.items():
            map_type = self.es_map.get(name, ESMapType.FROZEN)
            
            if map_type == ESMapType.FROZEN:
                new_params[name] = param
                continue
                
            # Estimate gradient
            gradient = torch.zeros_like(param)
            
            for member_idx in range(population_size):
                R_i = shaped_fitnesses[member_idx]
                base_seed = self.base_evo_keys[name].seed
                seed = self._get_perturbation_seed(base_seed, epoch, member_idx)
                
                if map_type == ESMapType.LORA and len(param.shape) == 2:
                    A, B = self._generate_lora_perturbation(param.shape, seed)
                    gradient += R_i * (A @ B.T)
                elif map_type == ESMapType.FULL:
                    gen = torch.Generator().manual_seed(seed)
                    noise = torch.randn_like(param, generator=gen) * self.config.sigma
                    gradient += R_i * noise.to(self.device) / self.config.sigma
                    
            gradient /= population_size
            
            # Apply optimizer
            update = self._apply_optimizer_step(name, gradient)
            
            # Update parameter (ADD because ES maximizes reward)
            new_param = param + update
            new_params[name] = new_param
            
            # Compute difference
            diff = torch.sqrt(torch.mean((new_param - param) ** 2)).item()
            
            if map_type == ESMapType.LORA:
                lora_diff_sum += diff
                lora_count += 1
            else:
                full_diff_sum += diff
                full_count += 1
                
            total_grad_norm_sq += torch.norm(gradient).item() ** 2
        
        # Update stored parameters
        self.params = new_params
        
        # Update model weights
        self._update_model_weights()
        
        stats['gradient_norm'] = math.sqrt(total_grad_norm_sq)
        stats['lora_param_diff'] = lora_diff_sum / max(lora_count, 1)
        stats['full_param_diff'] = full_diff_sum / max(full_count, 1)
        
        return stats
    
    def _apply_optimizer_step(
        self,
        name: str,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        """Apply optimizer step to gradient."""
        lr = self.config.lr_scale
        
        if self.config.optimizer_type == "adam":
            t = self.opt_state.step + 1
            beta1 = self.config.adam_beta1
            beta2 = self.config.adam_beta2
            eps = self.config.adam_eps
            
            m = self.opt_state.momentum.get(name, torch.zeros_like(gradient))
            v = self.opt_state.velocity.get(name, torch.zeros_like(gradient))
            
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            self.opt_state.momentum[name] = m
            self.opt_state.velocity[name] = v
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            return lr * m_hat / (torch.sqrt(v_hat) + eps)
            
        elif self.config.momentum > 0:
            m = self.opt_state.momentum.get(name, torch.zeros_like(gradient))
            m = self.config.momentum * m + gradient
            self.opt_state.momentum[name] = m
            return lr * m
            
        else:
            return lr * gradient
    
    def _update_model_weights(self):
        """Update model weights from params dictionary."""
        state_dict = self.model.state_dict()
        for name, param in self.params.items():
            if name in state_dict:
                state_dict[name] = param
        self.model.load_state_dict(state_dict)

    # ========================================================================
    # Single Epoch
    # ========================================================================
    
    def _single_epoch(
        self,
        train_data: List[Tuple[str, str]],
        epoch: int,
        val_data: Optional[List[Tuple[str, str]]] = None,
    ) -> TrainingStats:
        """
        Execute a single training epoch.
        
        Mirrors single_epoch from original code.
        """
        stats = TrainingStats(epoch=epoch)
        epoch_start = time.time()
        
        # Validation (periodic)
        if epoch % self.config.validate_every == 0 and val_data:
            val_start = time.time()
            stats.validation_score = self._validate(val_data)
            stats.validation_time = time.time() - val_start
            
            if stats.validation_score > self.best_validation_score:
                self.best_validation_score = stats.validation_score
                
        # Sample prompts for this epoch
        prompt_start = time.time()
        epoch_samples = self._sample_epoch_data(train_data, epoch)
        stats.prompt_time = time.time() - prompt_start
        
        # Generate with all population members
        gen_start = time.time()
        all_hypotheses = []
        all_references = []
        
        for source, reference in epoch_samples:
            # Tokenize
            inputs = self.tokenizer(
                source,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            # Generate for each population member
            for member_idx in range(self.config.generations_per_prompt):
                output_ids = self._generate_with_perturbation(
                    inputs["input_ids"],
                    epoch,
                    member_idx,
                )
                
                hypothesis = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                all_hypotheses.append(hypothesis)
                all_references.append(reference)
                
        stats.generation_time = time.time() - gen_start
        
        # Compute rewards (Step 4)
        fitness_start = time.time()
        raw_rewards = self._compute_rewards(all_hypotheses, all_references)
        
        # Aggregate rewards per direction
        raw_rewards = raw_rewards.reshape(
            self.config.prompts_per_epoch,
            self.config.generations_per_prompt
        ).sum(axis=0)
        
        stats.fitness_time = time.time() - fitness_start
        
        # Statistics
        stats.avg_fitness = float(np.mean(raw_rewards))
        stats.std_fitness = float(np.std(raw_rewards))
        stats.max_fitness = float(np.max(raw_rewards))
        stats.min_fitness = float(np.min(raw_rewards))
        stats.median_fitness = float(np.median(raw_rewards))
        
        # Shape fitnesses (Step 5a)
        shaped_fitnesses = self._shape_fitnesses(raw_rewards)
        
        # Estimate gradients and update (Steps 5b & 6)
        update_start = time.time()
        update_stats = self._estimate_and_update(shaped_fitnesses, epoch)
        stats.update_time = time.time() - update_start
        
        stats.lora_param_diff = update_stats['lora_param_diff']
        stats.full_param_diff = update_stats['full_param_diff']
        stats.gradient_norm = update_stats['gradient_norm']
        
        # Increment optimizer step
        self.opt_state.step += 1
        
        # Update running average
        self.true_train_fitness_sum += np.sum(raw_rewards)
        stats.true_train_avg_fitness = (
            self.true_train_fitness_sum / 
            ((epoch + 1) * self.config.generations_per_prompt)
        )
        
        # Save checkpoint (periodic)
        if self.config.save_model and epoch % self.config.save_every == 0:
            save_start = time.time()
            self._save_checkpoint(epoch, stats)
            stats.saving_time = time.time() - save_start
            
        stats.total_time = time.time() - epoch_start
        
        return stats
    
    def _sample_epoch_data(
        self,
        train_data: List[Tuple[str, str]],
        epoch: int,
    ) -> List[Tuple[str, str]]:
        """Sample data for this epoch."""
        # Use epoch as seed for reproducible sampling
        rng = np.random.RandomState(self.config.seed + epoch)
        indices = rng.choice(
            len(train_data),
            size=min(self.config.prompts_per_epoch, len(train_data)),
            replace=False,
        )
        return [train_data[i] for i in indices]
    
    # ========================================================================
    # Validation
    # ========================================================================
    
    @torch.no_grad()
    # def _validate(self, epoch: int) -> float:
    #     """Run validation without perturbation (σ=0)."""
    #     # This would use validation data
    #     # For now, return placeholder
    #     return 0.0
    
    def _validate(
        self,
        val_data: List[Tuple[str, str]],
    ) -> float:
        """
        Run validation on provided data.
        
        Uses base model without perturbation.
        """
        total_reward = 0.0
        count = 0

        for source, reference in tqdm(val_data, desc="Validating"):
            inputs = self.tokenizer(
                source,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    num_beams=self.config.num_beams,
                )
                
            hypothesis = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            reward = self.reward_function(hypothesis, reference)
            
            total_reward += reward
            count += 1
            
        return total_reward / max(count, 1)
    
    # ========================================================================
    # Checkpointing
    # ========================================================================
    
    def _save_checkpoint(self, epoch: int, stats: TrainingStats):
        """Save training checkpoint."""
        ckpt_dir = Path(self.config.save_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'params': {k: v.cpu() for k, v in self.params.items()},
            'opt_state': {
                'step': self.opt_state.step,
                'momentum': {k: v.cpu() for k, v in (self.opt_state.momentum or {}).items()},
                'velocity': {k: v.cpu() for k, v in (self.opt_state.velocity or {}).items()},
            },
            'es_map': self.es_map,
            'config': asdict(self.config),
            'stats': asdict(stats),
            'true_train_fitness_sum': self.true_train_fitness_sum,
            'best_validation_score': self.best_validation_score,
            'timestamp': datetime.now().isoformat(),
        }
        
        ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:05d}.pt"
        torch.save(checkpoint, ckpt_path)
        
        # Also save as latest
        latest_path = ckpt_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        self.model.save_pretrained(f"{ckpt_dir}/checkpoint_epoch_{epoch:05d}")
        self.tokenizer.save_pretrained(f"{ckpt_dir}/checkpoint_epoch_{epoch:05d}")

        self.model.save_pretrained(f"{ckpt_dir}/checkpoint_last")
        self.tokenizer.save_pretrained(f"{ckpt_dir}/checkpoint_last")
        
        print(f"  Checkpoint saved: {ckpt_path}")
        
    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.params = {
            k: v.to(self.device)
            for k, v in checkpoint['params'].items()
        }
        
        self.opt_state.step = checkpoint['opt_state']['step']
        if checkpoint['opt_state']['momentum']:
            self.opt_state.momentum = {
                k: v.to(self.device)
                for k, v in checkpoint['opt_state']['momentum'].items()
            }
        if checkpoint['opt_state']['velocity']:
            self.opt_state.velocity = {
                k: v.to(self.device)
                for k, v in checkpoint['opt_state']['velocity'].items()
            }
            
        self.current_epoch = checkpoint['epoch'] + 1
        self.true_train_fitness_sum = checkpoint.get('true_train_fitness_sum', 0.0)
        self.best_validation_score = checkpoint.get('best_validation_score', -float('inf'))
        
        self._update_model_weights()
        
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
     # ========================================================================
    # Logging
    # ========================================================================
    
    def _log_epoch(self, stats: TrainingStats):
        """Log epoch statistics."""
        # Console logging
        if stats.epoch % self.config.log_every == 0:
            print(f"\nEpoch {stats.epoch:5d} | "
                  f"Fitness: {stats.avg_fitness:.4f} ± {stats.std_fitness:.4f} | "
                  f"Best: {stats.max_fitness:.4f} | "
                  f"Grad: {stats.gradient_norm:.6f} | "
                  f"Time: {stats.total_time:.2f}s")
            
            if stats.validation_score is not None:
                print(f"           | Validation: {stats.validation_score:.4f} "
                      f"(Best: {self.best_validation_score:.4f})")
        
        # Wandb logging
        if self.wandb_run is not None:
            log_dict = {
                'epoch': stats.epoch,
                'avg_fitness': stats.avg_fitness,
                'std_fitness': stats.std_fitness,
                'max_fitness': stats.max_fitness,
                'min_fitness': stats.min_fitness,
                'median_fitness': stats.median_fitness,
                'lora_param_diff': stats.lora_param_diff,
                'full_param_diff': stats.full_param_diff,
                'gradient_norm': stats.gradient_norm,
                'generation_time': stats.generation_time,
                'update_time': stats.update_time,
                'total_time': stats.total_time,
                'true_train_avg_fitness': stats.true_train_avg_fitness,
            }
            
            if stats.validation_score is not None:
                log_dict['validation_score'] = stats.validation_score
                log_dict['best_validation_score'] = self.best_validation_score
                
            self.wandb_run.log(log_dict, step=stats.epoch)
    
    # ========================================================================
    # Main Training Loop
    # ========================================================================
    
    def train(
        self,
        train_data: List[Tuple[str, str]],
        val_data: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Main training loop.
        
        Args:
            train_data: List of (source, target) pairs
            val_data: Optional validation data
        """
        print("\n" + "=" * 70)
        print("Starting EGGROLL Training")
        print("=" * 70)
        
        self.start_time = time.time()
        
        try:
            for epoch in tqdm(range(self.current_epoch, self.config.num_epochs),
                            desc="Training", initial=self.current_epoch,
                            total=self.config.num_epochs):
                
                # Run single epoch
                if val_data:
                    stats = self._single_epoch(train_data, epoch, val_data)
                else:
                    stats = self._single_epoch(train_data, epoch)
                
                # Log
                self._log_epoch(stats)
                
                self.current_epoch = epoch + 1
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            
        finally:
            # Final save
            if self.config.save_model:
                print("\nSaving final checkpoint...")
                final_stats = TrainingStats(epoch=self.current_epoch - 1)
                self._save_checkpoint(self.current_epoch - 1, final_stats)
                
            # Cleanup wandb
            if self.wandb_run is not None:
                self.wandb_run.finish()
                
        total_time = time.time() - self.start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation score: {self.best_validation_score:.4f}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for EGGROLL training."""
    
    # Example configuration
    config = EggrollTrainerConfig(
        # Model
        model_name="/home/jovyan/nmt-srv-shared/users/binh/grpo_training/transflow/0_Base/en-vi-2.1.10.04-grpo-100k",
        
        # EGGROLL hyperparameters
        sigma=1e-3,
        lr_scale=1.0,
        rank=16,
        
        # Population
        generations_per_prompt=64,
        prompts_per_epoch=32,
        
        # Training
        num_epochs=100,
        validate_every=20,
        save_every=10,
        
        # Optimizer
        optimizer_type="sgd",
        
        # Reward
        reward_metric="bleu",
        fitness_shaping="centered_rank",
        
        # Paths
        output_directory="/home/jovyan/nmt-srv-shared/users/binh/EGGROLL/outputs",
        save_path="/home/jovyan/nmt-srv-shared/users/binh/EGGROLL/checkpoints",
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Example training data (source, target pairs)
    src_train = open("/home/jovyan/nmt-srv-shared/users/binh/eggroll_training/dataset/train.src", "r", encoding='utf-8').readlines()
    tgt_train = open("/home/jovyan/nmt-srv-shared/users/binh/eggroll_training/dataset/train.tgt", "r", encoding='utf-8').readlines()

    src_valid = open("/home/jovyan/nmt-srv-shared/users/binh/eggroll_training/dataset/valid.src", "r", encoding='utf-8').readlines()
    tgt_valid = open("/home/jovyan/nmt-srv-shared/users/binh/eggroll_training/dataset/valid.tgt", "r", encoding='utf-8').readlines()

    train_data = []
    valid_data = []
    for src, tgt in tqdm(zip(src_train, tgt_train), desc="Loading train dataset"):
        train_data.append((src.strip(), tgt.strip()))
    for src, tgt in tqdm(zip(src_valid, tgt_valid), desc="Loading valid dataset"):
        valid_data.append((src.strip(), tgt.strip()))

    # train_data = [
    #     ("Hello, how are you?", "Xin chào, bạn khỏe không?"),
    #     ("The weather is nice today.", "Thời tiết hôm nay đẹp."),
    #     ("I love learning new languages.", "Tôi thích học ngôn ngữ mới."),
    #     ("Machine translation is fascinating.", "Dịch máy thật thú vị."),
    #     ("Thank you very much.", "Cảm ơn bạn rất nhiều."),
    #     ("See you tomorrow.", "Hẹn gặp lại ngày mai."),
    #     ("What time is it?", "Bây giờ là mấy giờ?"),
    #     ("I am a student.", "Tôi là sinh viên."),
    #     # Add more training pairs...
    # ]
    
    # Create trainer
    trainer = EggrollTrainer(config)
    
    # Setup
    trainer.setup()
    
    # Train
    trainer.train(train_data, valid_data)


if __name__ == "__main__":
    main()
