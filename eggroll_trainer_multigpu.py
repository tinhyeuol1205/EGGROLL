"""
EGGROLL Multi-GPU Training Loop for Translation Model Finetuning

This module provides a complete multi-GPU training pipeline implementing the EGGROLL
algorithm from "Evolution Strategies at Hyperscale" (arXiv:2511.16652). 

Multi-GPU Strategy (mirrors JAX implementation):
- Data parallelism: Each GPU processes a subset of population members
- Sharding: Parameters replicated, data/perturbations sharded across GPUs
- All-gather: Aggregate fitness scores across all GPUs before update
- Synchronized updates: All GPUs apply the same parameter update

Based on: https://github.com/ESHyperscale/HyperscaleES/blob/main/llm_experiments/general_do_evolution_multi_gpu.py
"""

import os
import sys
import csv
import time
import math
import operator
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict, List, Any, Tuple, Callable, Union
from functools import partial
import warnings
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils. data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    raise ImportError("transformers is required.  Install with: pip install transformers")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EggrollMultiGPUConfig:
    """
    Complete configuration for EGGROLL multi-GPU training.
    Mirrors Args from the original JAX implementation. 
    """
    # Random seed
    seed: int = 0
    
    # Model configuration
    model_name: str = "Helsinki-NLP/opus-mt-en-vi"
    dtype: Optional[str] = "float32"
    
    # Output directories
    output_directory: str = "./outputs"
    save_path: str = "./checkpoints"
    load_path: Optional[str] = None
    
    # Save/Load options
    save_model: bool = True
    load_model: bool = False
    
    # Generation settings
    generation_length: int = 128
    max_source_length: int = 128
    num_beams: int = 1
    temperature: float = 1.0
    val_temperature: float = 1.0
    do_sample: bool = False
    
    # EGGROLL core hyperparameters
    sigma: float = 1e-3
    lr_scale: float = 1.0
    rank: int = 16
    noise_reuse: int = 1
    freeze_nonlora: bool = True
    
    # Population settings (per GPU)
    parallel_generations_per_gpu: int = 32
    parallel_validations_per_gpu: int = 16
    generations_per_prompt: int = 8
    
    # Training settings
    num_epochs: int = 100
    validate_every: int = 10
    save_every: int = 100
    log_every: int = 1
    log_output_every: int = 10
    validation_iterations: int = 10
    
    # Optimizer settings
    optimizer_type: str = "sgd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    momentum: float = 0.0
    
    # Reward settings
    reward_metric: str = "bleu"
    fitness_shaping: str = "group_normalize"
    
    # Distributed settings
    backend: str = "nccl"
    coord_addr: Optional[str] = None
    num_procs: Optional[int] = None
    proc_id: Optional[int] = None
    
    # Logging settings
    track: bool = False
    wandb_project: str = "EGGROLL-Translation"
    wandb_name: str = "eggroll-multigpu"
    wandb_mode: Literal["online", "offline"] = "online"
    wandb_directory: str = "./wandb_runs"
    
    @property
    def experiment_name(self) -> str:
        return f"{self.wandb_name}_lr={self.lr_scale}_sigma={self.sigma:. 2e}_rank={self.rank}"


# ============================================================================
# Distributed Utilities (mirrors JAX's sharding utilities)
# ============================================================================

class DistributedContext:
    """
    Manages distributed training context. 
    Mirrors JAX's mesh and sharding utilities. 
    """
    
    def __init__(self, config: EggrollMultiGPUConfig):
        self. config = config
        self.is_initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = None
        
    def initialize(self):
        """
        Initialize distributed context.
        Mirrors: jax.distributed.initialize(args. coord_addr, args. num_procs, args.proc_id)
        """
        if self.config.coord_addr is not None:
            # Multi-node setup
            os.environ['MASTER_ADDR'] = self. config.coord_addr. split(':')[0]
            os.environ['MASTER_PORT'] = self.config.coord_addr. split(':')[1]
            
            dist.init_process_group(
                backend=self.config. backend,
                world_size=self. config.num_procs,
                rank=self.config.proc_id,
            )
            self.world_size = self.config.num_procs
            self.rank = self. config.proc_id
            
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Single-node multi-GPU
            if 'RANK' in os.environ:
                # Launched with torchrun
                dist.init_process_group(backend=self.config. backend)
                self.world_size = dist.get_world_size()
                self.rank = dist. get_rank()
                self.local_rank = int(os.environ. get('LOCAL_RANK', 0))
            else:
                # Manual single-GPU fallback
                self. world_size = 1
                self. rank = 0
                self.local_rank = 0
        else:
            # CPU or single GPU
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            
        # Set device
        if torch.cuda.is_available():
            self.device = torch. device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
            
        self.is_initialized = True
        
        if self.is_main_process:
            print(f"Distributed context initialized:")
            print(f"  World size: {self.world_size}")
            print(f"  Rank: {self. rank}")
            print(f"  Local rank: {self. local_rank}")
            print(f"  Device: {self.device}")
            
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self. rank == 0
    
    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1
    
    def barrier(self, name: str = ""):
        """
        Synchronization barrier across all processes.
        Mirrors: mu.sync_global_devices("name")
        """
        if self.is_distributed:
            dist.barrier()
            
    def cleanup(self):
        """Cleanup distributed context."""
        if self.is_distributed:
            dist.destroy_process_group()


# ============================================================================
# Sharding Utilities (mirrors JAX's NamedSharding and shard_map)
# ============================================================================

class DataSharder:
    """
    Handles data sharding across GPUs.
    Mirrors JAX's NamedSharding and P('data') partition spec.
    """
    
    def __init__(self, dist_ctx: DistributedContext):
        self.dist_ctx = dist_ctx
        
    def shard_indices(self, total_size: int) -> Tuple[int, int]:
        """
        Get start and end indices for this GPU's shard.
        Mirrors: jax.device_put(data, NamedSharding(mesh, P('data')))
        """
        per_gpu = total_size // self.dist_ctx. world_size
        start = self.dist_ctx.rank * per_gpu
        end = start + per_gpu
        return start, end
    
    def get_local_size(self, total_size: int) -> int:
        """Get number of elements for this GPU."""
        return total_size // self.dist_ctx. world_size
    
    def create_local_indices(self, total_size: int, epoch: int = 0) -> np.ndarray:
        """
        Create local indices for this GPU.
        Mirrors: global_indices = replicate_matrix(np.arange(total_parallel_generations))
        """
        start, end = self.shard_indices(total_size)
        return np.arange(start, end)
    
    def create_direction_indices(self, generations_per_prompt: int) -> np.ndarray:
        """
        Create direction indices (population member indices). 
        Mirrors: global_indices % args.generations_per_prompt
        """
        local_size = self.get_local_size(
            self.dist_ctx.config.parallel_generations_per_gpu * self.dist_ctx. world_size
        )
        return np.arange(local_size) % generations_per_prompt


# ============================================================================
# All-Gather Operations (mirrors JAX's process_allgather)
# ============================================================================

class AllGatherOps:
    """
    All-gather operations for distributed training.
    Mirrors: process_allgather(local_fitness, tiled=True)
    """
    
    def __init__(self, dist_ctx: DistributedContext):
        self.dist_ctx = dist_ctx
        
    def all_gather_tensor(self, local_tensor: torch.Tensor) -> torch. Tensor:
        """
        Gather tensors from all GPUs.
        Mirrors: process_allgather(local_fitness, tiled=True)
        """
        if not self.dist_ctx.is_distributed:
            return local_tensor
            
        # Ensure tensor is contiguous
        local_tensor = local_tensor.contiguous()
        
        # Create output tensor
        gathered_tensors = [
            torch.zeros_like(local_tensor) 
            for _ in range(self.dist_ctx.world_size)
        ]
        
        # All-gather
        dist.all_gather(gathered_tensors, local_tensor)
        
        # Concatenate (tiled=True means concatenate along first dim)
        return torch.cat(gathered_tensors, dim=0)
    
    def all_gather_numpy(self, local_array: np.ndarray) -> np. ndarray:
        """Gather numpy arrays from all GPUs."""
        local_tensor = torch.from_numpy(local_array). to(self.dist_ctx.device)
        gathered = self.all_gather_tensor(local_tensor)
        return gathered.cpu().numpy()
    
    def all_reduce_sum(self, local_tensor: torch. Tensor) -> torch. Tensor:
        """Sum tensors across all GPUs."""
        if not self.dist_ctx.is_distributed:
            return local_tensor
            
        dist.all_reduce(local_tensor, op=dist. ReduceOp. SUM)
        return local_tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all ranks."""
        if not self.dist_ctx. is_distributed:
            return tensor
            
        dist.broadcast(tensor, src=src)
        return tensor


# ============================================================================
# ES Map Types and Random Key Generator
# ============================================================================

class ESMapType:
    FULL = 0
    LORA = 1
    FROZEN = 2
    NOOP = 3


class RandomKeyGenerator:
    """JAX-style random key generator."""
    
    def __init__(self, seed: int):
        self.seed = seed
        
    def fold_in(self, key_id: int) -> 'RandomKeyGenerator':
        new_seed = ((self.seed * 31337) + key_id) % (2**31)
        return RandomKeyGenerator(new_seed)


# ============================================================================
# Optimizer State
# ============================================================================

@dataclass
class OptimizerState:
    step: int = 0
    momentum: Optional[Dict[str, torch.Tensor]] = None
    velocity: Optional[Dict[str, torch. Tensor]] = None


# ============================================================================
# Training Statistics
# ============================================================================

@dataclass
class TrainingStats:
    epoch: int
    avg_fitness: float = 0.0
    std_fitness: float = 0.0
    max_fitness: float = 0.0
    min_fitness: float = 0. 0
    median_fitness: float = 0.0
    lora_updates: float = 0.0
    nonlora_updates: float = 0. 0
    prompt_preproc_time: float = 0.0
    token_gen_time: float = 0.0
    fitness_time: float = 0.0
    gather_time: float = 0.0
    update_time: float = 0.0
    saving_time: float = 0.0
    validation_time: float = 0. 0
    true_train_avg_fitness: float = 0.0
    validation_score: Optional[float] = None


# ============================================================================
# EGGROLL Multi-GPU Trainer
# ============================================================================

class EggrollMultiGPUTrainer:
    """
    Multi-GPU EGGROLL trainer for translation model finetuning. 
    
    Mirrors the structure of general_do_evolution_multi_gpu.py
    """
    
    def __init__(self, config: EggrollMultiGPUConfig):
        self.config = config
        
        # Distributed context
        self. dist_ctx = DistributedContext(config)
        self.sharder = None
        self.all_gather = None
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.params = None
        self.es_map = None
        self.base_evo_keys = None
        self.opt_state = None
        
        # Training state
        self. current_epoch = 0
        self.true_train_fitness_sum = 0.0
        self.best_validation_score = -float('inf')
        
        # Computed settings (set in setup)
        self.total_parallel_generations = 0
        self.total_validation_generations = 0
        self.prompts_per_epoch = 0
        
        # Logging
        self.wandb_run = None
        self.run_out_dir = None
        self. ckpt_dir = None
        
    # ========================================================================
    # Initialization (mirrors JAX initialization section)
    # ========================================================================
    
    def setup(self):
        """
        Complete setup for multi-GPU training. 
        Mirrors the initialization section of general_do_evolution_multi_gpu.py
        """
        # Initialize distributed context
        print("Starting distributed init")
        self.dist_ctx.initialize()
        
        if not self.dist_ctx.is_distributed:
            print("NOT DISTRIBUTED CONTEXT")
        
        # Initialize sharding utilities
        self.sharder = DataSharder(self.dist_ctx)
        self.all_gather = AllGatherOps(self.dist_ctx)
        
        # Compute derived settings
        # Mirrors: args.total_parallel_generations = total_num_devices * args.parallel_generations_per_gpu
        self.total_parallel_generations = (
            self.dist_ctx.world_size * self. config.parallel_generations_per_gpu
        )
        self.total_validation_generations = (
            self.dist_ctx.world_size * self.config.parallel_validations_per_gpu
        )
        self.prompts_per_epoch = (
            self.total_parallel_generations // self.config.generations_per_prompt
        )
        
        if self.dist_ctx.is_main_process:
            print()
            print(f"Global devices: {self.dist_ctx.world_size} GPUs")
            print(f"Process id: {self.dist_ctx.rank}")
            print()
            print(f"Per-device generations: {self. config.parallel_generations_per_gpu}")
            print(f"Full number of generations: {self.total_parallel_generations}")
            print()
        
        # Set random seeds
        self._set_seeds()
        
        # Load model
        self._load_model()
        
        # Build ES map
        self._build_es_map()
        
        # Initialize random keys
        self._init_random_keys()
        
        # Initialize noiser parameters (optimizer state)
        self._init_noiser()
        
        # Setup reward function
        self._setup_reward_function()
        
        # Setup output directories and wandb
        self._setup_logging()
        
        # Load checkpoint if specified
        if self. config.load_model and self.config.load_path:
            self._load_checkpoint()
            
        # Barrier to ensure all processes are ready
        self. dist_ctx.barrier("post-setup")
        
        if self.dist_ctx.is_main_process:
            self._print_setup_summary()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        # Mirrors: master_key = jax.random.key(args.seed)
        seed = self.config. seed + self.dist_ctx.rank  # Different seed per GPU
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Master keys (same across all GPUs for reproducibility)
        self.master_key = RandomKeyGenerator(self.config.seed)
        self.base_model_key = self.master_key. fold_in(0)
        self. base_gen_key = self. master_key.fold_in(1)
        self.base_valid_key = self. master_key.fold_in(2)
    
    def _load_model(self):
        """
        Load pre-trained model. 
        Mirrors: RWKV, full_params, tokenizer = get_model(...)
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float32)
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
        ). to(self.dist_ctx.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set model to eval mode
        self. model.eval()
        
        # Extract parameters
        # Mirrors: config, params, scan_map, es_map = full_params
        self.params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        if self.dist_ctx.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model: {self.model.__class__.__name__}")
            print(f"Total parameters: {total_params:,}")
    
    def _build_es_map(self):
        """
        Build ES parameter classification map.
        Mirrors: es_map from full_params
        """
        lora_targets = [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2", "self_attn", "encoder_attn",
        ]
        
        self.es_map = {}
        
        for name, param in self.params. items():
            if "embed" in name. lower() or "layer_norm" in name. lower():
                self.es_map[name] = ESMapType.FROZEN
            elif "bias" in name.lower():
                self. es_map[name] = ESMapType.FROZEN if self.config.freeze_nonlora else ESMapType.FULL
            elif any(t in name.lower() for t in lora_targets) and len(param.shape) == 2:
                self.es_map[name] = ESMapType.LORA
            else:
                self.es_map[name] = ESMapType. FROZEN if self.config.freeze_nonlora else ESMapType. FULL
    
    def _init_random_keys(self):
        """
        Initialize random keys for each parameter.
        Mirrors: base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)
        """
        self.base_evo_keys = {
            name: self.base_model_key.fold_in(i)
            for i, name in enumerate(self.params.keys())
        }
    
    def _init_noiser(self):
        """
        Initialize noiser parameters.
        Mirrors: frozen_noiser_params, noiser_params = NOISER.init_noiser(...)
        """
        self.frozen_noiser_params = {
            'group_size': self.config. generations_per_prompt,
            'freeze_nonlora': self.config.freeze_nonlora,
            'noise_reuse': self. config.noise_reuse,
            'rank': self.config.rank,
        }
        
        self.opt_state = OptimizerState(step=0)
        
        if self.config.optimizer_type == "adam":
            self. opt_state. momentum = {
                name: torch.zeros_like(p, device=self.dist_ctx.device)
                for name, p in self.params. items()
                if self.es_map. get(name) != ESMapType. FROZEN
            }
            self.opt_state. velocity = {
                name: torch.zeros_like(p, device=self.dist_ctx.device)
                for name, p in self.params. items()
                if self.es_map.get(name) != ESMapType. FROZEN
            }
    
    def _setup_reward_function(self):
        """Setup reward function."""
        try:
            import sacrebleu
            self._sacrebleu = sacrebleu
            self. reward_function = self._compute_bleu_sacrebleu
        except ImportError:
            self. reward_function = self._compute_bleu_simple
    
    def _setup_logging(self):
        """
        Setup logging directories and wandb.
        Mirrors the wandb initialization section. 
        """
        # Mirrors: mu.sync_global_devices("pre-wandb-init")
        self.dist_ctx.barrier("pre-wandb-init")
        
        experiment_id = f"{self. config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.dist_ctx.is_main_process:
            print(f"Run name: {self.config.experiment_name}")
        
        # Disable wandb on non-main processes
        # Mirrors: if jax.process_index() != 0: os.environ["WANDB_DISABLED"] = "true"
        if not self.dist_ctx.is_main_process:
            os.environ["WANDB_DISABLED"] = "true"
        elif self.config.track and WANDB_AVAILABLE:
            if self.config.wandb_mode == "offline":
                os. environ["WANDB_MODE"] = "offline"
            
            wandb_dir = Path(self.config.wandb_directory) / "wandb_runs"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            
            self.wandb_run = wandb.init(
                project=self. config.wandb_project,
                config=asdict(self.config),
                name=self.config.experiment_name,
                dir=str(wandb_dir),
            )
        
        self.dist_ctx.barrier("post-wandb-init")
        
        # Setup output directories
        if self.dist_ctx.is_main_process:
            run_id = self.wandb_run.id if self.wandb_run else experiment_id
            
            base_out_dir = Path(self.config. output_directory)
            self.run_out_dir = base_out_dir / run_id
            self. run_out_dir.mkdir(parents=True, exist_ok=True)
            
            if self.config.save_model:
                self. ckpt_dir = Path(self.config. save_path) / run_id
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    def _print_setup_summary(self):
        """Print setup summary."""
        lora_count = sum(1 for v in self.es_map.values() if v == ESMapType.LORA)
        full_count = sum(1 for v in self.es_map.values() if v == ESMapType. FULL)
        frozen_count = sum(1 for v in self.es_map.values() if v == ESMapType. FROZEN)
        
        print("\n" + "=" * 70)
        print("EGGROLL Multi-GPU Training Setup Complete")
        print("=" * 70)
        print(f"""
Distributed:
  World size: {self.dist_ctx.world_size} GPUs
  Total generations per epoch: {self.total_parallel_generations}
  Prompts per epoch: {self.prompts_per_epoch}
  
EGGROLL:
  σ (sigma): {self. config.sigma}
  α (learning rate): {self. config.lr_scale}
  r (rank): {self.config.rank}
  N (generations per prompt): {self.config.generations_per_prompt}
  
Parameters:
  LoRA: {lora_count}
  Full: {full_count}
  Frozen: {frozen_count}
""")

    # ========================================================================
    # Perturbation Generation (Step 2)
    # ========================================================================
    
    def _get_perturbation_seed(self, base_seed: int, epoch: int, member_idx: int) -> int:
        """Get deterministic seed for perturbation."""
        if self.config.noise_reuse > 0:
            effective_epoch = epoch // self.config.noise_reuse
        else:
            effective_epoch = epoch
        return ((base_seed * 31337) + effective_epoch * 10000 + member_idx) % (2**31)
    
    def _generate_lora_perturbation(
        self,
        param_shape: Tuple[int, int],
        seed: int,
    ) -> Tuple[torch. Tensor, torch. Tensor]:
        """Generate low-rank perturbation matrices."""
        out_features, in_features = param_shape
        rank = self.config. rank
        sigma = self.config. sigma
        scale = sigma / math.sqrt(rank)
        
        gen_A = torch.Generator(device='cpu').manual_seed(seed)
        gen_B = torch.Generator(device='cpu').manual_seed(seed + 1)
        
        A = torch.randn(out_features, rank, generator=gen_A) * scale
        B = torch.randn(in_features, rank, generator=gen_B)
        
        return A. to(self.dist_ctx.device), B.to(self.dist_ctx. device)

    # ========================================================================
    # Forward Pass with Perturbation (Step 3)
    # ========================================================================
    
    @torch.no_grad()
    def _generate_with_perturbation(
        self,
        input_ids: torch. Tensor,
        attention_mask: torch.Tensor,
        epoch: int,
        member_idx: int,
    ) -> torch.Tensor:
        """
        Generate with perturbed model.
        Mirrors: _generate_thread from build_generate_thread
        """
        original_weights = {}
        
        # Apply perturbations
        for name, param in self.model.named_parameters():
            map_type = self. es_map.get(name, ESMapType.FROZEN)
            
            if map_type == ESMapType.FROZEN:
                continue
                
            original_weights[name] = param.data. clone()
            base_seed = self. base_evo_keys[name].seed
            seed = self._get_perturbation_seed(base_seed, epoch, member_idx)
            
            if map_type == ESMapType.LORA and len(param.shape) == 2:
                A, B = self._generate_lora_perturbation(param.shape, seed)
                param.data = param.data + A @ B. T
            elif map_type == ESMapType.FULL:
                gen = torch.Generator(device='cpu').manual_seed(seed)
                noise = torch.randn(param.shape, generator=gen) * self.config.sigma
                param.data = param.data + noise. to(self.dist_ctx. device)
        
        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.config. generation_length,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
        )
        
        # Restore
        for name, original in original_weights.items():
            dict(self.model.named_parameters())[name]. data = original
            
        return output_ids

    # ========================================================================
    # Reward Computation (Step 4)
    # ========================================================================
    
    def _compute_bleu_sacrebleu(self, hypothesis: str, reference: str) -> float:
        if not hypothesis. strip() or not reference.strip():
            return 0.0
        try:
            bleu = self._sacrebleu.sentence_bleu(hypothesis, [reference], smooth_method='exp')
            return bleu.score / 100.0
        except:
            return 0.0
    
    def _compute_bleu_simple(self, hypothesis: str, reference: str) -> float:
        if not hypothesis.strip() or not reference.strip():
            return 0.0
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            hyp = hypothesis.lower(). split()
            ref = reference.lower(). split()
            return sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction(). method1)
        except:
            return 0.0

    # ========================================================================
    # Fitness Conversion (Step 5a)
    # ========================================================================
    
    def _convert_fitnesses(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Convert raw scores to shaped fitnesses.
        Mirrors: NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
        """
        group_size = self. frozen_noiser_params['group_size']
        
        if group_size == 0:
            mean = np.mean(raw_scores)
            std = np.sqrt(np.var(raw_scores) + 1e-5)
            return (raw_scores - mean) / std
        else:
            group_scores = raw_scores.reshape(-1, group_size)
            group_mean = np.mean(group_scores, axis=-1, keepdims=True)
            global_std = np.sqrt(np.var(raw_scores) + 1e-5)
            true_scores = (group_scores - group_mean) / global_std
            return true_scores. ravel()

    # ========================================================================
    # Gradient Estimation and Update (Steps 5b & 6)
    # ========================================================================
    
    def _do_update(
        self,
        raw_scores: np. ndarray,
        epoch: int,
        dir_indices: np.ndarray,
    ) -> Tuple[Dict[str, torch. Tensor], Dict[str, float]]:
        """
        Perform ES update.
        Mirrors: _do_update function from original code
        
        def _do_update(noiser_params, params, raw_scores, epoch_num, dir_indices):
            iterinfos = (jnp.full(raw_scores. size, epoch_num, dtype=jnp.int32), dir_indices)
            fitnesses = NOISER.convert_fitnesses(...)
            noiser_params, new_params = NOISER.do_updates(...)
            return noiser_params, new_params, parameter_differences
        """
        # Convert fitnesses
        fitnesses = self._convert_fitnesses(raw_scores)
        population_size = len(fitnesses)
        
        new_params = {}
        param_diffs = {}
        
        for name, param in self.params.items():
            map_type = self. es_map.get(name, ESMapType.FROZEN)
            
            if map_type == ESMapType.FROZEN:
                new_params[name] = param
                param_diffs[name] = 0. 0
                continue
            
            # Estimate gradient
            gradient = torch.zeros_like(param)
            
            for i, member_idx in enumerate(dir_indices[:population_size]):
                R_i = fitnesses[i]
                base_seed = self. base_evo_keys[name].seed
                seed = self._get_perturbation_seed(base_seed, epoch, int(member_idx))
                
                if map_type == ESMapType.LORA and len(param.shape) == 2:
                    A, B = self._generate_lora_perturbation(param.shape, seed)
                    gradient += R_i * (A @ B.T)
                elif map_type == ESMapType.FULL:
                    gen = torch.Generator(device='cpu').manual_seed(seed)
                    noise = torch.randn(param.shape, generator=gen) * self.config.sigma
                    gradient += R_i * noise. to(self.dist_ctx.device) / self.config.sigma
            
            gradient /= population_size
            
            # Apply optimizer step
            update = self._apply_optimizer_step(name, gradient)
            
            # Update parameter
            new_param = param + update
            new_params[name] = new_param
            
            # Compute RMSE difference
            # Mirrors: jax.tree. map(lambda x, y: jnp.sqrt(jnp. mean((x - y) ** 2)), params, new_params)
            param_diffs[name] = torch.sqrt(torch.mean((new_param - param) ** 2)). item()
        
        # Update stored params
        self.params = new_params
        self._update_model_weights()
        
        # Increment optimizer step
        self.opt_state. step += 1
        
        return new_params, param_diffs
    
    def _apply_optimizer_step(self, name: str, gradient: torch. Tensor) -> torch. Tensor:
        """Apply optimizer step."""
        lr = self.config. lr_scale
        
        if self.config.optimizer_type == "adam":
            t = self.opt_state.step + 1
            beta1, beta2 = self.config.adam_beta1, self.config. adam_beta2
            eps = self.config. adam_eps
            
            m = self. opt_state.momentum. get(name, torch.zeros_like(gradient))
            v = self.opt_state.velocity.get(name, torch. zeros_like(gradient))
            
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            self.opt_state. momentum[name] = m
            self. opt_state.velocity[name] = v
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            return lr * m_hat / (torch.sqrt(v_hat) + eps)
        else:
            return lr * gradient
    
    def _update_model_weights(self):
        """Update model weights from params dict."""
        state_dict = self. model.state_dict()
        for name, param in self.params.items():
            if name in state_dict:
                state_dict[name] = param
        self.model. load_state_dict(state_dict)

    # ========================================================================
    # Validation
    # ========================================================================
    
    @torch.no_grad()
    def _validate(self, epoch: int, val_data: List[Tuple[str, str]]) -> float:
        """
        Run validation.
        Mirrors: validate function with sigma=0
        """
        if not val_data:
            return 0. 0
            
        total = 0.0
        count = 0
        
        # Each GPU processes a portion
        local_size = len(val_data) // self.dist_ctx.world_size
        start_idx = self. dist_ctx.rank * local_size
        end_idx = start_idx + local_size
        local_data = val_data[start_idx:end_idx]
        
        for source, reference in local_data:
            inputs = self. tokenizer(
                source,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config. max_source_length,
            ).to(self. dist_ctx.device)
            
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs. get("attention_mask"),
                max_length=self.config. generation_length,
                num_beams=self.config. num_beams,
            )
            
            hypothesis = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            reward = self.reward_function(hypothesis, reference)
            
            total += reward
            count += 1
        
        # All-reduce sum
        local_sum = torch.tensor([total], device=self.dist_ctx.device)
        local_count = torch. tensor([count], device=self.dist_ctx.device)
        
        global_sum = self.all_gather. all_reduce_sum(local_sum)
        global_count = self.all_gather.all_reduce_sum(local_count)
        
        return (global_sum / global_count).item()

    # ========================================================================
    # Single Epoch (mirrors single_epoch function)
    # ========================================================================
    
    def _single_epoch(
        self,
        train_data: List[Tuple[str, str]],
        val_data: Optional[List[Tuple[str, str]]],
        epoch: int,
    ) -> TrainingStats:
        """
        Execute single training epoch.
        Mirrors: single_epoch function from original code
        """
        stats = TrainingStats(epoch=epoch)
        
        # Validation (periodic)
        # Mirrors: if epoch % args.validate_every == 0
        if epoch % self.config. validate_every == 0 and val_data:
            if self.dist_ctx.is_main_process:
                print("VALIDATION")
            start_time = time.time()
            stats.validation_score = self._validate(epoch, val_data)
            stats.validation_time = time.time() - start_time
            if self.dist_ctx.is_main_process:
                print(f"VALIDATION SCORE= {stats.validation_score}")
        
        # Prepare prompts
        start_time = time. time()
        
        # Sample prompts for this epoch
        # Mirrors: unique_indices = ...  + epoch * args.prompts_per_epoch
        rng = np.random.RandomState(self.config.seed + epoch)
        epoch_prompt_indices = rng.choice(
            len(train_data),
            size=min(self.prompts_per_epoch, len(train_data)),
            replace=True,
        )
        epoch_prompts = [train_data[i] for i in epoch_prompt_indices]
        
        stats.prompt_preproc_time = time.time() - start_time
        
        # Generate with all population members on this GPU
        # Mirrors: output_batch = generate_batch(noiser_params, params, batch_prompts, all_thread_idxes, epoch)
        start_time = time.time()
        
        if epoch == 0 and self.dist_ctx.is_main_process:
            print("generating batch")
        
        local_hypotheses = []
        local_references = []
        local_member_indices = []
        
        # This GPU handles a subset of directions
        local_directions = self.sharder.create_direction_indices(self.config.generations_per_prompt)
        
        for source, reference in epoch_prompts:
            inputs = self.tokenizer(
                source,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self. config.max_source_length,
            ).to(self. dist_ctx.device)
            
            for member_idx in local_directions:
                output_ids = self._generate_with_perturbation(
                    inputs["input_ids"],
                    inputs. get("attention_mask"),
                    epoch,
                    member_idx,
                )
                
                hypothesis = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                local_hypotheses.append(hypothesis)
                local_references.append(reference)
                local_member_indices.append(member_idx)
        
        stats.token_gen_time = time.time() - start_time
        
        # Compute local fitness
        # Mirrors: _local_fitness = [Task. get_batch_fitness(... )]
        start_time = time. time()
        
        if epoch == 0 and self.dist_ctx.is_main_process:
            print("calculating fitness")
        
        local_fitness = np.array([
            self.reward_function(hyp, ref)
            for hyp, ref in zip(local_hypotheses, local_references)
        ])
        
        stats.fitness_time = time.time() - start_time
        
        # Gather fitness from all GPUs
        # Mirrors: output_scores = process_allgather(local_fitness, True)
        start_time = time.time()
        
        if epoch == 0 and self.dist_ctx.is_main_process:
            print("gathering")
        
        global_fitness = self.all_gather.all_gather_numpy(local_fitness)
        
        stats.gather_time = time. time() - start_time
        
        # Aggregate scores per direction
        # Mirrors: output_scores = output_scores. reshape(args.prompts_per_epoch, args.generations_per_prompt). sum(axis=0)
        output_scores = global_fitness.reshape(
            self.prompts_per_epoch,
            self.config.generations_per_prompt
        ).sum(axis=0)
        
        # Compute statistics
        stats.avg_fitness = float(np.mean(output_scores))
        stats.std_fitness = float(np.std(output_scores))
        stats.max_fitness = float(np.max(output_scores))
        stats.min_fitness = float(np. min(output_scores))
        stats.median_fitness = float(np.median(output_scores))
        
        # Update parameters
        # Mirrors: noiser_params, params, parameter_differences = do_update(...)
        start_time = time.time()
        
        if epoch == 0 and self.dist_ctx.is_main_process:
            print("updating params")
        
        dir_indices = np. arange(self.config.generations_per_prompt)
        new_params, param_diffs = self._do_update(output_scores, epoch, dir_indices)
        
        stats. update_time = time.time() - start_time
        
        # Compute parameter update statistics
        # Mirrors: lora_updates = jax.tree.reduce(...)
        lora_diffs = [d for n, d in param_diffs.items() if self.es_map. get(n) == ESMapType. LORA]
        full_diffs = [d for n, d in param_diffs.items() if self.es_map. get(n) == ESMapType. FULL]
        
        stats.lora_updates = np.mean(lora_diffs) if lora_diffs else 0. 0
        stats. nonlora_updates = np.mean(full_diffs) if full_diffs else 0.0
        
        # Save checkpoint (periodic)
        if self.config.save_model and epoch % self.config.save_every == 0:
            if self.dist_ctx.is_main_process:
                start_time = time. time()
                self._save_checkpoint(epoch, stats)
                stats. saving_time = time. time() - start_time
        
        # Update running average
        self.true_train_fitness_sum += np.sum(output_scores)
        stats.true_train_avg_fitness = (
            self.true_train_fitness_sum / 
            ((epoch + 1) * self.config.generations_per_prompt)
        )
        
        return stats

    # ========================================================================
    # Logging (mirrors logging section)
    # ========================================================================
    
    def _log_epoch(self, stats: TrainingStats):
        """Log epoch statistics."""
        if self.dist_ctx.is_main_process:
            if self.wandb_run is not None:
                log_dict = {
                    'avg_fitness': stats. avg_fitness,
                    'std_fitness': stats.std_fitness,
                    'max_fitness': stats.max_fitness,
                    'min_fitness': stats.min_fitness,
                    'median_fitness': stats.median_fitness,
                    'lora_updates': stats.lora_updates,
                    'nonlora_updates': stats.nonlora_updates,
                    'prompt_preproc_time': stats.prompt_preproc_time,
                    'token_gen_time': stats.token_gen_time,
                    'fitness_time': stats. fitness_time,
                    'gather_time': stats. gather_time,
                    'update_time': stats.update_time,
                    'saving_time': stats.saving_time,
                    'validation_time': stats.validation_time,
                    'true_train_avg_fitness': stats.true_train_avg_fitness,
                }
                if stats.validation_score is not None:
                    log_dict['validation_score'] = stats.validation_score
                self.wandb_run.log(log_dict, step=stats.epoch)
            else:
                # Console logging
                print(f"Mean fitness: {stats. avg_fitness:.4f}; "
                      f"std: {stats.std_fitness:.4f}; "
                      f"max: {stats.max_fitness:.4f}; "
                      f"min: {stats.min_fitness:. 4f}")
                print(f"LoRA updates: {stats.lora_updates:.6f}")
                print(f"Full updates: {stats. nonlora_updates:.6f}")

    # ========================================================================
    # Checkpointing
    # ========================================================================
    
    def _save_checkpoint(self, epoch: int, stats: TrainingStats):
        """Save checkpoint."""
        if not self.dist_ctx.is_main_process:
            return
            
        ckpt_path = self.ckpt_dir / "latest. model"
        
        checkpoint = {
            'epoch': epoch,
            'params': {k: v.cpu() for k, v in self.params. items()},
            'opt_state': {
                'step': self.opt_state.step,
                'momentum': {k: v.cpu() for k, v in (self.opt_state.momentum or {}). items()},
                'velocity': {k: v.cpu() for k, v in (self.opt_state.velocity or {}).items()},
            },
            'es_map': self. es_map,
            'frozen_noiser_params': self.frozen_noiser_params,
            'true_train_fitness_sum': self.true_train_fitness_sum,
            'config': asdict(self.config),
        }
        
        torch. save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
    
    def _load_checkpoint(self):
        """Load checkpoint."""
        ckpt_path = Path(self.config.load_path)
        if ckpt_path. is_dir():
            ckpt_path = ckpt_path / "latest.model"
            
        if not ckpt_path. exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self. dist_ctx.device)
        
        self.params = {k: v.to(self.dist_ctx.device) for k, v in checkpoint['params'].items()}
        self.opt_state. step = checkpoint['opt_state']['step']
        
        if checkpoint['opt_state']['momentum']:
            self.opt_state.momentum = {
                k: v. to(self.dist_ctx.device) 
                for k, v in checkpoint['opt_state']['momentum']. items()
            }
        if checkpoint['opt_state']['velocity']:
            self.opt_state.velocity = {
                k: v. to(self.dist_ctx.device)
                for k, v in checkpoint['opt_state']['velocity']. items()
            }
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.true_train_fitness_sum = checkpoint. get('true_train_fitness_sum', 0.0)
        
        self._update_model_weights()
        
        if self.dist_ctx.is_main_process:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

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
        Mirrors: for epoch in tqdm. trange(args.num_epochs): ... 
        """
        if self.dist_ctx.is_main_process:
            print("\n" + "=" * 70)
            print("Starting EGGROLL Multi-GPU Training")
            print("=" * 70)
        
        iterator = range(self.current_epoch, self. config.num_epochs)
        if self.dist_ctx.is_main_process:
            iterator = tqdm(iterator, desc="Training")
        
        try:
            for epoch in iterator:
                stats = self._single_epoch(train_data, val_data, epoch)
                self._log_epoch(stats)
                self. current_epoch = epoch + 1
                
        except KeyboardInterrupt:
            if self.dist_ctx.is_main_process:
                print("\nTraining interrupted.")
        
        finally:
            # Cleanup
            # Mirrors: mu.sync_global_devices("before-wandb-finish")
            self.dist_ctx. barrier("before-wandb-finish")
            
            if self.wandb_run is not None and self.dist_ctx.is_main_process:
                self.wandb_run.finish()
                
            self.dist_ctx. barrier("after-wandb-finish")
            self.dist_ctx. cleanup()


# ============================================================================
# Launch Script
# ============================================================================

def main():
    """Main entry point."""
    config = EggrollMultiGPUConfig(
        model_name="Helsinki-NLP/opus-mt-en-vi",
        sigma=1e-3,
        lr_scale=1. 0,
        rank=16,
        parallel_generations_per_gpu=32,
        generations_per_prompt=8,
        num_epochs=100,
        validate_every=10,
        save_every=50,
    )
    
    # Sample training data
    train_data = [
        ("Hello, how are you?", "Xin chào, bạn khỏe không?"),
        ("The weather is nice today.", "Thời tiết hôm nay đẹp. "),
        ("I love learning new languages.", "Tôi thích học ngôn ngữ mới."),
        ("Machine translation is fascinating.", "Dịch máy thật thú vị."),
        ("Thank you very much.", "Cảm ơn bạn rất nhiều."),
        ("See you tomorrow.", "Hẹn gặp lại ngày mai."),
        ("What time is it?", "Bây giờ là mấy giờ? "),
        ("I am a student.", "Tôi là sinh viên."),
    ]
    
    trainer = EggrollMultiGPUTrainer(config)
    trainer.setup()
    trainer.train(train_data)


if __name__ == "__main__":
    main()