"""
EGGROLL Complete Training Loop for Translation Model Finetuning - ACCELERATE VERSION
"""

import os
import sys
import time
import math
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, List, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm

# --- [CHANGED] Import Accelerate ---
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

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
    seed: int = 42 # [CHANGED] Changed default seed
    model_name: str = "Helsinki-NLP/opus-mt-en-vi"
    dtype: Optional[str] = "float32"
    output_directory: str = "./outputs"
    save_path: str = "./checkpoints"
    load_path: Optional[str] = None
    save_model: bool = True
    load_model: bool = False
    
    generation_length: int = 128
    max_source_length: int = 128
    num_beams: int = 1
    temperature: float = 1.0
    do_sample: bool = False
    
    sigma: float = 1e-3
    lr_scale: float = 1.0
    rank: int = 16
    noise_reuse: int = 1
    freeze_nonlora: bool = True
    
    generations_per_prompt: int = 8
    prompts_per_epoch: int = 8  # This acts as Batch Size per GPU now
    
    num_epochs: int = 100
    validate_every: int = 10
    save_every: int = 100
    log_every: int = 1
    
    optimizer_type: str = "sgd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    momentum: float = 0.0
    
    reward_metric: str = "bleu"
    fitness_shaping: str = "centered_rank"
    
    # [CHANGED] Device is handled by Accelerator, removed manual config
    
    track: bool = False
    wandb_project: str = "EGGROLL-Translation"
    wandb_name: str = "eggroll-run"
    wandb_mode: Literal["online", "offline"] = "online"
    
    @property
    def experiment_name(self) -> str:
        return f"{self.wandb_name}_lr={self.lr_scale}_sigma={self.sigma:.2e}_rank={self.rank}"

# ============================================================================
# [CHANGED] Simple Dataset Wrapper
# ============================================================================
class TranslationDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]]):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

@dataclass
class TrainingStats:
    epoch: int
    avg_fitness: float = 0.0
    std_fitness: float = 0.0
    max_fitness: float = 0.0
    gradient_norm: float = 0.0
    total_time: float = 0.0
    validation_score: Optional[float] = None

@dataclass
class OptimizerState:
    step: int = 0
    momentum: Optional[Dict[str, torch.Tensor]] = None
    velocity: Optional[Dict[str, torch.Tensor]] = None

class RandomKeyGenerator:
    def __init__(self, seed: int):
        self.seed = seed
    def fold_in(self, key_id: int) -> 'RandomKeyGenerator':
        new_seed = ((self.seed * 31337) + key_id) % (2**31)
        return RandomKeyGenerator(new_seed)

class ESMapType:
    FULL = 0
    LORA = 1
    FROZEN = 2

# ============================================================================
# EGGROLL Trainer Class (Accelerate Enabled)
# ============================================================================

class EggrollTrainer:
    def __init__(self, config: EggrollTrainerConfig):
        self.config = config
        
        # [CHANGED] Initialize Accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device # Use accelerator's device
        
        self._set_seeds(config.seed)
        
        self.model = None
        self.tokenizer = None
        self.params = None
        self.es_map = None
        self.base_evo_keys = None
        self.opt_state = None
        self.reward_function = None
        
        self.current_epoch = 0
        self.best_validation_score = -float('inf')
        self.wandb_run = None
        
    def _set_seeds(self, seed: int):
        # [CHANGED] Add offset for each process to ensure different random behavior if needed
        # But for ES perturbations generated via RandomKeyGenerator, we keep base seed same
        # and vary by prompt/member index.
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def setup(self):
        if self.accelerator.is_main_process:
            print("=" * 70)
            print("EGGROLL Trainer Setup (Multi-GPU Enabled)")
            print("=" * 70)
        
        # 1. Load Model
        self._load_model()
        
        # 2. Build Maps & Init Keys
        self._build_es_map()
        self._init_random_keys()
        self._init_optimizer()
        self._setup_reward_function()
        
        if self.config.track and self.accelerator.is_main_process:
            self._init_wandb()
            
        if self.config.load_model and self.config.load_path:
            self._load_checkpoint(self.config.load_path)

        # [CHANGED] We do NOT prepare the model with accelerator.prepare() because 
        # we are manually manipulating weights (param.data) and doing custom updates.
        # DDP wrappers would interfere with this manual ES process.
        # We only prepare the DataLoader later.
            
    def _load_model(self):
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.config.dtype, torch.float32)
        
        # [CHANGED] Load to specific device using accelerator.device is handled by .to(self.device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model.eval()
        
        # Extract parameters (Shared pointers to model parameters)
        self.params = {name: param for name, param in self.model.named_parameters()}
        
        if self.accelerator.is_main_process:
            print(f"  Model loaded on: {self.device}")

    def _build_es_map(self):
        # ... (Same logic as original) ...
        lora_targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        self.es_map = {}
        for name, param in self.params.items():
            if "embed" in name.lower() or "layer_norm" in name.lower():
                self.es_map[name] = ESMapType.FROZEN
            elif any(target in name.lower() for target in lora_targets) and len(param.shape) == 2:
                self.es_map[name] = ESMapType.LORA
            elif "bias" in name.lower() and not self.config.freeze_nonlora:
                self.es_map[name] = ESMapType.FULL
            else:
                self.es_map[name] = ESMapType.FROZEN

    def _init_random_keys(self):
        master_key = RandomKeyGenerator(self.config.seed)
        self.base_model_key = master_key.fold_in(0)
        self.base_evo_keys = {
            name: self.base_model_key.fold_in(i) for i, name in enumerate(self.params.keys())
        }
        
    def _init_optimizer(self):
        self.opt_state = OptimizerState(step=0)
        # Initialize momentum buffers on correct device
        if self.config.optimizer_type == "adam" or self.config.momentum > 0:
            self.opt_state.momentum = {
                name: torch.zeros_like(p) for name, p in self.params.items()
                if self.es_map.get(name) != ESMapType.FROZEN
            }
        if self.config.optimizer_type == "adam":
            self.opt_state.velocity = {
                name: torch.zeros_like(p) for name, p in self.params.items()
                if self.es_map.get(name) != ESMapType.FROZEN
            }

    def _setup_reward_function(self):
        # Simple length ratio fallback for demo, replace with BLEU if needed
        self.reward_function = self._compute_length_ratio 
        
    def _compute_length_ratio(self, hypothesis: str, reference: str) -> float:
        ref_len = len(reference.split())
        if ref_len == 0: return 0.0
        return max(0.0, 1.0 - abs(len(hypothesis.split()) / ref_len - 1.0))

    def _init_wandb(self):
        if not WANDB_AVAILABLE: return
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.experiment_name,
            config=asdict(self.config),
        )

    # ========================================================================
    # Perturbation Logic (Run on each GPU independently)
    # ========================================================================
    
    def _get_perturbation_seed(self, base_seed: int, epoch: int, member_idx: int) -> int:
        effective_epoch = epoch // self.config.noise_reuse
        return ((base_seed * 31337) + effective_epoch * 1000 + member_idx) % (2**31)
    
    def _generate_lora_perturbation(self, param_shape, seed):
        gen_A = torch.Generator(device=self.device).manual_seed(seed)
        gen_B = torch.Generator(device=self.device).manual_seed(seed + 1)
        
        scale = self.config.sigma / math.sqrt(self.config.rank)
        A = torch.randn(param_shape[0], self.config.rank, generator=gen_A, device=self.device) * scale
        B = torch.randn(param_shape[1], self.config.rank, generator=gen_B, device=self.device)
        return A, B
    
    @torch.no_grad()
    def _generate_with_perturbation(self, input_ids, attention_mask, epoch, member_idx):
        # 1. Apply noise directly to weights
        original_weights = {}
        
        for name, param in self.model.named_parameters():
            map_type = self.es_map.get(name, ESMapType.FROZEN)
            if map_type == ESMapType.FROZEN: continue
            
            original_weights[name] = param.data.clone()
            base_seed = self.base_evo_keys[name].seed
            seed = self._get_perturbation_seed(base_seed, epoch, member_idx)
            
            if map_type == ESMapType.LORA:
                A, B = self._generate_lora_perturbation(param.shape, seed)
                # A@B.T result is on self.device
                param.data += A @ B.T
            elif map_type == ESMapType.FULL:
                gen = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn_like(param, generator=gen) * self.config.sigma
                param.data += noise
        
        # 2. Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.config.generation_length,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample
        )
        
        # 3. Restore weights
        for name, original in original_weights.items():
            dict(self.model.named_parameters())[name].data = original
            
        return output_ids

    # ========================================================================
    # Update Logic (Requires Synchronization)
    # ========================================================================
    
    def _estimate_and_update(self, shaped_fitnesses: np.ndarray, epoch: int):
        population_size = len(shaped_fitnesses)
        new_params = {}
        total_grad_norm_sq = 0.0
        
        # Convert fitnesses to tensor for easy calculation if needed, 
        # but here we iterate over params.
        
        for name, param in self.params.items():
            map_type = self.es_map.get(name, ESMapType.FROZEN)
            if map_type == ESMapType.FROZEN: continue
                
            # --- 1. Compute Gradient Estimate (Local to GPU) ---
            gradient = torch.zeros_like(param)
            
            for member_idx in range(population_size):
                R_i = float(shaped_fitnesses[member_idx])
                base_seed = self.base_evo_keys[name].seed
                seed = self._get_perturbation_seed(base_seed, epoch, member_idx)
                
                if map_type == ESMapType.LORA:
                    A, B = self._generate_lora_perturbation(param.shape, seed)
                    gradient += R_i * (A @ B.T)
                elif map_type == ESMapType.FULL:
                    gen = torch.Generator(device=self.device).manual_seed(seed)
                    noise = torch.randn_like(param, generator=gen) * self.config.sigma
                    gradient += R_i * noise / self.config.sigma
                    
            gradient /= population_size
            
            # --- [CHANGED] CRITICAL: SYNCHRONIZE GRADIENTS ---
            # We must average the gradients computed by each GPU (as they saw different data batches)
            # This ensures the update is consistent across all devices.
            gradient = self.accelerator.reduce(gradient, reduction="mean")
            
            # --- 2. Apply Optimizer (SGD/Adam) ---
            update = self._apply_optimizer_step(name, gradient)
            
            # Update parameter (In-place)
            param.data += update
            
            total_grad_norm_sq += torch.norm(gradient).item() ** 2
            
        return math.sqrt(total_grad_norm_sq)

    def _apply_optimizer_step(self, name, gradient):
        lr = self.config.lr_scale
        # (Standard Adam/SGD implementation logic here - same as original)
        if self.config.optimizer_type == "adam":
            t = self.opt_state.step + 1
            beta1, beta2 = self.config.adam_beta1, self.config.adam_beta2
            eps = self.config.adam_eps
            
            m = self.opt_state.momentum[name]
            v = self.opt_state.velocity[name]
            
            m.mul_(beta1).add_(gradient, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            return lr * m_hat / (torch.sqrt(v_hat) + eps)
        else:
            return lr * gradient

    # ========================================================================
    # Training Loop
    # ========================================================================
    
    def train(self, train_data: List[Tuple[str, str]]):
        if self.accelerator.is_main_process:
            print(f"Starting Training on {self.accelerator.num_processes} GPUs")
        
        # [CHANGED] Prepare DataLoader
        # Create Dataset
        dataset = TranslationDataset(train_data)
        
        # We want `prompts_per_epoch` prompts per optimization step.
        # In Multi-GPU, if we want Global Batch Size = X, and we have N GPUs, 
        # local batch size should be X / N.
        # Here we just let `prompts_per_epoch` be the batch size PER GPU for simplicity
        # or you can divide it. Let's assume prompts_per_epoch is per GPU.
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.prompts_per_epoch, 
            shuffle=True
        )
        
        # [CHANGED] Prepare dataloader with accelerator
        # It handles splitting the data among GPUs
        dataloader = self.accelerator.prepare(dataloader)
        
        self.model.train() # Technically we are in eval mode + perturbations, but good practice
        
        progress_bar = tqdm(
            range(self.config.num_epochs), 
            disable=not self.accelerator.is_main_process,
            desc="Training"
        )
        
        # We iterate epochs manually. 
        # Since dataloader provides batches, we treat one batch as "samples for this epoch" 
        # to match the original logic (sampling random prompts per epoch).
        
        data_iterator = iter(dataloader)
        
        for epoch in progress_bar:
            epoch_start = time.time()
            
            # Get data for this epoch (one batch)
            try:
                batch_data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch_data = next(data_iterator)
                
            # batch_data is a tuple of lists (sources, references) or stacked tensors if collated.
            # Default collate gives (tuple_of_src, tuple_of_ref) for list of tuples
            sources, references = batch_data
            
            # --- Generation Step ---
            inputs = self.tokenizer(
                list(sources), # Ensure it's a list
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_source_length
            ).to(self.device)
            
            all_hypotheses = []
            all_references_expanded = []
            
            # For each member in population
            for member_idx in range(self.config.generations_per_prompt):
                output_ids = self._generate_with_perturbation(
                    inputs["input_ids"],
                    inputs.get("attention_mask"),
                    epoch,
                    member_idx
                )
                hyps = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                all_hypotheses.extend(hyps)
                # Repeat references for alignment
                all_references_expanded.extend(list(references))
            
            # --- Reward Step ---
            # Calculates rewards locally on GPU
            raw_rewards = np.array([
                self.reward_function(h, r) for h, r in zip(all_hypotheses, all_references_expanded)
            ])
            
            # Shape: (generations, batch_size). Need to aggregate properly.
            # Original logic: sum rewards per direction.
            # Here: raw_rewards is flat list of (batch_size * generations)
            # We need to reshape to (generations, batch_size) ? 
            # Actually, `_estimate_and_update` expects fitness per MEMBER.
            # Each member ran on the WHOLE batch. So we average reward across the batch for that member.
            
            # Reshape: [Generations, Batch_Size]
            raw_rewards_reshaped = raw_rewards.reshape(self.config.generations_per_prompt, -1)
            # Fitness per member = Mean reward across all prompts in the batch
            member_fitnesses = np.mean(raw_rewards_reshaped, axis=1) # Shape: [Generations]
            
            # --- Shaping Step ---
            shaped_fitnesses = self._shape_fitnesses(member_fitnesses)
            
            # --- Update Step (With Sync) ---
            grad_norm = self._estimate_and_update(shaped_fitnesses, epoch)
            self.opt_state.step += 1
            
            # --- Logging (Main Process Only) ---
            if self.accelerator.is_main_process:
                avg_fit = np.mean(member_fitnesses)
                if self.wandb_run:
                    self.wandb_run.log({
                        "epoch": epoch, 
                        "fitness": avg_fit, 
                        "grad_norm": grad_norm
                    })
                
                if epoch % self.config.log_every == 0:
                    progress_bar.set_postfix({"fit": f"{avg_fit:.4f}", "grad": f"{grad_norm:.4f}"})
                    
            # Save Checkpoint
            if self.config.save_model and epoch % self.config.save_every == 0:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self._save_checkpoint(epoch, avg_fit)

    def _shape_fitnesses(self, raw_scores):
        # Centered Rank shaping
        n = len(raw_scores)
        ranks = np.argsort(np.argsort(raw_scores))
        return (ranks.astype(np.float32) + 0.5) / n - 0.5

    def _save_checkpoint(self, epoch, fitness):
        ckpt_dir = Path(self.config.save_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
        
        # Save only weights and opt state
        torch.save({
            'epoch': epoch,
            'params': {k: v.cpu() for k, v in self.params.items()}, # Params are shared with model
            'opt_state': self.opt_state, 
        }, path)
        print(f"Saved checkpoint to {path}")

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    config = EggrollTrainerConfig()
    
    # Dummy Data for demonstration
    train_data = [
        ("Hello world", "Xin chào thế giới"),
        ("Good morning", "Chào buổi sáng"),
        ("How are you", "Bạn khỏe không"),
        ("I love AI", "Tôi yêu AI"),
        ("Translate this", "Dịch cái này"),
        ("Computer science", "Khoa học máy tính"),
        ("Neural networks", "Mạng nơ ron"),
        ("Deep learning", "Học sâu"),
        ("Large language model", "Mô hình ngôn ngữ lớn"),
        ("Distributed training", "Huấn luyện phân tán"),
    ] * 10 # Duplicate to simulate more data
    
    trainer = EggrollTrainer(config)
    trainer.setup()
    trainer.train(train_data)

if __name__ == "__main__":
    main()
