"""
EGGROLL Step 3: Apply Perturbations & Forward Pass

This module implements efficient forward pass with low-rank perturbations
WITHOUT materializing the full perturbed weight matrices. 

Key insight: For y = x @ (W + A @ B. T). T
           = x @ W.T + x @ B @ A.T
           
This avoids storing the (d × d) perturbed matrix. 

Based on: https://github.com/ESHyperscale/HyperscaleES
Paper: "Evolution Strategies at Hyperscale" (arXiv:2511.16652)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import copy

# Import from previous steps
from eggroll_initialization import (
    EggrollConfig,
    FrozenNoiserParams,
    NoiserParams,
    ESMapType,
    RandomKeyGenerator,
)
from eggroll_perturbation import (
    LoRAPerturbation,
    FullPerturbation,
    PopulationMember,
    PerturbationGenerator,
    fold_in_key,
)


# ============================================================================
# Forward Pass Hooks for Memory-Efficient LoRA
# ============================================================================

class LoRAForwardHook:
    """
    Hook to add low-rank perturbation during forward pass.
    
    Instead of modifying weights directly, we intercept the output
    and add the low-rank contribution: output += x @ B @ A.T
    """
    
    def __init__(
        self,
        perturbation: LoRAPerturbation,
        param_name: str,
    ):
        self. perturbation = perturbation
        self.param_name = param_name
        self.input_cache = None
        
    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch. Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add low-rank perturbation to layer output.
        
        For Linear layer: y = x @ W.T + bias
        We add: y += x @ B @ A. T
        
        Total: y = x @ W.T + x @ B @ A.T + bias = x @ (W + A @ B.T). T + bias
        """
        x = input[0]  # Input tensor: (batch, seq_len, in_features)
        
        A = self.perturbation.A  # (out_features, rank)
        B = self.perturbation.B  # (in_features, rank)
        
        # Efficient computation: x @ B @ A.T
        # Step 1: x @ B -> (... , rank)  - reduces dimension
        # Step 2: result @ A.T -> (..., out_features)
        lora_output = (x @ B) @ A.T
        
        return output + lora_output


class FullPerturbationHook:
    """
    Hook to add full perturbation to weights before forward pass.
    
    Used for non-LoRA parameters (biases, small matrices).
    """
    
    def __init__(
        self,
        perturbation: FullPerturbation,
        param_name: str,
    ):
        self.perturbation = perturbation
        self.param_name = param_name
        self.original_weight = None
        
    def pre_hook(
        self,
        module: nn.Module,
        input: Tuple[torch. Tensor],
    ):
        """Store original and apply perturbation before forward."""
        # Get the parameter (weight or bias)
        param_attr = self.param_name.split('.')[-1]  # 'weight' or 'bias'
        
        if hasattr(module, param_attr):
            param = getattr(module, param_attr)
            if param is not None:
                self.original_weight = param.data.clone()
                param.data = param.data + self. perturbation.noise
                
    def post_hook(
        self,
        module: nn.Module,
        input: Tuple[torch. Tensor],
        output: torch.Tensor,
    ):
        """Restore original weight after forward."""
        param_attr = self. param_name.split('.')[-1]
        
        if self.original_weight is not None and hasattr(module, param_attr):
            param = getattr(module, param_attr)
            if param is not None:
                param.data = self.original_weight
                self.original_weight = None


# ============================================================================
# Perturbed Model Context Manager
# ============================================================================

@dataclass
class HookHandle:
    """Container for registered hooks"""
    handle: Any
    hook: Any
    hook_type: str  # 'forward' or 'pre_forward'


class PerturbedModelContext:
    """
    Context manager for applying perturbations to a model.
    
    Uses PyTorch hooks to efficiently add low-rank perturbations
    during forward pass without modifying the original weights.
    
    Usage:
        with PerturbedModelContext(model, perturbation_member, param_to_module):
            output = model. generate(...)
    """
    
    def __init__(
        self,
        model: nn.Module,
        population_member: PopulationMember,
        param_to_module: Dict[str, nn.Module],
        es_map: Dict[str, int],
    ):
        """
        Initialize perturbed model context. 
        
        Args:
            model: The base model
            population_member: Perturbations to apply
            param_to_module: Mapping from parameter names to their modules
            es_map: Parameter classification map
        """
        self.model = model
        self. population_member = population_member
        self.param_to_module = param_to_module
        self.es_map = es_map
        self.hook_handles: List[HookHandle] = []
        
    def __enter__(self):
        """Register hooks for all perturbations."""
        # Register LoRA hooks (forward hooks)
        for param_name, perturbation in self. population_member.lora_perturbations.items():
            if param_name in self.param_to_module:
                module = self.param_to_module[param_name]
                hook = LoRAForwardHook(perturbation, param_name)
                handle = module.register_forward_hook(hook)
                self.hook_handles.append(HookHandle(handle, hook, 'forward'))
                
        # Register full perturbation hooks (for non-LoRA params)
        for param_name, perturbation in self. population_member.full_perturbations. items():
            if param_name in self.param_to_module:
                module = self.param_to_module[param_name]
                hook = FullPerturbationHook(perturbation, param_name)
                # Pre-forward hook to modify weights
                pre_handle = module.register_forward_pre_hook(hook. pre_hook)
                # Post-forward hook to restore weights
                post_handle = module.register_forward_hook(hook.post_hook)
                self.hook_handles. append(HookHandle(pre_handle, hook, 'pre_forward'))
                self.hook_handles.append(HookHandle(post_handle, hook, 'forward'))
                
        return self. model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all registered hooks."""
        for hook_handle in self. hook_handles:
            hook_handle. handle.remove()
        self.hook_handles.clear()
        return False


# ============================================================================
# Parameter to Module Mapping
# ============================================================================

def build_param_to_module_map(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Build mapping from parameter names to their parent modules.
    
    This is needed to register hooks on the correct modules. 
    
    Args:
        model: The neural network model
        
    Returns:
        Dictionary mapping parameter names to modules
    """
    param_to_module = {}
    
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            param_to_module[full_name] = module
            
    return param_to_module


def build_weight_to_module_map(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Build mapping specifically for weight parameters to Linear/Conv modules.
    
    For LoRA, we need the Linear layer to hook into. 
    """
    weight_to_module = {}
    
    for name, module in model. named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            weight_name = f"{name}.weight" if name else "weight"
            weight_to_module[weight_name] = module
            
    return weight_to_module


# ============================================================================
# Direct Weight Modification (Alternative Approach)
# ============================================================================

class DirectPerturbationContext:
    """
    Alternative: Directly modify weights instead of using hooks. 
    
    Simpler but requires storing original weights.
    Better for small models or when hooks cause issues.
    """
    
    def __init__(
        self,
        model: nn.Module,
        population_member: PopulationMember,
    ):
        self. model = model
        self.population_member = population_member
        self.original_params: Dict[str, torch.Tensor] = {}
        
    def __enter__(self):
        """Apply perturbations to model weights."""
        state_dict = self. model.state_dict()
        
        # Apply LoRA perturbations
        for param_name, perturbation in self.population_member.lora_perturbations.items():
            if param_name in state_dict:
                self.original_params[param_name] = state_dict[param_name].clone()
                # Materialize: W_new = W + A @ B.T
                state_dict[param_name] = (
                    state_dict[param_name] + 
                    perturbation. A @ perturbation.B.T
                )
                
        # Apply full perturbations
        for param_name, perturbation in self. population_member.full_perturbations. items():
            if param_name in state_dict:
                self.original_params[param_name] = state_dict[param_name].clone()
                state_dict[param_name] = state_dict[param_name] + perturbation. noise
                
        # Load modified state dict
        self.model.load_state_dict(state_dict)
        
        return self. model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original weights."""
        if self.original_params:
            state_dict = self. model.state_dict()
            for param_name, original in self.original_params.items():
                state_dict[param_name] = original
            self.model.load_state_dict(state_dict)
            self.original_params. clear()
        return False


# ============================================================================
# Translation Generation with Perturbed Model
# ============================================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 128
    num_beams: int = 1  # 1 for greedy, >1 for beam search
    temperature: float = 1.0
    do_sample: bool = False
    top_k: int = 50
    top_p: float = 1.0
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None


@dataclass
class GenerationResult:
    """Result from generation with perturbed model"""
    member_index: int
    input_ids: torch.Tensor
    output_ids: torch. Tensor
    input_text: str
    output_text: str
    generation_time: float


class PerturbedModelGenerator:
    """
    Handles text generation with perturbed models.
    
    For each population member:
    1. Apply perturbations to model (via hooks or direct modification)
    2. Generate translation
    3.  Restore original weights
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        perturbation_generator: PerturbationGenerator,
        es_map: Dict[str, int],
        generation_config: Optional[GenerationConfig] = None,
        use_hooks: bool = True,  # True for memory efficiency, False for simplicity
        device: str = "cuda",
    ):
        """
        Initialize the generator.
        
        Args:
            model: Base translation model
            tokenizer: Tokenizer for encoding/decoding
            perturbation_generator: Generator for perturbations
            es_map: Parameter classification map
            generation_config: Configuration for text generation
            use_hooks: Whether to use hooks (memory efficient) or direct modification
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.perturbation_generator = perturbation_generator
        self.es_map = es_map
        self.generation_config = generation_config or GenerationConfig()
        self. use_hooks = use_hooks
        self.device = device
        
        # Build parameter to module mapping for hooks
        self.param_to_module = build_weight_to_module_map(model)
        
        # Set tokenizer special tokens in config
        if self.generation_config.pad_token_id is None:
            self. generation_config.pad_token_id = tokenizer.pad_token_id
        if self.generation_config.eos_token_id is None:
            self.generation_config. eos_token_id = tokenizer. eos_token_id
            
    def _get_context_manager(self, population_member: PopulationMember):
        """Get appropriate context manager for perturbation."""
        if self.use_hooks:
            return PerturbedModelContext(
                self.model,
                population_member,
                self. param_to_module,
                self. es_map,
            )
        else:
            return DirectPerturbationContext(
                self. model,
                population_member,
            )
    
    @torch.no_grad()
    def generate_single(
        self,
        population_member: PopulationMember,
        input_text: str,
    ) -> GenerationResult:
        """
        Generate translation with a single perturbed model. 
        
        Args:
            population_member: Perturbations to apply
            input_text: Source text to translate
            
        Returns:
            GenerationResult containing input/output text and metadata
        """
        import time
        start_time = time. time()
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.generation_config. max_length,
        ). to(self.device)
        
        # Generate with perturbed model
        with self._get_context_manager(population_member):
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs. get("attention_mask"),
                max_length=self.generation_config.max_length,
                num_beams=self. generation_config.num_beams,
                temperature=self. generation_config.temperature if self.generation_config. do_sample else 1.0,
                do_sample=self. generation_config. do_sample,
                top_k=self.generation_config. top_k if self.generation_config.do_sample else None,
                top_p=self.generation_config.top_p if self.generation_config.do_sample else None,
                early_stopping=self.generation_config.early_stopping,
                pad_token_id=self. generation_config.pad_token_id,
                eos_token_id=self.generation_config. eos_token_id,
            )
            
        generation_time = time. time() - start_time
        
        # Decode output
        output_text = self.tokenizer. decode(output_ids[0], skip_special_tokens=True)
        
        return GenerationResult(
            member_index=population_member. index,
            input_ids=inputs["input_ids"],
            output_ids=output_ids,
            input_text=input_text,
            output_text=output_text,
            generation_time=generation_time,
        )
    
    @torch.no_grad()
    def generate_batch(
        self,
        population_member: PopulationMember,
        input_texts: List[str],
    ) -> List[GenerationResult]:
        """
        Generate translations for a batch of inputs with perturbed model. 
        
        Args:
            population_member: Perturbations to apply
            input_texts: List of source texts to translate
            
        Returns:
            List of GenerationResult objects
        """
        import time
        start_time = time.time()
        
        # Tokenize batch
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self. generation_config.max_length,
        ).to(self. device)
        
        # Generate with perturbed model
        with self._get_context_manager(population_member):
            output_ids = self. model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=self.generation_config.max_length,
                num_beams=self. generation_config.num_beams,
                temperature=self. generation_config.temperature if self.generation_config.do_sample else 1.0,
                do_sample=self.generation_config.do_sample,
                early_stopping=self. generation_config.early_stopping,
                pad_token_id=self.generation_config.pad_token_id,
                eos_token_id=self.generation_config.eos_token_id,
            )
            
        total_time = time. time() - start_time
        time_per_sample = total_time / len(input_texts)
        
        # Create results
        results = []
        for i, (input_text, out_ids) in enumerate(zip(input_texts, output_ids)):
            output_text = self. tokenizer.decode(out_ids, skip_special_tokens=True)
            results.append(GenerationResult(
                member_index=population_member.index,
                input_ids=inputs["input_ids"][i:i+1],
                output_ids=out_ids. unsqueeze(0),
                input_text=input_text,
                output_text=output_text,
                generation_time=time_per_sample,
            ))
            
        return results
    
    @torch.no_grad()
    def generate_population(
        self,
        epoch: int,
        population_size: int,
        input_texts: List[str],
        verbose: bool = True,
    ) -> Dict[int, List[GenerationResult]]:
        """
        Generate translations for all population members.
        
        This is the main function for Step 3 of EGGROLL:
        For each member i in population:
            θ_i = θ + σ(A_i × B_i^T)
            outputs_i = generate(θ_i, inputs)
        
        Args:
            epoch: Current epoch
            population_size: Number of population members
            input_texts: Source texts to translate
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping member index to list of generation results
        """
        import time
        from tqdm import tqdm
        
        all_results = {}
        
        iterator = range(population_size)
        if verbose:
            iterator = tqdm(iterator, desc=f"Epoch {epoch} - Generating")
            
        for member_idx in iterator:
            # Generate perturbation for this member
            population_member = self.perturbation_generator.generate_population_member(
                epoch=epoch,
                member_idx=member_idx,
            )
            
            # Generate translations with perturbed model
            results = self.generate_batch(population_member, input_texts)
            all_results[member_idx] = results
            
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    "member": member_idx,
                    "time": f"{results[0].generation_time:. 3f}s",
                })
                
        return all_results


# ============================================================================
# Batch Processing for Efficiency
# ============================================================================

@dataclass
class BatchGenerationConfig:
    """Configuration for batched population generation"""
    batch_size: int = 8  # Number of samples per batch
    population_batch_size: int = 1  # Number of population members to process together
    

class EfficientPopulationGenerator:
    """
    Efficient generation across population with batching.
    
    Optimizations:
    1.  Batch multiple input samples together
    2.  Reuse KV cache where possible
    3.  Minimize perturbation regeneration
    """
    
    def __init__(
        self,
        generator: PerturbedModelGenerator,
        batch_config: Optional[BatchGenerationConfig] = None,
    ):
        self.generator = generator
        self.batch_config = batch_config or BatchGenerationConfig()
        
    def generate_epoch(
        self,
        epoch: int,
        population_size: int,
        input_texts: List[str],
        verbose: bool = True,
    ) -> Dict[int, List[GenerationResult]]:
        """
        Generate for entire population with batching.
        
        Args:
            epoch: Current epoch
            population_size: Number of population members
            input_texts: All source texts for this epoch
            verbose: Print progress
            
        Returns:
            Results for all population members
        """
        # Batch input texts
        batched_inputs = [
            input_texts[i:i + self.batch_config.batch_size]
            for i in range(0, len(input_texts), self.batch_config.batch_size)
        ]
        
        all_results = {}
        
        for member_idx in range(population_size):
            member_results = []
            
            # Generate perturbation once per member
            population_member = self. generator.perturbation_generator.generate_population_member(
                epoch=epoch,
                member_idx=member_idx,
            )
            
            # Process batches
            for batch in batched_inputs:
                batch_results = self. generator.generate_batch(population_member, batch)
                member_results.extend(batch_results)
                
            all_results[member_idx] = member_results
            
            if verbose:
                print(f"  Member {member_idx}: {len(member_results)} samples generated")
                
        return all_results


# ============================================================================
# Utility Functions
# ============================================================================

def verify_perturbation_applied(
    model: nn.Module,
    population_member: PopulationMember,
    param_to_module: Dict[str, nn.Module],
    sample_input: torch.Tensor,
) -> Dict[str, bool]:
    """
    Verify that perturbations are correctly applied during forward pass.
    
    Compares outputs with and without perturbation. 
    """
    results = {}
    
    # Get output without perturbation
    with torch.no_grad():
        base_output = model(sample_input)
        
    # Get output with perturbation
    with DirectPerturbationContext(model, population_member):
        with torch.no_grad():
            perturbed_output = model(sample_input)
            
    # Check if outputs are different
    if hasattr(base_output, 'logits'):
        base_logits = base_output.logits
        perturbed_logits = perturbed_output.logits
    else:
        base_logits = base_output
        perturbed_logits = perturbed_output
        
    diff = torch.abs(perturbed_logits - base_logits). mean(). item()
    results['output_difference'] = diff
    results['perturbation_applied'] = diff > 1e-6
    
    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from eggroll_initialization import initialize_eggroll, EggrollConfig
    
    print("=" * 70)
    print("EGGROLL Step 3: Apply Perturbations & Forward Pass")
    print("=" * 70)
    
    # Step 1: Initialize
    config = EggrollConfig(
        model_name="Helsinki-NLP/opus-mt-en-vi",
        sigma=1e-3,
        lr_scale=1.0,
        rank=16,
        population_size=4,  # Small for demo
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
    from eggroll_perturbation import PerturbationGenerator
    
    perturbation_generator = PerturbationGenerator(
        params=params,
        es_map=es_map,
        base_evo_keys=base_evo_keys,
        frozen_noiser_params=frozen_noiser_params,
        noiser_params=noiser_params,
        device=config.device,
    )
    
    # Step 3: Create generator and run forward pass
    print("\n" + "-" * 70)
    print("Setting up Perturbed Model Generator...")
    print("-" * 70)
    
    generation_config = GenerationConfig(
        max_length=128,
        num_beams=1,  # Greedy decoding for speed
        temperature=1.0,
        do_sample=False,
    )
    
    generator = PerturbedModelGenerator(
        model=model,
        tokenizer=tokenizer,
        perturbation_generator=perturbation_generator,
        es_map=es_map,
        generation_config=generation_config,
        use_hooks=False,  # Use direct modification for simplicity
        device=config.device,
    )
    
    # Test inputs
    test_inputs = [
        "Hello, how are you today?",
        "The weather is very nice.",
        "I love learning new languages.",
        "Machine translation is fascinating.",
    ]
    
    print(f"\nTest inputs ({len(test_inputs)} samples):")
    for i, text in enumerate(test_inputs):
        print(f"  {i+1}. {text}")
    
    # Generate with base model (no perturbation)
    print("\n" + "-" * 70)
    print("Generating with BASE model (no perturbation)...")
    print("-" * 70)
    
    model. eval()
    with torch.no_grad():
        inputs = tokenizer(test_inputs, return_tensors="pt", padding=True). to(config.device)
        base_outputs = model.generate(**inputs, max_length=128)
        base_translations = tokenizer.batch_decode(base_outputs, skip_special_tokens=True)
        
    print("\nBase model translations:")
    for src, tgt in zip(test_inputs, base_translations):
        print(f"  EN: {src}")
        print(f"  VI: {tgt}")
        print()
    
    # Generate with perturbed models
    print("-" * 70)
    print(f"Generating with {config.population_size} PERTURBED models...")
    print("-" * 70)
    
    epoch = 0
    all_results = generator.generate_population(
        epoch=epoch,
        population_size=config.population_size,
        input_texts=test_inputs,
        verbose=True,
    )
    
    # Show results for each population member
    print("\n" + "-" * 70)
    print("Results by Population Member")
    print("-" * 70)
    
    for member_idx in range(config.population_size):
        print(f"\n[Population Member {member_idx}]")
        results = all_results[member_idx]
        
        for result in results[:2]:  # Show first 2 examples
            print(f"  EN: {result. input_text}")
            print(f"  VI: {result.output_text}")
            print(f"  Time: {result.generation_time:.4f}s")
            print()
            
    # Compare translations across population
    print("-" * 70)
    print("Translation Diversity Across Population")
    print("-" * 70)
    
    print(f"\nInput: \"{test_inputs[0]}\"")
    print("\nTranslations:")
    print(f"  Base:     {base_translations[0]}")
    for member_idx in range(config.population_size):
        translation = all_results[member_idx][0].output_text
        is_different = translation != base_translations[0]
        diff_marker = " ←different" if is_different else ""
        print(f"  Member {member_idx}: {translation}{diff_marker}")
    
    # Verify perturbation is working
    print("\n" + "-" * 70)
    print("Verifying Perturbation Effect")
    print("-" * 70)
    
    population_member = perturbation_generator.generate_population_member(epoch=0, member_idx=0)
    param_to_module = build_weight_to_module_map(model)
    
    sample_input = tokenizer(test_inputs[0], return_tensors="pt").to(config.device)
    verification = verify_perturbation_applied(
        model, 
        population_member, 
        param_to_module,
        sample_input["input_ids"],
    )
    
    print(f"Output difference (perturbed vs base): {verification['output_difference']:. 6f}")
    print(f"Perturbation successfully applied: {verification['perturbation_applied']}")
    
    # Memory analysis
    print("\n" + "-" * 70)
    print("Memory Analysis")
    print("-" * 70)
    
    lora_memory = population_member.total_memory_bytes()
    print(f"Memory per population member: {lora_memory / 1024:. 2f} KB")
    print(f"Total for population of {config.population_size}: {lora_memory * config.population_size / 1024 / 1024:. 2f} MB")
    
    # Summary
    print("\n" + "=" * 70)
    print("Step 3 Complete!")
    print("=" * 70)
    print(f"""
Summary:
--------
• Generated translations for {config.population_size} population members
• Each member processed {len(test_inputs)} input samples
• Perturbation rank: {config.rank}
• Noise σ: {config. sigma}

Output format for Step 4:
-------------------------
all_results = {{
    member_idx: [GenerationResult(input_text, output_text, .. .), ... ],
    ... 
}}

Next: Step 4 - Compute Rewards (BLEU scores) for each translation
""")
