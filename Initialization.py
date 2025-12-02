"""
EGGROLL (Evolution Guided General Revolution Optimization via Low-rank Learning)
Bước 1: Initialization - Chuẩn bị mô hình và hyperparameters
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel
)


@dataclass
class EGGROLLConfig:
    """
    Cấu hình hyperparameters cho EGGROLL. 
    
    Attributes:
        sigma: Độ lệch chuẩn của nhiễu (noise standard deviation)
        alpha: Tốc độ học (learning rate)
        population_size: Kích thước quần thể N - số lượng biến thể mô hình
        rank: Hạng r của ma trận nhiễu low-rank (r << d)
        use_antithetic: Sử dụng Antithetic Sampling để giảm phương sai
        target_modules: Các module sẽ được finetune (None = tất cả linear layers)
        seed: Random seed để tái tạo kết quả
    """
    sigma: float = 0.01
    alpha: float = 1e-3
    population_size: int = 64
    rank: int = 16
    use_antithetic: bool = True
    target_modules: Optional[list] = None
    seed: Optional[int] = 42
    
    def __post_init__(self):
        """Validate hyperparameters."""
        assert self.sigma > 0, "sigma phải > 0"
        assert self.alpha > 0, "alpha phải > 0"
        assert self.population_size > 0, "population_size phải > 0"
        assert self.rank > 0, "rank phải > 0"
        
        if self.use_antithetic:
            # Với antithetic sampling, population_size phải là số chẵn
            assert self.population_size % 2 == 0, \
                "population_size phải là số chẵn khi dùng antithetic sampling"


@dataclass
class ParameterInfo:
    """
    Thông tin về một tham số cần finetune.
    
    Attributes:
        name: Tên của parameter
        shape: Kích thước của parameter (d1, d2)
        dtype: Kiểu dữ liệu
        device: Device (cpu/cuda)
        original_param: Reference đến parameter gốc
    """
    name: str
    shape: tuple
    dtype: torch.dtype
    device: torch.device
    original_param: nn.Parameter


class EGGROLLInitializer:
    """
    Class khởi tạo EGGROLL cho Translation Model.
    
    Quản lý việc:
    - Load pre-trained model
    - Freeze parameters
    - Xác định các layers cần finetune
    - Thiết lập cấu trúc low-rank
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        config: EGGROLLConfig,
        device: Optional[str] = None
    ):
        """
        Khởi tạo EGGROLL.
        
        Args:
            model_name_or_path: Tên hoặc đường dẫn đến pre-trained model
            config: Cấu hình EGGROLL
            device: Device để load model ('cuda', 'cpu', hoặc None để tự động)
        """
        self.config = config
        self.device = device or ('cuda' if torch. cuda.is_available() else 'cpu')
        
        # Set random seed cho reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
        
        # Load model và tokenizer
        print(f"[Bước 1. 1] Loading pre-trained model: {model_name_or_path}")
        self.model = self._load_model(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Freeze tất cả parameters
        print("[Bước 1.2] Freezing all model parameters...")
        self._freeze_model()
        
        # Xác định các parameters sẽ được perturb
        print("[Bước 1.3] Identifying target parameters for low-rank perturbation...")
        self. target_params = self._identify_target_parameters()
        
        # Tính toán và hiển thị thống kê
        self._print_statistics()
    
    def _load_model(self, model_name_or_path: str) -> PreTrainedModel:
        """Load pre-trained translation model."""
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,  # Có thể đổi sang float16 để tiết kiệm memory
        )
        model = model.to(self.device)
        model.eval()  # Set evaluation mode
        return model
    
    def _freeze_model(self):
        """Freeze tất cả parameters - ES không cần gradient."""
        for param in self.model. parameters():
            param.requires_grad = False
    
    def _identify_target_parameters(self) -> Dict[str, ParameterInfo]:
        """
        Xác định các parameters sẽ được perturb với low-rank matrices.
        
        Mặc định: Tất cả Linear layers (nn.Linear) trong Transformer. 
        Có thể customize qua config.target_modules. 
        """
        target_params = {}
        
        for name, module in self.model. named_modules():
            # Chỉ xét Linear layers (hoặc các modules được chỉ định)
            if isinstance(module, nn. Linear):
                # Kiểm tra nếu có target_modules filter
                if self. config.target_modules is not None:
                    if not any(target in name for target in self.config.target_modules):
                        continue
                
                # Lưu thông tin parameter weight
                param = module.weight
                param_info = ParameterInfo(
                    name=f"{name}.weight",
                    shape=param.shape,
                    dtype=param.dtype,
                    device=param. device,
                    original_param=param
                )
                target_params[f"{name}.weight"] = param_info
        
        return target_params
    
    def _print_statistics(self):
        """In thống kê về model và EGGROLL configuration."""
        # Tổng số parameters của model
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Số parameters sẽ được perturb
        target_params_count = sum(
            info.shape[0] * info.shape[1] 
            for info in self.target_params. values()
        )
        
        # Memory tiết kiệm được với low-rank
        # Full perturbation: N * target_params_count * 4 bytes (float32)
        # Low-rank: N * sum((d1 + d2) * r) * 4 bytes
        full_memory = self.config.population_size * target_params_count * 4
        lowrank_memory = self.config.population_size * sum(
            (info.shape[0] + info.shape[1]) * self. config.rank * 4
            for info in self.target_params. values()
        )
        memory_saved_ratio = (1 - lowrank_memory / full_memory) * 100
        
        print("\n" + "="*60)
        print("EGGROLL INITIALIZATION COMPLETE")
        print("="*60)
        print(f"\n📊 Model Statistics:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Target parameters (for perturbation): {target_params_count:,}")
        print(f"   - Number of target layers: {len(self.target_params)}")
        
        print(f"\n⚙️  EGGROLL Hyperparameters:")
        print(f"   - σ (noise std): {self.config.sigma}")
        print(f"   - α (learning rate): {self.config.alpha}")
        print(f"   - N (population size): {self.config.population_size}")
        print(f"   - r (rank): {self.config. rank}")
        print(f"   - Antithetic sampling: {self.config.use_antithetic}")
        
        print(f"\n💾 Memory Efficiency:")
        print(f"   - Full perturbation memory: {full_memory / 1e9:.2f} GB")
        print(f"   - Low-rank perturbation memory: {lowrank_memory / 1e9:. 4f} GB")
        print(f"   - Memory saved: {memory_saved_ratio:.2f}%")
        print("="*60 + "\n")
    
    def get_parameter_shapes(self) -> Dict[str, tuple]:
        """Trả về dictionary mapping tên parameter -> shape."""
        return {name: info.shape for name, info in self.target_params.items()}
    
    def get_original_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Trả về bản sao của parameters gốc θ.
        Dùng làm baseline cho perturbation.
        """
        return {
            name: info.original_param. data.clone()
            for name, info in self.target_params.items()
        }


def initialize_eggroll(
    model_name: str = "Helsinki-NLP/opus-mt-en-vi",
    sigma: float = 0.01,
    alpha: float = 1e-3,
    population_size: int = 64,
    rank: int = 16,
    use_antithetic: bool = True,
    target_modules: Optional[list] = None,
    seed: int = 42
) -> EGGROLLInitializer:
    """
    Convenience function để khởi tạo EGGROLL cho Translation Model.
    
    Args:
        model_name: Tên pre-trained model (ví dụ: Helsinki-NLP/opus-mt-en-vi)
        sigma: Độ lệch chuẩn của nhiễu
        alpha: Learning rate
        population_size: Kích thước quần thể
        rank: Hạng của low-rank matrices
        use_antithetic: Sử dụng antithetic sampling
        target_modules: List các module names để filter (None = all linear layers)
        seed: Random seed
    
    Returns:
        EGGROLLInitializer instance đã được khởi tạo
    
    Example:
        >>> eggroll = initialize_eggroll(
        ...     model_name="Helsinki-NLP/opus-mt-en-vi",
        ...      sigma=0. 01,
        ...      alpha=1e-3,
        ...     population_size=64,
        ...     rank=16
        ... )
    """
    config = EGGROLLConfig(
        sigma=sigma,
        alpha=alpha,
        population_size=population_size,
        rank=rank,
        use_antithetic=use_antithetic,
        target_modules=target_modules,
        seed=seed
    )
    
    return EGGROLLInitializer(
        model_name_or_path=model_name,
        config=config
    )


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # Ví dụ 1: Khởi tạo với model dịch Anh-Việt
    print("🚀 Initializing EGGROLL for English-Vietnamese Translation Model\n")
    
    eggroll = initialize_eggroll(
        model_name="Helsinki-NLP/opus-mt-en-vi",  # Model dịch Anh -> Việt
        sigma=0.01,           # Noise standard deviation
        alpha=1e-3,           # Learning rate
        population_size=64,   # Số candidates mỗi iteration
        rank=16,              # Low-rank dimension
        use_antithetic=True,  # Giảm variance
        seed=42
    )
    
    # Truy cập các thành phần
    print("📝 Accessible components:")
    print(f"   - Model: {type(eggroll. model).__name__}")
    print(f"   - Tokenizer: {type(eggroll.tokenizer).__name__}")
    print(f"   - Config: {eggroll.config}")
    print(f"   - Number of target params: {len(eggroll.target_params)}")
    
    # Lấy shapes để chuẩn bị cho Bước 2 (tạo low-rank matrices)
    param_shapes = eggroll.get_parameter_shapes()
    print(f"\n📐 First 5 parameter shapes (for low-rank matrix generation):")
    for i, (name, shape) in enumerate(list(param_shapes. items())[:5]):
        print(f"   {name}: {shape} -> A: ({shape[0]}, {eggroll.config. rank}), B: ({shape[1]}, {eggroll.config. rank})")
