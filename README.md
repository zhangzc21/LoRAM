# LoRAM: Magnitude-Driven LoRA Initialization for Effortless High Performance

The repository is still under development.

## Introduction

Tired of complex spectral initialization methods that slow down your LoRA (Low-Rank Adaptation) fine-tuning? LoRAM is your solution. Our lightweight initialization scheme for LoRA simplifies the process of optimizing large models, delivering spectral-level performance without the extra computational and storage headaches.

**Why LoRAM?**
- **Effortless Efficiency**: LoRAM ditches the cumbersome computations and storage demands of traditional spectral methods. It's a plug-and-play approach that integrates seamlessly with existing LoRA setups, saving you time and resources.
- **Insight-Driven Design**: We've cracked the code on what truly drives LoRA performance - the magnitude of weight updates (**Please see our Paper**!). LoRAM uses this insight to scale deterministic orthogonal bases with pretrained weight magnitudes, a simple yet powerful strategy that mimics spectral gains without the complexity.
- **Proven Results**: Across multiple benchmarksfor LLM, MLLM and Flow Model, LoRAM stands as a robust baseline, matching or surpassing spectral initialization while maintaining the full parameter efficiency of LoRA.

If you're looking for a straightforward, high-performing way to initialize LoRA for your large model tuning tasks, LoRAM is the way to go. No more juggling hyperparameters or dealing with excessive overhead - just fast convergence and top-notch performance, right out of the box. 


## How to Use: Integrating LoRAM with PEFT Package

### Step 1: Add LoRAM Initialization to `layer.py`

You need to add the LoRAM initialization code to the `src/peft/tuners/lora/layer.py` file in the PEFT package. Refer to the detailed implementation in the `layer.py` file from lines 155-157: [layer.py](https://github.com/zhangzc21/LoRAM/blob/8fdbd90546a94eb4a3d6562390ec62ef4446b78a/layer.py#L155). （Considering version compatibility, the most appropriate approach is to modify the code directly in your own active PEFT library）

### Step 2: Set Initialization Parameter During Training
When training your model, set the `init_lora_weights` parameter to `"loram"` in the `LoraConfig`. Here is an example:
```python
from peft import LoraConfig, get_peft_model

# Assume base_model is your pre-trained model
base_model = ...

config = LoraConfig(init_lora_weights="loram", ...)
model = get_peft_model(base_model, config)
```

### Step 3: Inference
No modifications are required during the LoRA inference process. The LoRAM initialization will be automatically applied.

## Key Features
- **Deterministic Initialization**: LoRAM uses the dst function for deterministic initialization, ensuring consistent results across different runs.
- **SOTA Performance**: LoRAM achieves SOTA performance comparable to PiSSA, which uses SVD initialization, with theoretical guarantees.
- **Reduced Computational Overhead**: Unlike PiSSA, LoRAM does not require SVD computation and saving initialization parameters, resulting in lower computational and storage costs.

## License
This integration follows the same license as the PEFT package. Please refer to the original PEFT repository for detailed license information.
