import os
import time
import GPUtil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Model paths
model_path = "/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16"  # Path to pre-trained model
save_path = "/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16_int4"  # Path to save quantized model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a configuration object to specify quantization parameters
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    load_in_8bit=False,  # Disable 8-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Computation precision setting
    bnb_4bit_quant_storage=torch.uint8,  # Storage format for quantized weights
    bnb_4bit_quant_type="nf4",  # Quantization type (NormalFloat4)
    bnb_4bit_use_double_quant=True,  # Enable double quantization (quantizes scaling factors)
    llm_int8_enable_fp32_cpu_offload=False,  # Enable CPU offloading with FP32
    llm_int8_has_fp16_weight=False,  # Enable mixed precision
    # llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # Modules to skip quantization
    llm_int8_threshold=6.0,  # Outlier threshold for llm.int8() algorithm
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,  # Map model to device
    quantization_config=quantization_config,
    trust_remote_code=True,
)

gpu_usage = GPUtil.getGPUs()[0].memoryUsed
start = time.time()
response = model.chat(
    tokenizer,
    "<User>Tell me a story<AI>",
    history=[],
    temperature=0.5,
    top_p=0.8,
    repetition_penalty=1.02,
)  # Model inference
print("Quantized output:", response)
print("Quantized inference time:", time.time() - start)
print(f"Quantized VRAM usage: {round(gpu_usage / 1024, 2)}GB")

# Save model and tokenizer
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
