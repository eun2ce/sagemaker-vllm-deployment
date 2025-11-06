"""
SageMaker inference script for LLM models using DJL Python + vLLM.
Non-streaming version.
Supports any HuggingFace model compatible with vLLM.
"""

from djl_python import Input, Output
import logging
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None


def load_model(properties):
    """
    Load the LLM model using vLLM.
    
    Supports various context lengths based on available VRAM:
    
    max_model_len Options:
    - 32,768 (32K):   ~16GB VRAM - Recommended for general chat, fast responses
    - 65,536 (64K):   ~24GB VRAM - Medium documents
    - 131,072 (128K): ~40GB VRAM - Long document analysis
    - 262,144 (256K): ~80GB VRAM - Maximum context (requires A100 80GB or similar)
    
    Default: 32,768 (balanced performance and memory usage)
    """

    model_location = properties.get("model_id", properties.get("model_dir", "/opt/ml/model"))
    
    logger.info(f"Loading model from {model_location}")
    
    # Configuration with memory-optimized defaults
    max_model_len = int(properties.get("max_model_len", 32768))  # 32K default
    gpu_memory_utilization = float(properties.get("gpu_memory_utilization", 0.85))
    tensor_parallel_size = int(properties.get("tensor_parallel_size", 1))
    
    logger.info(f"Model config - max_model_len: {max_model_len}, "
                f"gpu_memory_utilization: {gpu_memory_utilization}, "
                f"tensor_parallel_size: {tensor_parallel_size}")
    
    model = LLM(
        model=model_location,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    
    logger.info("Model loaded successfully")
    return model


def handle(inputs: Input):
    """Handle inference requests."""
    global model
    
    if model is None:
        model = load_model(inputs.get_properties())
    
    if inputs.is_empty():
        return None
    
    data = inputs.get_as_json()
    input_text = data.get("inputs", "")
    parameters = data.get("parameters", {})
    
    if not input_text:
        return Output().error("Input text is required")
    
    try:
        # Sampling parameters
        sampling_params = SamplingParams(
            max_tokens=parameters.get("max_new_tokens", 512),
            temperature=parameters.get("temperature", 0.7),
            top_p=parameters.get("top_p", 0.9),
            stop=parameters.get("stop", ["<|im_end|>", "<|endoftext|>"]),
        )
        
        # Generate
        outputs = model.generate([input_text], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        logger.info(f"Generated {len(generated_text)} characters")
        
        return Output().add_as_json([{"generated_text": generated_text}])
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return Output().error(f"Generation failed: {str(e)}")


def format_qwen_prompt(messages: list) -> str:
    """Format chat messages into Qwen3 prompt format."""
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)