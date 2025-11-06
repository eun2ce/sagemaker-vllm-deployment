"""
SageMaker inference script for LLM and embedding models.
Supports vLLM for text generation and transformers for embeddings.
"""

from djl_python import Input, Output
import logging
from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None
model_type = None  # "generation" or "embedding"


def load_model(properties):
    """Load model based on task type."""
    global model, tokenizer, model_type
    
    model_location = properties.get("model_id", properties.get("model_dir", "/opt/ml/model"))
    task_type = properties.get("task", "text-generation")
    
    logger.info(f"Loading model from {model_location}, task: {task_type}")
    
    if task_type == "embedding":
        # Load embedding model with transformers
        model_type = "embedding"
        tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_location, trust_remote_code=True)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        logger.info("Embedding model loaded successfully")
        
    else:
        # Load generation model with vLLM
        model_type = "generation"
        max_model_len = int(properties.get("max_model_len", 32768))
        gpu_memory_utilization = float(properties.get("gpu_memory_utilization", 0.85))
        tensor_parallel_size = int(properties.get("tensor_parallel_size", 1))
        
        model = LLM(
            model=model_location,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        logger.info("Generation model loaded successfully")
    
    return model


def handle(inputs: Input):
    """Handle inference requests."""
    global model, tokenizer, model_type
    
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
        if model_type == "embedding":
            # Generate embeddings
            inputs_tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs_tokenized = {k: v.cuda() for k, v in inputs_tokenized.items()}
            
            with torch.no_grad():
                outputs = model(**inputs_tokenized)
                # Use mean pooling for embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            return Output().add_as_json([{"embeddings": embeddings.tolist()}])
            
        else:
            # Text generation (existing code)
            sampling_params = SamplingParams(
                max_tokens=parameters.get("max_new_tokens", 512),
                temperature=parameters.get("temperature", 0.7),
                top_p=parameters.get("top_p", 0.9),
                stop=parameters.get("stop", ["<|im_end|>", "<|endoftext|>"]),
            )
            
            outputs = model.generate([input_text], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            return Output().add_as_json([{"generated_text": generated_text}])
            
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return Output().error(f"Inference failed: {str(e)}")