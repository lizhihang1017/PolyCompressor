
import os
from typing import Dict, Any, Optional

class Config:
    # Base paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
    
    # Model Paths (Adjust these to your actual model locations)
    # Defaulting to placeholders or the original paths from the code
    MODEL_PATHS = {
        "qwen": "Qwen2.5-7B-Instruct",
        "llama": "Llama-2-7b-hf",
        "bert": "all-mpnet-base-v2",
        "contriever": "contriever-msmarco",
        # For DAKE, if using local model
        "dake_model": "Qwen2.5-7B-Instruct" 
    }

    # Device Configuration
    DEVICE_ID = 0
    USE_ACCELERATE = False

    # Phase I: Multi-Query Generator (MQG)
    MQG_CONFIG = {
        "model_type": "qwen", # or "llama", etc.
        "max_tokens": 1024,
        "temperature": 0.7, # Increased for diversity
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "num_questions": 3
    }

    # Phase II: Hierarchical Evidence Refiner
    # MVIG Filter (Macro-Pruning)
    MVIG_CONFIG = {
        "initial_docs_num": 30, # Number of documents to use for initial reranking (Input to MVIG)
        "top_k": 5, # Number of documents to keep after reranking
        "instruction": "Write a high-quality answer for the given question using only the provided search results."
    }

    # Semantic Projector (Micro-Pruning)
    SEMANTIC_PROJECTOR_CONFIG = {
        "top_k": -1, # -1 means keep all sentences from the top_k docs
        "aggregation_method": "max" # "max" or "sum"
    }

    # DAKE Filter (Dynamic Awareness Knowledge Extractor)
    DAKE_CONFIG = {
        "model_type": "local", # "local" or "api"
        "model_name": "qwen", # Key in MODEL_PATHS if local, or model name if API
        "api_key": os.getenv("DAKE_API_KEY", ""),
        "api_base_url": os.getenv("DAKE_API_BASE_URL", ""),
        "max_new_tokens": 1024, # Reduced from 2048 for better focus
        "temperature": 0.1, # Reduced for factual extraction
        "top_p": 0.9,
        "top_k": 20 # Number of sentences to consider for extraction (Input to DAKE)
    }

    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        return cls.MODEL_PATHS.get(model_name, "")

    @classmethod
    def ensure_output_dir(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

