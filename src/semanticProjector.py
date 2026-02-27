
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
import os
from src.config import Config

class SemanticProjector:
    """Semantic Projector (Micro-Pruning) for sentence-level scoring."""

    def __init__(
        self,
        contriever_model_path: Optional[str] = None,
        probe_queries_path: Optional[str] = None,
        device_id: int = Config.DEVICE_ID,
        use_accelerate: bool = Config.USE_ACCELERATE
    ):
        self.contriever_model_path = contriever_model_path if contriever_model_path else Config.MODEL_PATHS.get("contriever")
        self.probe_queries_path = probe_queries_path

        # Setup device
        if use_accelerate:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
        else:
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        # Load model
        print(f"Loading Contriever model from {self.contriever_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.contriever_model_path)
        self.model = AutoModel.from_pretrained(self.contriever_model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load probe queries
        if probe_queries_path and os.path.exists(probe_queries_path):
            print(f"Loading probe queries from {probe_queries_path}...")
            self.probe_queries = self._load_json(probe_queries_path)
        else:
            print("Warning: Probe queries path not provided or invalid.")
            self.probe_queries = []

        print(f"SemanticProjector initialized on device: {self.device}")

    @staticmethod
    def _load_json(path: str) -> Any:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        return embeddings

    def compute_semantic_similarity(self, source_query: str, target_queries: List[str]) -> List[float]:
        if not target_queries:
            return []

        all_queries = [source_query] + target_queries
        all_embeddings = self.compute_embeddings(all_queries)

        source_embedding = all_embeddings[0:1]
        target_embeddings = all_embeddings[1:]

        source_embedding_norm = torch.nn.functional.normalize(source_embedding, p=2, dim=1)
        target_embeddings_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)

        similarities = torch.mm(source_embedding_norm, target_embeddings_norm.t()).squeeze(0)
        return similarities.cpu().tolist()

    def compute_document_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        if not documents:
            return []

        all_texts = [query] + documents
        all_embeddings = self.compute_embeddings(all_texts)
        all_embeddings_norm = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)

        query_embedding = all_embeddings_norm[0:1]
        document_embeddings = all_embeddings_norm[1:]

        scores = torch.mm(query_embedding, document_embeddings.t()).squeeze(0)
        return scores.cpu().tolist()

    def compute_weighted_scores(
        self,
        original_query: str,
        documents: List[str],
        probe_query_idx: int
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute scores using Max aggregation logic.
        S_sent(s) = max_k (w_k * Sim(E(s), E(q_k)))
        """
        # 1. Get probe queries
        if index := probe_query_idx < len(self.probe_queries):
             # Ensure we access the correct field
             # Assuming structure is list of dicts with "synonymous_question"
             item = self.probe_queries[probe_query_idx]
             if isinstance(item, dict):
                 current_probe_queries = item.get("synonymous_question", [])
             else:
                 current_probe_queries = []
        else:
            current_probe_queries = []

        # Extract questions if they are strings with numbering, or assume clean strings
        # The previous step produces clean strings in "synonymous_question"
        # But let's be safe and take top 3
        current_probe_queries = current_probe_queries[:3]

        # 2. Compute semantic weights
        if current_probe_queries:
            similarity_weights = self.compute_semantic_similarity(
                source_query=original_query,
                target_queries=current_probe_queries
            )
        else:
            similarity_weights = []

        # 3. Prepare all queries and weights
        all_queries = [original_query] + current_probe_queries
        all_weights = [1.0] + similarity_weights

        # 4. Compute scores and aggregate
        num_docs = len(documents)
        final_scores = [-float('inf')] * num_docs
        
        # Store original scores for reference
        original_scores = []

        for i, (query, weight) in enumerate(zip(all_queries, all_weights)):
            scores = self.compute_document_scores(query, documents)
            
            if i == 0:
                original_scores = scores

            for j in range(num_docs):
                weighted_score = weight * scores[j]
                if weighted_score > final_scores[j]:
                    final_scores[j] = weighted_score

        return original_scores, similarity_weights, final_scores

    def process_single_item(
        self,
        item: Dict,
        probe_query_idx: int,
        document_field: str = "evidence_units",
        top_k: int = -1,
        text_field: str = "text"
    ) -> Dict:
        if "question" not in item:
            raise ValueError("Item must contain 'question' field")
        if document_field not in item:
            raise ValueError(f"Item must contain '{document_field}' field")

        evidence_units = item[document_field]
        if top_k > 0 and len(evidence_units) > top_k:
            documents = [e.get(text_field, "") for e in evidence_units[:top_k]]
        else:
            documents = [e.get(text_field, "") for e in evidence_units]

        original_query = item["question"]

        original_scores, similarity_weights, weighted_scores = self.compute_weighted_scores(
            original_query=original_query,
            documents=documents,
            probe_query_idx=probe_query_idx
        )

        result = item.copy()
        result["original_scores"] = original_scores
        result["similarity_weights"] = similarity_weights
        result["weighted_scores"] = weighted_scores

        if isinstance(evidence_units, list) and evidence_units:
            for i, doc in enumerate(evidence_units):
                if i < len(weighted_scores):
                    doc["weighted_score"] = weighted_scores[i]
                if i < len(original_scores):
                    doc["original_score"] = original_scores[i]

        return result

    def process_file(
        self,
        input_file: str,
        output_file: str,
        document_field: str = "evidence_units",
        top_k: int = -1,
        text_field: str = "text",
        input_format: str = "json"
    ) -> pd.DataFrame:
        print(f"Loading data from {input_file}...")

        if input_format == "json":
            if input_file.endswith('.jsonl'):
                data = []
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                df = pd.DataFrame(data)
            else:
                df = pd.read_json(input_file)
        elif input_format == "csv":
            df = pd.read_csv(input_file)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        print(f"Loaded {len(df)} items")

        processed_rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing items"):
            try:
                item = row.to_dict()
                processed_item = self.process_single_item(
                    item=item,
                    probe_query_idx=idx,
                    document_field=document_field,
                    top_k=top_k,
                    text_field=text_field
                )
                processed_rows.append(processed_item)
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                processed_rows.append(row.to_dict())

        processed_df = pd.DataFrame(processed_rows)

        if output_file.endswith('.jsonl'):
            processed_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        else:
            processed_df.to_json(output_file, orient='records', indent=2, force_ascii=False)

        print(f"Results saved to: {output_file}")
        return processed_df

    def release_resources(self) -> None:
        print("Releasing SemanticProjector resources...")
        if hasattr(self, 'model'):
            self.model = self.model.cpu()
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("SemanticProjector resources released")

    def __del__(self):
        try:
            self.release_resources()
        except:
            pass
