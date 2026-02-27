
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM
import torch
from tqdm import tqdm
import os
from typing import List, Dict, Tuple, Optional, Any
from src.config import Config

class MVIGFilter:
    """MVIG Filter (Macro-Pruning) for hierarchical evidence refinement."""

    def __init__(
            self,
            llama_model_path: Optional[str] = None,
            bert_model_path: Optional[str] = None,
            probe_queries_path: Optional[str] = None,
            device_id: int = Config.DEVICE_ID,
            instruction: Optional[str] = None
    ):
        """
        Initialize MVIGFilter.

        Args:
            llama_model_path: Path to Llama model.
            bert_model_path: Path to BERT model.
            probe_queries_path: Path to probe queries file.
            device_id: GPU device ID.
            instruction: Instruction template.
        """
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.instruction = instruction if instruction else Config.MVIG_CONFIG.get("instruction", "")
        self.template = "Question: {question}\nAnswer:"

        # Load models
        llama_path = llama_model_path if llama_model_path else Config.MODEL_PATHS.get("llama")
        bert_path = bert_model_path if bert_model_path else Config.MODEL_PATHS.get("bert")
        
        print(f"Loading Llama model from {llama_path}...")
        self.model = LlamaForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path)

        print(f"Loading BERT model from {bert_path}...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModel.from_pretrained(
            bert_path,
            torch_dtype=torch.float16
        ).to(self.device)

        # Load probe queries
        if probe_queries_path and os.path.exists(probe_queries_path):
            print(f"Loading probe queries from {probe_queries_path}...")
            self.probe_queries = self._load_json(probe_queries_path)
        else:
            print("Warning: Probe queries path not provided or invalid.")
            self.probe_queries = []

        print("MVIGFilter initialization complete!")

    @staticmethod
    def _load_json(path: str) -> Any:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        with open(path, "r", encoding='utf-8') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines]

    @staticmethod
    def _save_jsonl(data: List[Dict], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    @staticmethod
    def normalize_question(question: str) -> str:
        if not question.endswith("?"):
            question = question + "?"
        return question[0].lower() + question[1:]

    @staticmethod
    def find_ground_truth_doc(ctxs: List[Dict]) -> Optional[int]:
        for idx, doc in enumerate(ctxs):
            if doc.get("hasanswer", False) and doc.get("isgold", False):
                return idx
        return None

    def get_token_length(self, text: str, add_special_tokens: bool = True) -> int:
        return len(
            self.tokenizer(text, add_special_tokens=add_special_tokens).input_ids
        )

    def get_ppl(
            self,
            text: str = "",
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple] = None,
            return_kv: bool = False,
            end: Optional[int] = None,
            condition_mode: str = "none",
            condition_pos_id: int = 0,
            granularity: str = "sentence"
    ) -> torch.Tensor:
        if input_ids is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt")
            input_ids = tokenized_text["input_ids"].to(self.device)
            attention_mask = tokenized_text["attention_mask"].to(self.device)

        past_length = 0
        if end is None:
            end = input_ids.shape[1]

        with torch.no_grad():
            response = self.model(
                input_ids[:, past_length:end],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values

        shift_logits = response.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1: end].contiguous()

        active_logits = shift_logits.view(-1, shift_logits.size(-1))
        active_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)

        if condition_mode == "before":
            loss = loss[:condition_pos_id]
        elif condition_mode == "after":
            loss = loss[condition_pos_id:]

        res = loss.mean() if granularity == "sentence" else loss
        return (res, past_key_values) if return_kv else res

    def get_condition_ppl(
            self,
            text: str,
            question: str,
            condition_in_question: str = "none",
            granularity: str = "sentence"
    ) -> torch.Tensor:
        if condition_in_question == "none":
            return self.get_ppl(
                text,
                granularity=granularity
            )
        elif condition_in_question == "before":
            return self.get_ppl(
                text=question + text,
                granularity=granularity,
                condition_mode="after",
                condition_pos_id=self.get_token_length(question) - 1
            )
        elif condition_in_question == "after":
            return self.get_ppl(
                text=text + question,
                granularity=granularity,
                condition_mode="after",
                condition_pos_id=self.get_token_length(text) - 1
            )

    def _get_coefficients(self, source_query: str, queries: List[str]) -> List[float]:
        all_queries = [source_query] + queries
        all_tokens = self.bert_tokenizer(
            all_queries,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            all_embeddings = self.bert_model(**all_tokens).last_hidden_state
            all_embeddings = all_embeddings.mean(dim=1)

        source_query_embedding = all_embeddings[0].unsqueeze(0)
        query_embeddings = all_embeddings[1:]
        similarities = torch.nn.functional.cosine_similarity(
            source_query_embedding,
            query_embeddings,
            dim=1
        )
        return similarities.tolist()

    @staticmethod
    def _extract_question(input_string: str) -> str:
        start_index = 0
        for i, char in enumerate(input_string):
            if char.isalpha():
                start_index = i
                break
        return input_string[start_index:]

    def compute_mvig_score(
            self,
            corpus: List[str],
            query: str,
            index: int
    ) -> List[Tuple[int, float]]:
        """
        Compute MVIG score for each document.
        MVIG(d) = Sum(w_k * (PPL(q_k) - PPL(q_k|d)))
        """
        # Get probe queries
        querys = []
        if index < len(self.probe_queries):
             # Try getting 'synonymous_question' first, fallback to 'synonymous_questions' if needed
             probe_item = self.probe_queries[index]
             syn_questions = probe_item.get("synonymous_question", [])
             if not syn_questions:
                 syn_questions = probe_item.get("synonymous_questions", [])
             
             # If syn_questions is a string (rare but possible), wrap in list
             if isinstance(syn_questions, str):
                 syn_questions = [syn_questions]
                 
             querys = [self._extract_question(e) for e in syn_questions[:3]]
        
        # Append original query
        querys.append(self.normalize_question(query))

        # Compute semantic weights (w_k)
        coefficients = self._get_coefficients(query, querys)

        # 1. Compute Prior Perplexity PPL(q_k)
        prior_ppls = []
        for q in querys:
            question_text = self.instruction + self.template.format(
                question=q) + " We can get the answer to this question in the given documents."

            prior_ppl = self.get_condition_ppl(
                text=question_text,
                question="",
                condition_in_question="none"
            ).item()

            if np.isnan(prior_ppl) or prior_ppl <= 0:
                prior_ppl = 100.0

            prior_ppls.append(prior_ppl)

        # 2. Compute MVIG for each document
        doc_scores = []

        for doc_idx, d in enumerate(corpus):
            mvig_score = 0

            for q, weight, prior_ppl in zip(querys, coefficients, prior_ppls):
                # Compute Posterior Perplexity PPL(q_k|d)
                # Text: Document + Question
                posterior_ppl = self.get_condition_ppl(
                    text=d,
                    question=self.instruction + self.template.format(
                        question=q) + " We can get the answer to this question in the given documents.",
                    condition_in_question="after"
                ).item()

                if np.isnan(posterior_ppl) or posterior_ppl <= 0:
                    # If invalid, assume no reduction (or high perplexity)
                    posterior_ppl = prior_ppl * 2.0

                # Reduction in perplexity
                reduction = prior_ppl - posterior_ppl
                mvig_score += weight * reduction

            doc_scores.append((doc_idx, mvig_score))

        # Sort descending (Maximize MVIG)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores

    def rerank_documents(
            self,
            documents: List[Dict],
            question: str,
            index: int
    ) -> List[Tuple[int, float]]:
        """Rerank documents using MVIG score."""
        org_docs = [
            f"Document [{idx + 1}](Title:" + doc.get("title", "") + ")" + doc.get("text", "")
            for idx, doc in enumerate(documents)
        ]

        return self.compute_mvig_score(
            org_docs,
            question,
            index
        )

    def process_dataset(
            self,
            dataset_path: str,
            output_path: str,
            format_docs: bool = True
    ) -> List[Dict]:
        dataset = self._load_json(dataset_path)
        all_rerank_results = []
        
        # Get limit from config, default to None (all docs) if not set or <= 0
        initial_limit = Config.MVIG_CONFIG.get("initial_docs_num", -1)
        if initial_limit <= 0:
            initial_limit = None

        for ii, ex in enumerate(tqdm(dataset, desc="Processing dataset")):
            query = ex["question"]
            
            # Robustly handle 'ctxs' or 'docs' or similar keys
            ctxs = ex.get("ctxs", [])
            if not ctxs:
                # Try fallback keys if 'ctxs' is missing
                if "docs" in ex:
                    ctxs = ex["docs"]
                elif "documents" in ex:
                    ctxs = ex["documents"]
            
            ground_truth_idx = self.find_ground_truth_doc(ctxs)

            # Limit the number of documents for reranking if configured
            docs_to_rerank = ctxs
            if initial_limit and len(docs_to_rerank) > initial_limit:
                docs_to_rerank = docs_to_rerank[:initial_limit]

            # Rerank
            if docs_to_rerank:
                rerank_order = self.rerank_documents(
                    documents=docs_to_rerank,
                    question=query,
                    index=ii
                )
                
                rerank_result = {
                    "question": query,
                    "original_docs_count": len(ctxs),
                    "processed_docs_count": len(docs_to_rerank),
                    "ground_truth_index": ground_truth_idx,
                    "rerank_order": [e[0] for e in rerank_order],
                    "rerank_scores": [e[1] for e in rerank_order]
                }
            else:
                 # Handle empty docs case
                 rerank_result = {
                    "question": query,
                    "original_docs_count": 0,
                    "processed_docs_count": 0,
                    "ground_truth_index": -1,
                    "rerank_order": [],
                    "rerank_scores": []
                }

            all_rerank_results.append(rerank_result)

        self._save_jsonl(all_rerank_results, output_path)
        print(f"Rerank results saved to: {output_path}")

        return all_rerank_results

    def release_resources(self):
        print("Releasing MVIGFilter resources...")
        if self.model is not None:
            self.model = self.model.cpu()
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if hasattr(self, 'bert_model') and self.bert_model is not None:
            self.bert_model = self.bert_model.cpu()
            del self.bert_model
            self.bert_model = None
        if hasattr(self, 'bert_tokenizer') and self.bert_tokenizer is not None:
            del self.bert_tokenizer
            self.bert_tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()
        print("MVIGFilter resources released.")

    def __del__(self):
        try:
            self.release_resources()
        except:
            pass
