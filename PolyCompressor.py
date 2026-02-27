
import torch
import json
import os
from typing import Dict, List, Optional, Any
from src.config import Config
from src.multiQueryGenerator import MultiQueryGenerator
from src.mvigFilter import MVIGFilter
from src.sentenceDecomposer import SentenceDecomposer
from src.semanticProjector import SemanticProjector
from src.dynamicAwarenessKnowledgeExtractor import DynamicAwarenessKnowledgeExtractor
from src.prompt import Probe_Queries_Prompt, Evidence_Extractor_Prompt
from tqdm import tqdm

class PolyCompressor:
    """PolyCompressor Pipeline: A Training-Free Framework for Context Compression."""

    def __init__(self, input_data_path: str = "./input_data/data.json", output_dir: Optional[str] = None):
        """
        Initialize the PolyCompressor pipeline.
        
        Args:
            input_data_path: Path to the initial input dataset.
            output_dir: Optional custom output directory. If None, uses Config.OUTPUT_DIR.
        """
        self.input_data_path = input_data_path
        self.output_dir = output_dir if output_dir else Config.OUTPUT_DIR
        self.cache_dir = os.path.join(self.output_dir, "cache")
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define file paths for each stage (in CACHE_DIR)
        self.probe_queries_path = os.path.join(self.cache_dir, "step1_probe_queries.json")
        self.rerank_output_path = os.path.join(self.cache_dir, "step2_1_rerank_results.jsonl")
        self.top_k_docs_path = os.path.join(self.cache_dir, f"step2_2_top{Config.MVIG_CONFIG['top_k']}_docs.json")
        self.split_docs_path = os.path.join(self.cache_dir, "step2_3_split_docs.json")
        self.scored_docs_path = os.path.join(self.cache_dir, "step2_4_scored_docs.json")
        self.extracted_evidence_path = os.path.join(self.cache_dir, "step3_raw_evidence.json")
        
        # Final output file (in OUTPUT_DIR)
        self.final_output_path = os.path.join(self.output_dir, "final_compressed_results.json")

        print(f"PolyCompressor initialized.")
        print(f"Output Directory: {self.output_dir}")
        print(f"Cache Directory: {self.cache_dir}")

    @staticmethod
    def load_json(path: str) -> Any:
        if path.endswith(".jsonl"):
            with open(path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        
        # Try loading as standard JSON first
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Fallback: Try loading as JSONL if standard JSON fails (e.g. NQ-2 might be .json but content is jsonl)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return [json.loads(line) for line in f]
            except Exception:
                 # Re-raise original error if fallback fails
                 raise

    @staticmethod
    def save_json(data: Any, path: str, indent: int = 2) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            
    @staticmethod
    def load_jsonl(path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def clear_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory cleared.")

    def _check_cache(self, file_path: str) -> bool:
        """Check if a cache file exists and is valid."""
        if os.path.exists(file_path):
            try:
                # Try reading to ensure it's not corrupted
                if file_path.endswith('.jsonl'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        line = f.readline()
                        if line: json.loads(line)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                print(f"Cache found and valid: {file_path}")
                return True
            except Exception:
                print(f"Cache found but invalid/corrupted: {file_path}")
                return False
        return False

    def step1_multi_query_generation(self) -> None:
        """Phase I: Multi-Query Generation (MQG)."""
        print("\n" + "=" * 60)
        print("Phase I: Multi-Query Generation (MQG)")
        print("=" * 60)

        if self._check_cache(self.probe_queries_path):
            print("Skipping Step 1 (Cache exists).")
            return
        
        # Check Ablation Flag
        if not Config.ABLATION_CONFIG.get("use_mqg", True):
            print("ABLATION: Skipping MQG (use_mqg=False). Generating dummy probe queries with original question only.")
            # Load input data
            input_data = self.load_json(self.input_data_path)
            # Create dummy output where synonymous_question is empty or just the original question
            processed_data = []
            for item in input_data:
                # We still need the structure, but with no extra queries
                # Or we can put the original question as the only "synonymous" one if downstream expects a list
                # Logic: If MQG is off, downstream (MVIG/Projector) should rely on original question.
                # However, code expects 'synonymous_question' list.
                # Let's populate it with just the original question to be safe and consistent.
                new_item = item.copy()
                new_item['synonymous_question'] = [item['question']] 
                processed_data.append(new_item)
            
            self.save_json(processed_data, self.probe_queries_path)
            print(f"Dummy probe queries saved to: {self.probe_queries_path}")
            return

        mqg = MultiQueryGenerator()
        
        mqg.process_dataset(
            input_path=self.input_data_path,
            output_path=self.probe_queries_path,
            prompt_template=Probe_Queries_Prompt,
            include_original=True # We need original data flow
        )

        mqg.release_resources()
        del mqg
        self.clear_gpu_memory()

    def step2_hierarchical_evidence_refinement(self) -> None:
        """Phase II: Hierarchical Evidence Refinement."""
        print("\n" + "=" * 60)
        print("Phase II: Hierarchical Evidence Refinement")
        print("=" * 60)

        # 2.1 MVIG Filtering (Macro-Pruning)
        print("\n2.1 MVIG Filtering (Macro-Pruning)...")
        if self._check_cache(self.rerank_output_path):
             print("Skipping Reranking (Cache exists).")
        elif not Config.ABLATION_CONFIG.get("use_mvig", True):
            print("ABLATION: Skipping MVIG (use_mvig=False). Using original retriever order.")
            # Load Step 1 output (which contains probe queries, or dummy ones if MQG was skipped)
            dataset = self.load_json(self.probe_queries_path)
            all_rerank_results = []
            
            # Simulate rerank results but keeping original order
            for i, ex in enumerate(dataset):
                ctxs = ex.get("ctxs", [])
                # If ctxs is empty, try fallback keys
                if not ctxs:
                    if "docs" in ex: ctxs = ex["docs"]
                    elif "documents" in ex: ctxs = ex["documents"]
                
                num_docs = len(ctxs)
                # Simply create indices [0, 1, 2, ..., N-1]
                rerank_order = list(range(num_docs))
                # Assign dummy scores
                if num_docs > 0:
                    rerank_scores = [1.0 - (j / num_docs) for j in range(num_docs)]
                else:
                    rerank_scores = []

                rerank_result = {
                    "question": ex.get("question", ""),
                    "original_docs_count": num_docs,
                    "processed_docs_count": num_docs, 
                    "ground_truth_index": -1, 
                    "rerank_order": rerank_order,
                    "rerank_scores": rerank_scores
                }
                all_rerank_results.append(rerank_result)
            
            # Save simulated results to jsonl
            with open(self.rerank_output_path, 'w', encoding='utf-8') as f:
                for item in all_rerank_results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Dummy rerank results saved to: {self.rerank_output_path}")

        elif not Config.ABLATION_CONFIG.get("use_mqg", True):
             # Logic: If MQG is OFF but MVIG is ON, we must ensure MVIG only uses the original query.
             # In Step 1 (when use_mqg=False), we generated dummy probe queries where 
             # 'synonymous_question' contains ONLY the original question.
             # The MVIGFilter.compute_mvig_score method (which calls _extract_question) 
             # reads 'synonymous_question' and then APPENDS the original query.
             # This would result in [original_query, original_query].
             # To fix this redundancy and adhere to "only original query", we should instantiate MVIGFilter normally,
             # but we need to trust that MVIGFilter handles this gracefully or update MVIGFilter logic.
             # Actually, MVIGFilter.compute_mvig_score appends the query to the list from probe_queries.
             # If probe_queries has [q], then querys becomes [q, q].
             # The coefficients calculation will be between q and [q]. 
             # This is mathematically fine (cosine sim is 1.0), but redundant computation.
             # However, it CORRECTLY implements "using original query" (just twice).
             # To be strictly cleaner, we could modify MVIGFilter, but for now, this behavior is acceptable 
             # as it relies *only* on the original query's semantic information.
             print("ABLATION: MQG is OFF. MVIG will use the original query from the dummy probe queries file.")
             mvig_filter = MVIGFilter(
                probe_queries_path=self.probe_queries_path
             )
             mvig_filter.process_dataset(
                dataset_path=self.probe_queries_path,
                output_path=self.rerank_output_path
             )
             mvig_filter.release_resources()
             del mvig_filter
             self.clear_gpu_memory()

        else:
            mvig_filter = MVIGFilter(
                probe_queries_path=self.probe_queries_path
            )
            # ... (standard logic)
            mvig_filter.process_dataset(
                dataset_path=self.probe_queries_path, 
                output_path=self.rerank_output_path
            )
            mvig_filter.release_resources()
            del mvig_filter
            self.clear_gpu_memory()

        # 2.2 Select Top-K Documents
        print(f"\n2.2 Selecting Top-{Config.MVIG_CONFIG['top_k']} Documents...")
        if self._check_cache(self.top_k_docs_path):
            print("Skipping Selection (Cache exists).")
        else:
            data = self.load_json(self.probe_queries_path)
            rerank_data = self.load_jsonl(self.rerank_output_path)
            
            # Ensure alignment
            if len(data) != len(rerank_data):
                print("Warning: Data length mismatch between Step 1 and Rerank results!")
            
            top_k = Config.MVIG_CONFIG['top_k']
            
            for i, sample in enumerate(tqdm(data, desc="Selecting Top-K")):
                if i < len(rerank_data):
                    rerank_info = rerank_data[i]
                    rerank_order = rerank_info['rerank_order']
                    original_docs = sample.get('ctxs', [])
                    
                    selected_docs = []
                    for idx in rerank_order[:top_k]:
                        if 0 <= idx < len(original_docs):
                            selected_docs.append(original_docs[idx])
                    
                    sample['top_k_docs'] = selected_docs
                    # Also store rerank info
                    sample['mvig_rerank_info'] = rerank_info
            
            self.save_json(data, self.top_k_docs_path)

        # 2.3 Sentence Decomposition
        print("\n2.3 Sentence Decomposition...")
        if self._check_cache(self.split_docs_path):
            print("Skipping Decomposition (Cache exists).")
        else:
            decomposer = SentenceDecomposer()
            decomposer.process_file(
                input_file=self.top_k_docs_path,
                output_file=self.split_docs_path,
                context_field="top_k_docs",
                output_field="evidence_units",
                keep_original_context=True,
                include_title_in_sentence=True
            )
            decomposer.release_resources()
            del decomposer

        # 2.4 Semantic Projection (Micro-Pruning)
        print("\n2.4 Semantic Projection (Micro-Pruning)...")
        if self._check_cache(self.scored_docs_path):
            print("Skipping Semantic Scoring (Cache exists).")
        elif not Config.ABLATION_CONFIG.get("use_semantic_projector", True):
            print("ABLATION: Skipping Semantic Projector (use_semantic_projector=False). Assigning default scores.")
            
            data = self.load_json(self.split_docs_path)
            processed_data = []
            
            for item in data:
                evidence_units = item.get("evidence_units", [])
                
                # Assign default score (e.g., 1.0) to all sentences
                num_units = len(evidence_units)
                # Weighted scores and original scores as all 1.0s or maybe decaying?
                # Let's use 1.0 to imply "no filtering based on score"
                default_scores = [1.0] * num_units
                
                # Update item structure similar to what SemanticProjector does
                new_item = item.copy()
                new_item["original_scores"] = default_scores
                new_item["similarity_weights"] = [] # No weights used
                new_item["weighted_scores"] = default_scores
                
                # Also update individual unit dicts
                for i, unit in enumerate(evidence_units):
                    unit["weighted_score"] = 1.0
                    unit["original_score"] = 1.0
                
                processed_data.append(new_item)
                
            self.save_json(processed_data, self.scored_docs_path)
            print(f"Dummy scored docs saved to: {self.scored_docs_path}")

        else:
            projector = SemanticProjector(
                probe_queries_path=self.probe_queries_path
            )
            projector.process_file(
                input_file=self.split_docs_path,
                output_file=self.scored_docs_path,
                document_field="evidence_units",
                top_k=Config.SEMANTIC_PROJECTOR_CONFIG['top_k'],
                text_field="text"
            )
            projector.release_resources()
            del projector
            self.clear_gpu_memory()

    def step3_dynamic_knowledge_extraction(self) -> None:
        """Phase III: Dynamic Awareness Knowledge Extraction (DAKE)."""
        print("\n" + "=" * 60)
        print("Phase III: Dynamic Awareness Knowledge Extraction (DAKE)")
        print("=" * 60)

        if self._check_cache(self.extracted_evidence_path):
            print("Skipping Extraction (Cache exists).")
            return

        if not Config.ABLATION_CONFIG.get("use_dake", True):
            print("ABLATION: Skipping DAKE (use_dake=False). Selecting Top-K sentences based on scores.")
            
            data = self.load_json(self.scored_docs_path)
            processed_data = []
            
            top_k = Config.DAKE_CONFIG['top_k']
            
            for item in data:
                evidence_units = item.get("evidence_units", [])
                # Sort by weighted_score descending
                # Ensure we handle potential missing scores safely
                sorted_units = sorted(
                    evidence_units, 
                    key=lambda x: x.get("weighted_score", -float('inf')), 
                    reverse=True
                )
                
                selected_docs = sorted_units[:top_k]
                
                # Construct compressed context
                # Deduplication logic similar to DAKE
                text = ""
                seen_texts = set()
                final_selected = []
                
                for doc in selected_docs:
                    doc_title = doc.get('title', '').strip()
                    doc_text = doc.get('text', '').strip()
                    full_content = f"{doc_title} {doc_text}".strip()
                    
                    if full_content not in seen_texts:
                        if text:
                            text += "\n"
                        text += full_content
                        seen_texts.add(full_content)
                        final_selected.append(doc)
                
                new_item = item.copy()
                new_item["selector_response"] = "SKIPPED_DAKE"
                new_item["selected_idxs"] = list(range(len(final_selected))) # Dummy indices
                new_item["selected_docs"] = final_selected
                new_item["compressed_context"] = text.strip()
                
                processed_data.append(new_item)
                
            self.save_json(processed_data, self.extracted_evidence_path)
            print(f"Dummy extracted evidence saved to: {self.extracted_evidence_path}")
            return

        dake = DynamicAwarenessKnowledgeExtractor(
            prompt_template=Evidence_Extractor_Prompt
        )
        
        # Load scored data
        data = self.load_json(self.scored_docs_path)
        
        dake.process_batch(
            data=data,
            top_k=Config.DAKE_CONFIG['top_k'],
            output_path=self.extracted_evidence_path
        )
        
        # DAKE model doesn't have explicit release method in current class, 
        # but garbage collection should handle it if we delete the instance
        del dake
        self.clear_gpu_memory()

    def run_full_pipeline(self) -> Dict:
        """Execute the full PolyCompressor pipeline."""
        print("Starting PolyCompressor Pipeline...")
        
        try:
            self.step1_multi_query_generation()
            self.step2_hierarchical_evidence_refinement()
            self.step3_dynamic_knowledge_extraction()
            
            # Aggregate final results
            print("\nAggregating Final Results...")
            raw_evidence = self.load_json(self.extracted_evidence_path)
            
            final_results = []
            for item in raw_evidence:
                # Keep only key information for the final output
                summary_item = {
                    "question": item.get("question"),
                    "answers": item.get("answers", []), # Preserve original answers
                    "final_compressed_context": item.get("compressed_context"),
                    "original_docs_count": len(item.get("ctxs", [])), # Assuming ctxs is preserved
                    "selected_sentences_count": len(item.get("selected_docs", [])),
                    # Optional: Add intermediate metrics if needed
                }
                final_results.append(summary_item)
            
            self.save_json(final_results, self.final_output_path)
            
            print("\nPipeline Completed Successfully!")
            print(f"Intermediate cache saved to: {self.cache_dir}")
            print(f"Final aggregated results saved to: {self.final_output_path}")
            
            return final_results
            
        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            raise

def main():
    # Example usage
    compressor = PolyCompressor(
        input_data_path="./input_data/data.json"
    )
    compressor.run_full_pipeline()

if __name__ == '__main__':
    main()
