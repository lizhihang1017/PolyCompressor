
import re
import json
import torch
import transformers
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from src.config import Config
from src.prompt import Evidence_Extractor_Prompt
import os

class DynamicAwarenessKnowledgeExtractor:
    """Dynamic Awareness Knowledge Extractor (DAKE) for information-need guided evidence extraction."""

    def __init__(
            self,
            model_path: Optional[str] = None,
            device: str = f"cuda:{Config.DEVICE_ID}",
            prompt_template: Optional[str] = None,
            system_prompt: str = "You are a helpful AI assistant"
    ):
        self.config = Config.DAKE_CONFIG
        self.model_type = self.config.get("model_type", "local")
        self.device = device
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template if prompt_template else Evidence_Extractor_Prompt
        
        # Generation parameters
        self.max_new_tokens = self.config.get("max_new_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.1)
        self.top_p = self.config.get("top_p", 0.9)

        # Initialize model
        if self.model_type == "local":
            self.model_path = model_path if model_path else Config.get_model_path(self.config.get("model_name", "qwen"))
            self._load_local_model()
        elif self.model_type == "api":
            self.api_key = self.config.get("api_key")
            self.api_base_url = self.config.get("api_base_url")
            print(f"Initialized DAKE with API mode (Base URL: {self.api_base_url})")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _load_local_model(self):
        print(f"Loading local model from {self.model_path}...")
        try:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Failed to load local model: {str(e)}")
            raise

    def _prepare_prompt(
            self,
            question: str,
            retrieved_docs: List[Dict],
            compressor_scores: List[float],
            top_k: int = 20,
            lat: int = 0
    ) -> Tuple[str, List[Tuple[Dict, float]]]:
        
        # Sort by scores descending
        # Ensure compressor_scores is same length as retrieved_docs
        if len(compressor_scores) != len(retrieved_docs):
             # Fallback if scores missing
             compressor_scores = [0.0] * len(retrieved_docs)

        # Enforce the logic: min(len(docs), top_k, 20)
        # This ensures we don't exceed the top_k parameter or the hard limit of 20
        # while respecting the actual number of documents available.
        actual_k = min(len(retrieved_docs), top_k, 20)

        sent_list = sorted(
            zip(retrieved_docs, compressor_scores),
            key=lambda x: x[1],
            reverse=True
        )[lat:lat + actual_k]

        sents = ""
        for j, (doc, _) in enumerate(sent_list):
            title = doc.get('title', '')
            text = doc.get('text', '')
            doc_str = f"{title} {text}"
            sents += f"[{j}] {doc_str}\n"

        prompt = self.prompt_template.format(
            question=question,
            num=len(sent_list),
            context=sents
        )

        return prompt, sent_list

    def _generate_response_local(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        if "Final Selection:" in response:
            response = response.split('Final Selection:')[-1].strip()

        return response

    def _generate_response_api(self, prompt: str) -> str:
        # Placeholder for API call
        # You would implement actual API call here using requests or openai library
        print("Warning: API mode selected but not implemented. Returning empty string.")
        return ""

    def _generate_response(self, prompt: str) -> str:
        if self.model_type == "local":
            return self._generate_response_local(prompt)
        else:
            return self._generate_response_api(prompt)

    def _parse_response(self, response: str) -> List[int]:
        try:
            if "No answer" in response:
                selected_idxs = []
            else:
                matches = re.findall(r'\[([\d, ]+)\]', response)
                selected_idxs = set()
                for match in matches:
                    for num in match.split(','):
                        num = num.strip()
                        if num.isdigit():
                            selected_idxs.add(int(num))
        except Exception as e:
            print(f"Error parsing indexes: {e}")
            selected_idxs = []

        return list(selected_idxs)

    def extract_evidence(
            self,
            question: str,
            retrieved_docs: List[Dict],
            compressor_scores: List[float],
            top_k: int = 20,
            lat: int = 0,
            verbose: bool = False
    ) -> Dict[str, Any]:
        
        prompt, sent_list = self._prepare_prompt(
            question, retrieved_docs, compressor_scores, top_k, lat
        )

        response = self._generate_response(prompt)
        if verbose:
            print(f"Response: {response}")
            
        selected_idxs = self._parse_response(response)

        text = ""
        selected_docs = []
        seen_texts = set()

        for idx in selected_idxs:
            if 0 <= idx < len(sent_list):
                doc, score = sent_list[idx]
                doc_title = doc.get('title', '').strip()
                doc_text = doc.get('text', '').strip()
                
                # Construct full sentence content
                full_content = f"{doc_title} {doc_text}".strip()
                
                # Deduplication logic
                if full_content not in seen_texts:
                    if text:
                        text += "\n" # Separate sentences with newline
                    text += full_content
                    seen_texts.add(full_content)
                    selected_docs.append(doc)

        return {
            "question": question,
            "selector_response": response,
            "selected_idxs": selected_idxs,
            "selected_docs": selected_docs,
            "compressed_context": text.strip(),
            # "sent_list": [(doc, score) for doc, score in sent_list] # Optional: keep for debugging
        }

    def process_batch(
            self,
            data: List[Dict],
            top_k: int = 20,
            lat: int = 0,
            output_path: Optional[str] = None,
            save_interval: int = 10
    ) -> List[Dict]:
        processed_data = []

        for i, sample in enumerate(tqdm(data, desc="Extracting evidence")):
            try:
                question = sample["question"]
                retrieved_docs = sample.get('evidence_units', [])
                compressor_scores = sample.get('weighted_scores', [])

                result = self.extract_evidence(
                    question=question,
                    retrieved_docs=retrieved_docs,
                    compressor_scores=compressor_scores,
                    top_k=top_k,
                    lat=lat
                )

                sample.update(result)
                processed_data.append(sample)

                if output_path and (i + 1) % save_interval == 0:
                    self._save_data(processed_data, output_path)
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                processed_data.append(sample) # Keep original if failed

        if output_path:
            self._save_data(processed_data, output_path)
            print(f"All data saved to {output_path}")

        return processed_data

    def _save_data(self, data: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

