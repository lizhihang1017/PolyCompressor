
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
import os
from src.config import Config

class MultiQueryGenerator:
    """Multi-Query Generator (MQG) for generating diverse query variations."""

    def __init__(
            self,
            model_path: Optional[str] = None,
            device_id: int = Config.DEVICE_ID,
            trust_remote_code: bool = True
    ):
        """
        Initialize MultiQueryGenerator.

        Args:
            model_path: Path to the model. If None, uses Config.MODEL_PATHS["qwen"].
            device_id: GPU device ID.
            trust_remote_code: Whether to trust remote code.
        """
        self.model_path = model_path if model_path else Config.MODEL_PATHS.get("qwen")
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code

        # Model and tokenizer
        self.model = None
        self.tokenizer = None

        # Generation config from Config
        self.generation_config = Config.MQG_CONFIG.copy()
        
        print(f"MultiQueryGenerator initialized with model path: {self.model_path}")

    def load_model(self) -> None:
        """Load model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            print(f"Loading local model: {self.model_path}")

            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=self.trust_remote_code
                )

                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map=self.device if self.device.type == "cuda" else None,
                    trust_remote_code=self.trust_remote_code
                )

                if self.device.type == "cpu":
                    self.model = self.model.to(self.device)

                self.model.eval()
                print(f"Model loaded successfully on device: {self.device}")

            except Exception as e:
                print(f"Model loading failed: {str(e)}")
                raise

    @staticmethod
    def _load_json(path: str) -> List[Dict]:
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
    def _save_json(data: Any, path: str, indent: int = 2) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def parse_questions(response_text: str) -> List[str]:
        """Parse the generated questions using regex."""
        questions = []

        # Method 1: Match [1], [2], [3] format
        pattern1 = r'\[(\d+)\]\s*(.+?)(?=\s*\[\d+\]|$)'
        matches1 = re.findall(pattern1, response_text, re.DOTALL)

        if matches1 and len(matches1) >= 3:
            for num, question in matches1:
                questions.append(question.strip())
            return questions[:3]

        # Method 2: Match 1., 2., 3. format
        pattern2 = r'(\d+)\.\s*(.+?)(?=\s*\d+\.|$)'
        matches2 = re.findall(pattern2, response_text, re.DOTALL)

        if matches2 and len(matches2) >= 3:
            for num, question in matches2:
                questions.append(question.strip())
            return questions[:3]

        # Method 3: Line by line
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            match = re.match(r'^(?:\[?\d+\]?|\(\d+\)|\d+\.)\s*(.+)', line)
            if match:
                question = match.group(1).strip()
                if question and not question.startswith('[') and not re.match(r'^\d', question):
                    questions.append(question)

        if len(questions) >= 3:
            return questions[:3]

        if not questions:
            sentences = re.split(r'[.!?]', response_text)
            questions = [s.strip() for s in sentences if len(s.strip()) > 10][:3]

        return questions

    def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None
    ) -> str:
        """Generate text using the local model."""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        config = self.generation_config.copy()
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if repetition_penalty is not None:
            config["repetition_penalty"] = repetition_penalty

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except AttributeError:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    do_sample=True, # Always sample for diversity
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=config["repetition_penalty"]
                )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            print(f"Generation failed: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "GENERATION_ERROR"

    def generate_synonymous_questions(
            self,
            question: str,
            prompt_template: str,
            num_questions: int = 3
    ) -> List[str]:
        """Generate synonymous questions."""
        prompt = prompt_template.format(question=question)
        generated = self.generate(prompt)
        synonymous_questions = self.parse_questions(generated)

        # Ensure we return the requested number of questions
        if len(synonymous_questions) < num_questions:
            last_question = synonymous_questions[-1] if synonymous_questions else question
            while len(synonymous_questions) < num_questions:
                synonymous_questions.append(f"{last_question} (variation {len(synonymous_questions) + 1})")
        elif len(synonymous_questions) > num_questions:
            synonymous_questions = synonymous_questions[:num_questions]

        return synonymous_questions

    def release_resources(self):
        """Release GPU resources."""
        print("Releasing MultiQueryGenerator resources...")
        if self.model is not None:
            self.model = self.model.cpu()
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()
        print("MultiQueryGenerator resources released.")

    def __del__(self):
        try:
            self.release_resources()
        except:
            pass

    def process_single_item(
            self,
            item: Dict,
            prompt_template: str,
            include_original: bool = True
    ) -> Dict:
        if "question" not in item:
            raise ValueError("Item must contain 'question' field")

        question = item["question"]
        synonymous_questions = self.generate_synonymous_questions(
            question=question,
            prompt_template=prompt_template,
            num_questions=Config.MQG_CONFIG.get("num_questions", 3)
        )

        result = {
            "question": question,
            "synonymous_question": synonymous_questions
        }

        if include_original:
            for key, value in item.items():
                if key not in result:
                    result[key] = value

        return result

    def process_dataset(
            self,
            input_path: str,
            output_path: str,
            prompt_template: str,
            include_original: bool = False
    ) -> List[Dict]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.endswith('.jsonl'):
            data = self._load_jsonl(input_path)
        else:
            data = self._load_json(input_path)

        processed_data = []
        for i, item in enumerate(tqdm(data, desc="Generating synonymous questions")):
            try:
                processed_item = self.process_single_item(
                    item=item,
                    prompt_template=prompt_template,
                    include_original=include_original
                )
                processed_data.append(processed_item)
            except Exception as e:
                print(f"Error processing item {i + 1}: {str(e)}")
                processed_data.append({
                    "question": item.get("question", f"Unknown_{i + 1}"),
                    "synonymous_question": []
                })

        self._save_json(processed_data, output_path)
        print(f"Processing complete! Results saved to: {output_path}")
        return processed_data
