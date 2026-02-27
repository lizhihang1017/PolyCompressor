
import json
from tqdm import tqdm
import spacy
from typing import List, Dict, Any, Optional, Generator
import os

class SentenceDecomposer:
    """Document Decomposer: Splits documents into sentence-level evidence units."""

    def __init__(
            self,
            spacy_model: str = "en_core_web_sm",
            disable_components: Optional[List[str]] = None
    ):
        if disable_components is None:
            disable_components = ["tok2vec", "tagger", "parser",
                                  "attribute_ruler", "lemmatizer", "ner"]

        self.spacy_model_name = spacy_model

        print(f"Loading spacy model: {spacy_model}")
        try:
            self.nlp = spacy.load(
                spacy_model,
                disable=disable_components
            )
            self.nlp.enable_pipe("senter")
            print("Spacy model loaded successfully")
        except OSError:
            print(f"Spacy model '{spacy_model}' not found. Downloading...")
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(
                spacy_model,
                disable=disable_components
            )
            self.nlp.enable_pipe("senter")
            print("Spacy model downloaded and loaded.")

    @staticmethod
    def load_json_array(file_path: str) -> Any:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_jsonl(file_path: str) -> Generator[Dict, None, None]:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def save_json(data: Any, file_path: str, indent: int = 2) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def save_jsonl(data: List[Dict], file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    def decompose_document(
            self,
            title: str,
            text: str,
            include_title_in_sentence: bool = True
    ) -> List[Dict]:
        sentences = []
        doc = self.nlp(text)

        for sent in doc.sents:
            sentence_text = sent.text.strip()

            if sentence_text:
                if include_title_in_sentence:
                    sentence_text = f"{title} {sentence_text}"

                sentence_dict = {
                    "title": title,
                    "text": sentence_text,
                    "original_text": sent.text.strip()
                }
                sentences.append(sentence_dict)

        return sentences

    def decompose_documents(
            self,
            documents: List[Dict],
            include_title_in_sentence: bool = True
    ) -> List[Dict]:
        all_sentences = []

        for doc in documents:
            title = doc.get('title', '')
            text = doc.get('text', '')

            if text:
                sentences = self.decompose_document(
                    title=title,
                    text=text,
                    include_title_in_sentence=include_title_in_sentence
                )
                all_sentences.extend(sentences)

        return all_sentences

    def process_item(
            self,
            item: Dict,
            context_field: str = "ctxs",
            output_field: str = "evidence_units",
            keep_original_context: bool = True,
            include_title_in_sentence: bool = True
    ) -> Dict:
        if context_field not in item:
            raise ValueError(f"Field not found in item: {context_field}")

        documents = item[context_field]
        sentences = self.decompose_documents(
            documents=documents,
            include_title_in_sentence=include_title_in_sentence
        )

        processed_item = item.copy()
        processed_item[output_field] = sentences

        if not keep_original_context:
            del processed_item[context_field]

        return processed_item

    def process_dataset(
            self,
            dataset: List[Dict],
            context_field: str = "ctxs",
            output_field: str = "evidence_units",
            keep_original_context: bool = True,
            include_title_in_sentence: bool = True,
            show_progress: bool = True
    ) -> List[Dict]:
        processed_data = []
        iterator = tqdm(dataset, desc="Decomposing documents") if show_progress else dataset

        for item in iterator:
            try:
                processed_item = self.process_item(
                    item=item,
                    context_field=context_field,
                    output_field=output_field,
                    keep_original_context=keep_original_context,
                    include_title_in_sentence=include_title_in_sentence
                )
                processed_data.append(processed_item)
            except Exception as e:
                print(f"Error processing item: {str(e)}")
                processed_data.append(item)

        return processed_data

    def process_file(
            self,
            input_file: str,
            output_file: str,
            context_field: str = "ctxs",
            output_field: str = "evidence_units",
            keep_original_context: bool = True,
            include_title_in_sentence: bool = True,
            input_format: str = "json",
            output_format: str = "json"
    ) -> List[Dict]:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if input_format == "jsonl":
            data = list(self.load_jsonl(input_file))
        else:
            data = self.load_json_array(input_file)

        print(f"Loaded {len(data)} items")

        processed_data = self.process_dataset(
            dataset=data,
            context_field=context_field,
            output_field=output_field,
            keep_original_context=keep_original_context,
            include_title_in_sentence=include_title_in_sentence
        )

        if output_format == "jsonl":
            self.save_jsonl(processed_data, output_file)
        else:
            self.save_json(processed_data, output_file)

        print(f"Processing complete! Results saved to: {output_file}")
        return processed_data

    def release_resources(self) -> None:
        if hasattr(self, 'nlp'):
            del self.nlp
            print("Spacy model resources released")

    def __del__(self):
        try:
            self.release_resources()
        except:
            pass
