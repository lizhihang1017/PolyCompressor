
import json
import torch
import re
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration & Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prompts
PROBE_QUERIES_PROMPT = """
Please perform the following task for the question: {question}

Generate three synonymous questions from the above, requiring the application of synonym replacement, syntactic variation, and semantic expansion methods, respectively.

You must output exactly three questions in the following format, with no other text:
[1] First question here
[2] Second question here
[3] Third question here
"""

EVIDENCE_EXTRACTOR_PROMPT = """
I will provide you with {num} sentences, each indicated by a numerical identifier []. Select the sentences based on their relevance to the search query: {question}

{context}

Search Query: {question}  

Please follow the steps below: 
Step 1. Please list up the information requirements to answer the query. 
Step 2. for each requirement in Step 1, find the sentences that has the information of the requirement. 
Step 3. Choose the sentences that mostly covers clear and diverse information to answer the query. Number of sentences is unlimited. 
Step 4. You must output the selected sentence identifiers in the format '### Final Selection: [] []', e.g., ### Final Selection: [2] [1].
"""

MVIG_INSTRUCTION = "Write a high-quality answer for the given question using only the provided search results."
MVIG_TEMPLATE = "Question: {question}\nAnswer:"

# --- Model Wrapper ---
class LlamaUnified:
    def __init__(self, model_path):
        print(f"Loading Llama model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("Model loaded.")

    def generate_text(self, prompt, max_new_tokens=1024, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def get_embedding(self, text):
        # Use last hidden state mean pooling
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last layer hidden states
            hidden_states = outputs.hidden_states[-1] 
            # Mean pooling
            mask = inputs.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        return embedding

    def get_perplexity(self, text):
        # Calculate perplexity of text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()

    def get_conditional_perplexity(self, context, target):
        # PPL(target | context)
        # We compute loss only on target tokens
        full_text = context + target
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        target_len = target_ids.shape[1]
        
        input_ids = inputs.input_ids
        labels = input_ids.clone()
        # Mask context (everything before target)
        labels[:, :-target_len] = -100 
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
        return torch.exp(loss).item()

# --- Helper Functions ---

def split_sentences(text):
    # Simple regex splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def parse_questions(response_text):
    pattern = r'\[(\d+)\]\s*(.+?)(?=\s*\[\d+\]|$)'
    matches = re.findall(pattern, response_text, re.DOTALL)
    questions = [m[1].strip() for m in matches]
    if not questions:
        # Fallback
        lines = response_text.split('\n')
        questions = [l.strip() for l in lines if '?' in l][:3]
    return questions[:3]

def parse_selection(response_text):
    if "Final Selection:" in response_text:
        selection_part = response_text.split("Final Selection:")[-1]
        matches = re.findall(r'\[(\d+)\]', selection_part)
        return [int(m) for m in matches]
    return []

# --- Main Logic ---

def process_file(input_path, output_path, model_path):
    # Load Model
    llm = LlamaUnified(model_path)
    
    # Load Data
    print(f"Loading data from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
            
    results = []
    
    for item in tqdm(data, desc="Processing"):
        question = item.get('question', '')
        # Handle various doc keys
        docs = []
        if 'ctxs' in item:
            docs = item['ctxs']
        elif 'docs' in item:
            docs = item['docs']
        elif 'documents' in item:
            docs = item['documents']
            
        answers = item.get('answers', [])
        
        # 1. MQG
        mqg_prompt = PROBE_QUERIES_PROMPT.format(question=question)
        mqg_response = llm.generate_text(mqg_prompt)
        synonyms = parse_questions(mqg_response)
        all_queries = [question] + synonyms
        
        # Cache query embeddings
        query_embeddings = []
        for q in all_queries:
            query_embeddings.append(llm.get_embedding(q))
            
        # 2. MVIG (Macro-Pruning)
        # Calculate weights (similarity to original question)
        q_embed = query_embeddings[0]
        weights = [1.0] # Weight for original question
        for i in range(1, len(all_queries)):
            syn_embed = query_embeddings[i]
            sim = torch.nn.functional.cosine_similarity(q_embed, syn_embed).item()
            weights.append(sim)
            
        # Score docs
        doc_scores = []
        for doc in docs:
            doc_text = doc.get('text', '')
            doc_title = doc.get('title', '')
            full_doc = f"Title: {doc_title}. {doc_text}"
            
            score = 0
            for i, (q, w) in enumerate(zip(all_queries, weights)):
                # PPL(q)
                prior_prompt = f"{MVIG_INSTRUCTION} {MVIG_TEMPLATE.format(question=q)} We can get the answer..."
                prior_ppl = llm.get_perplexity(prior_prompt)
                
                # PPL(q|d)
                # Note: MVIGFilter uses text=doc, question=formatted_q, condition_in_question="after"
                # This means PPL(formatted_q | doc)
                post_ppl = llm.get_conditional_perplexity(full_doc, prior_prompt)
                
                reduction = prior_ppl - post_ppl
                score += w * reduction
            doc_scores.append((doc, score))
            
        # Top 5 docs
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [d[0] for d in doc_scores[:5]]
        
        # 3. Semantic Projector (Micro-Pruning)
        # Split into sentences
        sentences = []
        for doc in top_docs:
            doc_title = doc.get('title', '')
            doc_text = doc.get('text', '')
            sents = split_sentences(doc_text)
            for s in sents:
                sentences.append(f"{doc_title} {s}")
                
        # Score sentences (Similarity to queries)
        sent_scores = []
        for sent in sentences:
            sent_embed = llm.get_embedding(sent)
            max_score = -float('inf')
            for i, (q, w) in enumerate(zip(all_queries, weights)):
                q_embed_curr = query_embeddings[i]
                sim = torch.nn.functional.cosine_similarity(sent_embed, q_embed_curr).item()
                weighted_sim = w * sim
                if weighted_sim > max_score:
                    max_score = weighted_sim
            sent_scores.append((sent, max_score))
            
        # Top 20 sentences for DAKE
        sent_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sent_scores[:20]
        
        # 4. DAKE (Evidence Extraction)
        context_str = ""
        for i, (sent, score) in enumerate(top_sentences):
            context_str += f"[{i}] {sent}\n"
            
        dake_prompt = EVIDENCE_EXTRACTOR_PROMPT.format(num=len(top_sentences), question=question, context=context_str)
        dake_response = llm.generate_text(dake_prompt, temperature=0.1) # Low temp for extraction
        
        selected_indices = parse_selection(dake_response)
        compressed_context = ""
        for idx in selected_indices:
            if 0 <= idx < len(top_sentences):
                compressed_context += top_sentences[idx][0] + " "
        
        # 5. Token Counting
        # Before: All docs tokens
        all_docs_text = " ".join([d.get('text', '') for d in docs])
        token_before = len(llm.tokenizer.encode(all_docs_text))
        
        # After: Compressed context tokens
        token_after = len(llm.tokenizer.encode(compressed_context))
        
        results.append({
            "question": question,
            "compressed_context": compressed_context.strip(),
            "answers": answers,
            "token_before": token_before,
            "token_after": token_after
        })
        
    # Save Output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    process_file(args.input_file, args.output_file, args.model_path)
