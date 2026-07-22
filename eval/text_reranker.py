"""Text-only cross-encoder reranker (Qwen3-Reranker-0.6B/4B/8B).

Same scoring mechanism as the VL reranker (yes/no logit at the last position → P(yes)), but text-only:
scores (query.text, doc.description) pairs — no image/video. Exposes `rerank_with_scores(query,
documents, top_k)` so it drops into `eval/rerank_from_run.py`'s two-pass flow unchanged.

Reference: official Qwen3-Reranker usage (causal LM + yes/no template).
"""
from __future__ import annotations
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
           'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
           '<|im_start|>user\n')
_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
_DEFAULT_INSTRUCT = 'Given a search query, retrieve the driving scenario whose description matches it'


class TextReranker:
    def __init__(self, model_path: str, device: str = 'cuda', max_length: int = 8192,
                 instruction: str = _DEFAULT_INSTRUCT):
        self.device = device
        self.max_length = max_length
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to(device).eval()
        self.token_true_id = self.tokenizer.convert_tokens_to_ids('yes')
        self.token_false_id = self.tokenizer.convert_tokens_to_ids('no')
        self.prefix_tokens = self.tokenizer.encode(_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(_SUFFIX, add_special_tokens=False)
        print(f'[TextReranker] loaded {model_path} (true_id={self.token_true_id}, '
              f'false_id={self.token_false_id})', flush=True)

    def _format(self, query_text: str, doc_text: str) -> str:
        return (f'<Instruct>: {self.instruction}\n<Query>: {query_text}\n'
                f'<Document>: {doc_text}')

    @torch.no_grad()
    def _score_pairs(self, pairs: List[str]) -> List[float]:
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        enc = self.tokenizer(pairs, padding=False, truncation='longest_first',
                             return_attention_mask=False, max_length=budget)
        for i, ids in enumerate(enc['input_ids']):
            enc['input_ids'][i] = self.prefix_tokens + ids + self.suffix_tokens
        enc = self.tokenizer.pad(enc, padding=True, return_tensors='pt', max_length=self.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits[:, -1, :]
        true_v = logits[:, self.token_true_id]
        false_v = logits[:, self.token_false_id]
        stacked = torch.stack([false_v, true_v], dim=1)
        probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return probs[:, 1].exp().float().tolist()  # P(yes)

    def rerank_with_scores(self, query, documents, top_k: int = 3, **kwargs) -> List[Tuple[object, float]]:
        """query: MultimodalQuery (uses .text); documents: List[ScenarioDocument] (uses .description)."""
        qtext = (getattr(query, 'text', None) or '')
        pairs = [self._format(qtext, (getattr(d, 'description', None) or '')) for d in documents]
        scores = self._score_pairs(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def rerank(self, query, documents, top_k: int = 3, **kwargs):
        return [d for d, _ in self.rerank_with_scores(query, documents, top_k=top_k)]
