"""
QwenVL reranker implementation for reordering retrieved documents.
"""

import logging
import torch
from typing import List, Optional, Any, Union
from langchain_core.documents import Document

from ..base import BaseReranker

logger = logging.getLogger(__name__)


class QwenVLReranker(BaseReranker):
    """
    Reranker for QwenVL models.
    
    Uses Qwen3-VL-Reranker to rerank documents based on multimodal queries.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        instruction: str = "Retrieval relevant image or text with user's query",
        **kwargs
    ):
        """
        Initialize QwenVL reranker.
        
        Args:
            model_path: Path to local Qwen3-VL-Reranker model directory (preferred for local models)
            model_name_or_path: HuggingFace model name or path (for backward compatibility)
                If model_path is provided, this is ignored
            instruction: Instruction text for the reranker
            fps: Frames per second for video processing
            device: Device to run the model on (default: auto-detect)
            torch_dtype: Data type for the model (default: auto)
            max_frames: Maximum number of frames for video processing
            **kwargs: Additional arguments passed to Qwen3VLReranker
        """
        if Qwen3VLReranker is None:
            raise ImportError(
                "Qwen3VLReranker model class not found. "
                "Please ensure qwen3_vl_reranker.py is available or install the required package."
            )
        self.model_path = model_path
        self.model = Qwen3VLReranker(
            model_name_or_path=self.model_path,
            **kwargs
        )
        self.instruction = instruction
        print(f"QwenVLReranker initialized successfully with model: {self.model_path}")

    def _document_to_dict(self, doc: Document) -> dict[str, Any]:
        """
        Convert a Document object to the dictionary format expected by the reranker.
        
        Args:
            doc: Document object with page_content and metadata
            
        Returns:
            Dictionary with text, image, video fields
        """
        doc_dict: dict[str, Any] = {}
        
        # Extract text from page_content
        if doc.page_content:
            doc_dict["text"] = doc.page_content
        
        # Extract multimodal fields from metadata
        if doc.metadata:
            # Check for image
            if "image" in doc.metadata:
                doc_dict["image"] = doc.metadata["image"]
            
            # Check for video
            if "video" in doc.metadata:
                doc_dict["video"] = doc.metadata["video"]
            
            # If we have both text and image, include both
            if "text" in doc.metadata and "image" in doc.metadata:
                # Create combined entry
                doc_dict = {
                    "text": doc.metadata.get("text", doc.page_content),
                    "image": doc.metadata["image"]
                }
        
        # If no content found, use empty text
        if not doc_dict:
            doc_dict = {"text": doc.page_content or ""}
        
        return doc_dict

    def _query_to_dict(self, query: Union[str, dict[str, Any]]) -> dict[str, Any]:
        """
        Convert query to the format expected by the reranker.
        
        Args:
            query: Query string or multimodal query dictionary
            
        Returns:
            Dictionary with text, image, video fields
        """
        if isinstance(query, str):
            return {"text": query}
        return query

    def rerank(
        self,
        query: Union[str, dict[str, Any]],
        documents: List[Document],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Rerank a list of documents based on a query.
        
        Args:
            query: Query string or multimodal query dictionary
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return after reranking
            **kwargs: Additional arguments (e.g., instruction, fps, max_frames)
            
        Returns:
            List of reranked Document objects, ordered by relevance (most relevant first)
        """
        if not documents:
            return []
        
        # Get scores using rerank_with_scores
        scored_docs = self.rerank_with_scores(query, documents, top_k=top_k, **kwargs)
        
        # Return only documents (without scores)
        return [doc for doc, _ in scored_docs]

    def rerank_with_scores(
        self,
        query: Union[str, dict[str, Any]],
        documents: List[Document],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Rerank documents and return them with relevance scores.
        
        Args:
            query: Query string or multimodal query dictionary
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return
            **kwargs: Additional arguments (e.g., instruction, fps, max_frames)
            
        Returns:
            List of (Document, score) tuples, ordered by score (highest first)
        """
        if not documents:
            return []
        
        # Get parameters from kwargs or use defaults
        instruction = kwargs.get("instruction", self.instruction)
        
        # Convert query to dictionary format
        query_dict = self._query_to_dict(query)
        
        # Convert documents to dictionary format
        doc_dicts = [self._document_to_dict(doc) for doc in documents]
        
        logger.debug(
            f"Reranking {len(documents)} documents with query type: "
            f"{'text' if isinstance(query, str) else 'multimodal'}"
        )
        
        # Prepare input for Qwen3VLReranker.process method
        inputs = {
            "instruction": instruction,
            "query": query_dict,
            "documents": doc_dicts,
        }
        
        # Get scores from the reranker model
        try:
            scores = self.model.process(inputs)
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise
        
        # Convert scores to list of floats if they're tensors
        scores_list = []
        for score in scores:
            if isinstance(score, torch.Tensor):
                scores_list.append(float(score.item()))
            else:
                scores_list.append(float(score))
        
        # Validate scores
        if not scores_list or len(scores_list) != len(documents):
            logger.warning(
                f"Score count mismatch: expected {len(documents)}, got {len(scores_list) if scores_list else 0}"
            )
            # Return documents with dummy scores if there's a mismatch
            return [(doc, 0.0) for doc in documents]
        
        # Create list of (document, score) tuples
        doc_score_pairs = list(zip(documents, scores_list))
        
        # Sort by score (highest first) - assuming higher scores are better
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit if specified
        if top_k is not None:
            doc_score_pairs = doc_score_pairs[:top_k]
        
        logger.info(
            f"Reranked {len(doc_score_pairs)} documents "
            f"(score range: {min(scores_list):.4f} - {max(scores_list):.4f})"
        )
        
        return doc_score_pairs

import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR  # 4 tokens
MAX_PIXELS = 640 * IMAGE_FACTOR * IMAGE_FACTOR  # 640 tokens
MAX_RATIO = 200

FRAME_FACTOR = 2
FPS = 0.5
MIN_FRAMES = 2
MAX_FRAMES = 64
MIN_TOTAL_PIXELS = 1 * FRAME_FACTOR * MIN_PIXELS  # 1 frames
MAX_TOTAL_PIXELS = 4 * FRAME_FACTOR * MAX_PIXELS  # 4 frames


def sample_frames(frames, num_segments, max_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            single_frame_path = frames[frame_idx]
        except:
            break
        sampled_frames.append(single_frame_path)
    # Pad with last frame if total frames less than num_segments
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames[:max_segments]

class Qwen3VLReranker():
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        num_frames: int = MAX_FRAMES,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
        **kwargs,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames

        self.default_instruction = default_instruction

        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=True, **kwargs
        ).to(self.device)

        self.model = lm.model
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True,
            padding_side='left'
        )
        self.model.eval()

        token_true_id = self.processor.tokenizer.get_vocab()["yes"]
        token_false_id = self.processor.tokenizer.get_vocab()["no"]
        self.score_linear = self.get_binary_linear(lm, token_true_id, token_false_id)
        self.score_linear.eval()
        self.score_linear.to(self.device).to(self.model.dtype)

    def get_binary_linear(self, model, token_yes, token_no):

        lm_head_weights = model.lm_head.weight.data

        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]

        D = weight_yes.size()[0]
        linear_layer = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            linear_layer.weight[0] = weight_yes - weight_no
        return linear_layer

    @torch.no_grad()
    def compute_scores(self, inputs):
        batch_scores = self.model(**inputs).last_hidden_state[:, -1]
        scores = self.score_linear(batch_scores)
        scores = torch.sigmoid(scores).squeeze(-1).cpu().detach().tolist()
        return scores

    def truncate_tokens_optimized(
        self,
        tokens: List[str],
        max_length: int,
        special_tokens: List[str]
    ) -> List[str]:
        if len(tokens) <= max_length:
            return tokens

        special_tokens_set = set(special_tokens)

        # Calculate budget: how many non-special tokens we can keep
        num_special = sum(1 for token in tokens if token in special_tokens_set)
        num_non_special_to_keep = max_length - num_special

        # Build final list according to budget
        final_tokens = []
        non_special_kept_count = 0
        for token in tokens:
            if token in special_tokens_set:
                final_tokens.append(token)
            elif non_special_kept_count < num_non_special_to_keep:
                final_tokens.append(token)
                non_special_kept_count += 1

        return final_tokens

    def tokenize(self, pairs: list, **kwargs):
        max_length = self.max_length
        text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)
        try:
            images, videos, video_kwargs = process_vision_info(
                pairs, image_patch_size=16, 
                return_video_kwargs=True, 
                return_video_metadata=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            videos = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}], 
                add_generation_prompt=True, tokenize=False
            )
        
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.truncate_tokens_optimized(
                inputs['input_ids'][i][:-5], max_length,
                self.processor.tokenizer.all_special_ids
            ) + inputs['input_ids'][i][-5:]
        temp_inputs = self.processor.tokenizer.pad(
            {'input_ids': inputs['input_ids']}, padding=True,
            return_tensors="pt", max_length=self.max_length
        )
        for key in temp_inputs:
            inputs[key] = temp_inputs[key]
        return inputs

    def format_mm_content(
        self, 
        text, image, video, 
        prefix='Query:', 
        fps=None, max_frames=None,
    ):
        content = []

        content.append({'type': 'text', 'text': prefix})
        if not text and not image and not video:
            content.append({'type': 'text', 'text': "NULL"})
            return content

        if video:
            video_content = None
            video_kwargs = { 'total_pixels': self.total_pixels }
            if isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    video_content = self._sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele) 
                    for ele in video_content
                ]
            elif isinstance(video, str):
                video_content = video if video.startswith(('http://', 'https://')) else 'file://' + video
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames,}
            else:
                raise TypeError(f"Unrecognized video type: {type(video)}")

            if video_content:
                content.append({
                    'type': 'video', 'video': video_content,
                    **video_kwargs
                })

        if image:
            image_content = None
            if isinstance(image, Image.Image):
                image_content = image
            elif isinstance(image, str):
                image_content = image if image.startswith(('http', 'oss')) else 'file://' + image
            else:
                raise TypeError(f"Unrecognized image type: {type(image)}")

            if image_content:
                content.append({
                    'type': 'image', 'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        if text:
            content.append({'type': 'text', 'text': text})
        return content

    def format_mm_instruction(
        self, 
        query_text, query_image, query_video, 
        doc_text, doc_image, doc_video,
        instruction=None, 
        fps=None, max_frames=None
    ):
        inputs = []
        inputs.append({
            "role": "system",
            "content": [{
                "type": "text",
                "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            }
            ]
        })
        if isinstance(query_text, tuple):
            instruct, query_text = query_text
        else:
            instruct = instruction
        contents = []
        contents.append({
            "type": "text",
            "text": '<Instruct>: ' + instruct
        })
        query_content = self.format_mm_content(
            query_text, query_image, query_video, prefix='<Query>:', 
            fps=fps, max_frames=max_frames
        )
        contents.extend(query_content)
        doc_content = self.format_mm_content(
            doc_text, doc_image, doc_video, prefix='\n<Document>:', 
            fps=fps, max_frames=max_frames
        )
        contents.extend(doc_content)
        inputs.append({
            "role": "user",
            "content": contents
        })
        return inputs

    def process(
        self,
        inputs,
    ) -> list[torch.Tensor]:
        instruction = inputs.get('instruction', self.default_instruction)

        query = inputs.get("query", {})
        documents = inputs.get("documents", [])
        if not query or not documents:
            return []

        pairs = [self.format_mm_instruction(
            query.get('text', None),
            query.get('image', None),
            query.get('video', None),
            document.get('text', None),
            document.get('image', None),
            document.get('video', None),
            instruction=instruction,
            fps=inputs.get('fps', self.fps),
            max_frames=inputs.get('max_frames', self.max_frames)
        ) for document in documents]

        final_scores = []
        for pair in pairs:
            inputs = self.tokenize([pair])
            inputs = inputs.to(self.model.device)
            scores = self.compute_scores(inputs)
            final_scores.extend(scores)
        return final_scores