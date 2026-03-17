"""
Multimodal document extractor for extracting text, images, videos, and other content from directories.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.services import BaseVLMModel
from ..prompt import load_prompt

logger = logging.getLogger(__name__)


class MultimodalDocumentInterpreter:
    """
    Interprets multimodal content (text, images, videos) from directories
    and converts them to dictionaries suitable for vector store.
    """
    
    # Image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    
    # Video extensions
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    # Text extensions (for reading content)
    TEXT_EXTENSIONS = {'.txt', '.md', '.scenic'}
    
    def __init__(
        self,
        default_instruction: str = "Retrieve text or image or video relevant to the user's query"
    ):
        """
        Initialize multimodal document interpreter.
        
        Args:
            default_instruction: Default instruction text for multimodal documents
        """
        self.default_instruction = default_instruction
    
    def extract_from_directory(
        self,
        directory_path: Union[str, Path],
        use_new_description: bool = False,  
    ) -> List[Dict[str, any]]:
        """
        Extract all multimodal content from a directory.
        
        For each subdirectory, creates a document dictionary
        containing:
            - instruction (str): Instruction text for multimodal documents
            - folder_path (str): Absolute path to the folder
            - description (str, optional): Content from description.txt file
            - description_json (str, optional): Content from new_description.json file, only exists when use_new_description is True
            - scenic_code (str, optional): Content from code.scenic file
            - text (str, optional): Combined text from other text files
            - video (str, optional): Path to video file if found
            - image (str, optional): Path to image file if found
        
        Args:
            directory_path: Path to the parent directory to extract from
            
        Returns:
            List of dictionaries, each representing a multimodal document
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory_path}")
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        documents = []
        
        # Process each subdirectory as a separate document
        subdirs = [d for d in directory_path.iterdir() if d.is_dir()]
        
        if not subdirs:
            logger.warning(f"No subdirectories found in {directory_path}")
            return []
        
        logger.info(f"Extracting content from {len(subdirs)} subdirectories")
        
        for subdir in subdirs:
            doc_dict = self._extract_dict_from_folder(subdir, use_new_description=use_new_description) # extract content from folder
            if doc_dict:
                documents.append(doc_dict)
        
        logger.info(f"Extracted {len(documents)} multimodal document(s)")
        return documents
    
    def _extract_dict_from_folder(self, folder_path: Path, use_new_description: bool = False) -> Optional[Dict[str, any]]:
        """
        Extract content from a single folder.
        
        Args:
            folder_path: Path to the folder
            use_new_description: Whether to use the new_description.txt file instead of description.txt file
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary with extracted content or None if folder is empty.
            When not None, the dictionary contains:
                - instruction (str): Instruction text for multimodal documents
                - folder_path (str): Absolute path to the folder
                - description (str, optional): Content from description.txt file or new_description.txt file
                - description_json (str, optional): Content from new_description.json file, only exists when use_new_description is True
                - text (str, optional): Combined text from other text files
                - video (str, optional): Path to video file if found
                - image (str, optional): Path to image file if found
            
            Returns None if the folder has no extractable content (no description, scenic_code, text, video, or image files).
        """
        folder_path = Path(folder_path)
        
        if not folder_path.is_dir():
            return None
        
        doc_dict = {
            "instruction": self.default_instruction,
            "folder_path": str(folder_path.resolve().as_posix())
        }
        
        # Extract description.txt separately
        if use_new_description:
            description_path = folder_path / "new_description.txt"
        else:
            description_path = folder_path / "description.txt"
        if description_path.exists() and description_path.is_file():
            try:
                with open(description_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        doc_dict["description"] = content
            except Exception as e:
                logger.warning(f"Error reading {description_path}: {e}")

        # if use_new_description is True, then extract the new_description.json file
        if use_new_description:
            new_description_path = folder_path / "new_description.json"
            if new_description_path.exists() and new_description_path.is_file():
                try:
                    with open(new_description_path, 'r', encoding='utf-8') as f:
                        # json.load() parses the file directly into a Python dictionary
                        content = json.load(f)
                        
                        if content: # Ensures the dictionary isn't empty
                            doc_dict["description_json"] = content
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Malformed JSON in {new_description_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error reading {new_description_path}: {e}")
        
        # Extract code.scenic separately
        scenic_path = folder_path / "code.scenic"
        if scenic_path.exists() and scenic_path.is_file():
            try:
                with open(scenic_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        doc_dict["scenic_code"] = content
            except Exception as e:
                logger.warning(f"Error reading {scenic_path}: {e}")
        
        # Extract other text files (excluding description.txt and code.scenic)
        text_parts = []
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.TEXT_EXTENSIONS:
                if file_path.name not in ["description.txt", "code.scenic"]:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                text_parts.append(content)
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
        
        # Combine other text parts if any
        if text_parts:
            doc_dict["text"] = " ".join(text_parts)
        
        # Extract video
        video_path = folder_path / "video.mp4"
        if video_path.exists() and video_path.is_file():
            doc_dict["video"] = str(video_path.resolve())
        else:
            # Look for other video files
            video_files = [
                f for f in folder_path.iterdir()
                if f.is_file() and f.suffix.lower() in self.VIDEO_EXTENSIONS
            ]
            if video_files:
                # Use the first video file found
                doc_dict["video"] = str(video_files[0].resolve())
        
        # Extract images
        image_files = [
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        if image_files:
            # Use the first image file found (can be extended to support multiple images)
            doc_dict["image"] = str(image_files[0].resolve())
        
        # Only return if we have description
        if "description" in doc_dict:
            return doc_dict
        
        return None
    
    def get_layer_model_description_by_vlm(
        self,
        content_dict: Dict[str, str],
        vlm_service: BaseVLMModel,  # VLMService when available
        prompt_template_name: Optional[str] = "describe_in_layer_model",
        **kwargs
    ) -> str:
        """
        Get layer model description by VLM (Vision Language Model).
        
        Takes a dictionary containing description, image, and/or video,
        formats it into a prompt, and calls the VLM service to generate a response.
        
        Args:
            content_dict: Dictionary with keys:
                - 'description' or 'text': Text description
                - 'image': Path to image file
                - 'video': Path to video file
            vlm_service: Optional VLMService instance. If None, creates one from config.
            prompt_template_name: Optional prompt template name. If None, uses default.
            instruction: Optional instruction text for the VLM
            **kwargs: Additional parameters for VLM (temperature, max_tokens, etc.)
            
        Returns:
            Generated response description from VLM
            
        Raises:
            ImportError: If VLM service is not available
            ValueError: If content_dict is empty or invalid
        """
        if not content_dict:
            raise ValueError("content_dict cannot be empty")
        
        # Extract content from dictionary
        description = content_dict.get("description") or content_dict.get("text", "")
        scenic_code = content_dict.get("scenic_code")
        image = content_dict.get("image")
        video = content_dict.get("video")
        
        # Validate that at least one content type is provided
        if not description and not scenic_code and not image and not video:
            raise ValueError("At least one of 'description', 'scenic_code', 'image', or 'video' must be provided")
        
        # Load prompt template
        prompt_template = load_prompt(prompt_template_name)
        prompt = prompt_template.format(
            scenic_code=scenic_code or "",
            description_text=description or ""
        )
        
        logger.info(f"Processing multimodal content with VLM: {vlm_service.model_name}")
        
        # Call VLM service
        response = vlm_service.chat(
            text=prompt,
            image=image,
            video=video,
            **kwargs
        )
        
        logger.info(f"VLM processing completed successfully, result: {response}")
        return response

    