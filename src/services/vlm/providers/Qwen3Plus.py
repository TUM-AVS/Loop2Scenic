import base64
import mimetypes
import os
from typing import Any, Optional

import dashscope
from dashscope import MultiModalConversation

from openai import OpenAI
from src.services.vlm.base import BaseVLMModel

import socket


class Qwen3Plus(BaseVLMModel):
    """Implementation of BaseVLMModel for Alibaba's Qwen VL models using OpenAI compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://{WorkspaceId}.eu-central-1.maas.aliyuncs.com/api/v1",
        **kwargs
    ):
        # --- IPv4 Force Patch ---
        old_getaddrinfo = socket.getaddrinfo
        def new_getaddrinfo(*args, **kwargs):
            responses = old_getaddrinfo(*args, **kwargs)
            # Filter out IPv6 (AF_INET6), keep only IPv4 (AF_INET)
            return [response for response in responses if response[0] == socket.AF_INET]
        socket.getaddrinfo = new_getaddrinfo
        # ------------------------

        self._model_name = kwargs.get("model_name", "qwen3-vl-flash")
        
        # Set the API key directly in the native dashscope module
        if api_key:
            dashscope.api_key = api_key
        elif "DASHSCOPE_API_KEY" in os.environ:
            dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        else:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set.")

        workspace_id = os.getenv("WORKSPACE_ID")
        dashscope.base_http_api_url = base_url.format(WorkspaceId=workspace_id)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _encode_local_file(self, file_path: str) -> str:
        import base64
        import mimetypes
        import os
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local file not found: {file_path}")
            
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
            
        with open(file_path, "rb") as file:
            base64_data = base64.b64encode(file.read()).decode("utf-8")
            
        return f"data:{mime_type};base64,{base64_data}"

    def chat(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        content = []
        
        # 1. Process local image using Native DashScope format (No 'type' key needed)
        if image:
            content.append({"image": self._encode_local_file(image)})
            
        # 2. Process local video using Native DashScope format
        if video:
            content.append({"video": self._encode_local_file(video)})
            
        # 3. Add text prompt
        if text:
            content.append({"text": text})

        # 4. Construct final message block
        messages = [{"role": "user", "content": content}]
        
        return self.chat_with_content(
            contents=messages, 
            system_instruction=instruction, 
            **kwargs
        )

    def chat_with_content(
        self, 
        contents: Any, 
        system_instruction: Optional[str] = None, 
        **kwargs
    ) -> str:
        
        # Standardize contents array
        if isinstance(contents, list) and len(contents) > 0 and "role" in contents[0]:
            messages = contents
        else:
            messages = [{"role": "user", "content": contents}]
            
        # Add system instruction in native format if provided
        if system_instruction:
            messages.insert(0, {"role": "system", "content": [{"text": system_instruction}]})

        # CRITICAL FIX: We call MultiModalConversation.call, NOT self.client.chat...
        response = MultiModalConversation.call(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        
        # Native SDK error handling
        if response.status_code != 200:
            raise Exception(
                f"DashScope API failed with code {response.status_code}. "
                f"Code: {response.code}, Message: {response.message}"
            )
            
        # Extract the text response from the native output object
        try:
            return response.output.choices[0].message.content[0]["text"]
        except (IndexError, KeyError) as e:
            raise ValueError(f"Unexpected response format from DashScope: {response.output}") from e

if __name__ == "__main__":
    # Ensure DASHSCOPE_API_KEY is exported in your environment
    vlm = Qwen3Plus(model_name="qwen3-vl-flash")
    
    # Example 1: Multimodal with local image upload
    response = vlm.chat(
        text="Please describe what happened in the scenario in the image and video.",
        image="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png",
        video="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4"
    )
    print(response)