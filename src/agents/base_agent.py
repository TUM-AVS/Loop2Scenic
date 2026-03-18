from abc import ABC, abstractmethod
import json
from typing import Any, Dict
import re
import logging

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, state: dict) -> dict:
        pass

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Clean and parse the raw text as a JSON object.
        """
        
        try: 
            json.loads(raw_text)
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.info(f"Failed to parse JSON from text, trying to match regex...")
            # try to find ```json {...} ``` or just ```{...}```
            match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", raw_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                json_obj = json.loads(json_str)
                return json_obj
            else:
                logger.error(f"Failed to find JSON in text: {raw_text}")
                return None
        except Exception as e:
            logger.error(f"Failed to parse JSON from text: {e}. Raw text: {raw_text}")
            return None

