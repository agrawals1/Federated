from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, List

from transformers import BatchEncoding, GenerationConfig, StoppingCriteriaList

from ..typing import TokenizerType


@dataclass
class LLMInput:
    prompt: str
    inputs: BatchEncoding
    num_prompt_tokens: int
    generation_config: GenerationConfig
    stopping_criteria: Optional[StoppingCriteriaList] = None
    stop: Optional[List[str]] = None


@dataclass
class LLMOutput:
    text: str  # text generated by the LLM
    token_ids: List[int]  # token IDs generated by the LLM

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any], tokenizer: TokenizerType) -> "LLMOutput":
        if "text" in input_dict:
            text = input_dict["text"]
        elif "generated_text" in input_dict:
            text = input_dict["generated_text"]
        else:
            raise ValueError("No `text` or `generated_text` in input")

        if "token_ids" not in input_dict:
            # if token_ids is empty or None
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            token_ids = input_dict["token_ids"]

        return cls(
            text=text,
            token_ids=token_ids
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
