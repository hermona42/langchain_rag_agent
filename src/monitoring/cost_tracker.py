from typing import Dict, Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard pricing rates per 1,000,000 tokens (USD)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "text-embedding-3-small": {"input": 0.020 / 1_000_000, "output": 0.0},
}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_PRICING.get(model_name, MODEL_PRICING["gpt-4o-mini"])
    input_cost = prompt_tokens * rates["input"]
    output_cost = completion_tokens * rates["output"]
    return input_cost + output_cost

class CostTrackerCallback(BaseCallbackHandler):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__()
        self.model_name = model_name
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def on_llm_end_tokens(self, prompt_tokens: int, completion_tokens: int):
        cost = calculate_cost(self.model_name, prompt_tokens, completion_tokens)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.total_cost_usd += cost
        
        logger.info(
            "LLM call usage tracked",
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            call_cost_usd=round(cost, 6),
            total_cost_usd=round(self.total_cost_usd, 6)
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            p_tokens = usage.get("prompt_tokens", 0)
            c_tokens = usage.get("completion_tokens", 0)
            self.on_llm_end_tokens(p_tokens, c_tokens)