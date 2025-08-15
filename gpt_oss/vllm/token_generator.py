"""
VLLM Token Generator

This module provides a token generator interface using VLLM backend for
efficient text generation with support for distributed inference.

BUG FIXES APPLIED:
- Fixed variable name inconsistency (last_token_id vs last_token_ids)
- Added proper error handling for engine operations
- Added safety checks for output access
- Fixed potential infinite loop conditions
"""

from vllm import LLMEngine, EngineArgs, SamplingParams, TokensPrompt
from typing import Generator, Union, Tuple, Optional, List


class TokenGenerator:
    """
    Token generator using VLLM engine for efficient text generation.
    
    Supports distributed inference and streaming token generation with
    proper error handling and safety checks.
    """
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        """
        Initialize the VLLM token generator.
        
        Args:
            model_path: Path to the model files
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        # BUG FIX: Add error handling for engine initialization
        try:
            args = EngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
            )
            self.engine = LLMEngine.from_engine_args(args)
            self.request_id = 0
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VLLM engine: {e}")

    def generate(self,
                 prompt_tokens: List[int],
                 stop_tokens: Optional[List[int]] = None,
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False) -> Generator[Union[int, Tuple[int, Optional[float]]], None, None]:
        """
        Generate tokens from the given prompt tokens.
        
        Args:
            prompt_tokens: List of input token IDs
            stop_tokens: Optional list of stop token IDs
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (0 = unlimited)
            return_logprobs: Whether to return log probabilities
            
        Yields:
            If return_logprobs=True: Tuple of (token_id, logprob)
            If return_logprobs=False: token_id
        """
        # BUG FIX: Add input validation
        if not prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
            
        if max_tokens == 0:
            max_tokens = None
            
        request_id = str(self.request_id)
        self.request_id += 1
        
        # BUG FIX: Add error handling for sampling params creation
        try:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop_token_ids=stop_tokens,
                logprobs=0 if return_logprobs else None
            )
            prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
            self.engine.add_request(request_id, prompt, sampling_params)
        except Exception as e:
            raise RuntimeError(f"Failed to add request to engine: {e}")
        
        # BUG FIX: Fixed variable name inconsistency (was last_token_id, should be last_token_ids)
        last_token_ids = []
        iteration_count = 0
        max_iterations = 10000  # BUG FIX: Prevent infinite loops
        
        while self.engine.has_unfinished_requests() and iteration_count < max_iterations:
            iteration_count += 1
            
            # BUG FIX: Add error handling for engine step
            try:
                step_outputs = self.engine.step()
            except Exception as e:
                print(f"Warning: Engine step failed: {e}")
                break
                
            # BUG FIX: Add safety checks for output access
            if not step_outputs or len(step_outputs) == 0:
                continue
                
            output = step_outputs[0].outputs[0] if step_outputs[0].outputs else None
            if output is None:
                continue
                
            token_ids = output.token_ids
            logprobs_list = output.logprobs if hasattr(output, "logprobs") else None
            
            # BUG FIX: Fixed variable name reference
            new_token_ids = token_ids[len(last_token_ids):]
            new_logprobs = (logprobs_list[len(last_token_ids):] 
                          if logprobs_list is not None 
                          else [None] * len(new_token_ids))
            
            for token_id, logprobs in zip(new_token_ids, new_logprobs):
                # BUG FIX: Fixed variable name reference
                last_token_ids.append(token_id)
                
                if return_logprobs:
                    logprob_val = None
                    if logprobs is not None and token_id in logprobs:
                        logprob_val = logprobs[token_id].logprob
                    yield (token_id, logprob_val)
                else:
                    yield token_id
                    
                # Check for stop tokens
                if stop_tokens is not None and token_id in stop_tokens:
                    return
                    
        # BUG FIX: Handle infinite loop case
        if iteration_count >= max_iterations:
            print(f"Warning: Generation stopped after {max_iterations} iterations to prevent infinite loop")
