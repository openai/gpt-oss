import sglang

class TokenGenerator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        mem_fraction_static: float = 0.7,
    ):
        # Create an LLM.
        self.engine = sglang.Engine(
            model_path=model_path,
            skip_tokenizer_init=True,
            tp_size=tensor_parallel_size,
            mem_fraction_static=mem_fraction_static,
        )

    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
    ):
        # https://docs.sglangang.ai/backend/sampling_params.html
        sampling_params = {
            "n": 1,  # number of samples to generate
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": stop_tokens,
        }
        if max_tokens > 0:
            sampling_params["max_new_tokens"] = max_tokens
        pre_len = 0
        gen_iter = self.engine.generate(
            input_ids=prompt_tokens,
            sampling_params=sampling_params,
            stream=True,
            return_logprob=return_logprobs,
        )
        for output in gen_iter:
            token_ids = output["output_ids"]
            logprobs_list = (
                output.logprobs
                if hasattr(output["meta_info"], "output_token_logprobs")
                else None
            )
            if return_logprobs is True:
                new_logprobs = logprobs_list[pre_len:]
            else:
                new_logprobs = [(None, token_id, None) for token_id in token_ids[pre_len:]]
            pre_len = len(token_ids)
            for logprob_val, token_id, _ in new_logprobs:
                if logprob_val is None:
                    yield token_id
                else:
                    yield (token_id, logprob_val)
                if stop_tokens is not None and token_id in stop_tokens:
                    break

    async def async_generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
    ):
        # https://docs.sglangang.ai/backend/sampling_params.html
        sampling_params = {
            "n": 1,  # number of samples to generate
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": stop_tokens,
        }
        if max_tokens > 0:
            sampling_params["max_new_tokens"] = max_tokens
        pre_len = 0
        gen_iter = await self.engine.async_generate(
            input_ids=prompt_tokens,
            sampling_params=sampling_params,
            stream=True,
            return_logprob=return_logprobs,
        )
        async for output in gen_iter:
            token_ids = output["output_ids"]
            logprobs_list = (
                output.logprobs
                if hasattr(output["meta_info"], "output_token_logprobs")
                else None
            )
            if return_logprobs is True:
                new_logprobs = logprobs_list[pre_len:]
            else:
                new_logprobs = [(None, token_id, None) for token_id in token_ids[pre_len:]]
            pre_len = len(token_ids)
            for logprob_val, token_id, _ in new_logprobs:
                if logprob_val is None:
                    yield token_id
                else:
                    yield (token_id, logprob_val)
                if stop_tokens is not None and token_id in stop_tokens:
                    break