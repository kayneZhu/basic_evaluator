import time
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import torch

from opd_eval.contract import vllm_sampling_kwargs


def _patch_qwen3_extra_special_tokens():
    """Patch Qwen3 tokenizer to handle extra_special_tokens correctly."""
    try:
        from transformers import AutoTokenizer
        # This is a workaround for Qwen3 tokenizer issue
        original_from_pretrained = AutoTokenizer.from_pretrained
        def patched_from_pretrained(*args, **kwargs):
            tokenizer = original_from_pretrained(*args, **kwargs)
            if hasattr(tokenizer, 'extra_special_tokens') and isinstance(tokenizer.extra_special_tokens, list):
                tokenizer.extra_special_tokens = {token: idx for idx, token in enumerate(tokenizer.extra_special_tokens)}
            return tokenizer
        AutoTokenizer.from_pretrained = patched_from_pretrained
    except ImportError:
        pass


class ModelInference:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 4096, 
                 enable_thinking: bool = False):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking
        self.llm = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        # Patch Qwen3 tokenizer if needed
        _patch_qwen3_extra_special_tokens()
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True
        )
        
        # Get tokenizer from vLLM
        self.tokenizer = self.llm.get_tokenizer()

    def _messages_to_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Convert prompt and system prompt to chat template with thinking mode support."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        apply_chat = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat):
            return self._messages_to_prompt_fallback(prompt, system_prompt)

        try:
            return apply_chat(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            try:
                return apply_chat(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return self._messages_to_prompt_fallback(prompt, system_prompt)

    def _messages_to_prompt_fallback(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Simple Qwen-style chat prompt if tokenizer has no apply_chat_template."""
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        return "".join(parts)

    def generate_batch(self, prompts: List[str], 
                      max_tokens: int = 2048,
                      temperature: float = 0.0,
                      top_p: float = 1.0,
                      stop: Optional[List[str]] = None, 
                      system_prompt: Optional[str] = None,
                      n: int = 1) -> List[Any]:
        # Convert prompts to chat format if tokenizer supports it
        formatted_prompts = []
        for prompt in prompts:
            formatted_prompts.append(self._messages_to_prompt(prompt, system_prompt))
        
        sampling_params = SamplingParams(
            **vllm_sampling_kwargs(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                n=n,
            )
        )
        
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        results = []
        for output in outputs:
            if n == 1:
                results.append(output.outputs[0].text)
            else:
                results.append([o.text for o in output.outputs])
        return results

    def generate_batch_with_metrics(self, prompts: List[str],
                                    max_tokens: int = 2048,
                                    temperature: float = 0.0,
                                    top_p: float = 1.0,
                                    stop: Optional[List[str]] = None, 
                                    system_prompt: Optional[str] = None,
                                    n: int = 1) -> Dict[str, Any]:
        start_time = time.time()
        
        # Convert prompts to chat format if tokenizer supports it
        formatted_prompts = []
        for prompt in prompts:
            formatted_prompts.append(self._messages_to_prompt(prompt, system_prompt))
        
        sampling_params = SamplingParams(
            **vllm_sampling_kwargs(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                n=n,
            )
        )
        
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results = []
        total_tokens = 0
        
        for output in outputs:
            if n == 1:
                generated_text = output.outputs[0].text
            else:
                generated_text = [o.text for o in output.outputs]
                
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = sum(len(o.token_ids) for o in output.outputs)
            total_tokens += completion_tokens
            
            results.append({
                'generated_text': generated_text,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            })
        
        metrics = {
            'total_time': total_time,
            'average_time_per_request': total_time / len(prompts),
            'total_tokens': total_tokens,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'requests_per_second': len(prompts) / total_time if total_time > 0 else 0
        }
        
        return {
            'results': results,
            'metrics': metrics
        }

    def generate_single(self, prompt: str,
                       max_tokens: int = 2048,
                       temperature: float = 0.0,
                       top_p: float = 1.0,
                       stop: Optional[List[str]] = None, 
                       system_prompt: Optional[str] = None) -> str:
        results = self.generate_batch([prompt], max_tokens, temperature, top_p, stop, system_prompt)
        return results[0] if results else ""

    def cleanup(self):
        if self.llm is not None:
            del self.llm
            torch.cuda.empty_cache()
