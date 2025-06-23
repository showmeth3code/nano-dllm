import atexit
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download
import os

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(
        self,
        config: Config,
        engine_args,
        tokenizer,
    ):
        self.config = config
        self.engine_args = engine_args
        self.tokenizer = tokenizer

        # For single-process execution, we use rank 0 and a dummy event.
        rank = 0
        event = mp.Event() # Dummy event for non-parallel execution
        event.set()  # Prevent blocking in single-process mode
        self.model_runner = ModelRunner(config, rank, [event])
        
        self.scheduler = Scheduler(self.config)

        # Ping the model runner to ensure it's alive.
        self.model_runner.call("ping")

        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = prompt
        seq = Sequence(prompt_token_ids, sampling_params, prompt)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()

        # Update temperature for each sequence
        for seq in seqs:
            if not is_prefill:  # Only step temperature during decoding
                seq.sampling_params.step()

        # Get token IDs from model runner
        # The run method should always return a list
        output_from_model = self.model_runner.call("run", seqs, is_prefill)
        
        # Safety check - if not a list, create one
        if isinstance(output_from_model, list):
            token_ids = output_from_model
        else:
            print(f"Warning: Expected list from model_runner, got {type(output_from_model)}. Using empty list.")
            token_ids = []
            
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.prompt) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq.token_ids) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, str | list[int]]]:
        """
        Generate completions for the given prompts.
        
        Args:
            prompts: List of prompts as strings or token IDs
            sampling_params: Sampling parameters (single or per prompt)
            use_tqdm: Whether to show progress bar
        
        Returns:
            List of dicts with 'text' and 'token_ids' keys
        """
        progress_bar = None
        # if use_tqdm:
        #     progress_bar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
            
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            if progress_bar:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                progress_bar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
                
            for seq_id, token_ids, prompt in output:
                outputs[seq_id] = {"token_ids": token_ids, "prompt": prompt}
                if progress_bar:
                    progress_bar.update(1)
                    
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        
        for output in outputs:
            token_ids = output["token_ids"]
            token_ids = self._flatten_token_ids(token_ids)
            output["text"] = self.tokenizer.decode(token_ids)
            
        if progress_bar:
            progress_bar.close()
            
        return outputs

    @classmethod
    def from_engine_args(cls, engine_args):
        # Download model from HF Hub if it is not a local directory
        model_path = engine_args.model
        if not os.path.isdir(model_path):
            model_path = snapshot_download(repo_id=model_path)
            
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_config = Config(model_path)
        
        # Apply engine args to config
        if hasattr(engine_args, "enforce_eager"):
            model_config.enforce_eager = engine_args.enforce_eager
            
        if hasattr(engine_args, "tensor_parallel_size"):
            model_config.tensor_parallel_size = engine_args.tensor_parallel_size
            
        # Handle additional common configs
        for param in ["kvcache_block_size", "max_num_seqs", "num_kvcache_blocks", "max_num_batched_tokens"]:
            if hasattr(engine_args, param) and hasattr(model_config, param):
                setattr(model_config, param, getattr(engine_args, param))
        
        tokenizer_vocab_size = len(tokenizer)
        if model_config.vocab_size != tokenizer_vocab_size:
            print(f"Warning: Vocab size mismatch found. Model: {model_config.vocab_size}, Tokenizer: {tokenizer_vocab_size}.")
            print("Resizing model vocab size to match tokenizer.")
            model_config.vocab_size = tokenizer_vocab_size

        engine = cls(model_config, engine_args, tokenizer)
        return engine

    @staticmethod
    def _flatten_token_ids(token_ids):
        flat = []
        for item in token_ids:
            if isinstance(item, (list, tuple)):
                flat.extend(LLMEngine._flatten_token_ids(item))
            else:
                flat.append(item)
        return flat