import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fileds = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fileds}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        result = self.model_runner.call("run", seqs, is_prefill)
        
        # Handle both list and dict returns from model_runner
        logits_data = {}
        if isinstance(result, dict):
            token_ids = result['tokens']
            # Store logits data per sequence
            for i, seq in enumerate(seqs):
                if 'logits' in result and 'indices' in result:
                    logits_data[seq.seq_id] = {
                        'logits': result['logits'][i] if i < len(result['logits']) else None,
                        'indices': result['indices'][i] if i < len(result['indices']) else None
                    }
        else:
            token_ids = result
            
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids, logits_data.get(seq.seq_id)) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens, logits_data

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        all_logits_data = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, step_logits_data = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # Collect logits data for each step
            for seq_id, logits_info in step_logits_data.items():
                if seq_id not in all_logits_data:
                    all_logits_data[seq_id] = {'logits': [], 'indices': []}
                if logits_info and logits_info['logits'] is not None:
                    all_logits_data[seq_id]['logits'].append(logits_info['logits'])
                    all_logits_data[seq_id]['indices'].append(logits_info['indices'])
            
            for seq_id, token_ids, _ in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # Format outputs
        results = []
        for seq_id in sorted(outputs):
            result = {
                "text": self.tokenizer.decode(outputs[seq_id]), 
                "token_ids": outputs[seq_id]
            }
            # Add logits data if available
            if seq_id in all_logits_data and all_logits_data[seq_id]['logits']:
                result['logits'] = all_logits_data[seq_id]['logits']
                result['indices'] = all_logits_data[seq_id]['indices']
            results.append(result)
        
        if use_tqdm:
            pbar.close()
        return results
