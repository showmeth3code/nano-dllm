import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.ar.sequence import Sequence
from nanovllm.engine.ar.model_runner import ModelRunner
from nanovllm.engine.ar.scheduler import Scheduler
from nanovllm.engine.diffusion.sequence import SequenceForDiffusionLM
from nanovllm.engine.diffusion.scheduler import SchedulerForDiffusionLM
from nanovllm.engine.diffusion.model_runner import ModelRunnerForDiffusionLM

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        # check if model is a dllm model
        self.config.is_dllm = True if "Dream" in model else False
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunnerForDiffusionLM if config.is_dllm else ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunnerForDiffusionLM(config, 0, self.events) if config.is_dllm else ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = SchedulerForDiffusionLM(config) if config.is_dllm else Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = SequenceForDiffusionLM(prompt, sampling_params, config=self.config) if self.config.is_dllm else Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if not self.config.is_dllm:
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        else:
            num_tokens = sum(seq.input_num_tokens + seq.new_tokens for seq in seqs) if is_prefill else -sum(seq.new_tokens for seq in seqs)
        return outputs, num_tokens

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
        prefill_throughput = decode_throughput = 0.
        acc_prefill_token_count = acc_decode_token_count = 0.
        acc_prefill_time = acc_decode_time = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    period = perf_counter() - t
                    prefill_throughput = num_tokens / period
                    acc_prefill_token_count += num_tokens
                    acc_prefill_time += period
                else:
                    period = perf_counter() - t
                    decode_throughput = -num_tokens / period
                    acc_decode_token_count += -num_tokens
                    acc_decode_time += period
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tps",
                    "Decode": f"{int(decode_throughput)}tps",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        # print(f"acc_prefill_throghput: {acc_prefill_token_count / acc_prefill_time}tps, acc_decode_throughput: {acc_decode_token_count / acc_decode_time} tps")
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # print(outputs)
        # print(self.tokenizer.decode(6395))
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        print(f"acc_decode_token_count: {acc_decode_token_count}, acc_decode_time: {acc_decode_time}")
        print(f"acc_prefill_throghput: {acc_prefill_token_count / acc_prefill_time}tps, acc_decode_throughput: {acc_decode_token_count / acc_decode_time}tps")
        return outputs
