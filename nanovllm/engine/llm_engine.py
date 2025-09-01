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
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
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
        """
        1. 将prompt转换为token_ids
        2. 为每一个请求单独设置一个Sequence对象，用于存储请求的token_ids和sampling_params
        3. 添加到scheduler的等待队列中
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 为每一个请求单独设置一个Sequence对象，用于存储请求的token_ids和sampling_params
        seq = Sequence(prompt, sampling_params)
        # 添加到scheduler的等待队列中
        self.scheduler.add(seq)

    def step(self):
        """
        1. 从scheduler的等待队列中获取等待的序列
        2. 将序列添加到模型运行器中
        3. 从模型运行器中获取输出
        4. 将输出添加到scheduler的运行队列中
        5. 返回输出
        """
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        1. 添加请求
        2. 循环执行step
        3. 返回输出
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 添加请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # self.is_finished(): 检查 waiting 和 running 队列是否都为空
        while not self.is_finished():
            t = perf_counter()
            # 执行step,
            output, num_tokens = self.step()
            #Prefill阶段 (num_tokens > 0): 处理新的prompt序列
            # 例如: 处理了100个token，耗时0.1秒 → 吞吐量 = 1000 tokens/s
            # Decode阶段 (num_tokens < 0): 逐个生成token
            # 例如: 生成了5个token，耗时0.05秒 → 吞吐量 = 100 tokens/s
            if use_tqdm:
                # 通过正负号区分处理模式，正数表示Prefill阶段，负数表示Decode阶段
                if num_tokens > 0:
                     # Prefill: 高吞吐量
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # Decode: 低吞吐量但低延迟
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
    
        # 按seq_id排序，获取对应的解码数据（需完成后）
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
