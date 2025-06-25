from argparse import Namespace

from nanovllm.engine.llm_engine import LLMEngine


class LLM:

    def __init__(self, model: str, enforce_eager=False, tensor_parallel_size=1, **kwargs):
        # Pass the arguments as attributes of the Namespace
        engine_args = Namespace(
            model=model,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs
        )
        self.engine = LLMEngine.from_engine_args(engine_args)

    def generate(self, *args, **kwargs):
        return self.engine.generate(*args, **kwargs)
