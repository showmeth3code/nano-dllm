from dataclasses import dataclass
from nanovllm.layers.temperature import TemperatureScheduler


@dataclass
class SamplingParams:
    max_tokens: int = 64
    ignore_eos: bool = False

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 100,
        ignore_eos: bool = False,
    ):
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.temp_scheduler = TemperatureScheduler(
            initial_temperature=temperature,
            min_temperature=0.1,
            max_temperature=1.0,
            warmup_steps=5,
            decay_rate=0.98,
        )

    @property
    def temperature(self) -> float:
        return self.temp_scheduler.get_temperature()

    def step(self):
        self.temp_scheduler.step()
