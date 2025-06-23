class TemperatureScheduler:
    def __init__(
        self, 
        initial_temperature: float = 0.7,
        min_temperature: float = 0.1,
        max_temperature: float = 1.0,
        warmup_steps: int = 0,
        decay_rate: float = 1.0
    ):
        """Initialize temperature scheduler with options for warmup and decay."""
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.min_temp = min_temperature
        self.max_temp = max_temperature
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.step_count = 0
        
    def step(self) -> float:
        """Update temperature based on current step.
        
        Returns:
            Current temperature value after update
        """
        # For now, use a fixed temperature for stability
        # This means temperature won't change during generation
        return self.temperature
    
    def get_temperature(self) -> float:
        """Get current temperature without updating"""
        return self.temperature
    
    def reset(self, temperature: float | None = None):
        """Reset step count and optionally set new temperature"""
        self.step_count = 0
        if temperature is not None:
            self.temperature = max(min(temperature, self.max_temp), self.min_temp)
        else:
            self.temperature = self.initial_temperature
