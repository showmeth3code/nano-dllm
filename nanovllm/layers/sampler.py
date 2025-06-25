import torch
from torch import nn


class Sampler(nn.Module):
    """Sampler module that handles temperature-based token sampling."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """Sample from logits using temperature-controlled sampling.
        
        Args:
            logits: The raw logits from the model, shape [batch_size, vocab_size]
            temperatures: The temperatures to use, shape [batch_size]
            
        Returns:
            The sampled token IDs, shape [batch_size]
        """
        # For safety, handle any NaN values in logits
        logits = torch.nan_to_num(logits)
        
        # Ensure dimensions are correct
        if logits.dim() == 1:  # Single set of logits
            logits = logits.unsqueeze(0)  # Add batch dimension [1, vocab_size]
            
        if isinstance(temperatures, (int, float)):
            temperatures = torch.tensor([temperatures], device=logits.device)
        elif isinstance(temperatures, list):
            temperatures = torch.tensor(temperatures, device=logits.device)
        
        # Make sure temperatures has the right shape [batch_size]
        if temperatures.dim() > 1:
            temperatures = temperatures.squeeze()
            
        # Ensure temperatures has right batch dimension
        if temperatures.shape[0] == 1 and logits.shape[0] > 1:
            temperatures = temperatures.repeat(logits.shape[0])
            
        # Get device - handle MPS special case
        is_mps = logits.device.type == 'mps'
        device = logits.device
        
        # Process for greedy selection (temperature == 0)
        greedy_indices = logits.argmax(dim=-1)
        
        # Check if we're doing all greedy
        if (temperatures <= 1e-6).all():
            return greedy_indices
        
        # Handle device conversion for proper sampling
        if is_mps:
            # MPS has issues with sampling, move to CPU 
            logits = logits.float().cpu()
            temperatures = temperatures.float().cpu()
        
        # Apply temperature and sample
        next_token_ids = []
        for i in range(logits.shape[0]):
            temp = temperatures[i].item()
            
            if temp <= 1e-6:  # Nearly zero temperature -> greedy
                next_token_id = greedy_indices[i].item()
            else:
                # Apply temperature scaling
                scaled_logits = logits[i] / temp
                
                # For numerical stability
                scaled_logits = scaled_logits - scaled_logits.max()
                
                # Convert to probabilities
                probs = torch.softmax(scaled_logits, dim=-1)
                
                # Ensure valid probability distribution
                if torch.isnan(probs).any() or (probs.sum() == 0):
                    # Fallback to uniform distribution
                    probs = torch.ones_like(probs) / probs.shape[-1]
                
                # Ensure probabilities sum to 1
                probs = probs / probs.sum()
                
                # Sample from distribution
                try:
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                except Exception as e:
                    print(f"Sampling error: {e}, falling back to greedy")
                    next_token_id = greedy_indices[i].item()
                    
            next_token_ids.append(next_token_id)
        
        # Convert to tensor on original device
        result = torch.tensor(next_token_ids, device=device)
        return result

