import torch
import torch.distributed as dist
import datetime

from nanovllm.config import Config
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.engine.sequence import Sequence


class ModelRunner:
    def __init__(self, config: Config, rank: int, event):
        print(f"[DEBUG] {datetime.datetime.now()} ModelRunner.__init__ start")
        self.config = config
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        self.event = event
        
        # KV cache for sequences
        self.kv_caches = {}
        # Track past hidden states
        self.past_key_values = {}

        # Determine the best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Metal Performance Shaders (MPS) device")
            # Ensure MPS is properly initialized
            torch.mps.empty_cache()
        else:
            self.device = torch.device("cpu")
            print("Using CPU for computations - this will be slow")

        print(f"[DEBUG] {datetime.datetime.now()} Initializing process group")
        dist.init_process_group("gloo", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        print(f"[DEBUG] {datetime.datetime.now()} Process group initialized")
        
        # Set default dtype based on device - bf16 works well on MPS, float32 on CPU
        if self.device.type == 'mps':
            default_dtype = torch.bfloat16
        elif self.device.type == 'cpu':
            default_dtype = torch.float32  
        else:
            default_dtype = torch.float16
            
        torch.set_default_dtype(getattr(config.hf_config, "torch_dtype", default_dtype))
        print(f"[DEBUG] {datetime.datetime.now()} Creating Qwen3ForCausalLM model with dtype: {torch.get_default_dtype()}")
        
        # Create the model and explicitly move to device
        self.model = Qwen3ForCausalLM(config.hf_config)
        print(f"[DEBUG] {datetime.datetime.now()} Moving model to device: {self.device}")
        self.model = self.model.to(self.device)
        print(f"[DEBUG] {datetime.datetime.now()} Model device after move: {next(self.model.parameters()).device}")
        
        print(f"[DEBUG] {datetime.datetime.now()} Model created, loading weights")
        self.load_weights()
        
        # Verify the model is on the correct device after weight loading
        actual_device = next(self.model.parameters()).device
        print(f"[DEBUG] Final model device: {actual_device}")
        if actual_device.type != self.device.type:
            print(f"WARNING: Model is on {actual_device.type} but requested device was {self.device.type}")
            # Force moving model to requested device again if needed
            self.model = self.model.to(self.device)
            
        # Set the model to evaluation mode for inference
        self.model.eval()
            
        print(f"[DEBUG] {datetime.datetime.now()} ModelRunner.__init__ end")

    def load_weights(self):
        """Load model weights and ensure they stay on the correct device"""
        # Get the initial device
        initial_device = next(self.model.parameters()).device
        print(f"[DEBUG] Device before loading weights: {initial_device}")
        
        # Load the model weights
        load_model(self.model, self.config.model_path)
        
        # Verify and fix device after loading
        current_device = next(self.model.parameters()).device
        print(f"[DEBUG] Device after loading weights: {current_device}")
        
        if current_device.type != self.device.type:
            print(f"[DEBUG] Moving model back to {self.device} after weight loading")
            self.model = self.model.to(self.device)
    
    def _setup_for_inference(self):
        """Ensure model is in the right mode and on the right device for inference"""
        self.model.eval()
        model_device = next(self.model.parameters()).device
        if model_device.type != self.device.type:
            print(f"WARNING: Model device mismatch detected in run(): Model on {model_device.type}, should be on {self.device.type}")
            self.model = self.model.to(self.device)

    def run(self, seqs: "list[Sequence]", is_prefill: bool) -> list[int]:
        """
        Run inference on the model for the given sequences.
        
        Args:
            seqs: List of sequences to run inference on
            is_prefill: Whether to run prefill (initial forward pass) or decode (next token prediction)
            
        Returns:
            List of next token predictions (int)
        """
        if not seqs:
            return []
            
        # Ensure model is ready for inference
        self._setup_for_inference()
        
        with torch.no_grad():
            if is_prefill:
                # Pre-fill phase (processing the entire prompt)
                next_token_ids = []
                
                for seq in seqs:
                    seq_id = seq.seq_id
                    
                    # Convert to tensor and ensure device
                    input_ids = torch.tensor(seq.token_ids, device=self.device).unsqueeze(0)  # [1, seq_len]
                    
                    # Generate positions starting from 0
                    positions = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)  # [1, seq_len]
                    
                    # Forward pass
                    logits = self.model(input_ids, positions)  # [1, seq_len, vocab_size]
                    
                    # Get the last token's logits for sampling
                    next_token_logits = logits[0, -1, :]  # [vocab_size]
                    
                    # Apply temperature
                    temperature = seq.sampling_params.temperature
                    
                    # Sample token
                    if temperature <= 1e-6:  # Nearly zero temperature -> greedy
                        next_token_id = torch.argmax(next_token_logits).item()
                    else:
                        # Apply temperature and convert to probability distribution
                        next_token_logits = next_token_logits / temperature
                        probs = torch.softmax(next_token_logits, dim=0)
                        
                        # Sample from the distribution
                        next_token_id = torch.multinomial(probs, num_samples=1).item()
                    
                    next_token_ids.append(next_token_id)
                    
                    # Store this sequence's context for future decoding
                    self.kv_caches[seq_id] = {
                        'position': input_ids.shape[1],  # Next position will be this
                        'tokens': seq.token_ids.copy()  # Store full token sequence
                    }
                    
                print(f"[DEBUG] Prefill next token predictions: {next_token_ids}")
                return next_token_ids
                
            else:
                # Decode phase (generating next token)
                next_token_ids = []
                
                for seq in seqs:
                    seq_id = seq.seq_id
                    
                    # Check if we have this sequence in our cache 
                    if seq_id not in self.kv_caches:
                        print(f"WARNING: Sequence {seq_id} not found in KV cache. Creating new entry.")
                        # Create new entry based on current sequence state
                        self.kv_caches[seq_id] = {
                            'position': len(seq.token_ids),
                            'tokens': seq.token_ids.copy()
                        }
                    
                    # Get the full token sequence from our cache
                    tokens = self.kv_caches[seq_id]['tokens']
                    
                    # Convert to tensor and ensure device 
                    input_ids = torch.tensor(tokens, device=self.device).unsqueeze(0)  # [1, seq_len] 
                    
                    # Generate positions for the full sequence starting from 0
                    positions = torch.arange(0, len(tokens), dtype=torch.long, device=self.device).unsqueeze(0)  # [1, seq_len]
                    
                    # Forward pass through model with full context
                    with torch.no_grad():
                        logits = self.model(input_ids, positions)  # [1, seq_len, vocab_size]
                    
                    # Get logits for the last token
                    next_token_logits = logits[0, -1, :]  # [vocab_size]
                    
                    # Penalize repetition to avoid getting stuck
                    # Simple heuristic: decrease probability of tokens that appear frequently in recent history
                    last_tokens = tokens[-20:] if len(tokens) > 20 else tokens  # Look at recent history
                    
                    # Count occurrences and penalize repeated tokens
                    for token_id in set(last_tokens):
                        # Count occurrences in recent history
                        count = last_tokens.count(token_id)
                        if count > 1:  # If token appears multiple times
                            # Apply repetition penalty: the more occurrences, the larger the penalty
                            penalty = 1.0 + 0.3 * count  # Adjust penalty strength as needed
                            # Decrease the logits for this token
                            if next_token_logits[token_id] > 0:
                                next_token_logits[token_id] = next_token_logits[token_id] / penalty
                            else:
                                next_token_logits[token_id] = next_token_logits[token_id] * penalty
                    
                    # Apply temperature
                    temperature = seq.sampling_params.temperature
                    
                    # Sample token
                    if temperature <= 1e-6:  # Nearly zero temperature -> greedy
                        next_token_id = torch.argmax(next_token_logits).item()
                    else:
                        # Apply temperature and convert to probability distribution
                        next_token_logits = next_token_logits / temperature
                        probs = torch.softmax(next_token_logits, dim=0)
                        
                        # Sample from the distribution
                        next_token_id = torch.multinomial(probs, num_samples=1).item()
                    
                    next_token_ids.append(next_token_id)
                    
                    # Update cache with the new token
                    self.kv_caches[seq_id]['tokens'].append(next_token_id)
                
                # print(f"[DEBUG] Decode next token predictions: {next_token_ids}")
                return next_token_ids

    def __call__(self, seqs, is_prefill=True):
        """Run model on input sequences. Delegates to run() method."""
        return self.run(seqs, is_prefill)
        
    def call(self, method_name, *args, **kwargs):
        """Call a method with arguments, handling any synchronization required."""
        if isinstance(self.event, list):
            for e in self.event:
                e.wait()
        else:
            self.event.wait()
        method = getattr(self, method_name, None)
        assert callable(method)
        result = method(*args, **kwargs)
        return result

    def ping(self):
        """Simple method to check if the model runner is alive."""
        pass

    def exit(self):
        """Clean up resources when exiting."""
        pass
