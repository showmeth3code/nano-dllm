from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List, Union, Tuple

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    Represents the current status of a sequence in the generation pipeline.
    
    WAITING: Sequence is waiting to be processed
    RUNNING: Sequence is actively being processed
    FINISHED: Sequence has completed generation (reached EOS or max tokens)
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    Represents a sequence of tokens for LLM processing, managing both input and generated tokens.
    
    The Sequence class handles token storage, position tracking, and integration with
    the BlockManager for KV cache management. It provides properties and methods to
    access various aspects of the sequence such as prompt tokens, completion tokens,
    and block allocation.
    
    Attributes:
        block_size (int): Number of tokens per block for KV cache allocation
        seq_id (int): Unique identifier for this sequence
        status (SequenceStatus): Current processing status
        prompt (Union[str, List[int]]): Original prompt text or token IDs
        token_ids (List[int]): Full list of token IDs (prompt + generated)
        last_token (int): Most recently generated token
        num_tokens (int): Total number of tokens in sequence
        num_prompt_tokens (int): Number of tokens in the original prompt
        num_cached_tokens (int): Number of tokens with valid cache entries
        block_table (List[int]): Mapping of sequence blocks to KV cache blocks
        sampling_params (SamplingParams): Parameters controlling token sampling
        current_position (int): Current position for the next token generation
        ignore_eos (bool): Whether to ignore end-of-sequence token
        max_tokens (int): Maximum number of tokens to generate
        generated_tokens (List[int]): List of generated token IDs (excluding prompt)
    """
    
    # Class variables
    block_size = 256
    counter = count()

    def __init__(self, token_ids: List[int], sampling_params: SamplingParams, 
                prompt: Union[str, List[int]]):
        """
        Initialize a new sequence with token IDs and sampling parameters.
        
        Args:
            token_ids: List of token IDs for the initial prompt
            sampling_params: Parameters controlling token generation
            prompt: Original prompt text or token IDs (for reference)
            
        Raises:
            ValueError: If token_ids is not a list of integers
        """
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.prompt = prompt
        
        # Validate token_ids
        if isinstance(token_ids, str):
            raise ValueError("token_ids must be a list of integers, not a string")
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)  # Convert to list if needed
            if not all(isinstance(x, int) for x in token_ids):
                raise ValueError("token_ids must be a list of integers")
        
        # Initialize sequence state
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1] if token_ids else 0
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        
        # Initialize an empty block table - will be filled by BlockManager
        self.block_table: List[int] = []
        
        # Store sampling parameters and extract frequently used ones
        self.sampling_params = sampling_params
        self.current_position = self.num_tokens  # Position for the NEXT token
        self.ignore_eos = self.sampling_params.ignore_eos
        self.max_tokens = self.sampling_params.max_tokens
        
        # Keep track of generated tokens separately for clear output
        self.generated_tokens: List[int] = []

    def __len__(self) -> int:
        """
        Get the total length of the sequence (number of tokens).
        
        Returns:
            int: Number of tokens in the sequence
        """
        return self.num_tokens

    def __getitem__(self, key) -> Union[int, List[int]]:
        """
        Access token(s) by index or slice.
        
        Args:
            key: Integer index or slice
            
        Returns:
            Union[int, List[int]]: Token ID or list of token IDs
            
        Raises:
            IndexError: If key is out of range
        """
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        """
        Check if this sequence has finished generation.
        
        Returns:
            bool: True if the sequence is in FINISHED status
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """
        Get the number of tokens generated after the prompt.
        
        Returns:
            int: Number of completion tokens
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> List[int]:
        """
        Get the token IDs from the original prompt.
        
        Returns:
            List[int]: List of prompt token IDs
        """
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> List[int]:
        """
        Get the token IDs generated after the prompt.
        
        Returns:
            List[int]: List of completion token IDs
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        """
        Get the number of fully cached blocks.
        
        Returns:
            int: Number of cached blocks
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        """
        Get the total number of blocks needed for this sequence.
        
        Returns:
            int: Number of blocks (including partially filled blocks)
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """
        Get the number of tokens in the last (potentially partially filled) block.
        
        Returns:
            int: Number of tokens in the last block
        """
        if self.num_blocks == 0:
            return 0
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    @property
    def position(self) -> int:
        """
        Get the current position for the next token generation.
        
        This is crucial for correct attention operations and KV cache indexing.
        
        Returns:
            int: Current position in the sequence
        """
        return self.current_position

    def block(self, i: int) -> List[int]:
        """
        Get the token IDs for a specific block.
        
        Args:
            i: Block index (0-based)
            
        Returns:
            List[int]: Token IDs in the specified block
            
        Raises:
            ValueError: If the block index is out of range
        """
        if not 0 <= i < self.num_blocks:
            raise ValueError(f"Block index {i} out of range (0 to {self.num_blocks-1})")
        
        start = i * self.block_size
        end = min((i + 1) * self.block_size, self.num_tokens)
        return self.token_ids[start:end]

    def append_token(self, token_id: int) -> None:
        """
        Add a new token to the sequence and update all tracking information.
        
        This method updates token_ids, last_token, num_tokens, current_position,
        and generated_tokens to maintain consistent state.
        
        Args:
            token_id: The token ID to append
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.current_position += 1  # Increment position for next token
        self.generated_tokens.append(token_id)  # Track generated tokens separately

    def __getstate__(self) -> Tuple:
        """
        Prepare the sequence for pickling.
        
        Returns:
            Tuple: Tuple of essential state variables
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, 
                self.block_table, self.token_ids, self.prompt, self.current_position, 
                self.generated_tokens, self.ignore_eos, self.max_tokens)

    def __setstate__(self, state: Tuple) -> None:
        """
        Restore the sequence from unpickling.
        
        Args:
            state: Tuple returned by __getstate__
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:4]
        self.token_ids = state[4]
        self.prompt = state[5]
        self.current_position = state[6]
        self.generated_tokens = state[7] if len(state) > 7 else []
        
        # Handle backwards compatibility with older pickled states
        if len(state) > 8:
            self.ignore_eos = state[8]
            self.max_tokens = state[9] if len(state) > 9 else 100  # Default if missing
        else:
            # Set defaults for old pickled states
            self.ignore_eos = False
            self.max_tokens = 100
            
        self.last_token = self.token_ids[-1] if self.token_ids else 0
