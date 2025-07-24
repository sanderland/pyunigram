import json
import os
from typing import List, Tuple, Dict, Optional, Union
from .train import train_unigram # Import the training function
from .model import Lattice, TrainerModel # Import core model components

class QwenUnigramTokenizer:
    """
    A Unigram tokenizer that uses the trained model from train.py.
    It encapsulates the vocabulary and provides encode/decode functionality.
    """

    def __init__(self, tokens: List[Tuple[str, float]], metadata: Optional[Dict] = None):
        """
        Initializes the tokenizer with trained pieces and optional metadata.

        Args:
            tokens: A list of (piece_string, score) tuples representing the vocabulary.
            metadata: Optional dictionary containing tokenizer information (e.g., training params).
        """
        if not tokens:
            raise ValueError("The 'tokens' list cannot be empty.")
        
        self.tokens = tokens
        self.metadata = metadata if metadata is not None else {}

        # --- Derived data for fast lookup ---
        # ID to token string mapping (ID is the index in the `tokens` list)
        self._id_to_token: List[str] = [piece for piece, _ in tokens]
        
        # Token string to ID mapping
        self._token_to_id: Dict[str, int] = {piece: i for i, (piece, _) in enumerate(tokens)}

        # --- Inference Model ---
        # Reuse TrainerModel for segmentation logic (populate, viterbi).
        # We initialize it with dummy frequencies (e.g., 1) because the core
        # segmentation logic in `populate` relies on the piece set and scores,
        # not the initial frequencies used during TrainerModel's own __init__ scoring.
        # The scores from `tokens` are used directly by the model after initialization.
        
        # Create a dictionary of piece strings to dummy frequencies for initialization
        piece_strings_for_model_init = {piece: 1.0 for piece, _ in tokens}
        
        # Assume max_piece_length was used during training or use a default.
        # It's crucial for Lattice.populate to work correctly.
        # Metadata could store this, or we assume it's implicit in the model.
        # Let's try to get it from metadata, otherwise assume a reasonable default or infer.
        # Inferring from the longest piece in the current vocab is a heuristic.
        max_piece_len_from_vocab = max((len(piece) for piece, _ in tokens), default=16)
        max_piece_length = self.metadata.get('training_params', {}).get('max_piece_len', max_piece_len_from_vocab)
        
        # Initialize the TrainerModel instance for inference
        self.model = TrainerModel(piece_strings_for_model_init, max_piece_length=max_piece_length)
        
        # Important: Overwrite the model's initial scores with the trained scores from `tokens`
        # This ensures the Viterbi algorithm uses the correct probabilities learned during training.
        self.model.set_pieces(tokens)

        # --- Handle Special Tokens (based on common conventions or metadata) ---
        # This implementation assumes the first token is often <unk>.
        # A more robust approach would be to have metadata specify special token IDs/names.
        self.unk_token = "<unk>"
        self.unk_id = self._token_to_id.get(self.unk_token, 0) # Default to ID 0 if not found


    def encode(self, text: str, return_tokens: bool = False) -> Union[List[int], Tuple[List[int], List[str]]]:
        """
        Encodes a text string into a list of token IDs.

        Args:
            text: The input text string.
            return_tokens: If True, also returns the list of token strings.

        Returns:
            A list of token IDs, or a tuple of (IDs, token_strings) if return_tokens is True.
        """
        if not isinstance(text, str):
             # Handle bytes input if needed, or raise TypeError
             raise TypeError("Input 'text' must be a string.")

        # 1. Create a lattice for the input text
        lattice = Lattice(text)

        # 2. Populate the lattice using the trained model's vocabulary and scores
        # This fills the lattice with possible segmentations and their scores.
        self.model.populate(lattice)

        # 3. Find the best segmentation path using the Viterbi algorithm
        viterbi_path, _score = lattice.viterbi()

        # 4. Extract token IDs from the Viterbi path
        # The piece_id from the lattice node corresponds to the index in the model's piece list,
        # which should match the index in our `self.tokens` and thus our ID mapping.
        token_ids = [node.piece_id for node in viterbi_path]

        # Basic handling for potential out-of-vocabulary segments
        # In a well-trained model with required_chars, this should be rare.
        # TrainerModel.populate should ideally handle unknown chars by mapping them
        # to a designated unknown piece (e.g., <unk>) whose ID is known.
        # If piece_id is somehow invalid (though unlikely from a valid model), map to unk.
        # This check might be redundant if TrainerModel/Lattice guarantees valid IDs.
        # final_token_ids = [tid if 0 <= tid < len(self._id_to_token) else self.unk_id for tid in token_ids]
        # For simplicity, and assuming model integrity, we use token_ids directly.
        # If issues arise, uncomment the check above.

        if return_tokens:
            # Map IDs back to token strings for return
            try:
                token_strings = [self._id_to_token[tid] for tid in token_ids]
            except IndexError:
                # Should ideally not happen if model is consistent
                raise RuntimeError("Internal error: Invalid token ID generated during encoding.")
            return token_ids, token_strings

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into a text string.

        Args:
            token_ids: A list of integer token IDs.

        Returns:
            The decoded text string.
        """
        if not token_ids:
            return ""
        try:
            # Map each ID to its corresponding token string and concatenate
            token_strings = [self._id_to_token[tid] for tid in token_ids]
            return "".join(token_strings)
        except IndexError:
            # Handle invalid token IDs gracefully
            raise ValueError("Invalid token ID encountered during decoding.")

    def __len__(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
            The number of tokens in the vocabulary.
        """
        return len(self.tokens)

    def save(self, path: str):
        """
        Save the tokenizer configuration to a JSON file.

        Args:
            path: The file path where the tokenizer should be saved.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            "tokens": self.tokens, # List of [piece_string, score] lists
            "metadata": self.metadata
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


    @classmethod
    def load(cls, path: str) -> 'QwenUnigramTokenizer':
        """
        Load a tokenizer from a JSON file.

        Args:
            path: The file path to load the tokenizer from.

        Returns:
            An instance of QwenUnigramTokenizer.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure tokens are tuples
        tokens = [tuple(item) for item in data["tokens"]]
        metadata = data.get("metadata", {})
        
        return cls(tokens, metadata)

    @classmethod
    def train(cls, **kwargs) -> 'QwenUnigramTokenizer':
        """
        Train a new tokenizer instance.

        Args:
            **kwargs: Keyword arguments passed to the training function.
                      Must include 'pretokens'. Other args like 'vocab_size',
                      'required_chars', etc. are optional.

        Returns:
            A new instance of QwenUnigramTokenizer.
        """
        # The training function returns the list of (piece_string, score) tuples
        tokens = train_unigram(**kwargs)
        
        # Capture training parameters for metadata, excluding the main data input
        params = {k: v for k, v in kwargs.items() if k != 'pretokens'}
        metadata = {'training_params': params}
        
        # Create and return a new tokenizer instance
        return cls(tokens, metadata)
