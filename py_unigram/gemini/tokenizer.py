import json
from typing import List, Tuple, Union

import regex as re


from .model import InternalModel


class GeminiUnigramTokenizer:
    def __init__(self, config: dict):
        self.metadata = config.get('metadata', {})
        self.vocab: dict[str, int] = config['vocab']
        self.scores: dict[int, float] = {int(k): v for k, v in config['scores'].items()}
        self.pre_tokenizer_regex: str = config['pre_tokenizer_regex']
        self.unk_token: str = config.get('unk_token', '<unk>')
        self.unk_id = self.vocab.get(self.unk_token)

        # Create reverse vocab for decoding
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pre_tokenizer = re.compile(self.pre_tokenizer_regex)

        self._model = InternalModel(
            vocab=self.vocab,
            scores=self.scores,
            unk_id=self.unk_id
        )

    def encode(self, text: str, return_tokens: bool = False) -> Union[List[int], Tuple[List[int], List[str]]]:
        """
        Encode text into token ids.

        Args:
            text: The text to encode
            return_tokens: If True, returns both ids and tokens. If False, returns only ids.

        Returns:
            If return_tokens is False, returns list of token ids.
            If return_tokens is True, returns tuple of (token ids, tokens).
        """
        tokens = []
        token_ids = []
        chunks = self.pre_tokenizer.findall(text)
        for chunk in chunks:
            pieces, ids, _ = self._model.encode_optimized(chunk)
            tokens.extend(pieces)
            token_ids.extend(ids)

        return (token_ids, tokens) if return_tokens else token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a sequence of token ids back into text.

        Args:
            token_ids: List of token ids to decode

        Returns:
            The decoded text string
        """
        return ''.join(self.id_to_token.get(id, self.unk_token) for id in token_ids)

    def save(self, path: str):
        """Save the tokenizer configuration to a JSON file."""
        config = {
            'metadata': self.metadata,
            'vocab': self.vocab,
            'scores': self.scores,
            'pre_tokenizer_regex': self.pre_tokenizer_regex,
            'unk_token': self.unk_token
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> 'GeminiUnigramTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return GeminiUnigramTokenizer(config)

    @classmethod
    def train(cls, **kwargs) -> 'GeminiUnigramTokenizer':
        return train_unigram_model(**kwargs)
