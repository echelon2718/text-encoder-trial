import torch
import re
from typing import Literal, Optional, List, Dict

class PhonemeTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"
    WORD_BOUNDARY = "|"

    def __init__(self, vocab: Optional[List[str]] = None):
        self.symbol2id: Dict[str, int] = {}
        self.id2symbol: Dict[int, str] = {}
        self._init_special()
        if vocab is not None:
            self.add_symbols(vocab)
    
    def _init_special(self) -> None:
        self.symbol2id = {self.PAD: 0, self.UNK: 1, self.WORD_BOUNDARY: 2}
        self.id2symbol = {0: self.PAD, 1: self.UNK, 2: self.WORD_BOUNDARY}
    
    def add_symbols(self, symbols: List[str]) -> None:
        for s in sorted(set(symbols)):
            if s not in self.symbol2id:
                idx = len(self.symbol2id)
                self.symbol2id[s] = idx
                self.id2symbol[idx] = s
    
    @classmethod
    def from_corpus(cls, phoneme_strings: List[str]) -> "PhonemeTokenizer":
        tok = cls()
        all_symbols: set = set()
        for s in phoneme_strings:
            all_symbols.update(cls._split(s))
        tok.add_symbols(sorted(all_symbols))
        return tok
    
    @staticmethod
    def _split(phoneme_str: str) -> List[str]:
        symbols: List[str] = []
        words = phoneme_str.split("|")
        for i, word in enumerate(words):
            symbols.extend(word.strip().split())
            if i < len(words) - 1:              # ← sisipkan "|" antar kata, bukan di akhir
                symbols.append("|")
        return symbols
    
    @property
    def vocab_size(self) -> int:
        return len(self.symbol2id)
    
    def encode_single(self, phoneme_str: str) -> torch.Tensor:
        symbols = self._split(phoneme_str)
        unk_id = self.symbol2id[self.UNK]
        ids = [self.symbol2id.get(s, unk_id) for s in symbols]
        return torch.tensor(ids, dtype=torch.long)
    
    def decode_single(self, ids: torch.Tensor) -> str:
        return " ".join(self.id2symbol.get(int(i), self.UNK) for i in ids.tolist())
