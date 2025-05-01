import sentencepiece as spm
from pathlib import Path
import itertools, os, tempfile
from collections import Counter
import torch
import pandas as pd


class BaseTokenizer:
    def get_sequence_statistics(self, sequence_lengths):
        # Get len statistics
        sequence_lengths = pd.Series(sequence_lengths)
        sequence_lengths_stats = sequence_lengths.describe().to_dict()
        return sequence_lengths_stats

    def training_preprocessing(self, tokens, sep_id, pad_id, min_summary_length, max_len):
        # Drop if summary part is too short
        sep_index = [i for i,a in enumerate(tokens) if a == sep_id]
        if not sep_index:
            return "", 0
        if sep_index[0] + min_summary_length > max_len:
            return "", 0
        # Normalize the rest of the tokens
        tokens = tokens[:max_len]
        tokens += [pad_id] * (max_len - len(tokens))
        seq_len = len(tokens)
        return tokens, seq_len


class SpaceTokenizer(BaseTokenizer):
    def __init__(self, sequence_raw, vocab_size=16000, max_len=1024, min_summary_length=64):
        # Tokenizing by space
        # NOTE: If tokenization is trained use only the train set
        train_tokens = [token for text in sequence_raw for token in text.split()]
        train_tokens = [token.strip() for token in train_tokens]
        tokens = train_tokens + ["<pad>", "<unk>"]

        # building vocabulary with counter
        counter = Counter(tokens)
        special_tokens = ["<pad>", "<unk>", "<bos>", "<sep>", "<eos>"]
        most_common = [tok for tok, _ in counter.most_common(vocab_size)]
        vocab = special_tokens + most_common
        self.token_to_idx = {tok: i for i, tok in enumerate(vocab)}
        self.idx_to_token = {i: tok for tok, i in self.token_to_idx.items()}
        self.vocab_size = len(vocab) 
        self.pad_id = self.token_to_idx["<pad>"]
        self.max_len = max_len
        self.min_summary_length = min_summary_length
        self.sep_id = self.token_to_idx["<sep>"]
        self.bos_id = self.token_to_idx["<bos>"]
        self.eos_id = self.token_to_idx["<eos>"]


    def tokenize_text(self, text, training):
        tokens = text.split()
        tokens = [token.strip() for token in tokens]
        tokens = [tok if tok in self.token_to_idx else "<unk>" for tok in tokens]
        seq_len = None
        tokens = [self.token_to_idx[tok] for tok in tokens]
        if training:
            tokens, seq_len = self.training_preprocessing(tokens, self.sep_id, self.pad_id, self.min_summary_length, self.max_len)
        return tokens, seq_len

    def tokenize(self, sequence_raw, training=False):
        sequence_lengths = []
        sequence = []
        # Processing the data
        for _, text in enumerate(sequence_raw):
            tokens, seq_len = self.tokenize_text(text, training)
            if tokens != "":
                sequence.append(tokens)
                sequence_lengths.append(seq_len)
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence, sequence_lengths

    def decode(self, idx):
        return self.idx_to_token[idx]



class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, sequence_raw, vocab_size=16000, max_len=1024, min_summary_length=64):
        SP_MODEL = f"spm_{vocab_size}.model"
        SPECIAL   = ["<pad>", "<bos>", "<sep>", "<eos>"]

        if not Path(SP_MODEL).exists():
            print("Training SentencePiece model …")
            # write a temporary corpus file one sentence per line
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
                for line in sequence_raw:          
                    # light clean-up
                    l = line.replace("<", " <").replace(">", "> ").strip()
                    tmp.write(l + "\n")
                corpus_path = tmp.name

            spm.SentencePieceTrainer.train(
                input = corpus_path,
                model_prefix = f"spm_{vocab_size}",
                vocab_size = vocab_size - len(SPECIAL),
                pad_id = 0,
                unk_id = 1,
                bos_id = -1, 
                eos_id=-1,
                user_defined_symbols = SPECIAL
            )
            os.remove(corpus_path)

        self.sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
        self.max_len = max_len
        self.min_summary_length = min_summary_length
        self.pad_id = self.sp.piece_to_id("<pad>")
        self.bos_id = self.sp.piece_to_id("<bos>")
        self.sep_id = self.sp.piece_to_id("<sep>")
        self.eos_id = self.sp.piece_to_id("<eos>")
        self.vocab_size = self.sp.get_piece_size()


    def tokenize_text(self, text, training):
        """token → list[int]  (optionally prepends BOS)"""
        tokens = self.sp.encode(text, out_type=int)
        seq_len = None
        if training:
            tokens, seq_len = self.training_preprocessing(tokens, self.sep_id, self.pad_id, self.min_summary_length, self.max_len)
        return tokens, seq_len

    def decode(self, idx):
        return self.sp.id_to_piece(idx)

    def tokenize(self, sequence_raw, training=False):
        """
        Returns:
            tokens (List[int]) padded/truncated to max_len
            token_length (int)  – actual #tokens BEFORE padding
            drop rows whose summary is too short
        """
        sequence = []
        sequence_lengths = []
        for _, text in enumerate(sequence_raw):
            tokens, seq_len = self.tokenize_text(text, training=training)
            if tokens != "":
                sequence.append(tokens)
                sequence_lengths.append(seq_len)
        sequence = torch.tensor(sequence, dtype=torch.long)
        if training:
            return sequence, sequence_lengths
        else:
            return sequence
    


tokenizers_dict = {
    "space":      SpaceTokenizer,
    "sentencepiece": SentencePieceTokenizer,
}


