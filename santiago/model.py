import torch
import torch.nn as nn
import torch.nn.functional as F

# Language Model
class LanguageModel(nn.Module):
    def __init__(self, tokenizer, dim_size, dim_feedforward, num_layers, num_heads, dropout, max_len, device):
        super(LanguageModel, self).__init__()

        self.device = device
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.sep_id = tokenizer.sep_id
        self.eos_id = tokenizer.eos_id
        self.max_len = max_len
        self.min_summary_length = tokenizer.min_summary_length
        # Architecture Config
        activation = nn.ReLU()
        layer_norm = nn.LayerNorm(dim_size)
        layer_norm_eps = 1e-5
        transformer_decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,   
        )
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, dim_size, padding_idx=self.pad_id)
        # Positional encoding
        self.positional_encoding = nn.Embedding(max_len, dim_size)        
        # Transformer decoder
        self.transformer_decoder = nn.TransformerEncoder(
            transformer_decoder_layer,
            num_layers=num_layers,
            norm=layer_norm,
        )
        # Output layer
        self.output_layers = nn.Linear(dim_size, self.vocab_size)

        # causal mask in buffer
        full = torch.triu(torch.ones(max_len, max_len, device=device) *
                          float("-inf"), diagonal=1)
        self.register_buffer("_causal_full", full, persistent=False)

    def causal_mask(self, seq_len):
        return self._causal_full[:seq_len, :seq_len]

    def forward(self, src):
        token_embs = self.embedding(src)
        batch_size, seq_len = src.size()
        positions = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embs = self.positional_encoding(positions)
        mask = self.causal_mask(src.size(1))
        x = token_embs + pos_embs
        x = self.transformer_decoder(x, mask=mask)
        x = self.output_layers(x)
        return x

    @torch.no_grad()
    def generate(self, diff_text):
        # Set model to evaluation mode
        self.eval()

        # Add BOS and SEP tokens
        if not diff_text.lstrip().startswith("<bos>"):
            diff_text = "<bos> " + diff_text
        if "<sep>" not in diff_text:
            diff_text = diff_text + " <sep> "

        # Encode 
        # If tokenizer has subword tokenizer, use it, otherwise use tokenize_text
        ids = (self.tokenizer.sp.encode(diff_text, out_type=int)
            if hasattr(self.tokenizer, "sp")
            else self.tokenizer.tokenize_text(diff_text, training=False)[0])

        # Guarantee SEP inside the window
        if self.sep_id not in ids:
            ids.append(self.sep_id)
        ids = ids[-(self.max_len - 1):]
        seq = torch.tensor(ids, device=self.device)

        # Predict one token at a time
        while len(seq) < self.max_len:
            next_id = self(seq.unsqueeze(0))[0, -1].argmax(-1)
            seq = torch.cat([seq, next_id.unsqueeze(0)])
            if next_id.item() == self.eos_id:
                break

        # Remove summary
        sep_mask = (seq == self.sep_id).nonzero(as_tuple=True)
        if len(sep_mask[0]) == 0:
            return ""
        start = sep_mask[0][0] + 1
        end = (seq == self.eos_id).nonzero(as_tuple=True)[0][0] if (seq == self.eos_id).any() else len(seq)
        ids = seq[start:end].tolist()

        # Decode into actual text
        if hasattr(self.tokenizer, "sp"):
            return self.tokenizer.sp.decode(ids)
        return " ".join(self.tokenizer.decode(i) for i in ids)
