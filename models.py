import torch
from torch import nn
import sys
sys.path.append('attention_is_all_you_need_pytorch')  # fix imports inside submodule
from transformer.Layers import EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads,
                 pf_dim, dropout, device, max_length=100):
        super().__init__()

        self.device = device
        assert hid_dim % n_heads == 0

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(d_model=hid_dim,
                                                  n_head=n_heads,
                                                  d_inner=pf_dim,
                                                  d_k=hid_dim//n_heads,
                                                  d_v=hid_dim//n_heads,
                                                  dropout=dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src, enc_slf_attn = layer(src, src_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers,
                 n_heads, pf_dim, dropout, device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(d_model=hid_dim,
                                                  n_head=n_heads,
                                                  d_inner=pf_dim,
                                                  d_k=hid_dim//n_heads,
                                                  d_v=hid_dim//n_heads,
                                                  dropout=dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, _, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.apply(self.initialize_weights)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention

    @staticmethod
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)


def build_model(args, src_lang, trg_lang, input_dim, output_dim, device):
    enc = Encoder(input_dim, args.hidden_size, args.n_layers,
                  args.n_heads, args.pf_dim, args.dropout,
                  device)

    dec = Decoder(output_dim, args.hidden_size, args.n_layers,
                  args.n_heads, args.pf_dim, args.dropout,
                  device)

    src_pad_idx = src_lang.vocab.stoi[src_lang.pad_token]
    trg_pad_idx = trg_lang.vocab.stoi[trg_lang.pad_token]

    model = Seq2Seq(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)

    return model
