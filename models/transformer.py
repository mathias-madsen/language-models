# import numpy as np
import torch


class MultiHeadedSelfAttentionLayer(torch.nn.Module):
  
    def __init__(self, dim, nheads, use_causal_masking=False, attn_dropout=0):

        super().__init__()
        self.dim = dim
        self.nheads = nheads
        self.use_causal_masking = use_causal_masking
        self.attn_dropout = attn_dropout
        if dim % nheads != 0:
            raise ValueError("nheads=%s does not divide dim=%s" % (nheads, dim))

        self.indense = torch.nn.Linear(dim, 3 * dim)
        self.outdense = torch.nn.Linear(dim, dim)

    def forward(self, x):

        # collect necessary shape information:
        bsize, length, _ = x.shape

        if self.use_causal_masking:
            mask = torch.tril(torch.ones(length, length))
            logmask = torch.log(mask)
        else:
            logmask = torch.zeros(length, length)

        if self.training and self.attn_dropout > 0:
            keep = torch.rand(length, length) >= self.attn_dropout
            logmask += torch.log(keep.to(logmask.dtype))

        # apply the initial affine function, which triples the size:
        qkv = self.indense(x).reshape([bsize, length, 3, self.dim]).moveaxis(2, 0)

        # split the information up into separate streams:
        qkvs = qkv.reshape([3, bsize, length, self.nheads, self.dim // self.nheads])
        qs, ks, vs = qkvs.moveaxis(3, 2)  # each [B, H, L, D // H]

        # carry out computation within each stream:
        dots = qs @ ks.swapaxes(-2, -1)  # [B, H, L, L], H attention mats
        scaled_dots = dots / qs.shape[-1]**0.5  # /= sqrt(small dim)
        weights = torch.softmax(scaled_dots + logmask, axis=-1)  # [B, H, L, L]
        separate_averages = (weights @ vs)  # [B, H, L, D // H]

        # move concatenate the results back into a single vector:
        stacked_aves = separate_averages.moveaxis(1, 2)
        concat_aves = stacked_aves.reshape([bsize, length, self.dim])
        
        return self.outdense(concat_aves), weights


def _test_multi_headed_self_attention_layer():

    num_heads = 2
    embed_dim = 6
    x = torch.rand(4, 5, embed_dim)

    their_layer = torch.nn.MultiheadAttention(
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                kdim=embed_dim,
                                vdim=embed_dim,
                                batch_first=True,  # NOTE! not the default
                                )

    my_layer = MultiHeadedSelfAttentionLayer(embed_dim,
                                             num_heads,
                                             use_causal_masking=False,
                                             attn_dropout=0.5)

    # transfer the parameters of the out-affine:
    my_layer.outdense.load_state_dict(their_layer.out_proj.state_dict())

    # transfer the parameters of the in-affine:
    state_dict = my_layer.state_dict()
    state_dict["indense.weight"] = their_layer.in_proj_weight.detach()
    state_dict["indense.bias"] = their_layer.in_proj_bias.detach()
    my_layer.load_state_dict(state_dict)
    
    my_layer.train(False)  # switch to evaluation mode

    attn_vals, attn_weights = their_layer(x, x, x, average_attn_weights=False)
    aves, weights = my_layer(x)

    assert attn_weights.shape == weights.shape
    assert attn_vals.shape == aves.shape

    assert torch.allclose(attn_weights, weights)
    assert torch.allclose(attn_vals, aves)
