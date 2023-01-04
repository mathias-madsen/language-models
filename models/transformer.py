# import numpy as np
import torch
import numpy as np
from tqdm import tqdm


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



def add_positional_encodings(x, dmodel):
    """ Add time-identifying dimensions to a batch of vector sequences. """
    batch_size, seq_length, dim = x.shape
    timespan = torch.linspace(0, torch.pi, seq_length)
    # timespan = torch.range(0, seq_length) / 1000
    # exponents = torch.range(dim // 2) / dmodel
    # cosfreqs = 1000 ** (dmodel)
    cosines = torch.stack([torch.cos(timespan * 2**k)
                           for k in range(dim // 2)], dim=-1)
    sines = torch.stack([torch.sin(timespan * 2**k)
                         for k in range(dim // 2)], dim=-1)
    sinusoids = torch.concat([cosines, sines], dim=-1)
    newdims = torch.tile(sinusoids, [batch_size, 1, 1])
    return x + newdims


class TransformerDecoderLayer(torch.nn.Module):

    def __init__(self, dim, nheads, use_causal_masking=False, attn_dropout=0,
                 relu_dropout=0):

        super().__init__()
        self.relu_dropout = relu_dropout
        self.mha = MultiHeadedSelfAttentionLayer(dim,
                                                 nheads,
                                                 use_causal_masking,
                                                 attn_dropout)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.affine1 = torch.nn.Linear(dim, dim)
        self.affine2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        y, weights = self.mha(x)
        x = self.norm1(x + y)
        x = self.affine1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, self.relu_dropout, self.training)
        return self.norm2(x + self.affine2(x))


class TransformerDecoder(torch.nn.Module):

    def __init__(self, dim, nheads, use_causal_masking=False, attn_dropout=0,
                 relu_dropout=0, num_layers=12):

        super().__init__()
        self.layers = []
        for idx in range(num_layers):
            layer = TransformerDecoderLayer(dim,
                                            nheads,
                                            use_causal_masking,
                                            attn_dropout,
                                            relu_dropout)
            self.layers.append(layer)

    def forward(self, x):
        x = add_positional_encodings(x, self.dim)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoderCharacterPredictor(torch.nn.Module):

    def __init__(self, num_classes, dim, nheads, use_causal_masking=True,
                 attn_dropout=0, relu_dropout=0, num_layers=12,
                 charprobs=None):

        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.charprobs = charprobs
        self.character_encodings = torch.nn.Parameter(
            12**0.5 * torch.rand(num_classes, dim) - 0.5  # mean 0, var 1
            )
        self.readout_layer = torch.nn.Linear(dim, num_classes)
        self.layers = []
        for idx in range(num_layers):
            layer = TransformerDecoderLayer(dim,
                                            nheads,
                                            use_causal_masking,
                                            attn_dropout,
                                            relu_dropout)
            self.layers.append(layer)

    def forward(self, character_indices):
        idx = character_indices.type(torch.int64)
        x = self.character_encodings[idx, :]
        x = add_positional_encodings(x, self.dim)
        for layer in self.layers:
            x = layer(x)
        return self.readout_layer(x)

    def sample(self, length):

        if length < 1:
            return []

        idx0 = np.random.choice(self.num_classes, p=self.charprobs)
        seq = [idx0]

        while len(seq) < length:
            context = torch.tensor([seq])  # 1-sequence batch
            batch_time_logdists = self(context)
            last_logdist = batch_time_logdists[0, -1]  # first batch, last time
            last_dist = torch.softmax(last_logdist, dim=0)
            cond_char_probs = last_dist.detach().numpy()
            idx = np.random.choice(self.num_classes, p=cond_char_probs)
            seq.append(idx)

        return "".join(chr(idx) for idx in seq)

    def fit(self, train, val, num_epochs, bsize=1, lr=0.05,
                  max_train_steps=None, max_val_steps=None):
        """ Train on stacks of sequences, of shape [N, T]. """

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        xent = torch.nn.CrossEntropyLoss()
        trainhist = []
        valhist = []

        for epoch_idx in range(num_epochs):

            print("======== EPOCH NUMBER %s ========" % (epoch_idx + 1,))
            print("")

            print("Training pass . . .")
            train = train[np.random.permutation(len(train)),]
            tuple_of_batches = torch.split(train, bsize)
            batches = torch.stack(tuple_of_batches[:-1][:max_train_steps], axis=0)
            tlosses = []
            for batch in tqdm(batches, unit_scale=bsize, unit=" sequences"):
                optimizer.zero_grad()
                logits = self(batch)[:, :-1, :]  # [bsize, len - 1, nclasses]
                targets = batch[:, 1:]  # logits[:, 0] has seen token 0
                assert logits.shape[:2] == targets.shape
                logits_flat = logits.reshape([-1, self.num_classes])
                targets_flat = targets.reshape([-1])
                meanloss = xent(logits_flat, targets_flat)
                tlosses.append(float(meanloss))
                meanloss.backward()
                optimizer.step()
            trainpair = np.mean(tlosses), np.std(tlosses)
            trainhist.append(trainpair)
            print("Mean training loss: %.5f +/- %.5f" % trainpair)
            print()

            print("Validation pass . . .")
            # in case we don't use the whole validation set, we shuffle:
            val = val[np.random.permutation(len(val)),]
            tuple_of_batches = torch.split(val, bsize)
            batches = torch.stack(tuple_of_batches[:-1][:max_val_steps], axis=0)
            vlosses = []
            for batch in tqdm(batches, unit_scale=bsize, unit=" sequences"):
                logits = self(batch)
                logits_flat = logits.reshape([-1, self.num_classes])
                targets_flat = batch.reshape([-1])
                meanloss = xent(logits_flat, targets_flat)
                vlosses.append(float(meanloss))
            valpair = np.mean(vlosses), np.std(vlosses)
            valhist.append(valpair)
            print("Mean validation loss: %.5f +/- %.5f" % valpair)
            print()

            print("Sample sequence:")
            print("----------------")
            print("%r" % self.sample(length=300))
            print()

        return trainhist, valhist



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


def _test_transformer_decoder():

    dim = 6
    nheads = 3

    decoder = TransformerDecoder(dim=dim,
                                 nheads=nheads,
                                 use_causal_masking=False,
                                 attn_dropout=0,
                                 relu_dropout=0,
                                 num_layers=12)
    
    x = torch.rand(4, 10, dim)
    y = decoder(x)
