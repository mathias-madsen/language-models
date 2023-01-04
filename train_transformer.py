import os
import torch
import numpy as np
from tqdm import tqdm

from reader import load_corpus, convert_to_snippets
from models.transformer import TransformerDecoderCharacterPredictor
from models.baselines import MonogramModel, DigramModel


num_classes = 256  # alphabet size
seq_length = 200  # length of the snippets to be predicted
num_heads = 8  # dim of the cell state and hidden state
dims_per_head = 64
encoding_size = num_heads * dims_per_head

batch_size = 16
learning_rate = 0.05
num_epochs = 100
max_train_steps = 1000  # `None` means unlimited
max_val_steps = 100  # `None` means unlimited


# load the data:
long_int_array = load_corpus("texts")
snippets_numpy = convert_to_snippets(long_int_array, seq_length, shuffle=True)
snippets = torch.as_tensor(snippets_numpy)

# split it into a test and a train set:
val_size = int(0.2 * len(snippets))
val, train = np.split(snippets, [val_size], axis=0)

print("Train shape: %s" % (train.shape,))
print("Val shape: %s" % (val.shape,))
print()


tseq = train.flatten()
vseq = val.flatten()

print("Monogram baseline:")
print("------------------")
monogram_model = MonogramModel()
(tmean, tstd), (vmean, vstd) = monogram_model.fit(tseq, vseq)
print("Training loss: %.5f +/- %.5f" % (tmean, tstd))
print("Validation loss: %.5f +/- %.5f" % (vmean, vstd))
print()
print("Sample:")
print("-------")
print(repr(monogram_model.sample(length=1000)))
print()

# print("Digram baseline:")
# print("----------------")
# digram_model = DigramModel()
# (tmean, tstd), (vmean, vstd) = digram_model.fit(tseq, vseq)
# print("Training loss: %.5f +/- %.5f" % (tmean, tstd))
# print("Validation loss: %.5f +/- %.5f" % (vmean, vstd))
# print()
# print("Sample:")
# print("-------")
# print(repr(digram_model.sample(length=1000)))
# print()


character_predictor = TransformerDecoderCharacterPredictor(
                            num_classes=num_classes,
                            dim=encoding_size,
                            nheads=num_heads,
                            num_layers=5,
                            charprobs=monogram_model.charprobs,
                            )

history = character_predictor.fit(
                train,
                val,
                num_epochs=1000,
                bsize=batch_size,
                lr=learning_rate,
                max_train_steps=max_train_steps,
                max_val_steps=max_val_steps,
                )
