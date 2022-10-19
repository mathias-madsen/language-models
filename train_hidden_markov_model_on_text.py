import os
import torch
import numpy as np
from tqdm import tqdm

from reader import load_corpus, convert_to_snippets
from models.hidden_markov_model import HiddenMarkovModel
from models.baselines import MonogramModel, DigramModel


num_classes = 256  # alphabet size
seq_length = 100  # length of the snippets to be predicted
num_hidden_states = 500  # number of hidden states in the HMM

old_weight = 10.0
num_epochs = 100
num_train_steps = 300  # `None` means unlimited
num_val_steps = 300  # `None` means unlimited


# load the data:
long_int_array = load_corpus("texts")
snippets = convert_to_snippets(long_int_array, seq_length, shuffle=True)

# split it into a test and a train set:
val_size = int(0.2 * len(snippets))
val, train = np.split(snippets, [val_size], axis=0)

print("Train shape: %s" % (train.shape,))
print("Val shape: %s" % (val.shape,))
print()

hmm = HiddenMarkovModel(num_hidden_states, num_classes)

trainhist, valhist = hmm.fit(train=train,
                             val=val,
                             num_epochs=num_epochs,
                             old_weight=old_weight,
                             num_train_steps=num_train_steps,
                             num_val_steps=num_val_steps)
