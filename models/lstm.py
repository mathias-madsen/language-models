import torch
import numpy as np
from tqdm import tqdm


class LSTMCharacterPredictor(torch.nn.Module):

    def __init__(self, encoding_size, recurrence_size, num_classes=256):

        super(LSTMCharacterPredictor, self).__init__()

        self.indim = encoding_size
        self.outdim = recurrence_size
        self.num_classes = num_classes

        self.output0 = torch.nn.Parameter(torch.rand(self.outdim))
        self.memory0 = torch.nn.Parameter(torch.rand(self.outdim))

        # This parameter holds the vectors we use to represent each
        # character in the character set. Indexing into this matrix
        # is equivalent to one-hot encoding the observed character
        # and then multiplying that one-hot vector by this matrix.
        self.encodings = torch.nn.Parameter(torch.rand(num_classes, self.indim))
        
        self.lstm = torch.nn.LSTM(self.indim,
                                  self.outdim,
                                  batch_first=False)

        self.decode = torch.nn.Linear(self.outdim, num_classes)

        # self.decode = torch.nn.Sequential(torch.nn.Linear(self.outdim, 300),
        #                                      torch.nn.ReLU(),
        #                                      torch.nn.Linear(300, 300),
        #                                      torch.nn.ReLU(),
        #                                      torch.nn.Linear(300, num_classes))

    def encode(self, idx):

        return self.encodings[idx.to(torch.int64), :]

    def forward(self, ints):

        _, bsize = ints.shape
        inputs = self.encode(ints)  # (seq_length, bsize, indim)
        output0s = torch.ones([1, bsize, self.outdim]) * self.output0
        memory0s = torch.ones([1, bsize, self.outdim]) * self.memory0
        outputs, _ = self.lstm(inputs, (output0s, memory0s))
        outputs_total = torch.cat([output0s, outputs[:-1,]], axis=0)

        return self.decode(outputs_total)

    def sample(self, length):

        input_t = None
        out_t = self.output0[None, :]
        mem_t = self.memory0[None, :]

        sentence = ""
        for _ in range(length):
            logits = self.decode(out_t)
            probs_torch = torch.softmax(logits, dim=-1)
            probs_numpy = probs_torch.detach().numpy().flatten()
            idx = np.random.choice(self.num_classes, p=probs_numpy)
            sentence += chr(idx)
            torch_int_vector = torch.as_tensor([idx], dtype=torch.int64)
            encoded_t = self.encode(torch_int_vector)
            _, (out_t, mem_t) = self.lstm(encoded_t, (out_t, mem_t))

        return sentence

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
                time_first_ints = batch.T  # so [seqlen, bsize]
                optimizer.zero_grad()
                logits = self(time_first_ints)
                logits_flat = logits.reshape([-1, self.num_classes])
                targets_flat = time_first_ints.reshape([-1])
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
                time_first_ints = batch.T  # so [seqlen, bsize]
                logits = self(time_first_ints)
                logits_flat = logits.reshape([-1, self.num_classes])
                targets_flat = time_first_ints.reshape([-1])
                meanloss = xent(logits_flat, targets_flat)
                vlosses.append(float(meanloss))
            valpair = np.mean(vlosses), np.std(vlosses)
            valhist.append(valpair)
            print("Mean validation loss: %.5f +/- %.5f" % valpair)
            print()

            print("Sample sequence:")
            print("----------------")
            print("%r" % self.sample(length=1000))
            print()

        return trainhist, valhist
