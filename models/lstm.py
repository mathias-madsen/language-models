import torch
import numpy as np
from tqdm import tqdm


class CharacterPredictor(torch.nn.Module):

    def __init__(self, encoding_size, recurrence_size, num_classes=256):

        super(CharacterPredictor, self).__init__()

        self.indim = encoding_size
        self.outdim = recurrence_size
        self.num_classes = num_classes

        self.output0 = torch.nn.Parameter(torch.rand(self.outdim))
        self.memory0 = torch.nn.Parameter(torch.rand(self.outdim))
        self.encodings = torch.nn.Parameter(torch.rand(num_classes, self.indim))

        self.lstm = torch.nn.LSTM(self.indim,
                                  self.outdim,
                                  batch_first=False)

        self.predictor = torch.nn.Sequential(torch.nn.Linear(self.outdim, 300),
                                             torch.nn.ReLU(),
                                             # torch.nn.Linear(300, 300),
                                             # torch.nn.ReLU(),
                                             torch.nn.Linear(300, num_classes))

    def forward(self, ints):

        _, bsize = ints.shape
        inputs = self.encodings[ints.to(torch.int64), :]  # (seq_length, bsize, indim)
        output0s = torch.ones([1, bsize, self.outdim]) * self.output0
        memory0s = torch.ones([1, bsize, self.outdim]) * self.memory0
        outputs, _ = self.lstm(inputs, (output0s, memory0s))
        outputs_total = torch.cat([output0s, outputs[:-1,]], axis=0)

        return self.predictor(outputs_total)

    def sample(self, length):

        input_t = None
        output_t = self.output0[None, :]
        memory_t = self.memory0[None, :]

        sentence = ""
        for _ in range(length):
            logits = self.predictor(output_t)
            probs_torch = torch.softmax(logits, dim=-1)
            probs_numpy = probs_torch.detach().numpy().flatten()
            idx = np.random.choice(self.num_classes, p=probs_numpy)
            sentence += chr(idx)
            input_t = self.encodings[idx, None, :]
            _, (output_t, memory_t) = self.lstm(input_t, (output_t, memory_t))

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
            batches = np.split(train, range(bsize, len(train), bsize))
            batches = torch.stack(batches[:-1], axis=0)
            tlosses = []
            for batch_first_ints in tqdm(batches[:max_train_steps]):
                time_first_ints = batch_first_ints.T  # lead with sequence axis, as per convention
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
            batches = np.split(val, range(bsize, len(val), bsize))
            batches = torch.stack(batches[:-1], axis=0)
            vlosses = []
            for batch_first_ints in tqdm(batches[:max_train_steps]):
                time_first_ints = batch_first_ints.T
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
