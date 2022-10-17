import numpy as np


class MonogramModel:

    def __init__(self, num_obs=0.5):

        self.num_obs = num_obs
        self.charprobs = np.ones([256])
        self.charprobs /= self.charprobs.sum()

    def fit(self, train, val, weight_prior=0.5):

        tprobs = weight_prior * self.num_obs * self.charprobs
        tidx, tcounts = np.unique(train, return_counts=True)
        tprobs[tidx,] += tcounts
        tprobs /= tprobs.sum()

        vprobs = weight_prior * self.num_obs * self.charprobs
        vidx, vcounts = np.unique(val, return_counts=True)
        vprobs[vidx,] += vcounts
        vprobs /= vprobs.sum()

        # store the updated parameter:
        self.num_obs += len(train)
        self.charprobs = tprobs

        mean_train_loss = np.sum(-tprobs*np.log(tprobs))
        mean_train_loss_squared = np.sum(tprobs*np.log(tprobs) ** 2)
        var_train_loss = mean_train_loss_squared - mean_train_loss ** 2
        train_stats = mean_train_loss, np.sqrt(var_train_loss)

        mean_val_loss = np.sum(-vprobs*np.log(tprobs))
        mean_val_loss_squared = np.sum(vprobs*np.log(tprobs) ** 2)
        var_val_loss = mean_val_loss_squared - mean_val_loss ** 2
        val_stats = mean_val_loss, np.sqrt(var_val_loss)

        return train_stats, val_stats

    def sample(self, length):

        assert self.charprobs is not None

        indices = np.random.choice(self.charprobs.size,
                                   p=self.charprobs,
                                   size=length)

        return "".join(chr(int(i)) for i in indices)


class DigramModel:

    def __init__(self, num_obs=0.5):

        self.num_obs = num_obs
        self.joints = np.ones([256, 256])

    def fit(self, train, val, weight_prior=0.5):

        tprobs = weight_prior * self.num_obs * self.joints
        tpairs = np.stack([train[:-1], train[1:]], axis=1)
        tidx, tcounts = np.unique(tpairs, axis=0, return_counts=True)
        tprobs[tidx[:, 0], tidx[:, 1]] += tcounts.astype(tprobs.dtype)
        tprobs /= tprobs.sum()

        vprobs = weight_prior * self.num_obs * self.joints
        vpairs = np.stack([val[:-1], val[1:]], axis=1)
        vidx, vcounts = np.unique(vpairs, axis=0, return_counts=True)
        vprobs[vidx[:, 0], vidx[:, 1]] += vcounts.astype(vprobs.dtype)
        vprobs /= vprobs.sum()

        # store the updated parameter:
        self.num_obs += len(train)
        self.joints = tprobs

        tcond = tprobs / np.sum(tprobs, axis=1, keepdims=True)
        mean_train_loss = np.sum(-tprobs*np.log(tcond))
        mean_train_loss_squared = np.sum(tprobs*np.log(tcond) ** 2)
        var_train_loss = mean_train_loss_squared - mean_train_loss ** 2
        train_stats = mean_train_loss, np.sqrt(var_train_loss)

        mean_val_loss = np.sum(-vprobs*np.log(tcond))
        mean_val_loss_squared = np.sum(vprobs*np.log(tcond) ** 2)
        var_val_loss = mean_val_loss_squared - mean_val_loss ** 2
        val_stats = mean_val_loss, np.sqrt(var_val_loss)

        return train_stats, val_stats

    def sample(self, length):

        if length == 0:
            return ""
        
        p = self.joints.sum(axis=1)
        idx = np.random.choice(p.size, p=p)
        string = chr(idx)
        for _ in range(length - 1):
            marginals = self.joints[idx, :]
            conditionals = marginals / marginals.sum()
            idx = np.random.choice(conditionals.size, p=conditionals)
            string += chr(idx)

        return string


if __name__ == "__main__":

    string = """
                Whoever has been blessed with the advantages of a religious education, and
                recurs to his own years of juvenile susceptibility, cannot forget the
                strong impressions he received by these means; and must have had frequent
                occasion to remark the tenaciousness with which they have lingered in his
                memory, and sprung up amidst his recollections at every subsequent period.
                In many cases they have proved the basis, of future eminence in piety,
                and blended delightfully with the gladdening retrospections of declining
                life. In those instances, where all the good effects which might be
                anticipated did not appear, these early lessons have checked the
                impetuousity of passion, neutralized the force of temptation, and
                cherished the convictions of an incipient piety.
            """

    string = " ".join(line.strip() for line in string.split("\n"))
    sequence = [ord(c) for c in string]

    splitpoint = len(sequence) // 5
    train = sequence[:splitpoint]
    val = sequence[splitpoint:]
    
    m1 = MonogramModel()
    print(m1.fit(train, val))
    print(m1.sample(100))
    print()

    m2 = MonogramModel()
    print(m2.fit(train, val))
    print(m2.sample(100))
    print()