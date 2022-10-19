from turtle import back
import numpy as np
from tqdm import tqdm

from models.baselines import MonogramModel, DigramModel


class HiddenMarkovModel:

    def __init__(self, nstates, nchars=256):

        self._trans = np.random.dirichlet(np.ones(nstates), size=nstates)
        self._emits = np.random.dirichlet(np.ones(nchars), size=nstates)

        self.initial = self.compute_initial_distribution()
        self.monochar = self.compute_stationary_emission_dist()
        self.dichar = self.compute_stationary_emission_digram_dist()

        self.nstates = nstates
        self.nchars = nchars

    @property
    def emits(self):
        return self._emits

    @emits.setter
    def emits(self, value):
        assert value.shape == self._emits.shape  # no broadcasting
        self._emits[:, :] = value / value.sum(axis=1, keepdims=True)
        self.monochar = self.compute_stationary_emission_dist()
        self.dichar = self.compute_stationary_emission_digram_dist()

    @property
    def trans(self):
        return self._trans

    @trans.setter
    def trans(self, value):
        assert value.shape == self._trans.shape  # no broadcasting
        self._trans[:, :] = value / value.sum(axis=1, keepdims=True)
        self.initial = self.compute_initial_distribution()
        self.monochar = self.compute_stationary_emission_dist()
        self.dichar = self.compute_stationary_emission_digram_dist()

    def compute_initial_distribution(self):

        vals, vecs = np.linalg.eig(self.trans.T)
        isone = np.isclose(vals, 1)
        assert np.any(isone)
        
        idx = np.argmax(isone)
        col = vecs[:, idx].real
        dist = col / col.sum()
        assert np.allclose(dist, dist @ self.trans)

        return dist
    
    def compute_stationary_emission_dist(self):

        joint = self.initial[:, None] * self.emits
        marginal = np.sum(joint, axis=0)

        return marginal / marginal.sum()
    
    def compute_stationary_emission_digram_dist(self):

        hidden_pair_prior = self.initial[:, None] * self.trans
        emit_likes = (self.emits[:, None, :, None] *
                      self.emits[None, :, None, :])
        joint = hidden_pair_prior[:, :, None, None] * emit_likes

        return np.sum(joint, axis=(0, 1))

    def prior_given_past(self, idx):
        """ The hidden-state distributions given the strict past.
        
        Parameters:
        -----------
        idx : sequence of T integers in {0, 1, ..., nchars - 1}
            A list, tuple, or array of observed emissions.
        
        Returns:
        --------
        forward : float array of shape [T, nstates]
            For each t, the number forward[t, :] contains the conditional
            state distribution given all observations strictly before but
            not including time t.
        """

        dists = np.zeros([len(idx), self.nstates])
        current_dist = self.compute_initial_distribution()
        emission_likelihoods = self.emits[:, idx].T

        for steps_since_start, elike in enumerate(emission_likelihoods):
            dists[steps_since_start, :] = current_dist
            current_dist *= elike
            current_dist /= current_dist.sum()
            current_dist = current_dist @ self.trans
        
        return dists

    def likelihood_of_future(self, idx):
        """ The hidden-state likelihoods given the present and future.
        
        Parameters:
        -----------
        idx : sequence of T integers in {0, 1, ..., nchars - 1}
            A list, tuple, or array of observed emissions.
        
        Returns:
        --------
        backward : float array of shape [T, nstates]
            For each t, the number forward[t, :] contains the relative
            likelihoods of the observations starting from and including
            time t, given each possible hidden state. The likelihoods
            are relative in the sense that they are scaled by a shared
            constant (so that, by arbitrary convention, they sum to 1).
        """

        dists = np.zeros([len(idx), self.nstates])
        current_dist = np.ones(self.nstates) / self.nstates
        emission_likelihoods = self.emits[:, idx].T

        for steps_from_last, elike in enumerate(emission_likelihoods[::-1]):
            current_dist *= elike
            current_dist /= current_dist.sum()  # hence the 'relative'
            dists[len(idx) - 1 - steps_from_last] = current_dist
            current_dist = current_dist @ self.trans.T

        return dists
    
    def sample_hidden_states(self, length):
        """ Sample a sequence of hidden states (an integer arrray). """

        sequence = []
        hdist = self.compute_initial_distribution()

        for _ in range(length):
            state = np.random.choice(self.nstates, p=hdist)
            sequence.append(state)
            hdist = self.trans[state, :]

        return np.int64(sequence)
    
    def sample_emissions(self, hidden_states):
        """ Sample sequence of emissions given sequence of hidden states. """

        dists = self.emits[hidden_states, :]
        dists = 1e-5*self.monochar + (1 - 1e-5)*dists  # avoid all-zero dists

        return np.int64([np.random.choice(self.nchars, p=p) for p in dists])

    def sample(self, length):
        """ Sample a sequence of emissions as an integer array. """

        return self.sample_emissions(self.sample_hidden_states(length))

    def nlogps(self, idx):
        """ Compute the negative log-probabilities of an observed sequence. """

        emission_dists = self.prior_given_past(idx) @ self.emits
        emission_probs = emission_dists[range(len(idx)), idx]

        return -np.log(emission_probs)

    def sum_trans_joints(self, priors, likes):

        joints = priors[:, :, None] * likes[:, None, :] * self.trans
        joints /= np.sum(joints, axis=(1, 2), keepdims=True)

        return np.sum(joints, axis=0)

    def sum_emits_joints(self, priors, likes, idx):

        post = priors * likes  # hidden-state posteriors given all evidence
        post /= np.sum(post, axis=1, keepdims=True)

        emits = np.zeros([self.nstates, self.nchars])
        for hdist, emission in zip(post, idx):
            emits[:, emission] += hdist

        return emits


def _test_that_hidden_markov_model_finds_the_right_stationary_distribution():

    model = HiddenMarkovModel(2, 1)

    dist = np.random.dirichlet(np.ones(2))
    model.trans = np.stack([dist, dist[::-1]], axis=1)  # symmetric
    assert np.allclose(model.initial, (0.5, 0.5))

    hsamples = [model.sample_hidden_states(1) for _ in range(1000)]
    assert np.isclose(np.mean(hsamples), 0.5, atol=0.1)


def _test_that_hidden_markov_model_samples_correct_hidden_sequences():

    model = HiddenMarkovModel(2, 1)

    model.trans = np.array([[0., 1.], [1., 0.]])  # 'always flip'
    assert np.allclose(model.initial, (0.5, 0.5))

    hsample = model.sample_hidden_states(10)
    assert (list(hsample) == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] or
            list(hsample) == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    hsamples = [model.sample_hidden_states(1) for _ in range(1000)]
    assert np.isclose(np.mean(hsamples), 0.5, atol=0.1)


def _test_hidden_markov_model_prior_and_likelihood_in_explicit_cases():

    model = HiddenMarkovModel(2, 3)

    model.trans[:, :] = (0, 1), (1, 0)
    model.emits[:, :] = (0.5, 0.5, 0), (0, 0.5, 0.5)
    
    all_ones = np.int64([1, 1, 1, 1, 1])
    forward_ones = model.prior_given_past(all_ones)
    assert np.allclose(forward_ones, 0.5)
    backward_ones = model.likelihood_of_future(all_ones)
    assert np.allclose(backward_ones, 0.5)

    revealed_end = np.int64([1, 1, 1, 1, 2])
    forward_end = model.prior_given_past(revealed_end)
    assert np.allclose(forward_end, 0.5)  # never revealed in time
    backward_end = model.likelihood_of_future(revealed_end)
    assert np.allclose(backward_end[0::2], (0, 1))  # known throughout
    assert np.allclose(backward_end[1::2], (1, 0))  # known throughout

    revealed_start = np.int64([2, 1, 1, 1, 1])
    forward_start = model.prior_given_past(revealed_start)
    assert np.allclose(forward_start[0], (0.5, 0.5))  # no known past at 0
    assert np.allclose(forward_start[1::2], (1, 0))  # but now there is
    assert np.allclose(forward_start[2::2], (0, 1))  # 
    backward_start = model.likelihood_of_future(revealed_start)
    assert np.allclose(backward_start[1:], (0.5, 0.5))  # no evidence yet
    assert np.allclose(backward_start[0], (0, 1))  # but now there is


def _test_that_stationary_character_probabilities_are_correct():

    model = HiddenMarkovModel(3, 2)
    sample = model.sample(length=1000)
    counts, _ = np.histogram(sample, range(model.nchars + 1))
    freqs = counts / np.sum(counts)

    assert np.sum(counts) == len(sample)
    assert np.allclose(model.monochar, counts / len(sample), atol=0.1)


def _test_that_stationary_character_digram_probabilities_are_correct():

    model = HiddenMarkovModel(3, 2)
    sample = model.sample(length=1000)
    sample_pairs = np.stack([sample[:-1], sample[1:]], axis=1)

    empirical = np.zeros([model.nchars, model.nchars])
    pairs, counts = np.unique(sample_pairs, axis=0, return_counts=True)
    for (i, j), k in zip(pairs, counts):
        empirical[i, j] += k

    theoretical = model.compute_stationary_emission_digram_dist()
    
    assert np.sum(empirical) == len(sample) - 1
    assert np.allclose(theoretical, empirical / empirical.sum(), atol=0.1)


if __name__ == "__main__":

    _test_that_hidden_markov_model_finds_the_right_stationary_distribution()
    _test_that_hidden_markov_model_samples_correct_hidden_sequences()
    _test_hidden_markov_model_prior_and_likelihood_in_explicit_cases()
    _test_that_stationary_character_probabilities_are_correct()
    _test_that_stationary_character_digram_probabilities_are_correct()
