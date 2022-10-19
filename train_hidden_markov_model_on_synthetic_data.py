import numpy as np
from models.hidden_markov_model import HiddenMarkovModel


if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    # FIX_TRANS = True
    FIX_TRANS = False

    # FIX_EMITS = True
    FIX_EMITS = False

    hdim = 5
    edim = 3

    true_model = HiddenMarkovModel(hdim, edim)
    approx_model = HiddenMarkovModel(hdim, edim)

    if FIX_TRANS:
        approx_model.trans = true_model.trans
    
    if FIX_EMITS:
        approx_model = true_model.emits

    for step_idx in range(30):
        
        x = true_model.sample(10000)

        true_loss = np.mean(true_model.nlogps(x))
        approx_loss = np.mean(approx_model.nlogps(x))
        print("True vs approx loss: %.5f vs %.5f (excess %.5f)"
              % (true_loss, approx_loss, approx_loss - true_loss))

        forward = approx_model.prior_given_past(x)
        backward = approx_model.likelihood_of_future(x)
        
        sum_trans = approx_model.sum_trans_joints(forward, backward)
        sum_emits = approx_model.sum_emits_joints(forward, backward, x)

        if not FIX_TRANS:
            # model.trans = 10.0*model.trans + sum_trans
            approx_model.trans = 0.3 * approx_model.trans + 0.7 * sum_trans / (len(x) - 1)
        
        if not FIX_EMITS:
            # model.emits = 10.0*model.emits + sum_emits
            approx_model.emits = 0.3 * approx_model.emits + 0.7 * sum_emits / len(x)

    print()

    print("Emission frequencies:")
    true_mono = true_model.compute_stationary_emission_dist()
    approx_mono = true_model.compute_stationary_emission_dist()
    for f1, f2 in zip(true_mono, approx_mono):
        print("%.8f, %.8f" % (f1, f2))
    print()

    print("State frequencies:")
    for f1, f2 in zip(sorted(true_model.initial), sorted(approx_model.initial)):
        print("%.8f, %.8f" % (f1, f2))
    print()
