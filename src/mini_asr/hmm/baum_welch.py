import numpy as np

from mini_asr.hmm.hmm import HMM
from mini_asr.hmm.algorithms import _validate_observations
from mini_asr.hmm.posterior import state_posteriors, transition_posteriors
from mini_asr.hmm.forward import forward_log

class BaumWelch:
    """
    Baum-Welch trainer for a discrete-emission HMM
    """
    def __init__(
            self,
            max_iterations: int=50,
            tolerance: float=1e-4
    )->None:
        if max_iterations<=0:
            raise ValueError("max_iterations must be positive")
        if tolerance<=0:
            raise ValueError("tolerance must be positive")

        self.max_iterations = max_iterations
        self.tolerance =  tolerance
        self.log_likelihood_history_:list[float] = []

    def fit(
            self,
            model: HMM,
            observations: np.ndarray
    )->HMM:
        """
        Train an HMM using Baum-Welch
        Parameter:
            model: HMM (to train)
            observations: Observation sequence of shape (T,)
        Return:
            HMM: updated model
        """

        observations = _validate_observations(observations)
        self.log_likelihood_history_ = []
        previous_log_likelihood = -np.inf

        for _ in range(self.max_iterations):
            gamma = state_posteriors(model, observations)
            xi = transition_posteriors(model, observations)
            log_likelihood = self._log_likelihood(model, observations)
            self.log_likelihood_history_.append(log_likelihood)

            #E-step has produced gamma and xi
            #M-step updates the HMM-params
            self._update_initial_prob(model, gamma)
            self._update_transition_prob(model, gamma, xi)

            improvement = log_likelihood - previous_log_likelihood

            if np.isfinite(previous_log_likelihood):
                if abs(improvement)<self.tolerance:
                    break
            previous_log_likelihood = log_likelihood
        return model

    @staticmethod
    def _log_likelihood(
            model: HMM,
            observations: np.ndarray,
    )->float:
        _,log_likelihood = forward_log(model, observations)
        return log_likelihood

    @staticmethod
    def _update_initial_prob(
            model: HMM,
            gamma: np.ndarray
    )->None:
        """
        Update: pi_i = gamma_1(i)
        """
        model.initial_prob = gamma[0].copy()

    @staticmethod
    def _update_transition_prob(
            model:HMM,
            gamma: np.ndarray,
            xi: np.ndarray
    )->None:
        """
        Update a_ij = sum_t xi_t(i,j)/sum_t gamma_t(i)
        """
        n_states = model.n_states
        if len(gamma)<2:
            return

        new_transition = np.zeros((n_states, n_states), dtype=np.float64)

        for i in range(n_states):
            denominator = gamma[:-1, i].sum()

            if denominator<=0:
                new_transition[i] = model.transition_prob[i]
                continue

            for j in range(n_states):
                numerator = xi[:, i,j].sum()
                new_transition[i,j] = numerator/denominator
        row_sums = new_transition.sum(axis=1)

        for i in range(n_states):
            if row_sums[i]>0:
                new_transition[i]/=row_sums[i]

        model.transition_prob = new_transition
