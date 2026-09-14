from unittest import result

import numpy as np

from mini_asr.probability.gmm import GMM

class GMMHMM:
    """
    HMM with GMM emission model for every hidden state
    Parameter:
        initial_prob: Initial state probabilities, shape=(n_states, )
        transition_prob: state transition probabilities, shape=(n_state, n_state)
        n_components: number of gaussian components per state
        max_gmm_iterations: maxm EM iterations for each GMM
        random_state: Random seed for GMM initialization
    """
    def __init__(
            self,
            initial_prob: np.ndarray,
            transition_prob: np.ndarray,
            n_components:int=2,
            max_gmm_iterations: int=100,
            random_state:int|None=None,
    )->None:
        self.initial_prob = np.asarray(initial_prob, dtype=np.float64)
        self.transition_prob = np.asarray(transition_prob, dtype=np.float64)
        self.n_components = n_components
        self.max_gmm_iterations = max_gmm_iterations
        self.random_state = random_state

        self._validate()
        self.gmms: list[GMM] = []

    @property
    def n_states(self)->int:
        return self.initial_prob.shape[0]

    def _validate(self) -> None:
        if self.initial_prob.ndim != 1:
            raise ValueError("initial_prob must be 1-D")
        if self.transition_prob.ndim != 2:
            raise ValueError("transition_prob must be 2-D")

        n_states = self.initial_prob.shape[0]

        if self.transition_prob.shape != (n_states, n_states):
            raise ValueError("transition_prob must have shape (n_states, n_states)")

        if np.any(self.initial_prob < 0):
            raise ValueError("initial_prob cannot contain negative values")
        if np.any(self.transition_prob < 0):
            raise ValueError("transition_prob cannot contain negative values")

        if not np.isclose(self.initial_prob.sum(), 1.0):
            raise ValueError("initial_prob must sum to 1")

        if not np.allclose(self.transition_prob.sum(axis=1), 1.0):
            raise ValueError("Each transition row must sum to 1")

    def fit(self,X: np.ndarray)->"GMMHMM":
        """
        Fit one GMM per HMM state.
        X: continuous acoustic feature vectors.
        Improvement Needed: using Baum-Welch Algorithm
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim!=2:
            raise ValueError("X must be a 2-D array")
        if X.shape[0]==0:
            raise ValueError("X cannot be empty")

        self.gmms = []
        for state in range(self.n_states):
            seed = None
            if self.random_state is not None:
                seed = self.random_state + state
            gmm = GMM(n_components=self.n_components, max_iterations=self.max_gmm_iterations, random_state=seed)
            gmm.fit(X)
            self.gmms.append(gmm)

        return self

    def _check_fitted(self)->None:
        if len(self.gmms)!=self.n_states:
            raise RuntimeError("Model has not beed fitted")

    def log_initial_prob(self)->np.ndarray:
        return np.log(np.maximum(self.initial_prob, 1e-300))

    def log_transition_prob(self)->np.ndarray:
        return np.log(np.maximum(self.transition_prob, 1e-300))

    def emission_log_probability(
            self,
            X: np.ndarray
    )->np.ndarray:
        """
        Calculate log P(x_t|state)
        Parameter:
            X: continuous feature vectors, shape: (T,D)
        return:
            log_probability: shape(T, n_states)
        """

        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim!=2:
            raise ValueError("X must be  2-D array")
        n_frames = X.shape[0]

        result = np.empty((n_frames, self.n_states), dtype=np.float64)

        for state, gmm in enumerate(self.gmms):
            result[:,state] = gmm.score_samples(X)

        return result