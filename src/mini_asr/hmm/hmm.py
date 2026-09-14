import numpy as np


class HMM:
    """
    Hidden Markov Model
    Parameter:
        initial_prob: Initial state probabilities, shape: (n_states, )
        transition_prob: state transition probabilities, shape: (n_state, n_states)
        emission_prob: observation emission probabilities shape: (n_states, n_observation)
    """

    def __init__(
            self,
            initial_prob: np.ndarray,
            transition_prob: np.ndarray,
            emission_prob: np.ndarray,
    ) -> None:
        self.initial_prob = np.asarray(initial_prob, dtype=np.float64)
        self.transition_prob = np.asarray(transition_prob, dtype=np.float64)
        self.emission_prob = np.asarray(emission_prob, dtype=np.float64)

        self._validate()

    @property
    def n_states(self)->int:
        return self.initial_prob.shape[0]

    @property
    def n_observations(self)->int:
        return self.emission_prob.shape[1]

    def _validate(self) -> None:
        if self.initial_prob.ndim != 1:
            raise ValueError("initial_prob must be 1-D")
        if self.transition_prob.ndim != 2:
            raise ValueError("transition_prob must be 2-D")
        if self.emission_prob.ndim != 2:
            raise ValueError("emission_prob must be 2-D")

        n_states = self.initial_prob.shape[0]

        if self.transition_prob.shape != (n_states, n_states):
            raise ValueError("transition_prob must have shape (n_states, n_states)")

        if self.emission_prob.shape[0] != n_states:
            raise ValueError("emission_prob must have one row per state")

        if np.any(self.initial_prob < 0):
            raise ValueError("initial_prob cannot contain negative values")
        if np.any(self.transition_prob < 0):
            raise ValueError("transition_prob cannot contain negative values")
        if np.any(self.emission_prob < 0):
            raise ValueError("emission_prob cannot contain negative values")

        if not np.isclose(self.initial_prob.sum(), 1.0):
            raise ValueError("initial_prob must sum to 1")

        if not np.allclose(self.transition_prob.sum(axis=1), 1.0):
            raise ValueError("Each transition row must sum to 1")

        if not np.allclose(self.emission_prob.sum(axis=1), 1.0):
            raise ValueError("Each emission row must sum to 1")

    def log_initial_prob(self)->np.ndarray:
        return np.log(np.maximum(self.initial_prob, 1e-300))

    def log_transition_prob(self)->np.ndarray:
        return np.log(np.maximum(self.transition_prob, 1e-300))

    def log_emission_prob(self)->np.ndarray:
        return np.log(np.maximum(self.emission_prob, 1e-300))

    def emission_log_probability(
            self,
            observations:np.ndarray,
    )->np.ndarray:
        """
        Parameters:
            observations: Integer Observation IDs, shape: (T,)
        Return log P(observation|state): shape(T,n_states)
        """
        observations = np.asarray(observations, dtype=np.int64)
        if observations.ndim!=1:
            raise ValueError("observations must be 1-D")
        if np.any(observations<0) or np.any(observations>=self.n_observations):
            raise ValueError("observation index out or range")

        log_emissions = self.log_emission_prob()
        return log_emissions[:, observations].T

    def probability_of_observation_sequence(
            self,
            observations:np.ndarray
    )->float:
        """
        calculate the probability of an observation sequence by direct enumeration
        Forward algorithm will later provide the efficient  solution
        """
        observations = np.asarray(observations, dtype=np.int64)
        if observations.ndim !=1:
            raise ValueError("observations must be 1-D")
        if len(observations)==0:
            raise ValueError("observations can be empty")

        #alpha:  probability of being in hidden state
        alpha = self.initial_prob*self.emission_prob[:, observations[0]]

        for observation in observations[1:]:
            alpha = alpha@self.transition_prob
            alpha*=self.emission_prob[:, observation,]
        return float(alpha.sum())