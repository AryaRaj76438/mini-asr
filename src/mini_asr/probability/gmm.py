import numpy as np
from mini_asr.probability.gaussian import log_multivariate_gaussian_pdf
from mini_asr.probability.numerical import logsumexp

class GMM:
    """
    Gaussian mixture model with diagonal covariance
    Parameters
    n_components: Number of gaussian components
    max_iterations: maximum number of EM iterations
    tolerance: Training stops when the change in log likelihood becomes smaller than this value.
    variance_floor: Minimum allowed variance. This prevents numerical instability when a component collapses onto a point.
    random_state: Random seed used for initialization.
    """

    def __init__(
            self,
            n_components: int,
            max_iterations: int=100,
            tolerance:float=1e-4,
            variance_floor:float=1e-6,
            random_state:int|None=None
    )->None:
        if n_components<=0:
            raise ValueError("n_components must be positive")
        if max_iterations<=0:
            raise ValueError("max_iterations must be positive")
        if tolerance<=0:
            raise ValueError("tolerance must be positive")
        if variance_floor<=0:
            raise ValueError("variance_floor must be positive")

        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.variance_floor = variance_floor
        self.random_state = random_state

        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None

        self.log_likelihood_: float | None = None
        self.n_iterations_: int = 0

    def _initialize_parameters(
            self,
            X:np.ndarray
    )->None:
        """
        Initialize GMM parameters.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        if self.n_components>n_samples:
            raise ValueError("n_components cannot exceed n_samples")

        indices = rng.choice(n_samples, size=self.n_components, replace=False)
        self.means_ = X[indices].copy()
        global_variance = np.var(X, axis = 0)
        global_variance = np.maximum(global_variance, self.variance_floor)
        self.variances_ = np.tile(global_variance, (self.n_components,1))
        self.weights_ = np.full(self.n_components, 1.0/self.n_components)

    def _estimate_log_prob(
            self,
            X:np.ndarray,
    )->np.ndarray:
        """
        compute p(x|component)
        return: log_prob (n_samples, n_components)
        """
        if self.means_ is None:
            raise RuntimeError("model params are not initialized")
        if self.variances_ is None:
            raise RuntimeError("model params is not initialized")
        n_samples = X.shape[0]
        log_prob = np.empty((n_samples, self.n_components))

        for k in range(self.n_components):
            log_prob[:,k] = log_multivariate_gaussian_pdf(X, self.means_[k], self.variances_[k])

        return log_prob

    def _e_step(
            self,
            X:np.ndarray
    )->np.ndarray:
        """
        Expectation steps
        Calculate responsibilities:
            gamma[n,k] = P(component_k|x_n)
        return:
            responsibilities: (n_samples, n_components)
            log_likelihood: Total log likelihood of the data
        """
        if self.weights_ is None:
            raise RuntimeError("model  params are not initialized")
        log_prob = self._estimate_log_prob(X)
        log_weights = np.log(self.weights_)
        weighted_log_prob = log_weights+log_prob
        log_likelihood_per_sample = logsumexp(weighted_log_prob, axis=1)
        log_responsibilities = weighted_log_prob - log_likelihood_per_sample[:, None]
        responsibilities = np.exp(log_responsibilities)
        total_log_likelihood = np.sum(log_likelihood_per_sample)

        return (
            responsibilities,
            float(total_log_likelihood),
        )

    def _m_step(
            self,
            X: np.ndarray,
            responsibilities: np.ndarray
    )->None:
        """
        Maximization steps
        update: weights, mean, variances
        """
        n_samples = X.shape[0]
        effective_counts = np.sum(responsibilities,axis=0)
        effective_counts = np.maximum(effective_counts, 1e-12)
        self.weights_ = effective_counts/n_samples
        self.means_ = (responsibilities.T @ X)/effective_counts[:,None]
        variances = np.empty_like(self.means_)

        for k in range(self.n_components):
            diff = X-self.means_[k]
            variances[k] = (
                    responsibilities[:, k, None]
                    * diff ** 2
            ).sum(axis=0)
            variances[k] /= effective_counts[k]

        self.variances_ = np.maximum(
            variances,
            self.variance_floor,
        )

    def fit(
            self,
            X:np.ndarray
    )->"GMM":
        """
        Fit the GMM using EM
        parameter:
        X: (n_samples, n_features)
        :return self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim!=2:
            raise ValueError("input(X) must be 2-D array")

        n_samples, n_features = X.shape
        if n_samples==0:
            raise ValueError("X cannot be empty")
        if n_features==0:
            raise ValueError("X must contain atleast one feature")

        self._initialize_parameters(X)
        previous_log_likelihood = None

        for iteration in range(self.max_iterations):
            responsibilities, log_likelihood = self._e_step(X)
            self._m_step(X, responsibilities)
            self.n_iterations_ = iteration+1
            self.log_likelihood_ = log_likelihood

            if previous_log_likelihood is not None:
                improvement = log_likelihood-previous_log_likelihood
                if abs(improvement)<self.tolerance:
                    break
            previous_log_likelihood = log_likelihood
        return self

    def predict_prob(
            self,
            X:np.ndarray
    )->np.ndarray:
        """
        calculate posterior prob for each component
        return:
            responsibilities: (n_samples, n_components)
        """
        if self.weights_ is None:
            raise RuntimeError("model has not been fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim!=2:
            raise ValueError("X must be 2-D array")
        responsibilities, _ = self._e_step(X)
        return responsibilities

    def score_samples(self, X:np.ndarray)->np.ndarray:
        """
        calculate loglikelihood for each sample
        """
        if self.weights_ is None:
            raise RuntimeError("model has not been fitted")
        X = np.asarray(X, dtype=np.float64)
        log_prob = self._estimate_log_prob(X)
        log_weight = np.log(self.weights_)
        return logsumexp(log_weight+log_prob, axis=1)



