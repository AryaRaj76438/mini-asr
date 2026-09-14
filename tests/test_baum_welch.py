import numpy as np

from mini_asr.hmm import HMM, BaumWelch


def create_model() -> HMM:
    return HMM(
        initial_prob=np.array([0.5, 0.5]),
        transition_prob=np.array([[0.5, 0.5], [0.5, 0.5]]),
        emission_prob=np.array([[0.9, 0.1], [0.2, 0.8]]),
    )


def test_baum_welch_updates_model():
    model = create_model()
    observations = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])

    original_transition = model.transition_prob.copy()

    trainer = BaumWelch(max_iterations=20, tolerance=1e-5)
    trained_model = trainer.fit(model, observations)

    assert trained_model is model
    assert not np.allclose(model.transition_prob, original_transition)


def test_baum_welch_produces_valid_model():
    model = create_model()
    observations = np.array([0, 0, 1, 1, 0, 1])

    trainer = BaumWelch(max_iterations=10)
    trainer.fit(model, observations)

    assert np.isclose(model.initial_prob.sum(), 1.0)
    assert np.allclose(model.transition_prob.sum(axis=1), 1.0)

    history = trainer.log_likelihood_history_
    assert len(history) > 0
    assert np.all(np.isfinite(history))