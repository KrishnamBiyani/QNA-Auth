import pytest

from model.evaluate import ModelEvaluator


@pytest.fixture
def evaluator() -> ModelEvaluator:
    return ModelEvaluator(embedder=None)


def test_vector_pair_scoring_uses_all_impostor_pairs_by_default():
    vectors = {
        "dev_a": [[1.0, 0.0], [0.9, 0.1]],
        "dev_b": [[0.0, 1.0], [0.1, 0.9]],
    }

    scores, labels = ModelEvaluator.compute_similarity_scores_from_vectors(vectors, metric="cosine")

    assert len(scores) == 6
    assert sum(labels) == 2
    assert len(labels) - sum(labels) == 4


def test_target_far_threshold_prefers_low_frr_not_highest_threshold(evaluator: ModelEvaluator):
    scores = [0.95, 0.88, 0.30, 0.20]
    labels = [1, 1, 0, 0]

    threshold, metrics = evaluator.find_threshold_for_target_far(scores, labels, target_far=0.0)

    assert metrics["far"] == 0.0
    assert metrics["frr"] == 0.0
    assert 0.30 < threshold < 0.88


def test_generate_score_report_includes_locked_threshold_metrics(evaluator: ModelEvaluator):
    scores = [0.92, 0.81, 0.25, 0.10]
    labels = [1, 1, 0, 0]

    report = evaluator.generate_score_report(scores, labels, target_far=0.0, deployed_threshold=0.5)

    assert report["pair_counts"] == {"total": 4, "genuine": 2, "impostor": 2}
    assert report["roc_auc"] == pytest.approx(1.0)
    assert report["pr_auc"] == pytest.approx(1.0)
    assert report["deployed_threshold_metrics"]["far"] == 0.0
    assert report["deployed_threshold_metrics"]["frr"] == 0.0
