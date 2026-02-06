from app.evaluation.statistics import (
    cohen_kappa,
    fleiss_kappa,
    mcnemar_bowker,
    paired_bootstrap_ci,
    holm_bonferroni,
)


def test_cohen_kappa_identical():
    a = ["A", "B", "C", "A", "B"]
    b = ["A", "B", "C", "A", "B"]
    assert cohen_kappa(a, b) == 1.0


def test_fleiss_kappa_bounds():
    ratings = [
        ["A", "A", "A", "A", "A"],
        ["B", "B", "B", "B", "B"],
        ["A", "A", "A", "A", "A"],
    ]
    k = fleiss_kappa(ratings)
    assert 0.0 <= k <= 1.0


def test_mcnemar_bowker_output():
    a = ["A", "A", "B", "B", "C", "C"]
    b = ["A", "B", "B", "C", "C", "A"]
    result = mcnemar_bowker(a, b)
    assert "statistic" in result
    assert "pvalue" in result
    assert result["df"] >= 0


def test_bootstrap_ci_shape():
    a = [0.8, 0.7, 0.9, 0.6]
    b = [0.7, 0.6, 0.85, 0.5]
    ci = paired_bootstrap_ci(a, b, n_bootstrap=200, seed=1)
    assert ci["ci_low"] <= ci["mean_delta"] <= ci["ci_high"]


def test_holm_bonferroni():
    pvals = {"a": 0.001, "b": 0.02, "c": 0.2}
    decisions = holm_bonferroni(pvals, alpha=0.05)
    assert decisions["a"] is True
    assert decisions["c"] is False
