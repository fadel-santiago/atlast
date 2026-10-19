from atlast.evaluation import DEFAULT_DATASET, run_evaluation


def test_packaged_evaluation_dataset_runs_reproducibly(embedder) -> None:
    report = run_evaluation(DEFAULT_DATASET, embedder, k=2)

    assert report["dataset"] == "atlast-synthetic-retrieval-v1"
    assert report["documents"] == 4
    assert report["answerable_cases"] == 6
    assert report["unanswerable_cases"] == 2
    assert len(report["cases"]) == 8
    assert 0.0 <= report["recall_at_k"] <= 1.0
    assert 0.0 <= report["mrr"] <= 1.0
    assert report["unanswerable_queries_with_candidates"] == 2


def test_evaluation_at_k_one_returns_one_candidate_per_case(embedder) -> None:
    report = run_evaluation(DEFAULT_DATASET, embedder, k=1)

    assert report["k"] == 1
    assert all(len(case["retrieved"]) == 1 for case in report["cases"])
