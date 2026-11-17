import json
from pathlib import Path

import numpy as np

import federated_knee_modal as experiment


def test_pooled_features_stay_float32_and_ignore_padding(tmp_path, monkeypatch):
    embeddings = np.array([[[2, 4], [900, 900], [4, 8]]], dtype=np.float16)
    np.savez(
        tmp_path / "features-000-of-256.npz",
        embeddings=embeddings,
        valid_mask=np.array([[True, False, True]]),
        study_ids=np.array(["study-a"]),
    )
    monkeypatch.setattr(experiment, "FEATURE_ROOT", str(tmp_path))
    ids, features = experiment._load_matrix(1)
    assert ids == ["study-a"]
    assert features.dtype == np.float32
    np.testing.assert_array_equal(features, [[3, 6]])


def test_repeated_report_group_never_crosses_clients():
    groups = ["report-a", "report-b", "report-a", "report-c", "report-b"]
    assigned = experiment._silo_assignment(groups, 5)
    assert assigned[0] == assigned[2]
    assert assigned[1] == assigned[4]
    assert all(0 <= value < 5 for value in assigned)
    assert assigned == experiment._silo_assignment(groups, 5)


def test_exported_results_are_consistent():
    payload = json.loads(
        Path(__file__).with_name("federated_knee_run.json").read_text()
    )
    assert payload["studies"] == sum(site["studies"] for site in payload["silos"])
    assert len(payload["history"]) == payload["rounds"]
    assert len(payload["silos"]) == payload["siloCount"]
    for site in payload["silos"]:
        assert site["studies"] == site["trainStudies"] + site["testStudies"]
    for round_result in payload["history"]:
        assert 0 <= round_result["globalMacroAuc"] <= 1
        assert len(round_result["clients"]) == payload["siloCount"]
    assert payload["finalMacroAuc"] == payload["history"][-1]["globalMacroAuc"]
    assert payload["limitations"]
