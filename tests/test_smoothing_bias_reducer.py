from scripts.research.reduce_finite_support_study import (
    NUMERIC_FIELDS,
    validate_task_metadata,
    validate_task_records,
)


def _records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for model in ("corrected", "naive"):
        for replicate in range(2):
            record = {
                "config_hash": "config-hash",
                "config_id": "config-a",
                "run_id": "run-a",
                "task_index": "0",
                "model": model,
                "replicate": str(replicate),
                "at_bound": "False",
            }
            record.update({field: "1.0" for field in NUMERIC_FIELDS})
            records.append(record)
    return records


def test_reducer_accepts_complete_finite_shard_and_metadata() -> None:
    reasons = validate_task_records(
        _records(),
        config_hash="config-hash",
        config_id="config-a",
        run_id="run-a",
        task_index=0,
        replicates=2,
    )
    assert reasons == []
    metadata = {
        "manifest_hash": "manifest-hash",
        "config_hash": "config-hash",
        "config_id": "config-a",
        "run_id": "run-a",
        "task_index": 0,
        "expected_rows": 4,
        "written_rows": 4,
    }
    assert (
        validate_task_metadata(
            metadata,
            manifest_hash="manifest-hash",
            config_hash="config-hash",
            config_id="config-a",
            run_id="run-a",
            task_index=0,
            expected_rows=4,
        )
        == []
    )


def test_reducer_rejects_nonfinite_and_incomplete_replicates() -> None:
    records = _records()
    records[0]["objective"] = "nan"
    records[1]["replicate"] = "0"
    reasons = validate_task_records(
        records,
        config_hash="config-hash",
        config_id="config-a",
        run_id="run-a",
        task_index=0,
        replicates=2,
    )
    assert "non-finite numeric field: objective" in reasons
    assert "model/replicate keys are incomplete, duplicated, or unexpected" in reasons
