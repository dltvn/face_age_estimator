from __future__ import annotations

import json
import sqlite3

from src.core.inference import AgePrediction, GenderPrediction, RacePrediction
from src.utils.prediction_logging import (
    get_recent_prediction_logs,
    initialize_prediction_log_db,
    log_prediction_request,
)


def test_log_prediction_request_persists_all_prediction_fields(
    monkeypatch,
    tmp_path,
) -> None:
    """Test prediction logger stores timestamped output metadata in SQLite."""
    db_path = tmp_path / "prediction_logs.db"
    monkeypatch.setenv("PREDICTION_LOG_DB_PATH", str(db_path))

    initialize_prediction_log_db()
    log_prediction_request(
        endpoint_name="predict_all_with_crop",
        gender_prediction=GenderPrediction(
            gender="female",
            confidence=0.92,
            prob_female=0.92,
            prob_male=0.08,
        ),
        age_agnostic_prediction=AgePrediction(
            distribution=[0.1, 0.2, 0.7],
            predicted_age=2,
            confidence=0.7,
        ),
        age_gender_specific_prediction=AgePrediction(
            distribution=[0.05, 0.15, 0.8],
            predicted_age=2,
            confidence=0.8,
        ),
        race_prediction=RacePrediction(
            race="asian",
            confidence=0.81,
            probabilities={"asian": 0.81, "white": 0.19},
        ),
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                endpoint_name,
                request_timestamp,
                predicted_gender,
                gender_confidence,
                prob_female,
                prob_male,
                age_agnostic_predicted_age,
                age_agnostic_confidence,
                age_agnostic_distribution_json,
                age_gender_specific_predicted_age,
                age_gender_specific_confidence,
                age_gender_specific_distribution_json,
                predicted_race,
                race_confidence,
                race_probabilities_json,
                error_message
            FROM prediction_logs
            """
        ).fetchone()

    assert row is not None
    assert row[0] == "predict_all_with_crop"
    assert row[1]
    assert row[2] == "female"
    assert row[3] == 0.92
    assert row[4] == 0.92
    assert row[5] == 0.08
    assert row[6] == 2
    assert row[7] == 0.7
    assert json.loads(row[8]) == [0.1, 0.2, 0.7]
    assert row[9] == 2
    assert row[10] == 0.8
    assert json.loads(row[11]) == [0.05, 0.15, 0.8]
    assert row[12] == "asian"
    assert row[13] == 0.81
    assert json.loads(row[14]) == {"asian": 0.81, "white": 0.19}
    assert row[15] is None


def test_log_prediction_request_persists_errors_without_predictions(
    monkeypatch,
    tmp_path,
) -> None:
    """Test prediction logger stores failed inference attempts."""
    db_path = tmp_path / "prediction_logs.db"
    monkeypatch.setenv("PREDICTION_LOG_DB_PATH", str(db_path))

    log_prediction_request(
        endpoint_name="predict_gender_with_crop",
        error_message="No face detected.",
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                endpoint_name,
                predicted_gender,
                age_agnostic_predicted_age,
                age_gender_specific_predicted_age,
                predicted_race,
                error_message
            FROM prediction_logs
            """
        ).fetchone()

    assert row == (
        "predict_gender_with_crop",
        None,
        None,
        None,
        None,
        "No face detected.",
    )


def test_get_recent_prediction_logs_returns_latest_rows(monkeypatch, tmp_path) -> None:
    """Test recent log query returns newest rows first with parsed JSON fields."""
    db_path = tmp_path / "prediction_logs.db"
    monkeypatch.setenv("PREDICTION_LOG_DB_PATH", str(db_path))

    log_prediction_request(
        endpoint_name="predict_gender_with_crop",
        gender_prediction=GenderPrediction(
            gender="male",
            confidence=0.88,
            prob_female=0.12,
            prob_male=0.88,
        ),
    )
    log_prediction_request(
        endpoint_name="predict_age_agnostic_with_crop",
        age_agnostic_prediction=AgePrediction(
            distribution=[0.25, 0.75],
            predicted_age=1,
            confidence=0.75,
        ),
    )

    rows = get_recent_prediction_logs(limit=1)

    assert len(rows) == 1
    assert rows[0]["endpoint_name"] == "predict_age_agnostic_with_crop"
    assert rows[0]["age_agnostic_predicted_age"] == 1
    assert rows[0]["age_agnostic_distribution"] == [0.25, 0.75]
