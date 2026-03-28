from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src.core.inference import AgePrediction, GenderPrediction, RacePrediction

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DB_PATH = PROJECT_ROOT / "logs" / "prediction_logs.db"

CREATE_PREDICTION_LOGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint_name TEXT NOT NULL,
    request_timestamp TEXT NOT NULL,
    predicted_gender TEXT,
    gender_confidence REAL,
    prob_female REAL,
    prob_male REAL,
    age_agnostic_predicted_age INTEGER,
    age_agnostic_confidence REAL,
    age_agnostic_distribution_json TEXT,
    age_gender_specific_predicted_age INTEGER,
    age_gender_specific_confidence REAL,
    age_gender_specific_distribution_json TEXT,
    predicted_race TEXT,
    race_confidence REAL,
    race_probabilities_json TEXT,
    error_message TEXT
);
"""

REQUIRED_COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "endpoint_name": "TEXT NOT NULL",
    "request_timestamp": "TEXT NOT NULL",
    "predicted_gender": "TEXT",
    "gender_confidence": "REAL",
    "prob_female": "REAL",
    "prob_male": "REAL",
    "age_agnostic_predicted_age": "INTEGER",
    "age_agnostic_confidence": "REAL",
    "age_agnostic_distribution_json": "TEXT",
    "age_gender_specific_predicted_age": "INTEGER",
    "age_gender_specific_confidence": "REAL",
    "age_gender_specific_distribution_json": "TEXT",
    "predicted_race": "TEXT",
    "race_confidence": "REAL",
    "race_probabilities_json": "TEXT",
    "error_message": "TEXT",
}


def _resolve_db_path() -> Path:
    """Resolve the SQLite log database path from environment or default."""
    configured = os.getenv("PREDICTION_LOG_DB_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_LOG_DB_PATH


def initialize_prediction_log_db() -> Path:
    """Create the SQLite database and prediction_logs table if missing."""
    db_path = _resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(CREATE_PREDICTION_LOGS_TABLE_SQL)
        existing_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(prediction_logs)"
            ).fetchall()
        }
        for column_name, column_type in REQUIRED_COLUMNS.items():
            if column_name not in existing_columns and column_name != "id":
                connection.execute(
                    f"ALTER TABLE prediction_logs ADD COLUMN {column_name} {column_type}"
                )
        connection.commit()
    return db_path


def log_prediction_request(
    endpoint_name: str,
    gender_prediction: GenderPrediction | None = None,
    age_agnostic_prediction: AgePrediction | None = None,
    age_gender_specific_prediction: AgePrediction | None = None,
    race_prediction: RacePrediction | None = None,
    error_message: str | None = None,
) -> None:
    """Persist inference request metadata and outputs to SQLite.

    Args:
        endpoint_name: Logical endpoint identifier handling the request.
        gender_prediction: Optional gender inference result.
        age_agnostic_prediction: Optional gender-agnostic age inference result.
        age_gender_specific_prediction: Optional gender-specific age inference result.
        race_prediction: Optional race inference result.
        error_message: Optional error message when inference fails.
    """
    request_timestamp = datetime.now(UTC).isoformat()

    try:
        db_path = initialize_prediction_log_db()
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """
                INSERT INTO prediction_logs (
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    endpoint_name,
                    request_timestamp,
                    gender_prediction.gender if gender_prediction else None,
                    gender_prediction.confidence if gender_prediction else None,
                    gender_prediction.prob_female if gender_prediction else None,
                    gender_prediction.prob_male if gender_prediction else None,
                    (
                        age_agnostic_prediction.predicted_age
                        if age_agnostic_prediction
                        else None
                    ),
                    (
                        age_agnostic_prediction.confidence
                        if age_agnostic_prediction
                        else None
                    ),
                    (
                        json.dumps(age_agnostic_prediction.distribution)
                        if age_agnostic_prediction
                        else None
                    ),
                    (
                        age_gender_specific_prediction.predicted_age
                        if age_gender_specific_prediction
                        else None
                    ),
                    (
                        age_gender_specific_prediction.confidence
                        if age_gender_specific_prediction
                        else None
                    ),
                    (
                        json.dumps(age_gender_specific_prediction.distribution)
                        if age_gender_specific_prediction
                        else None
                    ),
                    race_prediction.race if race_prediction else None,
                    race_prediction.confidence if race_prediction else None,
                    json.dumps(race_prediction.probabilities)
                    if race_prediction
                    else None,
                    error_message,
                ),
            )
            connection.commit()
    except sqlite3.Error:
        logger.exception("Failed to log prediction request for %s", endpoint_name)


def get_recent_prediction_logs(limit: int) -> list[dict[str, object | None]]:
    """Return the most recent prediction log rows from SQLite."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    db_path = initialize_prediction_log_db()
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id,
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
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    parsed_rows: list[dict[str, object | None]] = []
    for row in rows:
        parsed_rows.append(
            {
                "id": row["id"],
                "endpoint_name": row["endpoint_name"],
                "request_timestamp": row["request_timestamp"],
                "predicted_gender": row["predicted_gender"],
                "gender_confidence": row["gender_confidence"],
                "prob_female": row["prob_female"],
                "prob_male": row["prob_male"],
                "age_agnostic_predicted_age": row["age_agnostic_predicted_age"],
                "age_agnostic_confidence": row["age_agnostic_confidence"],
                "age_agnostic_distribution": (
                    json.loads(row["age_agnostic_distribution_json"])
                    if row["age_agnostic_distribution_json"]
                    else None
                ),
                "age_gender_specific_predicted_age": row[
                    "age_gender_specific_predicted_age"
                ],
                "age_gender_specific_confidence": row["age_gender_specific_confidence"],
                "age_gender_specific_distribution": (
                    json.loads(row["age_gender_specific_distribution_json"])
                    if row["age_gender_specific_distribution_json"]
                    else None
                ),
                "predicted_race": row["predicted_race"],
                "race_confidence": row["race_confidence"],
                "race_probabilities": (
                    json.loads(row["race_probabilities_json"])
                    if row["race_probabilities_json"]
                    else None
                ),
                "error_message": row["error_message"],
            }
        )
    return parsed_rows
