from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router


def _build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _mock_age_prediction() -> SimpleNamespace:
    return SimpleNamespace(
        predicted_age=34,
        confidence=0.62,
        distribution=[0.1, 0.2, 0.7],
    )


def test_get_recent_logs_returns_rows(monkeypatch) -> None:
    """Test recent logs endpoint returns helper rows unchanged."""
    expected_row = {
        "id": 1,
        "endpoint_name": "predict_gender_with_crop",
        "request_timestamp": "2026-03-28T12:00:00+00:00",
        "predicted_gender": "female",
        "gender_confidence": 0.91,
        "prob_female": 0.91,
        "prob_male": 0.09,
        "age_agnostic_predicted_age": None,
        "age_agnostic_confidence": None,
        "age_agnostic_distribution": None,
        "age_gender_specific_predicted_age": None,
        "age_gender_specific_confidence": None,
        "age_gender_specific_distribution": None,
        "predicted_race": None,
        "race_confidence": None,
        "race_probabilities": None,
        "error_message": None,
    }
    monkeypatch.setattr(
        "src.api.routes.get_recent_prediction_logs",
        lambda limit: [expected_row],
    )

    client = _build_test_client()
    response = client.get("/api/logs/recent", params={"limit": 1})

    assert response.status_code == 200
    assert response.json() == [expected_row]


def test_get_recent_logs_rejects_invalid_limit() -> None:
    """Test recent logs endpoint validates the limit query parameter."""
    client = _build_test_client()

    response = client.get("/api/logs/recent", params={"limit": 0})

    assert response.status_code == 422


def test_predict_gender_with_crop_returns_422_and_logs_value_errors(
    monkeypatch,
) -> None:
    """Test prediction route maps preprocessing ValueError to HTTP 422."""
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: (_ for _ in ()).throw(
            ValueError("No face detected.")
        ),
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/gender-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "No face detected."
    assert logged == {
        "endpoint_name": "predict_gender_with_crop",
        "error_message": "No face detected.",
    }


def test_predict_gender_with_crop_returns_prediction_payload(monkeypatch) -> None:
    """Test gender route returns prediction fields and writes a success log."""
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: SimpleNamespace(aligned_bgr="aligned"),
        predict_gender=lambda aligned_bgr: SimpleNamespace(
            gender="male",
            confidence=0.87,
            prob_female=0.13,
            prob_male=0.87,
        ),
        encode_jpeg=lambda aligned_bgr: b"jpeg-bytes",
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/gender-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["gender"] == "male"
    assert response.json()["confidence"] == 0.87
    assert logged["endpoint_name"] == "predict_gender_with_crop"
    assert logged["gender_prediction"].gender == "male"


def test_predict_age_agnostic_with_crop_returns_prediction_payload(
    monkeypatch,
) -> None:
    """Test age-agnostic route returns age output and logs the prediction."""
    age_prediction = _mock_age_prediction()
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: SimpleNamespace(aligned_bgr="aligned"),
        predict_age_agnostic=lambda aligned_bgr: age_prediction,
        encode_jpeg=lambda aligned_bgr: b"jpeg-bytes",
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/age-agnostic-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["predicted_age"] == 34
    assert response.json()["distribution"] == [0.1, 0.2, 0.7]
    assert logged["endpoint_name"] == "predict_age_agnostic_with_crop"
    assert logged["age_agnostic_prediction"].predicted_age == 34


def test_predict_age_agnostic_with_crop_returns_422_and_logs_value_errors(
    monkeypatch,
) -> None:
    """Test age-agnostic route maps preprocessing ValueError to HTTP 422."""
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: (_ for _ in ()).throw(
            ValueError("No face detected.")
        ),
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/age-agnostic-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "No face detected."
    assert logged == {
        "endpoint_name": "predict_age_agnostic_with_crop",
        "error_message": "No face detected.",
    }


def test_predict_age_gender_specific_with_crop_returns_prediction_payload(
    monkeypatch,
) -> None:
    """Test gender-specific age route returns gender and age outputs."""
    gender_prediction = SimpleNamespace(
        gender="female",
        confidence=0.93,
        prob_female=0.93,
        prob_male=0.07,
    )
    age_prediction = _mock_age_prediction()
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: SimpleNamespace(aligned_bgr="aligned"),
        predict_gender=lambda aligned_bgr: gender_prediction,
        predict_age_gender_specific=lambda aligned_bgr, gender: age_prediction,
        encode_jpeg=lambda aligned_bgr: b"jpeg-bytes",
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/age-gender-specific-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["gender"] == "female"
    assert response.json()["predicted_age"] == 34
    assert logged["endpoint_name"] == "predict_age_gender_specific_with_crop"
    assert logged["gender_prediction"].gender == "female"
    assert logged["age_gender_specific_prediction"].predicted_age == 34


def test_predict_age_gender_specific_with_crop_returns_422_and_logs_value_errors(
    monkeypatch,
) -> None:
    """Test gender-specific age route maps preprocessing ValueError to HTTP 422."""
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: (_ for _ in ()).throw(
            ValueError("No face detected.")
        ),
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/age-gender-specific-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "No face detected."
    assert logged == {
        "endpoint_name": "predict_age_gender_specific_with_crop",
        "error_message": "No face detected.",
    }


def test_predict_race_with_crop_returns_prediction_payload(monkeypatch) -> None:
    """Test race route returns probability payload and logs success."""
    race_prediction = SimpleNamespace(
        race="asian",
        confidence=0.78,
        probabilities={"asian": 0.78, "white": 0.22},
    )
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: SimpleNamespace(aligned_bgr="aligned"),
        predict_race=lambda aligned_bgr: race_prediction,
        encode_jpeg=lambda aligned_bgr: b"jpeg-bytes",
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/race-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["race"] == "asian"
    assert response.json()["probabilities"] == {"asian": 0.78, "white": 0.22}
    assert logged["endpoint_name"] == "predict_race_with_crop"
    assert logged["race_prediction"].race == "asian"


def test_predict_race_with_crop_returns_422_and_logs_value_errors(
    monkeypatch,
) -> None:
    """Test race route maps preprocessing ValueError to HTTP 422."""
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: (_ for _ in ()).throw(
            ValueError("No face detected.")
        ),
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/race-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "No face detected."
    assert logged == {
        "endpoint_name": "predict_race_with_crop",
        "error_message": "No face detected.",
    }


def test_predict_all_with_crop_returns_combined_prediction_payload(monkeypatch) -> None:
    """Test combined route returns all model outputs and logs them together."""
    gender_prediction = SimpleNamespace(
        gender="male",
        confidence=0.89,
        prob_female=0.11,
        prob_male=0.89,
    )
    age_agnostic_prediction = _mock_age_prediction()
    age_gender_specific_prediction = SimpleNamespace(
        predicted_age=36,
        confidence=0.66,
        distribution=[0.05, 0.25, 0.7],
    )
    race_prediction = SimpleNamespace(
        race="white",
        confidence=0.71,
        probabilities={"white": 0.71, "asian": 0.29},
    )
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: SimpleNamespace(aligned_bgr="aligned"),
        predict_gender=lambda aligned_bgr: gender_prediction,
        predict_age_agnostic=lambda aligned_bgr: age_agnostic_prediction,
        predict_age_gender_specific=lambda aligned_bgr, gender: (
            age_gender_specific_prediction
        ),
        predict_race=lambda aligned_bgr: race_prediction,
        encode_jpeg=lambda aligned_bgr: b"jpeg-bytes",
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/all-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["gender"]["gender"] == "male"
    assert payload["age_agnostic"]["predicted_age"] == 34
    assert payload["age_gender_specific"]["predicted_age"] == 36
    assert payload["race"]["race"] == "white"
    assert logged["endpoint_name"] == "predict_all_with_crop"
    assert logged["gender_prediction"].gender == "male"
    assert logged["age_agnostic_prediction"].predicted_age == 34
    assert logged["age_gender_specific_prediction"].predicted_age == 36
    assert logged["race_prediction"].race == "white"


def test_predict_all_with_crop_returns_422_and_logs_value_errors(
    monkeypatch,
) -> None:
    """Test combined route maps preprocessing ValueError to HTTP 422."""
    mock_service = SimpleNamespace(
        decode_image_bytes=lambda image_bytes: object(),
        preprocess_face=lambda image_bgr: (_ for _ in ()).throw(
            ValueError("No face detected.")
        ),
    )
    logged = {}

    monkeypatch.setattr("src.api.routes.get_inference_service", lambda: mock_service)
    monkeypatch.setattr(
        "src.api.routes.log_prediction_request",
        lambda **kwargs: logged.update(kwargs),
    )

    client = _build_test_client()
    response = client.post(
        "/api/predict/all-with-crop",
        files={"file": ("face.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "No face detected."
    assert logged == {
        "endpoint_name": "predict_all_with_crop",
        "error_message": "No face detected.",
    }
