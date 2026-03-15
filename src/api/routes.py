from __future__ import annotations

import base64

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.core.inference import get_inference_service

router = APIRouter(prefix="/api", tags=["inference"])


class GenderWithCropResponse(BaseModel):
    """Response schema for gender prediction + cropped image endpoint."""

    gender: str
    confidence: float
    prob_female: float
    prob_male: float
    cropped_image_base64: str
    cropped_image_mime_type: str = "image/jpeg"


class AgeAgnosticWithCropResponse(BaseModel):
    """Response schema for age-agnostic prediction + cropped image endpoint."""

    predicted_age: int
    confidence: float
    distribution: list[float] = Field(description="117-bin age distribution.")
    cropped_image_base64: str
    cropped_image_mime_type: str = "image/jpeg"


class GenderSpecificAgeWithCropResponse(BaseModel):
    """Response schema for gender-specific age + cropped image endpoint."""

    gender: str
    gender_confidence: float
    predicted_age: int
    confidence: float
    distribution: list[float] = Field(description="117-bin age distribution.")
    cropped_image_base64: str
    cropped_image_mime_type: str = "image/jpeg"


class RaceWithCropResponse(BaseModel):
    """Response schema for race prediction + cropped image endpoint."""

    race: str
    confidence: float
    probabilities: dict[str, float]
    cropped_image_base64: str
    cropped_image_mime_type: str = "image/jpeg"


class CombinedGenderResponse(BaseModel):
    """Gender prediction block for all-in-one endpoint."""

    gender: str
    confidence: float
    prob_female: float
    prob_male: float


class CombinedAgeResponse(BaseModel):
    """Age prediction block for all-in-one endpoint."""

    predicted_age: int
    confidence: float
    distribution: list[float] = Field(description="117-bin age distribution.")


class CombinedRaceResponse(BaseModel):
    """Race prediction block for all-in-one endpoint."""

    race: str
    confidence: float
    probabilities: dict[str, float]


class PredictAllWithCropResponse(BaseModel):
    """Response schema for all predictions + cropped image endpoint."""

    gender: CombinedGenderResponse
    age_agnostic: CombinedAgeResponse
    age_gender_specific: CombinedAgeResponse
    race: CombinedRaceResponse
    cropped_image_base64: str
    cropped_image_mime_type: str = "image/jpeg"


def _read_upload_bytes(upload: UploadFile) -> bytes:
    """Read uploaded file bytes with basic validation."""
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    return data


@router.post("/predict/all-with-crop", response_model=PredictAllWithCropResponse)
def predict_all_with_crop(file: UploadFile = File(...)) -> PredictAllWithCropResponse:
    """Crop once and return gender, age, and race predictions."""
    service = get_inference_service()
    try:
        image_bytes = _read_upload_bytes(file)
        image_bgr = service.decode_image_bytes(image_bytes)
        pre = service.preprocess_face(image_bgr)

        gender_pred = service.predict_gender(pre.aligned_bgr)
        age_agnostic_pred = service.predict_age_agnostic(pre.aligned_bgr)
        age_gender_specific_pred = service.predict_age_gender_specific(
            aligned_bgr=pre.aligned_bgr,
            gender=gender_pred.gender,
        )
        race_pred = service.predict_race(pre.aligned_bgr)

        cropped_jpeg = service.encode_jpeg(pre.aligned_bgr)
        cropped_b64 = base64.b64encode(cropped_jpeg).decode("utf-8")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictAllWithCropResponse(
        gender=CombinedGenderResponse(
            gender=gender_pred.gender,
            confidence=gender_pred.confidence,
            prob_female=gender_pred.prob_female,
            prob_male=gender_pred.prob_male,
        ),
        age_agnostic=CombinedAgeResponse(
            predicted_age=age_agnostic_pred.predicted_age,
            confidence=age_agnostic_pred.confidence,
            distribution=age_agnostic_pred.distribution,
        ),
        age_gender_specific=CombinedAgeResponse(
            predicted_age=age_gender_specific_pred.predicted_age,
            confidence=age_gender_specific_pred.confidence,
            distribution=age_gender_specific_pred.distribution,
        ),
        race=CombinedRaceResponse(
            race=race_pred.race,
            confidence=race_pred.confidence,
            probabilities=race_pred.probabilities,
        ),
        cropped_image_base64=cropped_b64,
    )


@router.post("/predict/gender-with-crop", response_model=GenderWithCropResponse)
def predict_gender_with_crop(file: UploadFile = File(...)) -> GenderWithCropResponse:
    """Predict gender and include cropped/aligned face image."""
    service = get_inference_service()
    try:
        image_bytes = _read_upload_bytes(file)
        image_bgr = service.decode_image_bytes(image_bytes)
        pre = service.preprocess_face(image_bgr)
        pred = service.predict_gender(pre.aligned_bgr)
        cropped_jpeg = service.encode_jpeg(pre.aligned_bgr)
        cropped_b64 = base64.b64encode(cropped_jpeg).decode("utf-8")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenderWithCropResponse(
        gender=pred.gender,
        confidence=pred.confidence,
        prob_female=pred.prob_female,
        prob_male=pred.prob_male,
        cropped_image_base64=cropped_b64,
    )


@router.post("/predict/age-agnostic-with-crop", response_model=AgeAgnosticWithCropResponse)
def predict_age_agnostic_with_crop(file: UploadFile = File(...)) -> AgeAgnosticWithCropResponse:
    """Predict age with gender-agnostic model and include cropped/aligned face image."""
    service = get_inference_service()
    try:
        image_bytes = _read_upload_bytes(file)
        image_bgr = service.decode_image_bytes(image_bytes)
        pre = service.preprocess_face(image_bgr)
        pred = service.predict_age_agnostic(pre.aligned_bgr)
        cropped_jpeg = service.encode_jpeg(pre.aligned_bgr)
        cropped_b64 = base64.b64encode(cropped_jpeg).decode("utf-8")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgeAgnosticWithCropResponse(
        predicted_age=pred.predicted_age,
        confidence=pred.confidence,
        distribution=pred.distribution,
        cropped_image_base64=cropped_b64,
    )


@router.post(
    "/predict/age-gender-specific-with-crop",
    response_model=GenderSpecificAgeWithCropResponse,
)
def predict_age_gender_specific_with_crop(
    file: UploadFile = File(...),
) -> GenderSpecificAgeWithCropResponse:
    """Predict gender-specific age and include cropped/aligned face image."""
    service = get_inference_service()
    try:
        image_bytes = _read_upload_bytes(file)
        image_bgr = service.decode_image_bytes(image_bytes)
        pre = service.preprocess_face(image_bgr)
        gender_pred = service.predict_gender(pre.aligned_bgr)
        age_pred = service.predict_age_gender_specific(
            aligned_bgr=pre.aligned_bgr,
            gender=gender_pred.gender,
        )
        cropped_jpeg = service.encode_jpeg(pre.aligned_bgr)
        cropped_b64 = base64.b64encode(cropped_jpeg).decode("utf-8")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenderSpecificAgeWithCropResponse(
        gender=gender_pred.gender,
        gender_confidence=gender_pred.confidence,
        predicted_age=age_pred.predicted_age,
        confidence=age_pred.confidence,
        distribution=age_pred.distribution,
        cropped_image_base64=cropped_b64,
    )


@router.post("/predict/race-with-crop", response_model=RaceWithCropResponse)
def predict_race_with_crop(file: UploadFile = File(...)) -> RaceWithCropResponse:
    """Predict race and include cropped/aligned face image."""
    service = get_inference_service()
    try:
        image_bytes = _read_upload_bytes(file)
        image_bgr = service.decode_image_bytes(image_bytes)
        pre = service.preprocess_face(image_bgr)
        pred = service.predict_race(pre.aligned_bgr)
        cropped_jpeg = service.encode_jpeg(pre.aligned_bgr)
        cropped_b64 = base64.b64encode(cropped_jpeg).decode("utf-8")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RaceWithCropResponse(
        race=pred.race,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
        cropped_image_base64=cropped_b64,
    )
