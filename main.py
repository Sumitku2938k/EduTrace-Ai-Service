import base64
import io
import os
import string
from typing import Any

import face_recognition
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image


app = Flask(__name__)

DEFAULT_DISTANCE_THRESHOLD = float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.45"))
MODEL_VERSION = "face_recognition_v1"


class FaceServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code  

# base64 string → actual image  
def decode_image(image_base64: str) -> np.ndarray:
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)
    except Exception as exc:
        raise FaceServiceError("Invalid image payload") from exc

# actual image → face embedding vector  
def get_single_face_embedding(image_array: np.ndarray) -> np.ndarray:
    face_locations = face_recognition.face_locations(image_array)

    if len(face_locations) == 0:
        raise FaceServiceError("No face detected. Please use a clear frontal image.")

    if len(face_locations) > 1:
        raise FaceServiceError("Multiple faces detected. Please capture one face only.")

    encodings = face_recognition.face_encodings(image_array, known_face_locations=face_locations)
    if len(encodings) == 0:
        raise FaceServiceError("Unable to generate face embedding from this image.")

    return encodings[0]

# request JSON hai ya nahi
def require_json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise FaceServiceError("Request body must be a JSON object")
    return payload


def require_non_empty_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise FaceServiceError(f"{field_name} is required")
    return value


def parse_candidates(payload: dict[str, Any]) -> tuple[list[str], list[np.ndarray]]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FaceServiceError("candidates must be a non-empty array")

    candidate_ids: list[str] = []
    candidate_vectors: list[np.ndarray] = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise FaceServiceError("Each candidate must be a JSON object")

        student_id = candidate.get("studentId")
        embedding = candidate.get("embedding")

        if not isinstance(student_id, str) or not student_id.strip():
            raise FaceServiceError("Each candidate studentId is required")

        if not isinstance(embedding, list) or not embedding:
            raise FaceServiceError("Each candidate embedding must be a non-empty array")

        try:
            candidate_vector = np.array(embedding, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise FaceServiceError("Candidate embedding contains invalid numeric values") from exc

        candidate_ids.append(student_id)
        candidate_vectors.append(candidate_vector)

    return candidate_ids, candidate_vectors


@app.errorhandler(FaceServiceError)
def handle_face_service_error(error: FaceServiceError):
    return jsonify({"detail": error.message}), error.status_code


@app.errorhandler(500)
def handle_internal_error(_error):
    return jsonify({"detail": "Internal server error"}), 500


@app.get("/health")
def health_check():
    return jsonify({"status": "ok", "service": "face-recognition"})

# enroll-face endpoint: image base64 string ko face embedding vector me convert karke return karta hai
@app.post("/enroll-face")
def enroll_face():
    payload = require_json_object()
    image_base64 = require_non_empty_string(payload, "imageBase64")
    image_array = decode_image(image_base64)
    embedding = get_single_face_embedding(image_array)

    return jsonify({
        "embedding": embedding.tolist(),
        "modelVersion": MODEL_VERSION,
    })


@app.post("/recognize-face")
def recognize_face():
    payload = require_json_object()
    image_base64 = require_non_empty_string(payload, "imageBase64")
    image_array = decode_image(image_base64)
    query_embedding = get_single_face_embedding(image_array)
    candidate_ids, candidate_vectors = parse_candidates(payload)

    distances = face_recognition.face_distance(candidate_vectors, query_embedding)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    confidence = max(0.0, min(1.0, 1.0 - best_distance))

    if best_distance > DEFAULT_DISTANCE_THRESHOLD:
        return jsonify({
            "isMatch": False,
            "studentId": None,
            "confidence": confidence,
            "distance": best_distance,
            "threshold": DEFAULT_DISTANCE_THRESHOLD,
        })

    return jsonify({
        "isMatch": True,
        "studentId": candidate_ids[best_index],
        "confidence": confidence,
        "distance": best_distance,
        "threshold": DEFAULT_DISTANCE_THRESHOLD,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=True)
