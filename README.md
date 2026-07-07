# EduTrace AI Service

Python Flask microservice for face enrollment and face recognition using pre-trained embeddings.

## Endpoints

- GET /health
- POST /enroll-face
- POST /recognize-face

## Setup

1. Create virtual environment:
   python -m venv .venv
2. Activate environment:
   .venv\Scripts\Activate.ps1
3. Install dependencies:
   pip install -r requirements.txt
4. Run server:
   python main.py

## Request format

### POST /enroll-face

```json
{
  "imageBase64": "...",
  "mimeType": "image/jpeg"
}
```

### POST /recognize-face

```json
{
  "imageBase64": "...",
  "mimeType": "image/jpeg",
  "candidates": [
    {
      "studentId": "64b...",
      "embedding": [0.01, -0.2, 0.3]
    }
  ]
}
```

## Environment variables

- FACE_DISTANCE_THRESHOLD (default: 0.45)
