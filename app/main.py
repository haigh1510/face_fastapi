import json
import base64

from facerec_module import (
    get_logger,
    FaceDetector,
    FaceEncoder,
    FaceVerifier,
    face_vector_from_base64_string,
    decode_image_from_base64_string
)

from fastapi import FastAPI, Request
import uvicorn

import cv2
import numpy as np

from pydantic import BaseModel, Field


app = FastAPI()

logger = get_logger()
face_detector = FaceDetector(face_confidence=0.995)
face_encoder = FaceEncoder()
face_verifier = FaceVerifier()


class FaceRect(BaseModel):
    left: float = Field(description="Relative left coordinate of the face bounding box")
    top: float = Field(description="Relative top coordinate of the the face bounding box")
    right: float = Field(description="Relative right coordinate of the face bounding box")
    bottom: float = Field(description="Relative bottom coordinate of the face bounding box")


class EncodeFaceRequest(BaseModel):
    image: str = Field(description="An image in base64 format")


class EncodeFaceResponse(BaseModel):
    success: bool
    message: str = Field(None, description="message if something has been gone wrong")
    face_rect: FaceRect = Field(None, description="bounding box of the face in format (x, y, width, height)")
    face_encoding: str = Field(None, description="base64 string with face descriptor")


class VerifyFaceRequest(BaseModel):
    image: str = Field(description="An image with a face to verify in base64 format")
    db_face_encoding: str = Field(description="A face descriptor from database in base64 format")


class VerifyFaceResponse(EncodeFaceResponse):
    verify_distance: float = Field(None, description="euclidean distance between 2 faces")


def face_encoding_func(request_json, response):
    image = decode_image_from_base64_string(request_json["image"])
    # print("image.shape:", image.shape)

    face_bboxes = face_detector.detect(image)
    if len(face_bboxes) == 0:
        response["message"] = "Faces not found"
    elif len(face_bboxes) > 1:
        response["message"] = f"Found {len(face_bboxes)} faces, expected 1 face"
    else:
        face_rect = face_bboxes[0]
        response["face_rect"] = face_rect

        face_encoding = face_encoder.encode(
            image[:, :, ::-1], [face_rect])[0]
        assert len(face_encoding) == 128, f"Wrong size of face encoding vector {len(face_encoding)}, expected 128"

        return face_encoding
    return None


@app.post("/api/v1/encode_face", response_model=EncodeFaceResponse)
async def encode_face_v1(item: EncodeFaceRequest):
    response = {"success": False}

    try:
        request = item.dict()

        face_encoding = face_encoding_func(request, response)
        if face_encoding is not None:
            face_base64 = base64.b64encode(face_encoding)
            response["face_encoding"] = face_base64.decode('utf-8')

            response["success"] = True
    except Exception as ex:
        logger.exception(f"Exception on encode_face_v1 occured")

    return EncodeFaceResponse(**response)


@app.post("/api/v1/compare_faces", response_model=VerifyFaceResponse)
async def compare_faces_v1(item: VerifyFaceRequest):
    response = {"success": False}

    try:
        request = item.dict()

        face_encoding = face_encoding_func(request, response)
        if face_encoding is not None:
            face_base64 = base64.b64encode(face_encoding)
            response["face_encoding"] = face_base64.decode('utf-8')

            db_face = face_vector_from_base64_string(request["db_face_encoding"])

            distance = face_verifier.verify([db_face], face_encoding)[0]
            response["verify_distance"] = distance

            response["success"] = True
    except Exception as ex:
        logger.exception(f"Exception on compare_faces_v1 occured")

    return VerifyFaceResponse(**response)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9876)
