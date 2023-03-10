import argparse
import base64
import os
import time

from facerec_module import (
    get_logger,
    FaceDetector,
    face_vector_from_base64_string,
    decode_image_from_base64_string
)
from .error import facerec_assert, FacerecError

from fastapi import FastAPI
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .data_types import (
    EncodeFaceRequest, EncodeFaceResponse,
    VerifyFaceRequest,
    VerificationResult, VerifyFaceResponse,
    TokenResponse,
)

from .get_encoder import get_encoder
from .config import init_params

from .jwt import TokenData, get_access_token, validate_token


app = FastAPI()

# https://fastapi.tiangolo.com/tutorial/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

logger = get_logger()
face_detector = FaceDetector(**init_params["detector"])
face_encoder = get_encoder(**init_params["encoder"])

max_distance_to_verify = face_encoder.verify_threshold()
face_encoding_length = face_encoder.encoding_length()

ENCODER_DTYPE = "float32"


def elapsed_seconds(start_tic):
    return time.time() - start_tic


def face_encoding_func(request_json, response):
    image = decode_image_from_base64_string(request_json["image"])

    facerec_assert(image.shape[2] == 3,
                   f"Unsupported number of image channels: {image.shape[2]}, expected: 3")

    face_bboxes = face_detector.detect(image)

    facerec_assert(len(face_bboxes) == 1, f"Found {len(face_bboxes)} faces, expected 1 face")

    face_rect = face_bboxes[0]
    response["face_rect"] = face_rect

    face_encoding = face_encoder.encode(image, [face_rect])[0]
    face_encoding = face_encoding.astype(ENCODER_DTYPE)

    facerec_assert(len(face_encoding) == face_encoding_length,
                   f"Wrong size of face encoding vector {len(face_encoding)}, expected {face_encoding_length}")

    return face_encoding


@app.post("/api/v1/encode_face", response_model=EncodeFaceResponse)
# @app.post("/api/v1/encode_face", response_model=EncodeFaceResponse, dependencies=[Depends(validate_token)])
async def encode_face_v1(item: EncodeFaceRequest):
    tic = time.time()

    response = {"success": False}
    try:
        request = item.dict()

        face_encoding = face_encoding_func(request, response)

        face_base64 = base64.b64encode(face_encoding)
        response["face_encoding"] = face_base64.decode('utf-8')

        response["success"] = True
    except FacerecError as ex:
        response["message"] = str(ex)
    except Exception as ex:
        logger.exception(f"Exception on encode_face_v1 occured:", str(ex))

    response["seconds"] = elapsed_seconds(tic)

    return EncodeFaceResponse(**response)


@app.post("/api/v1/compare_faces", response_model=VerifyFaceResponse)
# @app.post("/api/v1/compare_faces", response_model=VerifyFaceResponse, dependencies=[Depends(validate_token)])
async def compare_faces_v1(item: VerifyFaceRequest):
    tic = time.time()

    response = {"success": False}
    try:
        request = item.dict()

        face_encoding = face_encoding_func(request, response)

        face_base64 = base64.b64encode(face_encoding)
        response["face_encoding"] = face_base64.decode('utf-8')

        db_face = face_vector_from_base64_string(request["db_face_encoding"], dtype=ENCODER_DTYPE)

        facerec_assert(len(db_face) == len(face_encoding),
                       f"Incorrect size of db_face: {len(db_face)}, expected: {len(face_encoding)}")

        distance = face_encoder.verify([db_face], face_encoding)[0]
        verified = distance < max_distance_to_verify

        response["verification"] = VerificationResult(
            distance=distance,
            max_distance_to_verify=max_distance_to_verify,
            verified=verified).dict()

        response["success"] = True
    except FacerecError as ex:
        response["message"] = str(ex)
    except Exception as ex:
        logger.exception(f"Exception on compare_faces_v1 occured:", str(ex))

    response["seconds"] = elapsed_seconds(tic)

    return VerifyFaceResponse(**response)


@app.post("/api/v1/token", response_model=TokenResponse)
async def get_token_v1(item: TokenData):
    response = {"success": False}
    try:
        request = item.dict()

        user_id = request["user_id"]
        # TODO validate user_id

        response["token"] = get_access_token(user_id)

        response["success"] = True
    except Exception as ex:
        logger.exception(f"Exception on token occured:", str(ex))

    return TokenResponse(**response)
