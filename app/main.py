import argparse
import base64
import numpy as np
import time
from typing import Tuple

from fastapi import FastAPI
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware

from facerec_module import (
    get_logger,
    FaceDetector,
    face_vector_from_base64_string,
    decode_image_from_base64_string
)

from .schemas import (
    EncodeFaceRequest,
    EncodeFaceResponse,
    VerifyFaceRequest,
    VerificationResult,
    VerifyFaceResponse,
    TokenData,
    TokenResponse,
)

from .utils import (
    face_encoding_func,
    face_verify_func,
    elapsed_seconds,
    ENCODER_DTYPE,
    FacerecError,
)

from .jwt import get_access_token, validate_token


logger = get_logger()
app = FastAPI()

# https://fastapi.tiangolo.com/tutorial/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/api/v1/encode_face", response_model=EncodeFaceResponse)
# @app.post("/api/v1/encode_face", response_model=EncodeFaceResponse, dependencies=[Depends(validate_token)])
async def encode_face_v1(item: EncodeFaceRequest):
    tic = time.time()

    try:
        face_data = face_encoding_func(item)

        return EncodeFaceResponse(
            face_rect=face_data[0],
            face_encoding=base64.b64encode(face_data[1]).decode('utf-8'),
            success=True,
            seconds=elapsed_seconds(tic)
        )
    except FacerecError as ex:
        logger.exception(f"Exception on encode_face_v1 occured:", str(ex))

        return EncodeFaceResponse(
            success=False,
            message=str(ex),
            seconds=elapsed_seconds(tic)
        )


@app.post("/api/v1/compare_faces", response_model=VerifyFaceResponse)
# @app.post("/api/v1/compare_faces", response_model=VerifyFaceResponse, dependencies=[Depends(validate_token)])
async def compare_faces_v1(item: VerifyFaceRequest):
    tic = time.time()

    try:
        face_encoding = face_encoding_func(item)[1]

        db_face = face_vector_from_base64_string(item.db_face_encoding, dtype=ENCODER_DTYPE)

        if (len(db_face) != len(face_encoding)):
            raise FacerecError(
                f"Incorrect size of db_face: {len(db_face)}, expected: {len(face_encoding)}"
            )

        verification_result = face_verify_func(db_face, face_encoding)

        return VerifyFaceResponse(
            success=True,
            verification=verification_result,
            seconds=elapsed_seconds(tic)
        )
    except FacerecError as ex:
        logger.exception(f"Exception on compare_faces_v1 occured:", str(ex))

        return VerifyFaceResponse(
            success=False,
            seconds=elapsed_seconds(tic)
        )


@app.post("/api/v1/token", response_model=TokenResponse)
async def get_token_v1(item: TokenData):
    try:
        user_id = item.user_id
        # TODO validate user_id

        return TokenResponse(
            token=get_access_token(user_id),
            success=True
        )
    except Exception as ex:
        logger.exception(f"Exception on token occured:", str(ex))

        return TokenResponse(
            success=False,
            message=str(ex)
        )
