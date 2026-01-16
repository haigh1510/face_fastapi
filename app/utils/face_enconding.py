import argparse
import base64
import numpy as np
import time
from typing import Tuple

from facerec_module import (
    FaceDetector,
    decode_image_from_base64_string
)

from ..schemas import (
    FaceRect,
    EncodeFaceRequest,
    VerificationResult
)

from .error import facerec_assert
from .encoder import get_encoder


ENCODER_DTYPE = "float32"

init_params = {
    "detector": {
        "face_confidence": 0.9,
    },
    "encoder": {
        "encoder_name": "dlib",
    }
}

face_detector = FaceDetector(**init_params["detector"])
face_encoder = get_encoder(**init_params["encoder"])
max_distance_to_verify = face_encoder.verify_threshold()


def elapsed_seconds(start_tic):
    return time.time() - start_tic


def face_encoding_func(request: EncodeFaceRequest) -> Tuple[FaceRect, np.array]:
    image = decode_image_from_base64_string(request.image)

    facerec_assert(image.shape[2] == 3,
                   f"Unsupported number of image channels: {image.shape[2]}, expected: 3")

    face_bboxes = face_detector.detect(image)

    facerec_assert(len(face_bboxes) == 1, f"Found {len(face_bboxes)} faces, expected 1 face")

    face_rect = face_bboxes[0]

    face_encoding = face_encoder.encode(image, [face_rect], swapRB=False)[0]
    face_encoding = face_encoding.astype(ENCODER_DTYPE)

    facerec_assert(len(face_encoding) == face_encoder.encoding_length(),
                   f"Wrong size of face encoding vector {len(face_encoding)}, \
                    expected {face_encoder.encoding_length()}")

    return (face_rect, face_encoding)


def face_verify_func(db_face: np.array, face_encoding: np.array) -> VerificationResult:
    distance = face_encoder.verify([db_face], face_encoding)[0]

    return VerificationResult(
        distance=distance,
        max_distance_to_verify=max_distance_to_verify,
        verified=distance < max_distance_to_verify
    )
