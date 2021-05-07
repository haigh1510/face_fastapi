import base64
import time

from facerec_module import (
    get_logger,
    FaceDetector,
    face_vector_from_base64_string,
    decode_image_from_base64_string
)

from fastapi import FastAPI
import uvicorn

from data_types import (
    EncodeFaceRequest, EncodeFaceResponse,
    VerifyFaceRequest,
    VerificationResult, VerifyFaceResponse
)
from get_encoder import get_encoder
from config import init_params


app = FastAPI()

logger = get_logger()
face_detector = FaceDetector(**init_params["detector"])
face_encoder = get_encoder(**init_params["encoder"])

max_distance_to_verify = face_encoder.verify_threshold()
face_encoding_length = face_encoder.encoding_length()

ENCODER_DTYPE = "float32"


def elapsed_seconds(start_tic):
    return time.time() - start_tic


def face_encoding_func(request_json, response):
    try:
        image = decode_image_from_base64_string(request_json["image"])
        # print("image.shape:", image.shape)
    except Exception as ex:
        response["message"] = str(ex)
        return None

    face_bboxes = face_detector.detect(image)
    if len(face_bboxes) == 0:
        response["message"] = "Faces not found"
    elif len(face_bboxes) > 1:
        response["message"] = f"Found {len(face_bboxes)} faces, expected 1 face"
    else:
        face_rect = face_bboxes[0]
        response["face_rect"] = face_rect

        face_encoding = face_encoder.encode(image, [face_rect])[0]
        face_encoding = face_encoding.astype(ENCODER_DTYPE)

        assert len(face_encoding) == face_encoding_length, \
            f"Wrong size of face encoding vector {len(face_encoding)}, expected {face_encoding_length}"

        return face_encoding
    return None


@app.post("/api/v1/encode_face", response_model=EncodeFaceResponse)
async def encode_face_v1(item: EncodeFaceRequest):
    tic = time.time()

    response = {"success": False}
    try:
        request = item.dict()

        face_encoding = face_encoding_func(request, response)
        if face_encoding is not None:
            face_base64 = base64.b64encode(face_encoding)
            response["face_encoding"] = face_base64.decode('utf-8')

            response["success"] = True
    except Exception as ex:
        logger.exception(f"Exception on encode_face_v1 occured:", str(ex))

    response["seconds"] = elapsed_seconds(tic)

    return EncodeFaceResponse(**response)


@app.post("/api/v1/compare_faces", response_model=VerifyFaceResponse)
async def compare_faces_v1(item: VerifyFaceRequest):
    tic = time.time()

    response = {"success": False}
    try:
        request = item.dict()

        face_encoding = face_encoding_func(request, response)
        if face_encoding is not None:
            face_base64 = base64.b64encode(face_encoding)
            response["face_encoding"] = face_base64.decode('utf-8')

            db_face = face_vector_from_base64_string(request["db_face_encoding"], dtype=ENCODER_DTYPE)

            distance = face_encoder.verify([db_face], face_encoding)[0]
            verified = distance < max_distance_to_verify

            response["verification"] = VerificationResult(
                distance=distance,
                max_distance_to_verify=max_distance_to_verify,
                verified=verified).dict()

            response["success"] = True
    except Exception as ex:
        logger.exception(f"Exception on compare_faces_v1 occured:", str(ex))

    response["seconds"] = elapsed_seconds(tic)

    return VerifyFaceResponse(**response)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9876)
