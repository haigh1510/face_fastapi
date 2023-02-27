import io
import base64
import numpy as np
import cv2


def face_vector_from_base64_string(base64_str, dtype="float32"):
    data = base64_str.encode("utf-8")
    return np.frombuffer(base64.b64decode(data), dtype=dtype)


def _read_cv2_image(binaryimg, flags):
    with io.BytesIO(binaryimg) as stream:
        image = np.asarray(bytearray(stream.read()), dtype="uint8")

    image = cv2.imdecode(image, flags)

    return image


def decode_image_from_base64_string(base64_str, flags=cv2.IMREAD_COLOR):
    image_base64 = base64_str.encode("utf-8")
    image_data = base64.b64decode(image_base64)

    return _read_cv2_image(image_data, flags)
