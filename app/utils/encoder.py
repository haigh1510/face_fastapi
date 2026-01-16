from facerec_module import FaceEncoderDlib


encoders = {
    'dlib': FaceEncoderDlib,
}


def get_encoder(encoder_name: str):
    try:
        face_encoder = encoders[encoder_name]
    except KeyError:
        raise ValueError(f"Invalid encoder name {encoder_name}")

    return face_encoder()
