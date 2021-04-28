from pydantic import BaseModel, Field


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
    face_rect: FaceRect = Field(None, description="bounding box of the face")
    face_encoding: str = Field(None, description="base64 string with face descriptor")


class VerifyFaceRequest(BaseModel):
    image: str = Field(description="An image with a face to verify in base64 format")
    db_face_encoding: str = Field(description="A face descriptor from database in base64 format")


class VerifyFaceResponse(EncodeFaceResponse):
    verify_distance: float = Field(None, description="euclidean distance between 2 faces")
