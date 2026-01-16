from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: str


class TokenResponse(BaseModel):
    success: bool
    message: str = Field(None, description="message if something has gone wrong")
    token: Token = Field(None)
