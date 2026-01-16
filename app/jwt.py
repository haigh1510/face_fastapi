from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from .schemas import Token, TokenData


JWT_SECRET_KEY = "f580913fa68903c93a507c42e0136c1408e7700af9735165e87767aa5412beb9"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10

security = HTTPBearer()


def get_access_token(user_id: str) -> Token:
    expires = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "user_id": user_id,
        "expires": expires.timestamp()
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    return Token(access_token=token, token_type="bearer")


async def validate_token(
    auth: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    try:
        payload = jwt.decode(
            auth.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication error: invalid token")
    
    if datetime.utcnow().timestamp() > float(payload["expires"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication error: token expired")

    return TokenData(**payload)
