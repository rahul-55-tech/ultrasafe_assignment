from functools import wraps
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from configurations.config import settings
from configurations.logger_config import logger

security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials) -> bool:
    """
    Verify if the provided token matches the expected token.

    Args:
        credentials: The authorization credentials containing the token

    Returns:
        bool: True if token is valid, False otherwise
    """
    if credentials.credentials == settings.secret_key:
        return True
    return False


async def get_current_token(request: Request) -> Optional[str]:
    """
    Extract the token from the Authorization header.

    Args:
        request: The FastAPI request object

    Returns:
        Optional[str]: The token if present, None otherwise
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    try:
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            return None
        return token
    except Exception as err:
        logger.error(
            f"Error while authentication process"
            f"with header : {auth_header}, ERROR; {str(err)}"
        )
        return None


def require_auth():
    """
    Decorator to require authentication for API endpoints.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request:
                raise HTTPException(status_code=401, detail="Request object not found")

            token = await get_current_token(request)
            if not token:
                raise HTTPException(status_code=401, detail="No token provided")

            if token != settings.secret_key:
                raise HTTPException(status_code=401, detail="Invalid token")

            return await func(*args, **kwargs)

        return wrapper

    return decorator
