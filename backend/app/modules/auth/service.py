"""
AuthService — application use cases for authentication.
Contains all business logic; no HTTP concepts (Request/Response) enter here.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AuthenticationError,
    ConflictError,
    TokenInvalidError,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.repositories.user_repository import UserRepository


class AuthService:
    """
    Encapsulates:
        register()  — Create new user account.
        login()     — Verify credentials, issue token pair.
        refresh()   — Validate refresh token, issue new token pair.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._repo = UserRepository(session)

    async def register(
        self, full_name: str, email: str, password: str
    ) -> tuple[User, str, str]:
        """
        Create a new user.

        Raises:
            ConflictError: if email is already registered.

        Returns:
            Tuple of (user, access_token, refresh_token).
        """
        if await self._repo.email_exists(email):
            raise ConflictError(
                message=f"An account with email '{email}' already exists."
            )

        user = User(
            email=email,
            hashed_password=hash_password(password),
            full_name=full_name,
        )
        user = await self._repo.create(user)

        access_token = create_access_token(user.id, role=user.role)
        refresh_token = create_refresh_token(user.id)
        return user, access_token, refresh_token

    async def login(
        self, email: str, password: str
    ) -> tuple[User, str, str]:
        """
        Authenticate user with email/password.

        Raises:
            AuthenticationError: if credentials are invalid.

        Returns:
            Tuple of (user, access_token, refresh_token).
        """
        user = await self._repo.get_by_email(email)
        if user is None or user.hashed_password is None:
            raise AuthenticationError("Invalid email or password.")

        if not verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password.")

        if not user.is_active:
            raise AuthenticationError("This account has been deactivated.")

        access_token = create_access_token(user.id, role=user.role)
        refresh_token = create_refresh_token(user.id)
        return user, access_token, refresh_token

    async def refresh(self, refresh_token: str) -> tuple[User, str, str]:
        """
        Validate a refresh token and issue a new token pair.

        Raises:
            TokenInvalidError: if the token is malformed, expired, or wrong type.
            AuthenticationError: if the user no longer exists or is inactive.

        Returns:
            Tuple of (user, new_access_token, new_refresh_token).
        """
        try:
            payload = decode_token(refresh_token)
        except Exception:
            raise TokenInvalidError("Refresh token is invalid or expired.")

        if payload.get("type") != "refresh":
            raise TokenInvalidError("Provided token is not a refresh token.")

        user_id = payload.get("sub")
        if not user_id:
            raise TokenInvalidError("Refresh token is missing subject claim.")

        user = await self._repo.get_active_by_id(user_id)
        if user is None:
            raise AuthenticationError("User account not found or deactivated.")

        new_access = create_access_token(user.id, role=user.role)
        new_refresh = create_refresh_token(user.id)
        return user, new_access, new_refresh
