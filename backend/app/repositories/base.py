"""
Generic async repository base.
All concrete repositories extend BaseRepository[ModelT].
"""
from typing import Any, Generic, List, Optional, Type, TypeVar
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """
    Provides generic CRUD operations for an SQLAlchemy model.

    Subclasses declare:
        model: Type[ModelT] = MyModel

    Example:
        class UserRepository(BaseRepository[User]):
            model = User
    """

    model: Type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def session(self) -> AsyncSession:
        return self._session

    async def get_by_id(self, entity_id: UUID) -> Optional[ModelT]:
        """Return the entity with the given primary key or None."""
        result = await self._session.execute(
            select(self.model).where(self.model.id == entity_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 20, offset: int = 0) -> List[ModelT]:
        """Return a paginated list of all entities."""
        result = await self._session.execute(
            select(self.model).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def create(self, obj: ModelT) -> ModelT:
        """Persist a new entity and return it with DB-generated fields populated."""
        self._session.add(obj)
        await self._session.flush()   # Populate id / defaults without committing
        await self._session.refresh(obj)
        return obj

    async def update(self, obj: ModelT, **kwargs: Any) -> ModelT:
        """Apply kwargs as attribute updates and flush to DB."""
        for key, value in kwargs.items():
            setattr(obj, key, value)
        await self._session.flush()
        await self._session.refresh(obj)
        return obj

    async def delete(self, obj: ModelT) -> None:
        """Hard-delete the entity from the database."""
        await self._session.delete(obj)
        await self._session.flush()

    async def exists(self, entity_id: UUID) -> bool:
        """Return True if an entity with the given id exists."""
        result = await self._session.execute(
            select(self.model.id).where(self.model.id == entity_id)
        )
        return result.scalar_one_or_none() is not None
