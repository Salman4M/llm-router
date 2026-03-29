from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker,create_async_engine

from core.config import load_config
from core.recorder import Recorder
from models.request import Base
from router.proxy import Proxy
from routes.router import router


def _db_url()->str:
    import os
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DTABASE_URLA environment variable is not set")
    
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://","postgresql+asyncpg://",1)
    if url.startswith("postgres://"):
        url = url.replace("postgres://","postgresql+asyncpg://", 1)
    return url


@asynccontextmanager
async def lifespan(app:FastAPI):
    config = load_config("config.yaml")

    engine = create_async_engine(_db_url(),echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    
    app.state.config = config
    app.state.proxy = Proxy(config)
    app.state.recorder = Recorder(session_factory, config)

    yield
    

    await engine.dispose()


app = FastAPI(title="llm-router",lifespan=lifespan)
app.include_router
