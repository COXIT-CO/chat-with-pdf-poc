from typing import Optional

import aiohttp
import openai

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from core.repositories.repo_files import FilesRepository
from core.routers.router_experiments import ExperimentsRouter
from vectors.repositories.repo_milvus import MilvusRepository
from vectors.repositories.repo_redis import RedisRepository
from core.routers.router_base import BaseRouter
from core.routers.router_files import FilesRouter
from core.routers.router_mcpl import MCPLRouter


__all__ = ["App"]


class App(FastAPI):
    def __init__(
            self,
            files_repository: FilesRepository,
            redis_repository: Optional[RedisRepository],
            milvus_repository: Optional[MilvusRepository],
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.http_session: aiohttp.ClientSession
        self.files_repository = files_repository
        self.redis_repository = redis_repository
        self.milvus_repository = milvus_repository

        self.openai = openai.OpenAI()

        self._setup_middlewares()
        self.add_event_handler("startup", self._startup_events)
        self.add_event_handler("shutdown", self._shutdown_events)

    def _setup_middlewares(self):
        self.add_middleware(
            CORSMiddleware,  # type: ignore[arg-type]
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.add_middleware(NoCacheMiddleware)  # type: ignore[arg-type]

    async def _startup_events(self):
        self.http_session = aiohttp.ClientSession()

        for router in self._routers():
            self.include_router(router)


    async def _shutdown_events(self):
        if self.http_session:
            await self.http_session.close()
        if self.redis_repository is not None:
            self.redis_repository.close()

    def _routers(self):
        return [
            BaseRouter(),
            MCPLRouter(
                self.http_session,
                self.files_repository,
                self.redis_repository,
                self.milvus_repository,

                self.openai,
            ),
            FilesRouter(
                self.files_repository,
            ),
            ExperimentsRouter(),
        ]


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response
