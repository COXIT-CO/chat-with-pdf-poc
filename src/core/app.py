import aiohttp

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from core.repositories.repo_files import FilesRepository
from core.routers.router_base import BaseRouter
from core.routers.router_files import FilesRouter
from core.routers.router_mcpl import MCPLRouter


__all__ = ["App"]


class App(FastAPI):
    def __init__(
            self,
            files_repository: FilesRepository,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.http_session: aiohttp.ClientSession
        self.files_repository = files_repository

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
        # self.add_middleware(NoCacheMiddleware)  # type: ignore[arg-type]

    async def _startup_events(self):
        self.http_session = aiohttp.ClientSession()

        for router in self._routers():
            self.include_router(router)


    async def _shutdown_events(self):
        if self.http_session:
            await self.http_session.close()

    def _routers(self):
        return [
            BaseRouter(),
            MCPLRouter(
                self.http_session,
                self.files_repository,
            ),
            FilesRouter(
                self.files_repository,
            ),
        ]


# class NoCacheMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request, call_next):
#         response = await call_next(request)
#         response.headers["Cache-Control"] = "no-cache"
#         return response
