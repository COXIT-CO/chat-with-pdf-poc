from fastapi import APIRouter
from src.custom_get_scalar_api_reference import get_scalar_api_reference as custom_get_scalar_api_reference


class BaseRouter(APIRouter):
    def __init__(self):
        super().__init__()

        self.add_api_route("/docs", self._scalar, methods=["GET"], include_in_schema=False)

    async def _scalar(self):
        try:
            from scalar_fastapi import get_scalar_api_reference
            return get_scalar_api_reference(
                openapi_url="/v1/openapi.json",
                title="Doc RAG Api Ref",
            )
        except TypeError:
            return custom_get_scalar_api_reference(
                openapi_url="/v1/openapi.json",
                title="Doc RAG Api Ref",
            )
