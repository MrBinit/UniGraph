from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RouteMatchingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if response.status_code not in (404, 405):
            return response

        detail = (
            "No route matched this path."
            if response.status_code == 404
            else "Method is not allowed for this route."
        )
        return JSONResponse(
            status_code=response.status_code,
            content={
                "detail": detail,
                "method": request.method,
                "path": request.url.path,
            },
        )
