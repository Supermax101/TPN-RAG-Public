"""
FastAPI application factory.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routes import router
from .dependencies import check_services_health
from .schemas import HealthResponse
from ..config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    print(f"RAG API starting...")
    print(f"Data directory: {settings.data_dir}")
    print(f"ChromaDB directory: {settings.chromadb_dir}")
    yield
    print("RAG API shutting down...")


def create_app() -> FastAPI:
    """Creates and configures FastAPI application."""
    application = FastAPI(
        title="RAG API",
        description="Document retrieval and question answering API",
        version="2.0.0",
        lifespan=lifespan
    )
    
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        # Wildcard origin with credentials is invalid in browsers.
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    application.include_router(router)
    
    @application.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        try:
            services = await check_services_health()
            all_healthy = all(services.values())
            return HealthResponse(
                status="healthy" if all_healthy else "degraded",
                version="2.0.0",
                services=services
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    
    @application.get("/")
    async def root():
        return {
            "name": "RAG API",
            "version": "2.0.0",
            "endpoints": {
                "health": "/health",
                "search": "/api/v1/search",
                "ask": "/api/v1/ask",
                "stats": "/api/v1/stats"
            },
            "docs": "/docs"
        }
    
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
