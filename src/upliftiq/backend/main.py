"""ASGI entry point: ``uvicorn upliftiq.backend.main:app``."""

from upliftiq.backend.app import create_app

app = create_app()
