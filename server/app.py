# server/app.py — OpenEnv multi-mode entry point
# This file is required by the openenv validator.
# It re-exports the FastAPI app from the main server module.

from customer_support_env.server import app

__all__ = ["app"]