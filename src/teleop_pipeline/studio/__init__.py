"""The desktop app: run the pipeline and read the result without a terminal."""

from .pipeline import STAGES, Runner
from .server import make_server, serve

__all__ = ["STAGES", "Runner", "make_server", "serve"]
