"""API and web interface modules."""

try:
    from orgnet.api.app import create_app

    __all__ = ["create_app"]
except ImportError:
    # Flask is an optional dependency
    create_app = None
    __all__ = []
