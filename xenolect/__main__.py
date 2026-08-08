"""Allow ``python -m xenolect`` as a console-script fallback."""

from xenolect.cli.main import app


if __name__ == "__main__":
    app()
