"""Command-line interface for vidscribe."""

import typer

app = typer.Typer(help="Local video transcription with CLI-provider correction.")


@app.callback()
def main() -> None:
    """Run vidscribe commands."""

