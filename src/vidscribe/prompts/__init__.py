"""Prompt rendering utilities."""

from __future__ import annotations

from importlib import resources
from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateError, meta


class PromptRenderError(ValueError):
    """Raised when a prompt cannot be rendered from supplied values."""


def render(name: str, **kwargs: Any) -> str:
    """Render a markdown prompt template by name.

    ``name`` can be passed with or without the ``.md`` suffix. Templates use
    strict undefined handling so missing slots fail before provider execution.
    """

    template_name = name if name.endswith(".md") else f"{name}.md"
    try:
        template_text = (
            resources.files(__package__).joinpath(template_name).read_text()
        )
    except FileNotFoundError as exc:
        raise PromptRenderError(f"unknown prompt template: {name}") from exc

    env = Environment(
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    parsed = env.parse(template_text)
    required = meta.find_undeclared_variables(parsed)
    missing = sorted(required.difference(kwargs))
    if missing:
        raise PromptRenderError(
            f"missing prompt template values for {template_name}: {', '.join(missing)}"
        )

    try:
        return env.from_string(template_text).render(**kwargs)
    except TemplateError as exc:
        raise PromptRenderError(
            f"failed to render prompt template {template_name}: {exc}"
        ) from exc


__all__ = ["PromptRenderError", "render"]
