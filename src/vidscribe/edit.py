"""Transcript editing helpers — apply LLM-driven text edits to a markdown transcript."""

from __future__ import annotations

from typing import Sequence

from vidscribe.provider import Provider


EDIT_SYSTEM_PROMPT = """\
You will edit a markdown transcript file. The user provided one or more
instructions in Russian or English.

CONSTRAINTS:
- Preserve markdown structure (## headers with timestamps, blockquotes for
  screen events, regular paragraphs for speech).
- Apply the instructions exactly. Do not add new content.
- Return the edited markdown as plain text, no surrounding fences.

INSTRUCTIONS:
{instructions}

TRANSCRIPT:
{transcript}

Return the edited markdown."""


def apply_edits(
    transcript_text: str,
    instructions: Sequence[str],
    provider: Provider,
    *,
    timeout: int = 600,
) -> str:
    """Apply one or more edit instructions to *transcript_text* via *provider*.

    Parameters
    ----------
    transcript_text:
        The full markdown transcript to edit.
    instructions:
        One or more edit instructions (Russian or English).
    provider:
        An instantiated :class:`~vidscribe.provider.Provider`.
    timeout:
        Provider timeout in seconds.

    Returns
    -------
    str
        The edited markdown text.
    """
    numbered = "\n".join(
        f"{idx}. {instruction.strip()}"
        for idx, instruction in enumerate(instructions, start=1)
        if instruction.strip()
    )
    prompt = EDIT_SYSTEM_PROMPT.format(
        instructions=numbered,
        transcript=transcript_text,
    )
    response = provider.correct(prompt, frame_paths=[], timeout=timeout)
    # The provider may return the edited text directly (not as JSON).
    # Try raw_json["text"] / response.text, fall back to stripping fences.
    edited = response.text.strip()
    if not edited:
        # Last resort: join all string values from raw_json
        edited = " ".join(
            str(v)
            for v in response.raw_json.values()
            if isinstance(v, str)
        ).strip()
    # Strip markdown fences if the model wrapped the output anyway
    if edited.startswith("```"):
        lines = edited.splitlines()
        # drop first line (```markdown / ``` etc.) and last line (```)
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        edited = "\n".join(inner)
    return edited
