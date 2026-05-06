import pytest

from vidscribe.prompts import PromptRenderError, render


def test_correct_chunk_prompt_renders_json_contract() -> None:
    rendered = render(
        "correct_chunk",
        transcript="SPEAKER_00: привет алиса",
        frame_paths=["/tmp/frame-001.jpg"],
        glossary={"OpenAI": "canonical spelling"},
        speaker_map={"SPEAKER_00": "Иван"},
    )

    assert "SPEAKER_00: привет алиса" in rendered
    assert "/tmp/frame-001.jpg" in rendered
    assert '"segments"' in rendered
    assert '"corrected_text"' in rendered
    assert '"glossary_delta"' in rendered
    assert "Respond with exactly one JSON object" in rendered


def test_identify_speakers_prompt_renders_context() -> None:
    rendered = render(
        "identify_speakers.md",
        transcript="SPEAKER_01: Алиса, расскажи про демо.",
        speakers=["SPEAKER_00", "SPEAKER_01"],
        frame_paths=[],
    )

    assert "SPEAKER_01: Алиса, расскажи про демо." in rendered
    assert "- SPEAKER_00" in rendered
    assert "- none" in rendered
    assert '"speakers"' in rendered


def test_render_detects_missing_slots_before_provider_call() -> None:
    with pytest.raises(PromptRenderError, match="speaker_map"):
        render(
            "correct_chunk",
            transcript="text",
            frame_paths=[],
            glossary={},
        )


def test_render_rejects_unknown_template() -> None:
    with pytest.raises(PromptRenderError, match="unknown prompt template"):
        render("missing")
