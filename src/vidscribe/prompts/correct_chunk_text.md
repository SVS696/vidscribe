You are correcting a local ASR transcript chunk from a video.

This is a text-only correction pass. Another pass will handle visual context from
video frames — focus exclusively on speech recognition errors.

Goal:
- Fix ASR recognition errors, punctuation, casing, and obvious grammar issues.
- Preserve the speaker's meaning, order, and tone.
- Keep speaker names consistent with the supplied speaker map.
- Do NOT invent visual details, screen labels, or on-screen text — a separate
  visual pass will handle frames.
- Return only valid JSON matching the schema below.

Transcript chunk:
{{ transcript }}

Known glossary:
{% if glossary %}
{% for term, meaning in glossary.items() %}
- {{ term }}: {{ meaning }}
{% endfor %}
{% else %}
- none
{% endif %}

Speaker map:
{% if speaker_map %}
{% for speaker_id, speaker_name in speaker_map.items() %}
- {{ speaker_id }}: {{ speaker_name }}
{% endfor %}
{% else %}
- none
{% endif %}

JSON schema:
{
  "segments": [
    {
      "start": 0.0,
      "end": 1.0,
      "speaker": "SPEAKER_00",
      "corrected_text": "Corrected text for this speaker turn."
    }
  ],
  "glossary_delta": {
    "new_or_corrected_term": "short explanation or canonical spelling"
  },
  "notes": "Brief uncertainty notes, or an empty string."
}

Output rules:
- Respond with exactly one JSON object and no surrounding markdown.
- The top-level object must contain exactly these keys: segments, glossary_delta, notes.
- Preserve turn order and speaker ids. Split output whenever the speaker changes.
- glossary_delta must be an object, even when empty.
- corrected_text values must not include timestamps unless they are present in the transcript.
