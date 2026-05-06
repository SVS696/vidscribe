You are correcting a local ASR transcript chunk from a video.

Goal:
- Preserve the speaker's meaning, order, and tone.
- Fix ASR recognition errors, punctuation, casing, and obvious grammar issues.
- Use frame paths only as visual context; do not invent facts that are not supported.
- Keep speaker names consistent with the supplied speaker map.
- Return only valid JSON matching the schema below.

Transcript chunk:
{{ transcript }}

Frame paths:
{% for frame_path in frame_paths %}
- {{ frame_path }}
{% else %}
- none
{% endfor %}

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
