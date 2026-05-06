You are identifying speaker names in a diarized video transcript.

Goal:
- Map diarization labels like SPEAKER_00 to real names when the evidence is strong.
- Use transcript clues such as greetings, direct address, introductions, and questions like "X, tell us".
- Use frame paths as visual context for on-screen names in Zoom, Teams, slides, or lower thirds.
- If a speaker cannot be identified confidently, use a stable fallback like s00, s01, s02.
- Return only valid JSON.

Transcript excerpts:
{{ transcript }}

Speakers to identify:
{% for speaker_id in speakers %}
- {{ speaker_id }}
{% endfor %}

Frame paths:
{% for frame_path in frame_paths %}
- {{ frame_path }}
{% else %}
- none
{% endfor %}

JSON schema:
{
  "speakers": {
    "SPEAKER_00": "Name or fallback",
    "SPEAKER_01": "Name or fallback"
  },
  "notes": "Brief evidence or uncertainty notes."
}

Output rules:
- Respond with exactly one JSON object and no surrounding markdown.
- Include every requested speaker exactly once under speakers.
- Do not guess names from voice, gender, age, appearance, or unsupported context.
- Prefer fallbacks over weak guesses.
