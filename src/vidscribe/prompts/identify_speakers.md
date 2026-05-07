You are identifying speaker names in a diarized video transcript.

Goal:
- Map diarization labels like SPEAKER_00 to real names when the evidence is strong.
- Use transcript clues such as greetings, direct address, introductions, and questions like "X, tell us".
- Use frame paths as visual context for on-screen names in Zoom, Teams, slides, lower thirds, name badges,
  app login names (e.g. Excel user, Windows taskbar), or active window titles.
- If a speaker cannot be identified confidently, use a stable fallback like s00, s01, s02.
- Return only valid JSON.

CRITICAL — distinguish a speaker from a mentioned name:
- A speaker is the person whose voice produces the segments labelled with that diarization ID.
  Look at frames captured near those segments — UI logins, Zoom name tags, active windows, and
  name badges often reveal who is speaking.
- A mentioned name is a name that appears in the speech itself (e.g. "let's ask Andrew",
  "Andrew's report", "as Алексей said") — do NOT assign it as the speaker identity.
- If visual context is absent or insufficient to confirm identity → return null (the pipeline will
  substitute a fallback label such as s00/s01). Never guess from voice, gender, or appearance.

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
