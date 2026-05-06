You are performing a visual correction pass on an already text-corrected transcript chunk.

CRITICAL FIRST STEP: Before doing anything, open EVERY frame path listed below using
your Read tool. The frames are JPG screenshots from the video at the same timestamps
as the transcript.

Your job is NARROW — do NOT rewrite or restructure the transcript. Only:
1. Correct numbers, figures, or codes visible on screen that differ from the transcript.
2. Fix proper nouns, names, and technical terms where the canonical spelling is shown
   on screen (UI labels, document titles, lower-thirds, name badges, code identifiers).
3. Fill the `notes` field with a brief description of what is visible in the frames
   (what is on screen, what the speaker is demonstrating, any relevant context).
4. Add entries to `glossary_delta` when a frame reveals a corrected or canonical term.
{% if enable_screen_context %}
5. ADDITIONAL TASK: describe visual events shown across the frames that are NOT already
   verbalized in the transcript:
   - screen/tab switches
   - cell selections, highlighting
   - mouse-driven scrolls or zooms
   - application opens/closes
   - modal dialogs

   Return them in `screen_events` array in the JSON. Each ts should be within the
   chunk's time window. Skip when frames show nothing notable beyond what speech
   already conveys.
{% endif %}

Keep all speech-level corrections from the previous pass intact unless they directly
contradict something clearly visible in a frame.

Original ASR transcript:
{{ asr_transcript }}

Text-corrected transcript (output of previous pass — preserve these corrections):
{{ text_corrected_transcript }}

Frame paths (open ALL of these with Read tool before answering):
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
{% if enable_screen_context %}
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
  "notes": "What is visible in the frames: screen content, demonstrated UI, context.",
  "screen_events": [
    {"ts": 12.3, "description": "Switched from 'Data' to 'Main' tab"},
    {"ts": 45.0, "description": "Selected row 'Бабич Алексей' with value 544"}
  ]
}
{% else %}
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
  "notes": "What is visible in the frames: screen content, demonstrated UI, context."
}
{% endif %}

Output rules:
- Respond with exactly one JSON object and no surrounding markdown.
{% if enable_screen_context %}
- The top-level object must contain exactly these keys: segments, glossary_delta, notes, screen_events.
- screen_events must be an array, even when empty. Each item: {"ts": float, "description": string}.
{% else %}
- The top-level object must contain exactly these keys: segments, glossary_delta, notes.
{% endif %}
- Preserve turn order and speaker ids from the text-corrected transcript.
- glossary_delta must be an object, even when empty.
- corrected_text values must not include timestamps unless they are present in the transcript.
