import datetime
import logging
from typing import List, Any
import uuid
from arklex.env.tools.tools import register_tool
from arklex.env.tools.types import Transcript

logger = logging.getLogger(__name__)

description = "Save the transcription of the conversation to the database. Pass the transcription of the whole conversation to this tool."

slots = [
    {
        "name": "transcription",
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "enum": ["user", "bot"],
                    "description": "The speaker of the utterance",
                },
                "text": {
                    "type": "string",
                    "description": "The text spoken by the speaker",
                },
            },
        },
        "prompt": "Please provide the transcription of the conversation",
        "required": True,
    }
]

outputs = []

errors = []


@register_tool(description, slots, outputs, lambda x: x not in errors)
def get_transcript(transcription: List[dict], **kwargs: Any) -> List[Transcript]:
    logger.info(f"Recieved transcription: {transcription}")

    transcripts = []
    start_time = datetime.datetime.now(datetime.timezone.utc)
    for t in transcription:
        if t.get("origin") and t.get("text"):
            transcripts.append(
                Transcript(
                    id=str(uuid.uuid4()),
                    text=t["text"],
                    origin=t["origin"],
                    created_at=start_time,
                )
            )
        else:
            logger.error(f"Transcription is missing origin or text: {t}")
        start_time += datetime.timedelta(seconds=1)

    return transcripts
