from typing import List, Tuple, TYPE_CHECKING
from pathlib import Path
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModelSettings

if TYPE_CHECKING:
    from kg_gen.kg_gen import ModelConfig


_TEXT_SYSTEM_PROMPT = """\
Extract key entities from the source text. Extracted entities are subjects or objects.
This is for an extraction task, please be THOROUGH and accurate to the reference text.

Consider both explicit entities and participants in the conversation when applicable.
Preserve entity names in their original script (e.g. Chinese, Arabic, Japanese — do not translate).\
"""

_CONVERSATION_SYSTEM_PROMPT = """\
Extract key entities from the conversation. Extracted entities are subjects or objects.
Consider both explicit entities and participants in the conversation.
This is for an extraction task, please be THOROUGH and accurate.

Preserve entity names in their original script (e.g. Chinese, Arabic, Japanese — do not translate).\
"""


class EntitiesResponse(BaseModel):
    """A thorough list of key entities extracted from the source text."""

    entities: List[str]


def _build_agent(model_config: "ModelConfig", is_conversation: bool) -> Agent:
    system_prompt = _CONVERSATION_SYSTEM_PROMPT if is_conversation else _TEXT_SYSTEM_PROMPT
    pai_model = model_config.build()

    settings: ModelSettings = OpenAIChatModelSettings(
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        **(
            {"openai_reasoning_effort": model_config.reasoning_effort}
            if model_config.reasoning_effort
            else {}
        ),
    )

    return Agent(
        pai_model,
        output_type=EntitiesResponse,
        system_prompt=system_prompt,
        model_settings=settings,
    )


def get_entities(
    input_data: str,
    is_conversation: bool = False,
    model_config: "ModelConfig" = None,
) -> Tuple[List[str], object]:
    """Extract entities from text or conversation.

    Returns:
        Tuple of (list of entity strings, RunUsage for token accounting)
    """
    agent = _build_agent(model_config, is_conversation)

    user_prompt = f"""
Here is the {'conversation' if is_conversation else 'text'} to extract entities from:

<{'conversation' if is_conversation else 'article'}>
{input_data}
</{'conversation' if is_conversation else 'article'}>
"""

    result = agent.run_sync(user_prompt)
    return result.output.entities, result.usage
