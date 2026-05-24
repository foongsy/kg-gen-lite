from typing import List, Tuple, TYPE_CHECKING
import logging
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModelSettings

if TYPE_CHECKING:
    from kg_gen.kg_gen import ModelConfig

logger = logging.getLogger(__name__)

_RELATIONS_SYSTEM_PROMPT_TEMPLATE = """\
Extract subject-predicate-object triples from the source text.
Subject and object must be exact matches to items in the provided entities list.
Entities were previously extracted from the same source text.
This is for an extraction task, please be thorough, accurate, and faithful to the reference text.

Preserve all entity names and predicates in their original script — do not translate them.
{context}\
"""

_CONVERSATION_RELATIONS_SYSTEM_PROMPT_TEMPLATE = """\
Extract subject-predicate-object triples from the conversation, including:
1. Relations between concepts discussed
2. Relations between speakers and concepts (e.g. user asks about X)
3. Relations between speakers (e.g. assistant responds to user)

Subject and object must be exact matches to items in the provided entities list.
Entities were previously extracted from the same source text.
This is for an extraction task, please be thorough, accurate, and faithful to the reference text.

Preserve all entity names and predicates in their original script — do not translate them.
{context}\
"""


class RelationItem(BaseModel):
    """A single subject-predicate-object triple."""

    subject: str
    predicate: str
    object: str


class RelationsResponse(BaseModel):
    """A thorough list of subject-predicate-object triples from the source text."""

    relations: List[RelationItem]


def _filter_entities(entities: List[str]) -> List[str]:
    """Remove entities that contain double-quotes (rejected by some APIs)."""
    return [e for e in entities if '"' not in e]


def _build_agent(
    model_config: "ModelConfig",
    is_conversation: bool,
    context: str,
    entities: List[str],
) -> Agent:
    """Build a PydanticAI Agent for relation extraction.

    Entities are captured in the validator closure before the agent is
    returned, so there is no need to mutate agent state after construction.
    """
    template = (
        _CONVERSATION_RELATIONS_SYSTEM_PROMPT_TEMPLATE
        if is_conversation
        else _RELATIONS_SYSTEM_PROMPT_TEMPLATE
    )
    system_prompt = template.format(context=context)
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

    agent: Agent[None, RelationsResponse] = Agent(
        pai_model,
        output_type=RelationsResponse,
        system_prompt=system_prompt,
        model_settings=settings,
        retries=2,
    )

    entities_set: set[str] = set(entities)

    @agent.output_validator
    def validate_entities(output: RelationsResponse) -> RelationsResponse:
        """Reject the response if any subject/object is not in the entity list."""
        bad = [
            r
            for r in output.relations
            if r.subject not in entities_set or r.object not in entities_set
        ]
        if bad:
            invalid = {r.subject for r in bad if r.subject not in entities_set} | {
                r.object for r in bad if r.object not in entities_set
            }
            raise ModelRetry(
                f"The following subjects/objects are not in the entities list: {sorted(invalid)}. "
                "Ensure every subject and object is an exact match to an entity in the list."
            )
        return output

    return agent


def get_relations(
    input_data: str,
    entities: List[str],
    is_conversation: bool = False,
    context: str = "",
    *,
    model_config: "ModelConfig",
) -> Tuple[List[Tuple[str, str, str]], object]:
    """Extract subject-predicate-object relations from text.

    Args:
        input_data: Text or conversation string to extract relations from.
        entities: List of entities previously extracted from the same text.
        is_conversation: Set True when input_data is a formatted conversation.
        context: Optional description of the data context.
        model_config: Required — holds model name, API credentials, and settings.

    Returns:
        Tuple of (list of (subject, predicate, object) triples, RunUsage)
    """
    entities = _filter_entities(entities)
    entities_str = "\n".join(f"- {e}" for e in entities)

    agent = _build_agent(model_config, is_conversation, context, entities)

    tag = "conversation" if is_conversation else "text"
    user_prompt = (
        f"\nHere is the list of entities that were previously extracted from the source text:\n\n"
        f"<entities>\n{entities_str}\n</entities>\n\n"
        f"Here is the {tag} to analyze:\n\n"
        f"<{tag}>\n{input_data}\n</{tag}>\n"
    )

    result = agent.run_sync(user_prompt)
    triples = [(r.subject, r.predicate, r.object) for r in result.output.relations]
    return triples, result.usage
