from typing import List, Tuple, Optional, TYPE_CHECKING
from pathlib import Path
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


def _build_agent(model_config: "ModelConfig", is_conversation: bool, context: str) -> Agent:
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

    entities_set: set[str] = set()

    @agent.output_validator
    def validate_entities(output: RelationsResponse) -> RelationsResponse:
        """Reject the response if any subject/object is not in the entity list."""
        bad = [
            r
            for r in output.relations
            if r.subject not in entities_set or r.object not in entities_set
        ]
        if bad:
            bad_subjects = {r.subject for r in bad if r.subject not in entities_set}
            bad_objects = {r.object for r in bad if r.object not in entities_set}
            invalid = bad_subjects | bad_objects
            raise ModelRetry(
                f"The following subjects/objects are not in the entities list: {sorted(invalid)}. "
                "Ensure every subject and object is an exact match to an entity in the list."
            )
        return output

    # Attach entities_set as a mutable cell so the closure captures it
    agent._entities_set = entities_set  # type: ignore[attr-defined]
    return agent


def get_relations(
    input_data: str,
    entities: List[str],
    is_conversation: bool = False,
    context: str = "",
    model_config: "ModelConfig" = None,
) -> Tuple[List[Tuple[str, str, str]], object]:
    """Extract subject-predicate-object relations from text.

    Returns:
        Tuple of (list of (subject, predicate, object) triples, RunUsage)
    """
    entities = _filter_entities(entities)
    entities_str = "\n".join(f"- {e}" for e in entities)

    agent = _build_agent(model_config, is_conversation, context)
    # Populate the entities set that the output_validator closure uses
    agent._entities_set.update(entities)  # type: ignore[attr-defined]

    user_prompt = f"""
Here is the list of entities that were previously extracted from the source text:

<entities>
{entities_str}
</entities>

Here is the {'conversation' if is_conversation else 'source text'} to analyze:

<{'conversation' if is_conversation else 'text'}>
{input_data}
</{'conversation' if is_conversation else 'text'}>
"""

    result = agent.run_sync(user_prompt)
    triples = [(r.subject, r.predicate, r.object) for r in result.output.relations]
    return triples, result.usage
