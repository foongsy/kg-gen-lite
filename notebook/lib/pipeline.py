"""Self-contained KG extraction pipeline (vendored from kg-gen-lite fork)."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import httpx
import inflect
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.vercel import VercelProvider
from pydantic_ai.settings import ModelSettings
from semhash import SemHash

if TYPE_CHECKING:
    from semhash.utils import Encoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

_DEFAULT_HTTP_TIMEOUT = httpx.Timeout(timeout=600, connect=5)
_DEFAULT_SEMHASH_MODEL = "minishlab/potion-multilingual-128M"

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
    (0x3000, 0x303F),
    (0xFF00, 0xFFEF),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0xAC00, 0xD7AF),
)


def _is_cjk_char(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def is_cjk_text(text: str, threshold: float = 0.10) -> bool:
    """Return True if more than *threshold* fraction of characters are CJK."""
    if not text:
        return False
    cjk_count = sum(1 for c in text if _is_cjk_char(ord(c)))
    return cjk_count / len(text) > threshold


def _http_client(ssl_verify: bool) -> httpx.AsyncClient | None:
    if ssl_verify:
        return None
    return httpx.AsyncClient(verify=False, timeout=_DEFAULT_HTTP_TIMEOUT)


def _parse_vercel_model(model: str) -> str | None:
    if model.startswith("vercel:"):
        return model.removeprefix("vercel:")
    if model.startswith("vercel_ai_gateway/"):
        return model.removeprefix("vercel_ai_gateway/")
    return None


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class ModelConfig:
    """Holds all parameters needed to instantiate a PydanticAI model."""

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 16000
    reasoning_effort: Optional[str] = None
    disable_cache: bool = False
    ssl_verify: bool = True

    def build(self) -> OpenAIChatModel:
        http_client = _http_client(self.ssl_verify)
        provider_kwargs = (
            {"http_client": http_client} if http_client is not None else {}
        )

        vercel_model = _parse_vercel_model(self.model)
        if vercel_model is not None:
            provider = VercelProvider(api_key=self.api_key, **provider_kwargs)
            model_name = vercel_model
        elif "/" in self.model:
            provider = LiteLLMProvider(
                api_key=self.api_key,
                api_base=self.api_base,
                **provider_kwargs,
            )
            model_name = self.model
        else:
            provider = OpenAIProvider(
                api_key=self.api_key,
                base_url=self.api_base,
                **provider_kwargs,
            )
            model_name = self.model

        return OpenAIChatModel(model_name, provider=provider)


class Graph(BaseModel):
    entities: set[str] = Field(
        ..., description="All entities including additional ones from response"
    )
    edges: set[str] = Field(..., description="All edges")
    relations: set[Tuple[str, str, str]] = Field(
        ..., description="List of (subject, predicate, object) triples"
    )
    entity_clusters: Optional[dict[str, set[str]]] = None
    edge_clusters: Optional[dict[str, set[str]]] = None
    entity_metadata: dict[str, set[str]] | None = None

    @staticmethod
    def from_file(file_path: str) -> "Graph":
        with open(file_path, "r", encoding="utf-8") as f:
            graph = Graph.model_validate(json.load(f))

        for relation in graph.relations:
            if relation[0] not in graph.entities:
                graph.entities.add(relation[0])
            if relation[1] not in graph.edges:
                graph.edges.add(relation[1])
            if relation[2] not in graph.entities:
                graph.entities.add(relation[2])

        return graph

    def to_file(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))


class EntitiesResponse(BaseModel):
    """A thorough list of key entities extracted from the source text."""

    entities: List[str]


class RelationItem(BaseModel):
    subject: str
    predicate: str
    object: str


class RelationsResponse(BaseModel):
    relations: List[RelationItem]


def _model_settings(model_config: ModelConfig) -> ModelSettings:
    return OpenAIChatModelSettings(
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        **(
            {"openai_reasoning_effort": model_config.reasoning_effort}
            if model_config.reasoning_effort
            else {}
        ),
    )


def _build_entities_agent(model_config: ModelConfig) -> Agent:
    return Agent(
        model_config.build(),
        output_type=EntitiesResponse,
        system_prompt=_load_prompt("entities.txt"),
        model_settings=_model_settings(model_config),
    )


def get_entities(
    input_data: str,
    is_conversation: bool = False,
    *,
    model_config: ModelConfig,
) -> Tuple[List[str], object]:
    agent = _build_entities_agent(model_config)

    tag = "conversation" if is_conversation else "article"
    user_prompt = (
        f"\nHere is the {'conversation' if is_conversation else 'text'} to extract entities from:\n\n"
        f"<{tag}>\n{input_data}\n</{tag}>\n"
    )

    result = agent.run_sync(user_prompt)
    return result.output.entities, result.usage


def _filter_entities(entities: List[str]) -> List[str]:
    return [e for e in entities if '"' not in e]


def _relations_system_prompt(context: str) -> str:
    prompt = _load_prompt("relations.txt")
    if context:
        prompt += f"\n\n## Domain context\n{context}"
    return prompt


def _build_relations_agent(
    model_config: ModelConfig,
    context: str,
    entities: List[str],
) -> Agent:
    agent: Agent[None, RelationsResponse] = Agent(
        model_config.build(),
        output_type=RelationsResponse,
        system_prompt=_relations_system_prompt(context),
        model_settings=_model_settings(model_config),
        retries=2,
    )

    entities_set: set[str] = set(entities)

    @agent.output_validator
    def validate_entities(output: RelationsResponse) -> RelationsResponse:
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
    model_config: ModelConfig,
) -> Tuple[List[Tuple[str, str, str]], object]:
    entities = _filter_entities(entities)
    entities_str = "\n".join(f"- {e}" for e in entities)

    agent = _build_relations_agent(model_config, context, entities)

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


def _load_semhash_encoder(model_name: str | None) -> "Encoder":
    from model2vec import StaticModel

    return StaticModel.from_pretrained(model_name or _DEFAULT_SEMHASH_MODEL)


class DeduplicateList:
    def __init__(
        self,
        threshold: float = 0.95,
        semhash_model: str | None = None,
        semhash_encoder: "Encoder | None" = None,
    ):
        self.threshold = threshold
        self.semhash_model = semhash_model
        self.semhash_encoder = semhash_encoder
        self.inflect_engine = inflect.engine()
        self.original_map: dict[str, str] = {}
        self.items_map: dict[str, str] = {}
        self.duplicates: dict[str, str] = {}
        self.deduplicated: list[str] = []
        self.total_items = 0
        self.deduplicated_items = 0
        self.duplicate_items = 0
        self.reduction = 0.0

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def singularize(self, text: str) -> str:
        if is_cjk_text(text):
            return text

        tokens = []
        for tok in text.split():
            sing = self.inflect_engine.singular_noun(tok)
            tokens.append(sing if isinstance(sing, str) and sing else tok)
        return " ".join(tokens).strip()

    def deduplicate(self, items: list[str]) -> list[str]:
        self.total_items = len(items)

        normalized_items = set()
        for item in items:
            normalized = self.normalize(item)
            singular = self.singularize(normalized)
            self.original_map[item] = singular
            self.items_map[singular] = item
            normalized_items.add(singular)

        encoder = self.semhash_encoder
        if encoder is None:
            encoder = _load_semhash_encoder(self.semhash_model)

        semhash = SemHash.from_records(
            records=list(normalized_items),
            model=encoder,
        )
        deduplication_result = semhash.self_deduplicate(threshold=self.threshold)

        self.deduplicated_items = len(deduplication_result.selected)
        self.duplicate_items = len(deduplication_result.duplicates)
        self.reduction = (self.duplicate_items / self.total_items) * 100

        for duplicate in deduplication_result.duplicates:
            original = duplicate.record
            if (
                duplicate.duplicates
                and len(duplicate.duplicates) > 0
                and len(duplicate.duplicates[0]) > 0
            ):
                duplicate_value = duplicate.duplicates[0][0]
                self.items_map[original] = self.items_map[duplicate_value]
                if original not in self.duplicates:
                    self.duplicates[original] = duplicate_value

        self.deduplicated = deduplication_result.selected
        return self.deduplicated


def run_semhash_deduplication(
    graph: Graph,
    similarity_threshold: float = 0.95,
    semhash_model: str | None = None,
) -> Graph:
    encoder = _load_semhash_encoder(semhash_model)

    entities_dedup = DeduplicateList(similarity_threshold, semhash_encoder=encoder)
    if graph.entities:
        entities_dedup.deduplicate(list(graph.entities))
    edges_dedup = DeduplicateList(similarity_threshold, semhash_encoder=encoder)
    if graph.edges:
        edges_dedup.deduplicate(list(graph.edges))

    def _get_relation(relation: tuple[str, str, str]) -> list[str]:
        first_entity_original = relation[0]
        if first_entity_original in entities_dedup.original_map:
            first_entity = entities_dedup.items_map[
                entities_dedup.original_map[first_entity_original]
            ]
        else:
            first_entity = first_entity_original

        second_entity_original = relation[2]
        if second_entity_original in entities_dedup.original_map:
            second_entity = entities_dedup.items_map[
                entities_dedup.original_map[second_entity_original]
            ]
        else:
            second_entity = second_entity_original

        edge_original = relation[1]
        if edge_original in edges_dedup.original_map:
            edge = edges_dedup.items_map[edges_dedup.original_map[edge_original]]
        else:
            edge = edge_original

        return [first_entity, edge, second_entity]

    new_entities = [
        entities_dedup.items_map[item] for item in entities_dedup.deduplicated
    ]
    new_edges = [edges_dedup.items_map[item] for item in edges_dedup.deduplicated]
    new_relations = [_get_relation(relation) for relation in graph.relations]
    new_relations = list(set(tuple(relation) for relation in new_relations))

    new_entity_metadata: dict[str, set[str]] | None = None
    if graph.entity_metadata:
        new_entity_metadata = {}
        for original_entity, metadata_set in graph.entity_metadata.items():
            if original_entity in entities_dedup.original_map:
                deduped_entity = entities_dedup.items_map[
                    entities_dedup.original_map[original_entity]
                ]
            else:
                deduped_entity = original_entity
            if deduped_entity in new_entity_metadata:
                new_entity_metadata[deduped_entity].update(metadata_set)
            else:
                new_entity_metadata[deduped_entity] = metadata_set.copy()

    return Graph(
        entities=set(new_entities),
        edges=set(new_edges),
        relations=set(new_relations),
        entity_metadata=new_entity_metadata,
    )
