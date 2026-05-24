from typing import Union, List, Dict, Optional
from typing_extensions import deprecated
from dataclasses import dataclass, field

from kg_gen.steps._1_get_entities import get_entities
from kg_gen.steps._2_get_relations import get_relations
from kg_gen.steps._3_deduplicate import run_deduplication, DeduplicateMethod
from kg_gen.utils.chunk_text import chunk_text
from kg_gen.utils.visualize_kg import visualize as visualize_kg
from kg_gen.models import Graph
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.usage import RunUsage
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)


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

    def build(self) -> OpenAIChatModel:
        """Build a PydanticAI OpenAIChatModel from this config.

        Uses LiteLLMProvider when the model name contains a provider prefix
        (e.g. ``openai/gpt-4o``, ``anthropic/claude-3-5-sonnet-20241022``),
        which routes the call through LiteLLM just as the original DSPy path
        did. Falls back to OpenAIProvider for bare model names.
        """
        if "/" in self.model:
            provider = LiteLLMProvider(
                api_key=self.api_key,
                api_base=self.api_base,
            )
            model_name = self.model
        else:
            provider = OpenAIProvider(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            model_name = self.model

        return OpenAIChatModel(model_name, provider=provider)


class KGGen:
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_tokens: int = 16000,
        temperature: float = 0.0,
        reasoning_effort: str = None,
        api_key: str = None,
        api_base: str = None,
        retrieval_model: Optional[str] = None,
        disable_cache: bool = False,
    ):
        """Initialize KGGen with optional model configuration

        Args:
            model: LiteLLM-style model name (e.g. ``'openai/gpt-4o'``)
            max_tokens: Maximum tokens for model output
            temperature: Temperature for model sampling
            reasoning_effort: Reasoning effort for o-series models (``'low'``/``'medium'``/``'high'``)
            api_key: API key for model access
            api_base: Custom base URL endpoint for the language model service
            retrieval_model: SentenceTransformer model name for retrieval/dedup
            disable_cache: Disable LiteLLM response caching
        """
        self.retrieval_model: Optional[SentenceTransformer] = None
        self._usage_history: List[RunUsage] = []

        self.init_model(
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            retrieval_model=retrieval_model,
            disable_cache=disable_cache,
        )

    def validate_temperature(self, temperature: float):
        if "gpt-5" in self.model_config.model and temperature < 1.0:
            raise ValueError("Temperature must be 1.0 for gpt-5 family models")

    def validate_max_tokens(self, max_tokens: int):
        if "gpt-5" in self.model_config.model and max_tokens < 16000:
            raise ValueError("Max tokens must be 16000 for gpt-5 family models")

    def init_model(
        self,
        model: str = None,
        reasoning_effort: str = None,
        max_tokens: int = None,
        temperature: float = None,
        retrieval_model: str = None,
        api_key: str = None,
        api_base: str = None,
        disable_cache: bool | None = None,
    ):
        """Initialize or reinitialize the model config with new parameters."""

        # Carry forward existing values for unspecified params
        existing = getattr(self, "model_config", None)

        self.model_config = ModelConfig(
            model=model if model is not None else (existing.model if existing else "openai/gpt-4o"),
            api_key=api_key if api_key is not None else (existing.api_key if existing else None),
            api_base=api_base if api_base is not None else (existing.api_base if existing else None),
            temperature=temperature if temperature is not None else (existing.temperature if existing else 0.0),
            max_tokens=max_tokens if max_tokens is not None else (existing.max_tokens if existing else 16000),
            reasoning_effort=reasoning_effort if reasoning_effort is not None else (existing.reasoning_effort if existing else None),
            disable_cache=disable_cache if disable_cache is not None else (existing.disable_cache if existing else False),
        )

        self.validate_temperature(self.model_config.temperature)
        self.validate_max_tokens(self.model_config.max_tokens)

        if retrieval_model is not None:
            self.retrieval_model = SentenceTransformer(retrieval_model)

    # Convenience property kept for backward compatibility with code that
    # previously accessed ``self.model`` or ``self.lm``.
    @property
    def model(self) -> str:
        return self.model_config.model

    @staticmethod
    def from_file(file_path: str) -> Graph:
        with open(file_path, "r") as f:
            graph = Graph(**json.load(f))
        return graph

    @staticmethod
    def from_dict(graph_dict: dict) -> Graph:
        return Graph(**graph_dict)

    def generate(
        self,
        input_data: Union[str, List[Dict]],
        model: str = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
        chunk_size: Optional[int] = None,
        reasoning_effort: str = None,
        deduplication_method: DeduplicateMethod | None = DeduplicateMethod.SEMHASH,
        temperature: float = None,
        output_folder: Optional[str] = None,
        # Kept for backward compatibility; PydanticAI is now always used.
        no_dspy: bool = False,
    ) -> Graph:
        """Generate a knowledge graph from input text or messages.

        Args:
            input_data: Text string or list of message dicts
            model: Override the model for this call
            api_key: Override the API key for this call
            api_base: Override the API base for this call
            context: Description of data context
            chunk_size: Max size of text chunks in characters to process
            reasoning_effort: Reasoning effort for o-series models
            deduplication_method: Deduplication strategy (default: SEMHASH)
            temperature: Override temperature for this call
            output_folder: Path to save partial progress
            no_dspy: Deprecated; kept for backward compatibility only

        Returns:
            Graph: Generated knowledge graph
        """
        is_conversation = isinstance(input_data, list)
        if is_conversation:
            text_content = []
            for message in input_data:
                if (
                    not isinstance(message, dict)
                    or "role" not in message
                    or "content" not in message
                ):
                    raise ValueError(
                        "Messages must be dicts with 'role' and 'content' keys"
                    )
                if message["role"] in ["user", "assistant"]:
                    text_content.append(f"{message['role']}: {message['content']}")
            processed_input = "\n".join(text_content)
        else:
            processed_input = input_data

        if any(x is not None for x in [model, temperature, api_key, api_base, reasoning_effort]):
            self.init_model(
                model=model if model is not None else self.model_config.model,
                temperature=temperature if temperature is not None else self.model_config.temperature,
                api_key=api_key if api_key is not None else self.model_config.api_key,
                api_base=api_base if api_base is not None else self.model_config.api_base,
                reasoning_effort=reasoning_effort if reasoning_effort is not None else self.model_config.reasoning_effort,
            )

        cfg = self.model_config

        def _process(content: str):
            entities, e_usage = get_entities(
                content,
                is_conversation=is_conversation,
                model_config=cfg,
            )
            relations, r_usage = get_relations(
                content,
                entities,
                is_conversation=is_conversation,
                context=context,
                model_config=cfg,
            )
            self._usage_history.extend([e_usage, r_usage])
            return entities, relations

        if not chunk_size:
            try:
                entities, relations = _process(processed_input)
            except Exception as e:
                if "context length" in str(e).lower():
                    logger.warning(
                        f"Context length error: {e}. Chunking text with chunk size 16384."
                    )
                    chunk_size = 16384
                else:
                    raise e

        if chunk_size:
            chunks = chunk_text(processed_input, chunk_size)
            entities = set()
            relations = set()

            with ThreadPoolExecutor() as executor:
                future_to_chunk = {
                    executor.submit(_process, chunk): chunk for chunk in chunks
                }
                for future in as_completed(future_to_chunk):
                    chunk_entities, chunk_relations = future.result()
                    entities.update(chunk_entities)
                    relations.update(chunk_relations)

        graph = Graph(
            entities=entities,
            relations=relations,
            edges={relation[1] for relation in relations},
        )

        if deduplication_method:
            graph = self.deduplicate(
                graph, method=deduplication_method, context=context
            )

        if output_folder:
            self.export_graph(graph, os.path.join(output_folder, "graph.json"))
        return graph

    @deprecated("Use KGGen.deduplicate() method instead")
    def cluster(
        self,
        graph: Graph,
        **kwargs,
    ) -> Graph:
        return self.deduplicate(graph, **kwargs)

    def deduplicate(
        self,
        graph: Graph,
        method: DeduplicateMethod = DeduplicateMethod.FULL,
        semhash_similarity_threshold: float = 0.95,
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
    ) -> Graph:
        if any(x is not None for x in [model, temperature, api_key, api_base]):
            self.init_model(
                model=model if model is not None else self.model_config.model,
                temperature=temperature if temperature is not None else self.model_config.temperature,
                api_key=api_key if api_key is not None else self.model_config.api_key,
                api_base=api_base if api_base is not None else self.model_config.api_base,
            )

        return run_deduplication(
            model_config=self.model_config,
            graph=graph,
            method=method,
            retrieval_model=self.retrieval_model,
            semhash_similarity_threshold=semhash_similarity_threshold,
        )

    def aggregate(self, graphs: list[Graph]) -> Graph:
        all_entities = set()
        all_relations = set()
        all_edges = set()
        all_entity_metadata: dict[str, set[str]] = {}

        for graph in graphs:
            all_entities.update(graph.entities)
            all_relations.update(graph.relations)
            all_edges.update(graph.edges)
            if graph.entity_metadata:
                for entity, metadata_set in graph.entity_metadata.items():
                    if entity in all_entity_metadata:
                        all_entity_metadata[entity].update(metadata_set)
                    else:
                        all_entity_metadata[entity] = metadata_set.copy()

        return Graph(
            entities=all_entities,
            relations=all_relations,
            edges=all_edges,
            entity_metadata=all_entity_metadata if all_entity_metadata else None,
        )

    @staticmethod
    def visualize(graph: Graph, output_path: str, open_in_browser: bool = False):
        visualize_kg(graph, output_path, open_in_browser=open_in_browser)

    # ====== Retrieval Methods ======

    def _parse_embedding_model(
        self, model: Optional[SentenceTransformer] = None
    ) -> Optional[SentenceTransformer]:
        if model is None:
            model = self.retrieval_model
        if model is None:
            raise ValueError("No retrieval model provided")
        return model

    @staticmethod
    def to_nx(graph: Graph) -> nx.DiGraph:
        G = nx.DiGraph()
        for entity in graph.entities:
            G.add_node(entity)
        for relation in graph.relations:
            source, rel, target = relation
            G.add_edge(source, target, relation=rel)
        return G

    def generate_embeddings(
        self,
        graph: Union[Graph, nx.DiGraph],
        model: Optional[SentenceTransformer] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        model = self._parse_embedding_model(model)
        if isinstance(graph, Graph):
            graph = self.to_nx(graph)

        node_embeddings = {node: model.encode(node).tolist() for node in graph.nodes}
        relation_embeddings = {
            rel: model.encode(rel).tolist()
            for rel in set(edge[2]["relation"] for edge in graph.edges(data=True))
        }
        return node_embeddings, relation_embeddings

    def retrieve(
        self,
        query: str,
        node_embeddings: dict[str, np.ndarray],
        graph: nx.DiGraph,
        model: Optional[SentenceTransformer] = None,
        k: int = 8,
        verbose: bool = False,
    ) -> tuple[list[tuple[str, float]], set[str], str]:
        model = self._parse_embedding_model(model)
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)
        context = set()
        for node, _ in top_nodes:
            node_context = self.retrieve_context(node, graph)
            if verbose:
                print(f"Context for node {node}: {node_context}")
            context.update(node_context)
        context_text = " ".join(context)
        if verbose:
            print(f"Combined context: '{context_text}'\n---")
        return top_nodes, context, context_text

    @staticmethod
    def retrieve_relevant_nodes(
        query: str,
        node_embeddings: dict[str, np.ndarray],
        model: SentenceTransformer,
        k: int = 8,
    ) -> list[tuple[str, float]]:
        query_embedding = model.encode(query).reshape(1, -1)
        similarities = []
        for node, embed in node_embeddings.items():
            target_embedding = np.array(embed).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, target_embedding)[0][0]
            similarities.append((node, similarity))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:k]

    @staticmethod
    def retrieve_context(node: str, graph: nx.DiGraph, depth: int = 2) -> list[str]:
        context = set()

        def explore_neighbors(current_node, current_depth):
            if current_depth > depth:
                return
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]["relation"]
                context.add(f"{current_node} {rel} {neighbor}.")
                explore_neighbors(neighbor, current_depth + 1)
            for neighbor in graph.predecessors(current_node):
                rel = graph[neighbor][current_node]["relation"]
                context.add(f"{neighbor} {rel} {current_node}.")
                explore_neighbors(neighbor, current_depth + 1)

        explore_neighbors(node, 1)
        return list(context)

    @staticmethod
    def export_graph(graph: Graph, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        graph_dict = {
            "entities": list(graph.entities),
            "relations": list(graph.relations),
            "edges": list(graph.edges),
            "entity_clusters": {k: list(v) for k, v in graph.entity_clusters.items()}
            if graph.entity_clusters
            else None,
            "edge_clusters": {k: list(v) for k, v in graph.edge_clusters.items()}
            if graph.edge_clusters
            else None,
            "entity_metadata": graph.entity_metadata,
        }
        with open(output_path, "w") as f:
            json.dump(graph_dict, f, indent=2)

    # ====== Token Usage ======

    def reset_token_usage(self):
        """Reset accumulated token usage counters."""
        self._usage_history = []

    def extract_token_usage_from_history(self) -> Dict[str, int]:
        """Sum token usage across all calls made since the last reset."""
        total_input = sum(u.input_tokens for u in self._usage_history)
        total_output = sum(u.output_tokens for u in self._usage_history)
        total = total_input + total_output
        return {
            "prompt_tokens": total_input,
            "completion_tokens": total_output,
            "total_tokens": total,
        }
