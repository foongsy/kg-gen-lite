"""LLM-assisted deduplication via KMeans clustering + PydanticAI per-cluster inference.

Non-Latin text support (step 3 of 3):
    BM25 tokenization now uses character-level splitting for CJK text instead of
    whitespace splitting, which has no word-boundary signal for Chinese/Japanese/Korean.
    Detection is done by checking if more than 10 % of characters fall in CJK Unicode
    ranges — a fast, dependency-free heuristic that avoids requiring jieba or similar.
"""

from typing import List, TYPE_CHECKING
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
from kg_gen.models import Graph
import logging
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModelSettings

if TYPE_CHECKING:
    from kg_gen.kg_gen import ModelConfig


# ---------------------------------------------------------------------------
# CJK-aware tokenization helper (non-Latin text support — step 3)
# ---------------------------------------------------------------------------

# Unicode ranges that cover CJK Unified Ideographs and common extensions
_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Supplement
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms (includes ｆｕｌｌ-ｗｉｄｔｈ)
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
)


def _is_cjk_char(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_cjk_text(text: str, threshold: float = 0.10) -> bool:
    """Return True if more than *threshold* fraction of characters are CJK."""
    if not text:
        return False
    cjk_count = sum(1 for c in text if _is_cjk_char(ord(c)))
    return cjk_count / len(text) > threshold


def _tokenize(text: str) -> list[str]:
    """Tokenize *text* for BM25.

    For CJK text each character is its own token (character-level n-gram
    baseline). For Latin/other scripts whitespace splitting is used as before.
    """
    if _is_cjk_text(text):
        return [c for c in text if not c.isspace()]
    return text.lower().split()


# ---------------------------------------------------------------------------
# PydanticAI deduplication models
# ---------------------------------------------------------------------------


class DeduplicateResponse(BaseModel):
    """Duplicate detection result for a single item against a candidate set."""

    duplicates: List[str]
    """Exact matches to items in the candidate set that mean the same thing."""

    alias: str
    """Best name to represent the group, ideally chosen from the candidate set."""


# ---------------------------------------------------------------------------
# LLMDeduplicate
# ---------------------------------------------------------------------------


class LLMDeduplicate:
    graph: Graph
    nodes: list[str]
    edges: list[str]
    node_clusters: list[list[str]]
    edge_clusters: list[list[str]]
    retrieval_model: SentenceTransformer
    model_config: "ModelConfig"

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        retrieval_model: SentenceTransformer,
        model_config: "ModelConfig",
        graph: Graph,
    ):
        self.graph = graph
        self.nodes = list(graph.entities)
        self.edges = list(graph.edges)
        self.node_clusters = graph.entity_clusters or []
        self.edge_clusters = graph.edge_clusters or []
        self.retrieval_model = retrieval_model
        self.model_config = model_config

        self.node_embeddings = retrieval_model.encode(self.nodes, show_progress_bar=True)
        # Character-aware BM25 tokenization (non-Latin step 3)
        self.node_bm25_tokenized = [_tokenize(text) for text in self.nodes]
        self.node_bm25 = BM25Okapi(self.node_bm25_tokenized)

        self.edge_embeddings = retrieval_model.encode(self.edges, show_progress_bar=True)
        self.edge_bm25_tokenized = [_tokenize(text) for text in self.edges]
        self.edge_bm25 = BM25Okapi(self.edge_bm25_tokenized)

    def _build_dedup_agent(self, plural_type: str, singular_type: str) -> Agent:
        system_prompt = (
            f"Find duplicate {plural_type} for the item against a candidate set. "
            f"Duplicates are {plural_type} that are the same in meaning, such as with "
            "variation in tense, plural form, stem form, case, abbreviation, or shorthand. "
            "This includes duplicates in non-Latin scripts — judge semantic equivalence "
            "across scripts or transliterations if present. "
            "Return an empty duplicates list if there are none."
        )
        pai_model = self.model_config.build()
        settings: ModelSettings = OpenAIChatModelSettings(
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            **(
                {"openai_reasoning_effort": self.model_config.reasoning_effort}
                if self.model_config.reasoning_effort
                else {}
            ),
        )
        return Agent(
            pai_model,
            output_type=DeduplicateResponse,
            system_prompt=system_prompt,
            model_settings=settings,
        )

    def get_relevant_items(
        self, query: str, top_k: int = 50, type: str = "node"
    ) -> list[str]:
        """Rank-fusion of BM25 + embedding similarity to retrieve top-k candidates."""
        query_tokens = _tokenize(query)

        bm25_scores = (
            self.node_bm25.get_scores(query_tokens)
            if type == "node"
            else self.edge_bm25.get_scores(query_tokens)
        )

        query_embedding = self.retrieval_model.encode([query], show_progress_bar=False)
        embeddings = self.node_embeddings if type == "node" else self.edge_embeddings
        embedding_scores = cosine_similarity(query_embedding, embeddings).flatten()

        combined_scores = 0.5 * bm25_scores + 0.5 * embedding_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        items = self.nodes if type == "node" else self.edges
        return [items[i] for i in top_indices]

    def cluster(self):
        cluster_size = 128
        embedding_sets = {"node": self.node_embeddings, "edge": self.edge_embeddings}

        for embedding_type, embeddings in embedding_sets.items():
            n_samples = len(embeddings)
            num_clusters = max(1, n_samples // cluster_size)

            kmeans = KMeans(
                n_clusters=num_clusters,
                init="random",
                n_init=1,
                max_iter=20,
                tol=0.0,
                algorithm="lloyd",
                verbose=True,
            )
            kmeans.fit(embeddings.astype(np.float32))
            centroids = kmeans.cluster_centers_

            distances = cdist(embeddings, centroids)
            assignments = np.argsort(distances, axis=1)

            clusters: List[List[int]] = [[] for _ in range(num_clusters)]
            assigned = np.zeros(n_samples, dtype=bool)

            for rank in range(num_clusters):
                for i in range(n_samples):
                    if assigned[i]:
                        continue
                    cluster_id = assignments[i, rank]
                    if len(clusters[cluster_id]) < cluster_size:
                        clusters[cluster_id].append(i)
                        assigned[i] = True

            unassigned = np.where(~assigned)[0]
            if len(unassigned) > 0:
                self.logger.debug(
                    "Adding %s unassigned items as a separate cluster", len(unassigned)
                )
                clusters.append(unassigned.tolist())

            self.logger.debug("Number of %s clusters: %s", embedding_type, len(clusters))

            if embedding_type == "node":
                self.node_clusters = [
                    [self.nodes[idx] for idx in cluster] for cluster in clusters
                ]
            else:
                self.edge_clusters = [
                    [self.edges[idx] for idx in cluster] for cluster in clusters
                ]

    def deduplicate_cluster(
        self, cluster: list[str], type: str = "node"
    ) -> tuple[set, dict[str, list[str]]]:
        cluster = cluster.copy()
        items = set()
        item_clusters = {}
        plural_type = "entities" if type == "node" else "edges"
        singular_type = "entity" if type == "node" else "edge"

        self.logger.info(
            "Starting deduplication of %s %s in cluster", len(cluster), plural_type
        )

        agent = self._build_dedup_agent(plural_type, singular_type)
        processed_count = 0

        while cluster:
            processed_count += 1
            item = cluster.pop()

            self.logger.debug(
                "[%s/%s] Processing %s: '%s'",
                processed_count,
                len(cluster),
                singular_type,
                item,
            )

            relevant_items = self.get_relevant_items(item, 16, type)

            user_prompt = (
                f"Item: {item}\n\n"
                f"Candidate {plural_type} set:\n"
                + "\n".join(f"- {it}" for it in relevant_items)
            )

            result = agent.run_sync(user_prompt)
            response = result.output
            items.add(response.alias)

            duplicates = [dup for dup in response.duplicates if dup in cluster]

            if duplicates:
                self.logger.info(
                    "  → Using alias '%s' to represent: '%s' and %s",
                    response.alias,
                    item,
                    duplicates,
                )
                item_clusters[response.alias] = {item}
                for duplicate in duplicates:
                    cluster.remove(duplicate)
                    item_clusters[response.alias].add(duplicate)
            else:
                self.logger.debug("  ✗ No duplicates found for '%s'", item)
                item_clusters[item] = {item}

        return items, item_clusters

    def deduplicate(self) -> Graph:
        entities: set = set()
        edges: set = set()
        entity_clusters: dict = {}
        edge_clusters: dict = {}

        pool = ThreadPoolExecutor(max_workers=64)

        node_futures = [
            pool.submit(self.deduplicate_cluster, cluster, "node")
            for cluster in self.node_clusters
        ]
        edge_futures = [
            pool.submit(self.deduplicate_cluster, cluster, "edge")
            for cluster in self.edge_clusters
        ]

        for i, future in enumerate(node_futures):
            try:
                cluster_entities, cluster_entity_map = future.result()
                entities.update(cluster_entities)
                entity_clusters.update(cluster_entity_map)
            except Exception as e:
                self.logger.error("Error processing node cluster %s: %s", i, e)

        for i, future in enumerate(edge_futures):
            try:
                cluster_edges, cluster_edge_map = future.result()
                edges.update(cluster_edges)
                edge_clusters.update(cluster_edge_map)
            except Exception as e:
                self.logger.error("Error processing edge cluster %s: %s", i, e)

        # Remap relations through deduplicated representatives
        relations: set[tuple[str, str, str]] = set()
        for s, p, o in self.graph.relations:
            if s not in entities:
                for rep, cluster in entity_clusters.items():
                    if s in cluster:
                        s = rep
                        break
            if p not in edges:
                for rep, cluster in edge_clusters.items():
                    if p in cluster:
                        p = rep
                        break
            if o not in entities:
                for rep, cluster in entity_clusters.items():
                    if o in cluster:
                        o = rep
                        break
            relations.add((s, p, o))

        new_entity_metadata: dict[str, set[str]] | None = None
        if self.graph.entity_metadata:
            new_entity_metadata = {}
            for original_entity, metadata_set in self.graph.entity_metadata.items():
                deduped_entity = original_entity
                for rep, cluster in entity_clusters.items():
                    if original_entity in cluster:
                        deduped_entity = rep
                        break
                if deduped_entity in new_entity_metadata:
                    new_entity_metadata[deduped_entity].update(metadata_set)
                else:
                    new_entity_metadata[deduped_entity] = metadata_set.copy()

        return Graph(
            entities=entities,
            edges=edges,
            relations=relations,
            entity_clusters=entity_clusters,
            edge_clusters=edge_clusters,
            entity_metadata=new_entity_metadata,
        )
