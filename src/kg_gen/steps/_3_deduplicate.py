from typing import TYPE_CHECKING
from kg_gen.models import Graph
from kg_gen.utils.deduplicate import run_semhash_deduplication
from kg_gen.utils.llm_deduplicate import LLMDeduplicate
from sentence_transformers import SentenceTransformer
import enum

if TYPE_CHECKING:
    from kg_gen.kg_gen import ModelConfig


class DeduplicateMethod(enum.Enum):
    SEMHASH = "semhash"
    LM_BASED = "lm_based"
    FULL = "full"


def run_deduplication(
    model_config: "ModelConfig",
    graph: Graph,
    method: DeduplicateMethod = DeduplicateMethod.FULL,
    retrieval_model: SentenceTransformer | None = None,
    semhash_similarity_threshold: float = 0.95,
    semhash_model: str | None = None,
) -> Graph:
    if method != DeduplicateMethod.SEMHASH and retrieval_model is None:
        raise ValueError("No retrieval model provided")

    if method == DeduplicateMethod.SEMHASH:
        return run_semhash_deduplication(
            graph, semhash_similarity_threshold, semhash_model=semhash_model
        )

    if method == DeduplicateMethod.LM_BASED:
        llm_deduplicate = LLMDeduplicate(retrieval_model, model_config, graph)
        llm_deduplicate.cluster()
        return llm_deduplicate.deduplicate()

    # FULL: semhash first, then LLM
    deduplicated_graph = run_semhash_deduplication(
        graph, semhash_similarity_threshold, semhash_model=semhash_model
    )
    llm_deduplicate = LLMDeduplicate(retrieval_model, model_config, deduplicated_graph)
    llm_deduplicate.cluster()
    return llm_deduplicate.deduplicate()
