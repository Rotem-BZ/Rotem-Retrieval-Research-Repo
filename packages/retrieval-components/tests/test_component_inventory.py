import ast
import re
from pathlib import Path

COMPONENT_MODULES = {
    "retrieval_components.cascade.chunk_cascade": {"ChunkCascade"},
    "retrieval_components.cascade.top_k_documents": {"TopKDocuments"},
    "retrieval_components.chunking.langchain_document_splitter": {"LangChainDocumentSplitter"},
    "retrieval_components.experimental.elasticsearch_bm25_retriever": {
        "ElasticsearchBM25Retriever"
    },
    "retrieval_components.experimental.elasticsearch_document_indexer": {
        "ElasticsearchDocumentIndexer"
    },
    "retrieval_components.filtering.document_content_filter": {"DocumentContentFilter"},
    "retrieval_components.fusion.normalized_score_fusion": {"LinearScoreFusion", "ZScoreFusion"},
    "retrieval_components.fusion.reciprocal_rank_fusion": {"ReciprocalRankFusion"},
    "retrieval_components.fusion.score_fusion": {"ScoreFusion"},
    "retrieval_components.indexing.persisted_in_memory_document_indexer": {
        "PersistedInMemoryDocumentIndexer"
    },
    "retrieval_components.interfaces.indexing": {"IndexingInput", "IndexingOutput"},
    "retrieval_components.interfaces.inference": {"InferenceInput", "InferenceOutput"},
    "retrieval_components.models.sentence_transformers_similarity_ranker": {
        "SentenceTransformersSimilarityRanker"
    },
    "retrieval_components.models.sentence_transformers_text_embedder": {
        "SentenceTransformersTextEmbedder"
    },
    "retrieval_components.models.transformers_similarity_ranker": {"TransformersSimilarityRanker"},
    "retrieval_components.preprocessing.document_text_prefixer": {"DocumentTextPrefixer"},
    "retrieval_components.preprocessing.file_metadata_enricher": {
        "DocumentMetadataEnricher",
        "QueryMetadataEnricher",
    },
    "retrieval_components.preprocessing.identity_parser": {"IdentityParser"},
    "retrieval_components.preprocessing.query_text_preprocessor": {"QueryTextPreprocessor"},
    "retrieval_components.preprocessing.query_to_string": {"QueryToString"},
    "retrieval_components.ranking.embedding_similarity_ranker": {"EmbeddingSimilarityRanker"},
    "retrieval_components.reformulation.http_query_reformulator": {"HttpQueryReformulator"},
    "retrieval_components.retrieval.persisted_in_memory_embedding_retriever": {
        "PersistedInMemoryEmbeddingRetriever"
    },
}


def test_readme_component_inventory_matches_defining_modules() -> None:
    package_root = Path(__file__).parents[1]
    readme = (package_root / "README.md").read_text(encoding="utf-8")
    inventory = readme.split("## Available components", 1)[1].split("## Haystack overlap", 1)[0]
    documented: dict[str, set[str]] = {}

    for line in inventory.splitlines():
        if not line.startswith("| `retrieval_components."):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        module_path = cells[0].strip("`")
        documented[module_path] = set(re.findall(r"`([A-Za-z][A-Za-z0-9]*)`", cells[1]))

    assert documented == COMPONENT_MODULES
    for module_path, documented_names in documented.items():
        relative_module = module_path.removeprefix("retrieval_components.").replace(".", "/")
        source_path = package_root / "src" / "retrieval_components" / f"{relative_module}.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        defined_names = {
            node.name for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        assert documented_names <= defined_names, module_path


def test_category_initializers_contain_only_docstrings() -> None:
    package_root = Path(__file__).parents[1] / "src" / "retrieval_components"

    for init_path in package_root.rglob("__init__.py"):
        if init_path == package_root / "__init__.py":
            continue
        module = ast.parse(init_path.read_text(encoding="utf-8"))
        assert len(module.body) == 1, init_path
        assert (
            isinstance(module.body[0], ast.Expr)
            and isinstance(module.body[0].value, ast.Constant)
            and isinstance(module.body[0].value.value, str)
        )
        assert module.body[0].value.value, init_path


def test_package_root_does_not_import_component_modules() -> None:
    package_root = Path(__file__).parents[1] / "src" / "retrieval_components"
    module = ast.parse((package_root / "__init__.py").read_text(encoding="utf-8"))

    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(module))
