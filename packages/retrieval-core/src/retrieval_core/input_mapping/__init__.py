"""Candidate input mappings for inference runs."""

from retrieval_core.input_mapping.operations import (
    GENERATION_KEYS,
    INPUT_MAPPING_FILENAME,
    INPUT_MAPPING_METADATA_FILENAME,
    GeneratedInputMapping,
    InferenceMapping,
    configured_input_mapping_path,
    discover_input_mapping_ids,
    generate_input_mapping,
    input_mapping_generation_params,
    metadata_path_for,
    prepared_mapping_dir,
    prepared_mapping_path,
    prepare_generated_input_mapping,
    resolve_inference_mapping,
    validate_input_mapping_id,
    write_generated_mapping,
)

__all__ = [
    "GENERATION_KEYS",
    "INPUT_MAPPING_FILENAME",
    "INPUT_MAPPING_METADATA_FILENAME",
    "GeneratedInputMapping",
    "InferenceMapping",
    "configured_input_mapping_path",
    "discover_input_mapping_ids",
    "generate_input_mapping",
    "input_mapping_generation_params",
    "metadata_path_for",
    "prepared_mapping_dir",
    "prepared_mapping_path",
    "prepare_generated_input_mapping",
    "resolve_inference_mapping",
    "validate_input_mapping_id",
    "write_generated_mapping",
]
