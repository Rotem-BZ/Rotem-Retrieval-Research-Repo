import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_haystack_imports_component_defining_module_on_demand() -> None:
    package_root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(package_root / "src"), env.get("PYTHONPATH"))
        if value
    )
    script = textwrap.dedent(
        """
        import sys

        from haystack import Pipeline, component

        import retrieval_components

        module_path = (
            "retrieval_components.chunking.langchain_document_splitter"
        )
        component_path = f"{module_path}.LangChainDocumentSplitter"

        assert module_path not in sys.modules
        assert component_path not in component.registry

        pipeline = Pipeline.from_dict(
            {
                "components": {
                    "splitter": {
                        "type": component_path,
                        "init_parameters": {
                            "chunk_size": 100,
                            "chunk_overlap": 10,
                        },
                    }
                },
                "connections": [],
            }
        )

        assert module_path in sys.modules
        assert component_path in component.registry
        assert type(pipeline.get_component("splitter")).__module__ == module_path
        """
    )

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=package_root,
        env=env,
        check=True,
    )
