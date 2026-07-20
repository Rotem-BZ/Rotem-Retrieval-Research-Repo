import ast
import re
from pathlib import Path


CATEGORY_MODULES = {
    "retrieval_components.cascade",
    "retrieval_components.chunking",
    "retrieval_components.filtering",
    "retrieval_components.fusion",
    "retrieval_components.indexing",
    "retrieval_components.interfaces",
    "retrieval_components.models",
    "retrieval_components.preprocessing",
    "retrieval_components.ranking",
    "retrieval_components.reformulation",
    "retrieval_components.retrieval",
    "retrieval_components.sources",
}


def test_readme_component_inventory_matches_category_exports() -> None:
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

    assert set(documented) == CATEGORY_MODULES
    for module_path, documented_names in documented.items():
        category = module_path.rsplit(".", 1)[1]
        init_path = package_root / "src" / "retrieval_components" / category / "__init__.py"
        module = ast.parse(init_path.read_text(encoding="utf-8"))
        exports = next(
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            )
        )
        imported_names = {
            alias.asname or alias.name
            for node in module.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert documented_names == set(exports), module_path
        assert set(exports) == imported_names, module_path
