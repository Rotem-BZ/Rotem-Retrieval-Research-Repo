"""Leave the generated project with a real, empty experiments directory."""

from pathlib import Path


(Path.cwd() / "experiments" / ".gitkeep").unlink()
