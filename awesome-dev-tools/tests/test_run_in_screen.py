from pathlib import Path

import pytest

import run_in_screen


def test_launches_arbitrary_command_in_detached_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    launched: list[dict[str, object]] = []
    monkeypatch.setattr(run_in_screen.sys, "platform", "linux")
    monkeypatch.setattr(run_in_screen, "require_screen", lambda: "/usr/bin/screen")
    monkeypatch.setattr(
        run_in_screen,
        "list_screen_sessions",
        lambda **_kwargs: set(),
    )
    monkeypatch.setattr(
        run_in_screen,
        "launch_screen",
        lambda **kwargs: launched.append(kwargs),
    )

    run_in_screen.main(
        [
            "--name",
            "toy-index",
            "--cwd",
            str(project),
            "--log-file",
            "logs/indexing.log",
            "--",
            "uv",
            "run",
            "stage",
            "indexing",
            "dataset=toy",
        ]
    )

    assert launched == [
        {
            "session_name": "toy-index",
            "log_file": (project / "logs" / "indexing.log").resolve(),
            "command": ["uv", "run", "stage", "indexing", "dataset=toy"],
            "cwd": project.resolve(),
            "executable": "/usr/bin/screen",
        }
    ]
    output = capsys.readouterr().out
    assert "Launched detached screen: toy-index" in output
    assert "screen -r toy-index" in output


def test_generates_stage_session_name_and_default_log(tmp_path: Path) -> None:
    name = run_in_screen.default_session_name(
        ["uv", "run", "stage", "inference", "dataset=toy"],
        timestamp="20260722-010203",
    )

    assert name == "stage-inference-20260722-010203"
    assert run_in_screen.resolve_log_file(None, cwd=tmp_path, session_name=name) == (
        tmp_path / "artifacts" / "screens" / f"{name}.log"
    )


def test_rejects_duplicate_screen_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_in_screen.sys, "platform", "linux")
    monkeypatch.setattr(run_in_screen, "require_screen", lambda: "screen")
    monkeypatch.setattr(
        run_in_screen,
        "list_screen_sessions",
        lambda **_kwargs: {"existing"},
    )

    with pytest.raises(SystemExit, match="Screen session already exists: existing"):
        run_in_screen.main(
            ["--name", "existing", "--cwd", str(tmp_path), "--", "echo", "hello"]
        )


def test_requires_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_in_screen.sys, "platform", "win32")

    with pytest.raises(SystemExit, match="requires Linux"):
        run_in_screen.main(["--", "echo", "hello"])
