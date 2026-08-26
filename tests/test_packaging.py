from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

from lerobot.policies import get_policy_class, make_policy_config, make_pre_post_processors


def test_distribution_name_is_importable_lerobot_policy_plugin() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    distribution_name = project["name"]

    assert distribution_name.startswith("lerobot_policy_")
    assert importlib.import_module(distribution_name).__name__ == distribution_name


def test_lerobot_dependency_matches_required_processor_api() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]

    assert "lerobot>=0.6.2,<0.7" in project["dependencies"]


def test_console_entry_point_resolves_to_callable() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    module_name, function_name = project["scripts"]["open-trajectory"].split(":", maxsplit=1)

    function = getattr(importlib.import_module(module_name), function_name)

    assert callable(function)


def test_lerobot_factories_discover_plugin_conventions() -> None:
    config = make_policy_config("open_trajectory")

    policy_class = get_policy_class("open_trajectory")
    preprocessor, postprocessor = make_pre_post_processors(config)

    assert config.__class__.__name__ == "OpenTrajectoryConfig"
    assert policy_class.__name__ == "OpenTrajectoryPolicy"
    assert len(preprocessor.steps) == 4
    assert len(postprocessor.steps) == 2
