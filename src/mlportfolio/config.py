"""Configuration loading and a typed view over the YAML config.

The config is intentionally a thin wrapper around a nested dict so it stays easy
to serialize, override from the CLI, and inspect in tests.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml"
)


@dataclass
class Config:
    """A lightweight, dictionary-backed configuration object."""

    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        path = path or _DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(raw=data)

    @classmethod
    def default(cls) -> "Config":
        return cls.load()

    # ------------------------------------------------------------------ #
    # Access helpers
    # ------------------------------------------------------------------ #
    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup: cfg.get('optimize', 'risk_aversion')."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def override(self, updates: Dict[str, Any]) -> "Config":
        """Return a deep-copied config with top-level sections updated.

        ``updates`` may be a flat dict of dotted paths, e.g.
        ``{"dates.start": "2015-01-01"}``.
        """
        new_raw = copy.deepcopy(self.raw)
        for dotted, value in updates.items():
            if value is None:
                continue
            parts = dotted.split(".")
            node = new_raw
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(raw=new_raw)

    # Convenient typed properties used throughout the codebase. ----------- #
    @property
    def seed(self) -> int:
        return int(self.get("seed", default=42))

    @property
    def tickers(self) -> list[str]:
        return list(self.get("universe", "tickers", default=[]))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config(tickers={self.tickers}, seed={self.seed})"
