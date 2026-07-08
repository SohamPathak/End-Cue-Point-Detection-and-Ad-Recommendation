"""Loader for externalised LLM prompt templates.

Prompts live as editable ``.txt`` files under ``configs/prompts`` so they can be
tuned per use-case without code changes — outro/ad styles are not one-size-fits-all.
Templates use ``str.format`` placeholders filled by the caller.
"""

from __future__ import annotations

import functools

from brahma.config import get_config


@functools.lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load a prompt template by file stem (without extension).

    Args:
        name: The template name, e.g. ``"outro_detection"``.

    Returns:
        The raw template text.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    cfg = get_config()
    path = cfg.abs_path(cfg.paths.prompts_dir) / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs: object) -> str:
    """Load and format a prompt template.

    Args:
        name: The template name.
        **kwargs: Placeholder values for ``str.format``.

    Returns:
        The formatted prompt.
    """
    return load_prompt(name).format(**kwargs)
