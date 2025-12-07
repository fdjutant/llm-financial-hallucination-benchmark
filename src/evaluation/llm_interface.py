"""LLM interface abstraction for benchmarking UK financial QA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    """Protocol describing the interface expected from LLM backends."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - interface only
        """Return an LLM response for a given prompt."""
        ...


@dataclass
class LoggedResponse:
    """Represents a single LLM response tied to a QA example."""

    question: str
    model_name: str
    response: str


def query_model(client: LLMClient, question: str, model_name: str) -> LoggedResponse:
    """Query a model and capture metadata for UK benchmark runs."""
    return LoggedResponse(question=question, model_name=model_name, response=client.generate(question))
