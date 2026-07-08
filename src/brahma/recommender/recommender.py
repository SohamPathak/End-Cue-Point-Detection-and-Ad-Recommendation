"""Ad recommender — the system's decision-maker.

The recommender consumes a :class:`FeatureSet` and the ad catalog, then chooses
an ad and explains why. It is *fixed per experiment* via ``config.yaml``:

* ``strategy: rule`` — deterministic tag matching (reproducible, no LLM). This
  is also the safe fallback when the LLM strategy fails.
* ``strategy: llm``  — Gemini reasons over the features + catalog metadata and
  returns a structured choice.

Ad selection is deliberately *not* a hard-coded ``weather -> ad`` map: ads carry
metadata tags, so new features/ads slot in without touching this logic.
"""

from __future__ import annotations

import json

from brahma.clients.gemini import get_gemini_client, parse_json_response
from brahma.config import AppConfig, get_config
from brahma.exceptions import RecommendationError
from brahma.models import AdRecommendation, FeatureSet


class Recommender:
    """Selects an ad for a collected feature set."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialise the recommender from config.

        Args:
            config: Optional config override.
        """
        self._config = config or get_config()

    def recommend(self, features: FeatureSet) -> AdRecommendation:
        """Recommend an ad for the given features using the configured strategy.

        Args:
            features: The collected signals.

        Returns:
            The chosen :class:`AdRecommendation`.

        Raises:
            RecommendationError: If no ad can be chosen.
        """
        # agent.enabled=false forces the deterministic rule strategy for a
        # fully reproducible, LLM-free run.
        strategy = (
            self._config.recommender.strategy.lower()
            if self._config.agent.enabled
            else "rule"
        )
        if strategy == "llm":
            try:
                return self._recommend_llm(features)
            except Exception as exc:  # noqa: BLE001 - fall back, never dead-end
                # LLM strategy is best-effort; degrade to the deterministic rule.
                fallback = self._recommend_rule(features)
                fallback.reason = (
                    f"{fallback.reason} "
                    f"(LLM strategy failed: {exc}; used rule fallback)"
                )
                return fallback
        return self._recommend_rule(features)

    def _recommend_rule(self, features: FeatureSet) -> AdRecommendation:
        """Deterministic tag-matching strategy.

        Scores each ad by how many of its ``tags`` match the collected feature
        values; highest score wins.
        """
        flat = features.as_dict()
        scores: dict[str, float] = {}
        for ad in self._config.ads:
            matches = sum(1 for k, v in ad.tags.items() if flat.get(k) == v)
            scores[ad.id] = float(matches)

        best_id = max(scores, key=lambda k: scores[k]) if scores else None
        if best_id is None or scores[best_id] == 0:
            raise RecommendationError(f"No ad matched the collected features: {flat}")

        chosen = self._config.ad_by_id(best_id)
        if chosen is None:  # unreachable: best_id came from the ad catalog
            raise RecommendationError(f"Chosen ad id not in catalog: {best_id}")
        matched = {k: v for k, v in chosen.tags.items() if flat.get(k) == v}
        reason = (
            f"Selected '{chosen.id}' — its tags {matched} match the current signals "
            f"{flat}."
        )
        return AdRecommendation(
            ad_id=chosen.id,
            ad_path=chosen.path,
            reason=reason,
            strategy="rule",
            experiment_id=self._config.recommender.experiment_id,
            scores=scores,
        )

    def _recommend_llm(self, features: FeatureSet) -> AdRecommendation:
        """Gemini-reasoning strategy — returns a structured ad choice."""
        catalog = [
            {"id": a.id, "tags": a.tags, "description": a.description}
            for a in self._config.ads
        ]
        prompt = (
            "You are an ad recommendation engine. Given the current signals and an "
            "ad catalog, pick the single best ad to show. Respond with ONLY a JSON "
            'object: {"ad_id": <id>, "reason": <short justification>}.\n\n'
            f"Signals: {json.dumps(features.as_dict())}\n"
            f"Ad catalog: {json.dumps(catalog)}\n"
        )
        raw = get_gemini_client().generate_text(
            prompt,
            model=self._config.recommender.llm_model,
            temperature=0.0,
        )
        choice = parse_json_response(raw)
        ad = self._config.ad_by_id(choice["ad_id"])
        if ad is None:
            raise RecommendationError(
                f"LLM chose unknown ad id: {choice.get('ad_id')!r}"
            )
        return AdRecommendation(
            ad_id=ad.id,
            ad_path=ad.path,
            reason=str(choice.get("reason", "")).strip(),
            strategy="llm",
            experiment_id=self._config.recommender.experiment_id,
        )
