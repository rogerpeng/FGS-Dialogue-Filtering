"""Bounded filtering/refinement with final-output verification and abstention.

Topic relevance filters do NOT establish doctrinal correctness or harmlessness.
The optional independent guard is applied to every variant, including baselines.
"""
from dataclasses import dataclass, asdict
from itertools import product
import time

FALLBACK = "這次無法提供經確認適切的回答。請改寫問題，或向具資格的法師／相關專業人士確認。"


@dataclass(frozen=True)
class Variant:
    name: str
    query_filter: bool
    answer_filter: bool
    refinement: bool


VARIANTS = {"b1": Variant("b1", False, False, False),
            "b2": Variant("b2", True, True, False),
            "full": Variant("full", True, True, True),
            "no_query": Variant("no_query", False, True, True),
            "no_answer": Variant("no_answer", True, False, True),
            "refinement_only": Variant("refinement_only", False, False, True),
            "query_only": Variant("query_only", True, False, False),
            "answer_only": Variant("answer_only", False, True, False)}


class DialoguePipeline:
    def __init__(self, backend, question_filter=None, answer_filter=None, guard=None,
                 max_generation_attempts=3, max_refinements=3):
        if max_generation_attempts < 1 or max_refinements < 1:
            raise ValueError("Attempt limits must be positive")
        self.backend, self.question_filter, self.answer_filter, self.guard = backend, question_filter, answer_filter, guard
        self.max_generation_attempts, self.max_refinements = max_generation_attempts, max_refinements

    def run(self, question, variant=VARIANTS["full"], sample_id="demo", replicate=0):
        if not isinstance(question, str) or not question.strip() or len(question) > 10000:
            raise ValueError("Question must contain 1–10000 characters")
        if variant.query_filter and self.question_filter is None:
            raise ValueError("This variant requires a question filter")
        if variant.answer_filter and self.answer_filter is None:
            raise ValueError("This variant requires an answer filter")
        trace, started = [], time.perf_counter()

        def result(answer, status):
            return {"id": sample_id, "replicate": replicate, "system": variant.name, "question": question,
                    "answer": answer, "status": status, "variant": asdict(variant), "trace": trace,
                    "wall_seconds": time.perf_counter() - started}

        def acceptable(answer, stage):
            if not isinstance(answer, str) or not answer.strip():
                trace.append({"stage": stage, "accepted": False, "reason": "empty_output"})
                return False
            if self.guard is not None and not self.guard(question, answer):
                trace.append({"stage": stage, "accepted": False, "reason": "independent_guard"})
                return False
            ok = not variant.answer_filter or bool(self.answer_filter.accepts(answer))
            trace.append({"stage": stage, "accepted": ok, "reason": "domain_filter" if variant.answer_filter else "filter_disabled"})
            return ok

        try:
            if variant.query_filter:
                ok = bool(self.question_filter.accepts(question))
                trace.append({"stage": "query_filter", "accepted": ok})
                if not ok:
                    return result(FALLBACK, "query_rejected")
            answer = None
            for attempt in range(self.max_generation_attempts):
                # Shared initial draws across variants through backend request cache.
                candidate = self.backend.generate(question, sample_id, replicate, attempt)
                if acceptable(candidate, "generated_answer"):
                    answer = candidate
                    break
            if answer is None:
                return result(FALLBACK, "generation_exhausted")
            if not variant.refinement:
                return result(answer, "accepted")
            for attempt in range(self.max_refinements):
                refined = self.backend.refine(question, answer, sample_id, replicate, attempt)
                if acceptable(refined, "refined_answer"):
                    return result(refined, "accepted")
                # Always refine the last accepted candidate, not a rejected draft.
            return result(FALLBACK, "refinement_exhausted")
        except Exception as exc:
            # Retain error type without leaking credentials/provider response bodies.
            trace.append({"stage": "error", "error_type": type(exc).__name__})
            return result(FALLBACK, "provider_or_filter_error")


class FixtureBackend:
    """Deterministic offline demonstration, never a scored LLM experiment."""
    def generate(self, question, sample_id, replicate, attempt):
        return "可以先平靜心情，理解彼此的需要，再以尊重的態度溝通。"

    def refine(self, question, answer, sample_id, replicate, attempt):
        return "建議先讓心情平靜，傾聽彼此的需要，再用尊重且具體的方式溝通。這是一般性建議，並非星雲大師的原話。"
