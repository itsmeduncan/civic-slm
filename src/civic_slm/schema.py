"""Pydantic data contracts for the civic-slm pipeline.

Every artifact that crosses a stage boundary — crawled docs, chunked text,
synthetic instruction pairs, preference pairs, eval examples, eval results —
serializes through these models. Validation is the gate; if it doesn't round-trip,
it doesn't land in `data/`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DocType(StrEnum):
    """Common U.S. local government document types.

    Naming choices reflect the most common term across the country, but cover
    regional variations: California uses "general plan", most other states use
    "comprehensive plan" or "master plan" for the same artifact, so all three
    are accepted. `municipal_code` covers ordinances codified into a single
    document; `ordinance` and `resolution` are individual legislative items.
    """

    AGENDA = "agenda"
    MINUTES = "minutes"
    STAFF_REPORT = "staff_report"
    GENERAL_PLAN = "general_plan"
    COMPREHENSIVE_PLAN = "comprehensive_plan"
    MASTER_PLAN = "master_plan"
    ZONING_ORDINANCE = "zoning_ordinance"
    MUNICIPAL_CODE = "municipal_code"
    ORDINANCE = "ordinance"
    RESOLUTION = "resolution"
    BUDGET = "budget"
    RFP = "rfp"
    NOTICE = "notice"
    OTHER = "other"


class TaskType(StrEnum):
    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    QA_GROUNDED = "qa_grounded"
    REFUSAL = "refusal"
    DIFF = "diff"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class CivicDocument(_Frozen):
    """A single source document crawled from a local-government website.

    Scoped to U.S. cities, counties, and townships at v0. The `state` field is
    a 2-letter postal abbreviation (`CA`, `TX`, `NY`...) and is required so
    downstream filters (e.g. "train only on Sun Belt cities") work without
    parsing the city slug.
    """

    id: str = Field(
        min_length=1, description="Stable id, typically `{state}/{city}/{sha256[:12]}`."
    )
    jurisdiction: str = Field(
        min_length=1, description="Slug of city, county, or other unit; e.g. `san-clemente`."
    )
    state: str = Field(
        min_length=2,
        max_length=2,
        pattern=r"^[A-Z]{2}$",
        description="2-letter U.S. state postal code, e.g. `CA`, `TX`, `NY`.",
    )
    doc_type: DocType
    source_url: HttpUrl
    retrieved_at: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_path: str = Field(description="Path under data/raw/, gitignored.")
    text: str = Field(min_length=1, description="Extracted text (no formatting).")


class DocumentChunk(_Frozen):
    """A chunk of a `CivicDocument` ready for embedding or training."""

    doc_id: str
    chunk_idx: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    section_path: list[str] = Field(
        default_factory=list,
        description="Heading trail, e.g. ['Land Use', 'Goals', 'LU-1'].",
    )


class Provenance(_Frozen):
    """Where a generated example came from — model id, prompt hash, timestamp."""

    generator: Literal["claude", "human", "model_v0"]
    model: str | None = None
    prompt_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


class InstructionExample(_Frozen):
    """A single (system, input, output) triple for SFT."""

    id: str = Field(min_length=1)
    task: TaskType
    system: str = Field(min_length=1)
    input: str = Field(min_length=1)
    output: str = Field(min_length=1)
    source_chunk_ids: list[str] = Field(default_factory=list)
    provenance: Provenance

    def to_chat_record(self) -> dict[str, list[dict[str, str]]]:
        """Convert to the chat-format row `mlx_lm.lora --data` expects.

        `{messages: [{role, content}, ...]}`. This is the contract between the
        SFT corpus and MLX-LM's `format: chat` mode. Keep this method and the
        `configs/sft.yaml` `data.format` field in lock-step.
        """
        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.input},
                {"role": "assistant", "content": self.output},
            ]
        }


class PreferencePair(_Frozen):
    """Chosen/rejected pair for DPO."""

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    rationale: str | None = None
    source_example_id: str | None = None


# --- Eval examples — discriminated union over `bench` -------------------------


class _EvalBase(_Frozen):
    id: str = Field(min_length=1)


class FactualityExample(_EvalBase):
    bench: Literal["factuality"] = "factuality"
    question: str
    context: str
    gold_answer: str
    gold_citations: list[str] = Field(default_factory=list)


class RefusalExample(_EvalBase):
    bench: Literal["refusal"] = "refusal"
    question: str
    context: str = Field(description="Context that does NOT contain the answer.")
    expected_refusal: bool = True


class ExtractionExample(_EvalBase):
    bench: Literal["extraction"] = "extraction"
    document_text: str
    gold_json: dict[str, object]
    schema_name: str = Field(min_length=1)


class SideBySideExample(_EvalBase):
    bench: Literal["side_by_side"] = "side_by_side"
    prompt: str
    rubric: str | None = None


EvalExample = Annotated[
    FactualityExample | RefusalExample | ExtractionExample | SideBySideExample,
    Field(discriminator="bench"),
]


# --- Eval results -------------------------------------------------------------


class EvalResult(_Frozen):
    """One model's output on one eval example, with score and judge notes."""

    model_id: str = Field(min_length=1, description="e.g. `qwen-civic-sft-v1`.")
    bench: Literal["factuality", "refusal", "extraction", "side_by_side"]
    example_id: str
    prediction: str
    score: float = Field(ge=0.0, le=1.0)
    judge_notes: str | None = None
    latency_ms: float = Field(ge=0.0)
