"""
The shape of every PharmaScan API answer (blueprint job B5), so /openapi.json documents responses, not only requests.
Models allow extra fields: new fields can be added without breaking clients that validate against these.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Evidence = Literal["visual"]


class Out(BaseModel):
    model_config = ConfigDict(extra="allow")


class InputCheck(Out):
    type: Literal["loose_pills", "blister", "package", "printed", "document", "unchecked"]
    caption: str | None = Field(None, description="What the photo looked like to the checker")
    checked: bool
    abstained: bool = Field(description="True: not loose pills, so nothing was compared")
    message: str | None = None


class Error(Out):
    code: str = Field(description="unauthorised, not_found, version_mismatch, invalid_input, upstream_failed, "
                                  "not_ready or error")
    detail: str | list | dict = Field(description="What went wrong (for invalid_input: the fields at fault)")
    retry_after: int | None = Field(None, description="Seconds to wait before retrying (with not_ready)")


# ---- health and model info -------------------------------------------------------------------------------------------
class Part(Out):
    loaded: bool
    error: str | None = None


class ModelPart(Part):
    version: str | None = None
    embedding_space_id: str | None = None
    device: str | None = Field(None, description="zerogpu, or cpu (slower, when no GPU is granted)")


class CataloguePart(Part):
    products: int | None = None
    model_version: str | None = None


class SafetyPart(Part):
    built_at: str | None = None
    pairs: str | None = None


class TextPart(Part):
    model: str | None = Field(None, description="The reader on the CPU")
    gpu_model: str | None = Field(None, description="The reader on the lent GPU (the Space only); answers say which "
                                                    "ran")
    gpu_imprint_adapter: str | None = Field(None, description="The adapter trained on pill imprints that the GPU "
                                                              "reader uses for imprints only, or null")


class Health(Out):
    status: Literal["ok"]
    ready: bool = Field(description="The model, catalogue and safety database are all loaded")
    api_version: str
    started_at: str
    model_repo: str
    model: ModelPart
    catalogue: CataloguePart
    safety_db: SafetyPart
    text_reading: TextPart


class ErrorRate(Out):
    id: str
    what: str = Field(description="The test: which endpoint, which photos")
    rates: dict[str, float] = Field(description="Fractions (0.012 = 1.2%)")
    meaning: str | None = None
    source: str = Field(description="The evaluation output in the repository the rates are copied from")
    keys: dict[str, str] = Field(description="Where each rate sits inside `source` ('/'-separated)")


class ModelInfo(Out):
    api_version: str
    model_version: str
    embedding_space_id: str = Field(description="Compare embeddings only within one space")
    preprocess_version: str
    embedding_dim: int
    normalized: bool
    crop_window: float = Field(description="Crop contract: square window = this x the pill box's long side")
    threshold: float
    margin: float
    calibration: dict
    imprint: dict
    training_coverage: str
    known_limits: list[str]
    similarity: dict[str, str] = Field(description="How similarities are computed, and why they aren't probabilities")
    error_rates: list[ErrorRate] = Field(description="Measured error rates of the served model, each with its source; "
                                                     "empty for a model they weren't measured on")
    evidence: Evidence


# ---- Stage 2 ---------------------------------------------------------------------------------------------------------
class Embedding(Out):
    embedding: list[float]
    dim: int
    model_version: str
    embedding_space_id: str
    preprocess_version: str
    normalized: bool
    framed_by: Literal["box", "as sent"]
    imprint_reads: list[str] | None = None
    text_reading: Literal["used", "not_requested", "unavailable", "skipped"] = Field(
        "not_requested", description="Whether the imprint was read: unavailable (the text reader isn't working) "
                                     "or skipped (GPU busy, too slow on the CPU) mean nothing imprint-based was applied")
    evidence: Evidence


class Ref(Out):
    medication_id: str
    name: str
    reference_set_id: str | None = None
    similarity: float


class MatchResult(Out):
    decision: Literal["match", "unknown", "ambiguous"]
    medication_id: str | None
    name: str | None
    reference_set_id: str | None = Field(None, description="As sent in the gallery, for the matched medication")
    similarity: float
    closest: Ref
    runner_up: Ref | None
    message: str
    imprint_check: str | None = Field(None, description="confirmed, resolved, conflict, not_read, not_used, unresolved, "
                                                       "not_checked, unavailable or skipped (no imprint read); null when "
                                                       "no reads were sent")
    imprint_read_for: list[str] = []


class MatchResponse(Out):
    results: list[MatchResult]
    model_version: str
    embedding_space_id: str
    threshold: float
    margin: float
    evidence: Evidence
    limits: str


class Picture(Out):
    url: str
    kind: str


class Strength(Out):
    ingredient: str
    strength: str


class Product(Out):
    pill_id: str
    name: str | None
    rxnorm_name: str | None
    generic_name: str | None
    manufacturer: str | None
    ingredients: list[str]
    strength: str | None = Field(None, description="e.g. '500 MG', or '40 MG / 1100 MG' for a combination")
    strengths: list[Strength] = []
    dose_form: str | None = Field(None, description="e.g. 'Oral Tablet', 'Extended Release Oral Capsule', 'Pack'")
    release: Literal["IR", "ER", "DR"] | None = Field(None, description="Immediate, extended or delayed release")
    brand: str | None = None
    brand_or_generic: Literal["brand", "generic"] | None = None
    pack: list[dict] | None = Field(None, description="For packs: each kind of pill, with count and strength")
    look_alike_strengths: list[dict] = Field([], description="Catalogue products with the same ingredients in "
                                                             "another strength or release and the same recorded "
                                                             "shape and colour: {pill_id, strength, release}. A photo "
                                                             "can't tell these apart; the strength needs the label")
    ndc9: str | None = None
    ndc11: str | None = None
    rxcui: str | None = None
    rxnorm_status: str | None = None
    market: Literal["US"] = "US"
    shape: str | None
    colour: str | None
    size_mm: float | None
    imprint: str | None
    images: list[Picture]
    image_licence: str | None = None


class Candidate(Product):
    similarity: float = Field(description="Cosine similarity of appearance: not a probability")
    imprint_match: bool = Field(description="The typed imprint fits this product")
    imprint_read: float = Field(description="How well the imprint read on the photo fits this product, 0-1")


class Identification(Out):
    candidates: list[Candidate]
    notes: list[str]
    note: str
    confidence: float | None = Field(None, description="Chance the first candidate is the right product, if the pill "
                                                       "is a US catalogue product (calibrated; null with filters)")
    catalogue_fit: Literal["good", "weak", "poor"] | None = None
    verdict: Literal["possible_match", "uncertain", "no_confident_match"] | None = None
    verdict_message: str | None = None


class IdentifyResponse(Identification):
    imprint_reads: list[str] | None = None
    input_check: InputCheck | None = None
    text_reading: Literal["used", "not_requested", "unavailable", "skipped"] = Field(
        "not_requested", description="Whether the imprint was read: unavailable (the text reader isn't working) "
                                     "or skipped (GPU busy, too slow on the CPU) mean nothing imprint-based was applied")
    model_version: str
    evidence: Evidence
    limits: str


class Detection(Out):
    boxes: list[list[float]] = Field(description="[x0, y0, x1, y1] per pill, in the photo's pixels")
    box_flags: list[list[str]] = Field(description="Per box: touches_edge, merged, irregular, small")
    pill_count: int
    count_reliable: bool = Field(description="False when any box is flagged or blobs were left out: don't count "
                                             "pills from this photo")
    flags: list[str] = Field(description="Per photo: no_pills, count_uncertain, too_many")
    found_by: str


class DetectResponse(Detection):
    input_check: InputCheck | None = None
    message: str | None = None


class CheckInputResponse(Out):
    input_check: InputCheck
    next_tools: list[str] = Field(description="Agent tools (GET /v1/tools) that fit this photo; empty: ask the user "
                                              "for the box, label or prescription, or the pharmacist")
    message: str


class Alert(Out):
    type: str
    severity: str
    message: str
    medication_ids: list[str]
    source: str
    severity_graded: bool
    colour: str
    speak: bool
    notify_caregiver: bool
    details: dict


class DoseSummary(Out):
    pills_seen: int
    matched: int
    unknown: int
    unsure: int
    all_ok: bool = Field(description="Deprecated name of all_visually_matched")
    all_visually_matched: bool
    count_reliable: bool = True
    text: str


class DoseResponse(Out):
    alerts: list[Alert]
    summary: DoseSummary


class ScanDose(DoseResponse):
    deprecated: bool = True
    note: str


class ScanPill(Out):
    box: list[float]
    quality_flags: list[str] = []
    counted: bool = Field(True, description="False for a flagged box: not counted in the dose check")
    imprint_reads: list[str] | None = None


class ScanResponse(Out):
    found_by: str
    model_version: str
    detection: Detection
    pills: list[ScanPill] = Field(description="With a gallery: /v1/match's fields per pill; without: /v1/identify's")
    dose: ScanDose | None = None
    input_check: InputCheck | None = None
    message: str | None = None
    text_reading: Literal["used", "not_requested", "unavailable", "skipped"] = Field(
        "not_requested", description="Whether the imprint was read: unavailable (the text reader isn't working) "
                                     "or skipped (GPU busy, too slow on the CPU) mean nothing imprint-based was applied")
    evidence: Evidence
    limits: str


class Ingredient(Out):
    rxcui: str | None
    name: str


class Printed(Out):
    value: str
    line: str = Field(description="The line(s) it was read from")


class Dose(Out):
    amount: float = Field(description="Per dose, e.g. 1, 0.5")
    amount_max: float | None = Field(None, description="For a range: 'Take 1-2 tablets' -> 2")
    unit: str | None = Field(None, description="tablet, capsule, ml, puff, drop, sachet, ...")


class Instructions(Out):
    text: str = Field(description="As printed, e.g. 'Take ONE tablet TWICE a day AFTER FOOD'")
    line: str = Field(description="The line(s) it was read from")
    dose: Dose | None = None
    times_per_day: int | None = Field(None, description="Only when printed ('twice a day', 'BD', 'daily')")
    every_hours: float | None = Field(None, description="'every 6 hours'; never turned into times a day")
    as_needed: bool = Field(False, description="'when necessary', 'as needed', 'PRN'")
    times_of_day: list[Literal["morning", "afternoon", "evening", "night", "bedtime"]] = []
    food: Literal["before_food", "after_food", "with_food", "with_or_without_food", "empty_stomach"] | None = None
    duration: str | None = Field(None, description="'7 days'")
    max_per_day: str | None = Field(None, description="'8 tablets in 24 hours'")


class Caution(Printed):
    kind: Literal["storage", "side_effect", "how_to_take", "warning"] = Field(
        "warning", description="As MOH's 'Know about medication label' groups them: store your medication, manage "
                               "common side-effects, how to take it; anything else is a warning")


class Quantity(Printed):
    amount: float | None = Field(None, description="e.g. 60")
    unit: str | None = Field(None, description="e.g. tablet, capsule, ml")


class DocumentFields(Out):
    directions: Printed | None = Field(None, description="As printed, e.g. 'Take 1 tablet twice a day after meals'")
    instructions: Instructions | None = Field(None, description="The directions, parsed into their parts; a part "
                                                                "that isn't printed is null")
    indication: Printed | None = Field(None, description="What the medicine is for, as printed ('For high blood "
                                                         "pressure' -> 'high blood pressure')")
    cautions: list[Caution] = Field([], description="Cautionary notes as printed ('May cause drowsiness', 'Avoid "
                                                    "alcohol', 'Swallow whole'), each with its kind")
    quantity: Quantity | None = Field(None, description="Total supplied, e.g. 'Qty: 60 TAB' -> 60 tablet")
    dose_form: Printed | None = None
    date: Printed | None = Field(None, description="Dispensing or prescription date, as printed")
    expiry_date: Printed | None = None


class ListCheck(Out):
    status: Literal["same_medicine", "different_strength", "contains_listed", "not_on_list"]
    medication_id: str | None
    message: str


class LabelCandidate(Out):
    name: str
    brand: str | None
    ingredients: list[Ingredient]
    strength: str | None
    release: Literal["ER", "DR"] | None = Field(None, description="Extended or delayed release, when printed (LA, SR, "
                                                                 "XR, XL, CR, MR, EC, ...); the RxNorm product always "
                                                                 "has this release, or none is given")
    release_printed: str | None = Field(None, description="The release marker as printed, e.g. 'LA'")
    rxcui: str | None
    rxnorm_name: str | None
    found_in: list[str] = Field(description="Only the lines that name the medicine")
    first_line: int | None = None
    fields: DocumentFields | None = Field(None, description="What is printed about this medicine; null when not "
                                                            "printed, never inferred")
    list_check: ListCheck | None = Field(None, description="With `medications`: compared with the patient's list")


class ReadLabelResponse(Out):
    candidates: list[LabelCandidate]
    label_kind: Literal["dispensing", "package", "none"] = Field(
        "none", description="dispensing: a pharmacy sticker or prescription with instructions for the patient; "
                            "package: a manufacturer's box or bottle (no instructions); none: no medicine found")
    dispensed_by: Printed | None = Field(None, description="The dispensing pharmacy or clinic as printed")
    note: str
    ocr_model: str
    source: Literal["photo", "pdf_text", "pdf_scan"] = "photo"
    message: str | None = None


class SearchProduct(Product):
    match: Literal["same", "contains", "name"]


class SearchResponse(Out):
    resolution: dict
    count: int
    products: list[SearchProduct]
    source: str | None


# ---- Stage 3 and feedback --------------------------------------------------------------------------------------------
class Coverage(Out):
    medications: int
    screened: int = Field(description="Medications with at least one ingredient in the interaction database")
    not_screened: list[str]
    ingredients_not_in_interaction_database: list[str]
    pairs: int
    pairs_checked: int
    pairs_not_checkable: int


class MedicationsResponse(Out):
    alerts: list[Alert]
    medications: list[dict]
    all_identified: bool
    coverage: Coverage
    note: str = Field(description="No alert doesn't mean the medicines are safe together")
    sources: dict


class LabelResponse(Out):
    brand_name: str | None
    generic_name: str | None
    label_date: str | None = Field(description="The label's effective date (YYYYMMDD)")
    matched_by: str
    sections: dict[str, str]
    disclaimer: str
    source: str | None = None
    set_id: str | None = None
    label_id: str | None = None
    url: str | None = Field(None, description="The label on DailyMed, for citation")


class FeedbackStored(Out):
    stored: bool
    item: str
    note: str


class FeedbackDeleted(Out):
    deleted: bool = Field(description="True only once checked: nothing of this consent id is left here or in the "
                                      "repo")
    local_photos_removed: int
    removed_from_repo: bool
    history_squashed: bool = Field(False, description="The repo's history was rewritten, so old commits no "
                                                      "longer hold the photos")
    verified: bool = False
