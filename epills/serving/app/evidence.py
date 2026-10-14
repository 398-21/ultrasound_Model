"""
What /v1/model-info says about the served model (blueprint B25): how similarity is computed, and the measured error
rates with their sources. Every rate is copied from an evaluation output in this repository: `source` is the file and
`keys` the path of each rate inside it ("/"-separated; `scale` 100 when the file has percent).
tests/test_serving.py checks the copies against those files when they are present.
Rates are fractions (0.012 = 1.2%). All are for run 4 (outputs/runs/stage2_run4) at its served threshold and margin.
"""

SIMILARITY = {
    "measure": "cosine similarity of L2-normalised 256-number embeddings (a dot product); 1.0 = same direction",
    "personal_gallery": "/v1/match: each medication is the normalised mean of its enrolment embeddings; a pill takes "
                        "the most similar medication, `unknown` below the threshold, `ambiguous` when the runner-up "
                        "is within the margin; the imprint read on the pill can then confirm or overturn it",
    "catalogue": "/v1/identify: per product, the best similarity over its catalogue photos (averaged over the photos "
                 "sent of one pill); the closest products are re-ranked by the imprint read; exact search over every "
                 "photo, no approximate index",
    "not_a_probability": "Similarities are not probabilities: pills that aren't in the catalogue still score about "
                         "0.93. /v1/identify's `confidence` is the calibrated probability, `verdict` its reading.",
}

MODEL_VERSION = "stage2_run4@2026-10-09"  # the rates below hold for this served model only (its version string)
RUN4 = "outputs/runs/stage2_run4/eval/"
IDENTIFY_KEYS = {"top1_right": "test/top1_right",
                 "possible_match_share": "test/catalogue_pills_by_verdict/possible_match/share",
                 "possible_match_right": "test/catalogue_pills_by_verdict/possible_match/top1_right",
                 "not_in_catalogue_called_possible_match": "test/other_pills_by_verdict/possible_match"}
GALLERY = {"correct_match": "correct_match", "wrong_match": "wrong_match", "own_flip": "own_flip",
           "own_rejected": "own_rejected", "unknown_matched": "unknown_matched", "unknown_rejected": "unknown_rejected"}

ERROR_RATES = [
    {
        "id": "personal_gallery",
        "what": "/v1/match, a patient with 10 medications and 3 enrolment photos each, on C3PI photos of 200 "
                "medicines the model never saw in training",
        "rates": {"correct_match": 0.7599, "wrong_match": 0.0004, "own_flip": 0.095, "own_rejected": 0.1447,
                  "unknown_matched": 0.0377, "unknown_rejected": 0.9483},
        "meaning": "wrong_match: named as another of the patient's medicines; own_flip: asked to photograph again; "
                   "own_rejected: their own pill called unknown; unknown_matched: a pill that isn't theirs named "
                   "as one of theirs",
        "source": RUN4 + "c3pi_holdout.json",
        "keys": {k: "open_set/gallery_10/at_calibration/" + v for k, v in GALLERY.items()},
    },
    {
        "id": "personal_gallery_with_imprint",
        "what": "As personal_gallery, with the imprint check on (the GPU reader: Florence-2-large, framed as served; "
                "the pill's own imprint is read on 47% of photos)",
        "rates": {"correct_match": 0.7826, "wrong_match": 0.0006, "own_flip": 0.0721, "unknown_matched": 0.0351},
        "source": RUN4 + "imprint_gallery_large_c92_distinct.json",
        "keys": {k: "test/with_check/" + k for k in ("correct_match", "wrong_match", "own_flip", "unknown_matched")},
    },
    {
        "id": "personal_gallery_with_imprint_cpu",
        "what": "As personal_gallery_with_imprint, read by the CPU reader (Florence-2-base, framed as served): most "
                "live calls, since the free GPU quota is per caller; the pill's own imprint is read on 43% of photos",
        "rates": {"correct_match": 0.7731, "wrong_match": 0.0006, "own_flip": 0.0815, "unknown_matched": 0.0351},
        "source": RUN4 + "imprint_gallery_base_c92_distinct.json",
        "keys": {k: "test/with_check/" + k for k in ("correct_match", "wrong_match", "own_flip", "unknown_matched")},
    },
    {
        "id": "personal_gallery_other_phones",
        "what": "As personal_gallery on MEDISEG: 32 Hong Kong products, other phones, lighting and backgrounds",
        "rates": {"correct_match": 0.1872, "wrong_match": 0.0069, "own_flip": 0.1021, "own_rejected": 0.7038,
                  "unknown_matched": 0.0306, "unknown_rejected": 0.9309},
        "meaning": "Outside C3PI's photo style most of a patient's own pills are called unknown (safe but unhelpful): "
                   "enrol with the patient's own phone and background",
        "source": RUN4 + "mediseg.json",
        "keys": {k: "open_set/gallery_10/at_calibration/" + v for k, v in GALLERY.items()},
    },
    {
        "id": "same_ingredient_siblings",
        "what": "Both siblings in one gallery (same ingredients, other strength or release form), with the imprint "
                "check (GPU reader); look_alike: both have the same recorded shape and colour",
        "rates": {"named_as_sibling": 0.0011, "named_as_sibling_look_alike": 0.0058},
        "meaning": "A stress test: in look-alike pairs the imprint check settles more ties (correct 53% -> 65%) and "
                   "occasionally picks the sibling (0.35% -> 0.58%); strength is never confirmed from a photo",
        "source": RUN4 + "lookalike_strengths_large_c92_distinct.json",
        "keys": {"named_as_sibling": "all/with imprint/named as the sibling",
                 "named_as_sibling_look_alike": "look_alike/with imprint/named as the sibling"},
        "scale": 100,
    },
    {
        "id": "same_maker_imprints",
        "what": "Two different medicines from one maker whose imprints share a token (Teva's '93', 'TEVA', 'TV') in one "
                "gallery, with the imprint check (GPU reader): 3,149 pairs; look_alike: same recorded shape and colour",
        "rates": {"named_as_the_other": 0.0001, "named_as_the_other_look_alike": 0.001,
                  "named_as_the_other_shared_tv": 0.0015},
        "meaning": "Rare overall; '93' pairs 0.00%. The worst pair: tramadol 50 mg (TV 58) named as carvedilol 12.5 mg "
                   "(TV 7295) on 5.8% of its photos. A token two of the patient's medicines share never confirms either",
        "source": RUN4 + "same_maker_imprints_large_c92_distinct.json",
        "keys": {"named_as_the_other": "all/with imprint/named as the other",
                 "named_as_the_other_look_alike": "look_alike/with imprint/named as the other",
                 "named_as_the_other_shared_tv": "shared 'TV'/with imprint/named as the other"},
        "scale": 100,
    },
    {
        "id": "catalogue_identify",
        "what": "/v1/identify on held-out C3PI photos of catalogue products (photos not in the catalogue), and on "
                "MEDISEG pills that aren't in the catalogue; imprints read by the GPU reader",
        "rates": {"top1_right": 0.802, "possible_match_share": 0.758, "possible_match_right": 0.867,
                  "not_in_catalogue_called_possible_match": 0.07},
        "source": RUN4 + "identify_confidence_large_c92.json",
        "keys": IDENTIFY_KEYS,
    },
    {
        "id": "catalogue_identify_cpu",
        "what": "As catalogue_identify with imprints read by the CPU reader (most live calls), scored with the served "
                "confidence rule as it is",
        "rates": {"top1_right": 0.792, "possible_match_share": 0.751, "possible_match_right": 0.864,
                  "not_in_catalogue_called_possible_match": 0.074},
        "source": RUN4 + "identify_confidence_base_c92_served.json",
        "keys": IDENTIFY_KEYS,
    },
    {
        "id": "pill_count",
        "what": "Stand-in pill finder (/v1/detect, /v1/scan without boxes) on 1,000 one-pill C3PI photos (test case "
                "D01), and 2- and 3-pill photos made from them (D02)",
        "rates": {"exact_count_1": 0.878, "no_boxes_1": 0.068, "over_count_1": 0.054,
                  "over_count_while_reliable_1": 0.014, "exact_count_2": 0.86, "exact_count_3": 0.8511},
        "meaning": "over_count_while_reliable: an extra box while count_reliable is true. The backend confirms with "
                   "the patient before recording an extra pill (docs/dose_rules.md)",
        "source": "outputs/eval/pill_finder.json",
        "keys": {"exact_count_1": "by_pills/1/exact_count", "no_boxes_1": "by_pills/1/no_boxes",
                 "over_count_1": "by_pills/1/over_count",
                 "over_count_while_reliable_1": "by_pills/1/over_count_while_reliable",
                 "exact_count_2": "by_pills/2/exact_count", "exact_count_3": "by_pills/3/exact_count"},
    },
    {
        "id": "input_check",
        "what": "Input guard (B23) on a small labelled set of web photos: 14 loose-pill photos, 8 blisters, 7 bottles "
                "or boxes, 5 printed labels",
        "rates": {"loose_pills_wrongly_refused": 0.0, "blister_not_caught": 0.125, "package_not_caught": 0.143,
                  "printed_not_caught": 0.0},
        "meaning": "A small set: a sanity check, not a measured error rate",
        "source": "src/epills/core/inference/input_type.py (docstring)",
        "keys": {},
    },
]


def error_rates(model_version):
    """The rates as /v1/model-info shows them; none for a model they weren't measured on."""
    if model_version != MODEL_VERSION:
        return []
    return [{k: v for k, v in e.items() if k != "scale"} for e in ERROR_RATES]


def lookup(data, path):
    for key in path.split("/"):
        data = data[key]
    return data
