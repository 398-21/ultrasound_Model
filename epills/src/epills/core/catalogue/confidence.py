"""
Honest confidence for catalogue identification (blueprint job B4). Raw similarities are not probabilities: pills that
aren't in the catalogue at all still score about 0.93. Fitted by scripts/calibrate_identify.py for one model; for any
other model version, assess() returns None (refit after every new model).

  confidence     chance the first candidate is the right product *if the pill is a US catalogue product*: logistic
                 fit of (top-1 similarity, gap to the runner-up, imprint read on the top-1) on C3PI validation photos
  catalogue_fit  poor / weak / good: poor below what 95% of rightly identified catalogue pills reach, good above what
                 90% of non-catalogue pills (MEDISEG) reach
  verdict        no_confident_match (poor fit or confidence < LOW), possible_match (good fit and confidence >= HIGH),
                 uncertain (the rest)
Refitted 2026-10-13 with the served GPU reader's imprint reads and the re-ranking re-chosen on the catalogue
(calibrate_identify.py --ocr large_c92; IMPRINT_ALPHA 0.05, IMPRINT_TOP_K 100). Test (run 4, held-out medicines;
outputs/runs/stage2_run4/eval/identify_confidence_large_c92.json): first candidate right 80.2%; stated 0.96 -> 93%
right, 0.70 -> 75%, 0.51 -> 49%, 0.36 -> 39%. Catalogue pills: possible_match 76% (87% right), uncertain 20%,
no_confident_match 4%. Non-catalogue pills: no_confident_match 70%, uncertain 23%, possible_match 7%.
"""
import math

MODEL_VERSION = "stage2_run4@2026-10-09"
WEIGHTS = (-12.5019, 12.5011, 72.1318, 3.9206)  # intercept, top-1 similarity, gap to the runner-up, imprint read
T_POOR, T_GOOD = 0.9462, 0.9658
LOW, HIGH = 0.2, 0.5
MESSAGES = {
    "possible_match": "The first candidate looks most like this pill. A photo can't confirm it. Check with your "
                      "pharmacist.",
    "uncertain": "Several catalogue pills look alike and none stands out. Type the imprint, or check with your "
                 "pharmacist.",
    "no_confident_match": "No catalogue pill matches this one well: it may not be a US catalogue product, or the photo "
                          "differs from the catalogue photos. Check with your pharmacist.",
}


def assess(candidates, model_version):
    """candidates: best first, each with similarity and imprint_read (at least two) -> dict, or None."""
    if model_version != MODEL_VERSION or len(candidates) < 2:
        return None
    s1, s2 = candidates[0]["similarity"], candidates[1]["similarity"]
    z = WEIGHTS[0] + WEIGHTS[1] * s1 + WEIGHTS[2] * (s1 - s2) + WEIGHTS[3] * candidates[0].get("imprint_read", 0.0)
    confidence = 1 / (1 + math.exp(-z))
    fit = "poor" if s1 < T_POOR else "good" if s1 >= T_GOOD else "weak"
    verdict = ("no_confident_match" if fit == "poor" or confidence < LOW else
               "possible_match" if fit == "good" and confidence >= HIGH else "uncertain")
    return {"confidence": round(confidence, 3), "catalogue_fit": fit, "verdict": verdict,
            "verdict_message": MESSAGES[verdict]}
