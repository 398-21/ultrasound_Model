"""
Is this a photo of loose pills? (blueprint job B23; test cases P02 "unprinted clear blister: no Stage 2 match" and P03
"opaque foil: abstain"). Pill recognition is only valid on loose pills; blister packs, bottles, boxes and printed
labels need the label instead (/v1/read-label) or a pharmacist.

The app asks the patient what they are photographing first (blueprint: the user's intent comes first); this check is
a safety net on top. It reads Florence-2's caption of the whole photo and looks for strong words only:
  blister   "blister", "foil", "pack of pills", "strip of"
  package   "bottle", "box of", "jar", "vial", "carton", "packet", "tube of", "sachet"
  printed   a strength or tablet count in the caption (Florence transcribes printed labels: "paracetamol 500mg tablet")
Measured on a small set (13 Oct 2026): 14 loose-pill photos (C3PI, MEDISEG pills in a clear organiser) gave no false
alarm; of 8 blister photos 5 were caught as blister and 2 as packaging, and the foil back of a pack ("a keyboard")
was missed; 6 of 7 bottle photos and all 5 printed labels were caught. Small numbers: a net, not a guarantee.
"""
import re

BLISTER = re.compile(r"\b(blister|foil|strip of|pack of (pills|tablets|capsules|medicine|medication))\b", re.I)
PACKAGE = re.compile(r"\b(bottle|box of|jar|vial|carton|packet|package of|tube of|sachet)\b", re.I)
PRINTED = re.compile(r"\d+(\.\d+)?\s?(mg|mcg|ml)\b|\b\d+\s+(tablets|capsules|caplets)\b", re.I)

MESSAGES = {
    "blister": "This looks like a blister pack. Pill recognition works on loose pills only: photograph the box or "
               "pharmacy label instead, or ask your pharmacist.",
    "package": "This looks like a bottle or box. Photograph its label to read it, or put the pills on a plain surface "
               "and photograph them.",
    "printed": "This looks like printed text (a label or box). Use label reading for it, or photograph the loose pills.",
    "document": "This is a document or label: use label reading for it.",
    "unchecked": "The photo couldn't be checked automatically (text reading is unavailable): make sure it shows loose "
                 "pills.",
}


def classify_input(caption):
    """A Florence-2 caption -> 'blister', 'package', 'printed' or 'loose_pills' (the default: no strong word)."""
    caption = caption or ""
    if BLISTER.search(caption):
        return "blister"
    if PACKAGE.search(caption):
        return "package"
    if PRINTED.search(caption):
        return "printed"
    return "loose_pills"
