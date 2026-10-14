# Dose rules: spec for the main backend (blueprint job B11)

The blueprint (decision D3) moves the dose decision to the main backend: only it knows the active regimen, the
dose history, PRN medicines, stopped courses and what the patient confirmed. This is the rule set the PharmaScan API
used so far, written so the backend can implement it, plus the cases only the backend can handle.

The API keeps a **stateless reference implementation**: `POST /v1/safety/dose` (code:
`src/epills/core/safety/rules.py`, `check_dose`; tests: `tests/test_safety.py`). `/v1/scan`'s `schedule` option still
runs it, marked `deprecated`, and only when the pill count is reliable.

## Inputs

- **This dose's schedule:** `[{medication_id, count}]`, from the active regimen version only.
- **The pills seen:** per pill, the visual decision from `/v1/match` (`match` with a `medication_id`, `unknown`,
  `ambiguous`) and whether it may be counted: `counted` is false for a box the finder flagged.
- **The detection:** `count_reliable` from `/v1/detect` or `/v1/scan`. When it is false, don't count at all.

## Rules (visual evidence only)

Each alert ends with "Check with your pharmacist." The colours, voice and caregiver flags follow severity (docs/API.md,
ALERT).

| # | Condition | Alert `type` | Severity | Message (English default; the agent may reword) |
|---|---|---|---|---|
| 0 | `count_reliable` is false | `count_uncertain` | moderate | I can't count the pills in this photo reliably. Spread them apart on a plain surface and photograph them again. *(no other rule runs)* |
| 1 | any pill `unknown` | `unknown_pill` | major | One pill doesn't look like any of your medicines. Please set it aside. |
| 2 | pills matched to a medicine with count 0 in this dose | `not_scheduled` | major | One pill looks like your X, which is not part of this dose. Please set it aside. |
| 3 | more pills matched to a medicine than scheduled | `double_dose` | major | This photo shows two pills that look like your X instead of one. Please count them: if there are two, set the extra one aside. *(`details.confirm_count` is true and `notify_caregiver` false: see backend item 8)* |
| 4 | fewer pills matched than scheduled | `not_seen` | moderate | I can't see your X in this photo. / I can see only one pill that looks like your X, instead of two. |
| 5 | any pill `ambiguous` | `unsure_pill` | moderate | I'm not sure about one pill. Please turn it over and photograph the dose again. |

Alerts are sorted by severity (major first). `summary`: `pills_seen`, `matched`, `unknown`, `unsure`,
`all_visually_matched` (no alerts; `all_ok` is its deprecated name), `count_reliable`, `text`.

**Never:** call a pill "the right medicine" or "safe to take"; turn "not seen" into "missed" (a missing photo or a
missing pill in a photo is not a missed dose); suggest taking an extra dose, skipping or changing a dose.

## Test cases (the reference implementation passes these)

| Schedule | Pills seen | Expected alerts |
|---|---|---|
| metformin ×1, amlodipine ×1, atorvastatin ×1 | metformin, metformin, amlodipine, vitamin D, unknown, amlodipine (ambiguous) | `unknown_pill`, `double_dose` (metformin), `not_scheduled` (vitamin D), `not_seen` (atorvastatin), `unsure_pill` |
| metformin ×2 | metformin, metformin | none; text "I see two pills. They all look like the medicines of this dose." |
| metformin ×1 | (count not reliable) | `count_uncertain` only |
| metformin ×1 | metformin, plus a box the finder flagged (`small`) | `count_uncertain` only: one flagged box makes the whole count unreliable |
| metformin ×1 | metformin, metformin (count reliable) | `double_dose`, with `details.confirm_count` true and `notify_caregiver` false |

## What the backend adds (only it has the data)

1. **Actual time:** compare with the scheduled time window; a check far outside it is a question to the patient,
   not an alert.
2. **Already recorded:** if this dose already has `intake_status = taken`, a second photo is not a second dose;
   writes must be idempotent.
3. **PRN (as needed) medicines:** never `not_seen`; `not_scheduled` only if the patient's PRN limit for the day is
   reached (from the prescription text, never inferred).
4. **Stopped or changed regimens:** a pill matching a medicine whose regimen version ended is `not_scheduled` with
   the reason "stopped on <date>"; strengths that changed between versions need the new reference photos.
5. **Two separate statuses:** `vision_status` (what this check saw: the summary above) and `intake_status`
   (taken / not_taken / unsure, from the patient). Never derive one from the other.
6. **Unconfirmed events:** a reminder with no photo and no answer is "no record", not "missed"; long-term views
   count it separately (blueprint, long-term care flow).
7. **Gallery version:** galleries embedded in another `embedding_space_id` are rejected by the API (`409`): re-embed
   the stored enrolment photos, then retry.
8. **Confirm an extra pill before recording it (blueprint B24):** a `double_dose` alert can come from one pill boxed
   twice. On 1,000 one-pill C3PI photos the stand-in finder reported an extra pill while calling the count reliable
   on **1.4%** (test case D01; `outputs/eval/pill_finder.json` → `by_pills/1/over_count_while_reliable`; also in
   `/v1/model-info` → `error_rates` → `pill_count`). So when an alert has `details.confirm_count`:
   - ask the patient how many pills of that medicine there are before writing a dose event with the extra pill;
   - tell a linked caregiver only after the patient confirms the extra pill (the API sends `notify_caregiver` false
     on this alert for that reason);
   - if the patient says there is one, record the dose as seen once and keep the alert in the audit trail as corrected
     by the patient; don't ask for another photo.
