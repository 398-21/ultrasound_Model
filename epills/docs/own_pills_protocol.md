# Own pills: collection and photo protocol

These photos are the only test of the real workflow: enrol a patient's medications, then recognise them in new
photos and reject pills that were never enrolled. They also give the only images with a real mm scale and
mat-corrected colour. Plan about two short sessions on different days.

## 1. Choose the pills
- **15–30 enrolled pills:** over-the-counter meds and vitamins you can legally buy. Pick deliberately hard
  cases, not just easy ones:
  - several **white round tablets** of different sizes, e.g. ibuprofen 200 mg, acetaminophen 325 / 500 mg,
    aspirin 81 / 325 mg, loratadine 10 mg, cetirizine 10 mg, famotidine
  - **look-alike pairs** that differ mainly in size or imprint
  - **two-tone capsules**, **softgels** (shiny or transparent), **coated caplets**
  - **scored** tablets
  - **printed and debossed** imprints, and pills with an imprint on **one side only**
- **5–10 never-enrolled pills** (the "unknowns"). Include some that look like enrolled ones, e.g. another
  brand's version of the same drug.
- A different brand of the same drug is a **different pill** for the model. Treat it as its own entry.
- Safety: don't take the pills, keep them away from children, and dispose of them at a pharmacy take-back.
  Don't use anyone else's prescription medication.

## 2. Record each product (`labels.csv`)
Fill in one row per product **before** taking photos. Columns:
`pill_id, enrolled (yes/no), product_name, manufacturer, active_ingredients, strength, ndc (from the package,
or "none"), lot, expiry, shape, colour, imprint_front, imprint_back, imprint_type (printed/debossed/none),
length_mm, width_mm, notes`.

Measure length and width with calipers or a ruler if you can. This gives ground truth for the measured-size
input.

## 3. The mat
- Use the Stage 1 mat, which has scale markers and colour reference patches.
- If it isn't ready, Claude can design a printable one. Print it at 100% scale on matte paper, then check one
  marker distance with a ruler and record it.
- Put the mat on a flat surface, and keep all pills inside its marked area.

## 4. Enrolment session (day 1)
- For each enrolled pill, take **5 photos of the front and 5 of the back**. Move and rotate the pill between
  shots.
- Use one phone and normal indoor light.
- Phone settings: full resolution, no filters, no portrait mode, no digital zoom, flash off.

## 5. Query session (another day: different light, and a second phone if you have one)
- For each **enrolled** pill: 5 photos per side, spread over conditions:
  - daylight near a window
  - warm indoor light
  - dim light
  - slightly tilted phone
  - a few with flash, as hard cases
- For each **never-enrolled** pill: 5 photos per side (query only, never enrolled).
- **Multi-pill photos:** about 20 photos of a "daily dose", with 3–6 pills on the mat at once. Mix enrolled
  pills, a duplicate of one pill (for the double-dose rule) and sometimes an unknown pill. Write down
  immediately which pills, and how many of each, are in each photo.

About 30 × 2 × 5 enrolment + 30 × 2 × 5 query + 10 × 2 × 5 unknown + 20 multi-pill ≈ **720 photos**.

## 6. Check before you finish a session
- [ ] Pill in focus and the imprint readable by eye
- [ ] Whole pill inside the mat area; no fingers or shadows over it
- [ ] Every photo can be traced to a `pill_id` and side

## 7. Folder layout and handover
```
data/raw/own_pills/
├── labels.csv
├── enroll/<pill_id>/{front,back}/*.jpg
├── query/<pill_id>/{front,back}/*.jpg       (enrolled and never-enrolled pills)
├── multi/*.jpg
└── multi_labels.csv                         (photo, pill_id, count)
```
Copy the folder to the desktop, and tell Claude it's there. Claude checks the counts and labels and reports
anything missing.
