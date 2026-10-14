# PharmaScan tools for the Medication Care Agent (blueprint job B14)

`GET <BASE_URL>/v1/tools` returns the tool definitions, generated from the API's OpenAPI document so they always
match it: `name`, `description` (written for the agent: when to call, what not to conclude), `input_schema`,
`output_schema` (`$ref`s into the returned `components`), and `call` (method, path, content type). Base URL:
`https://davidpansx-pharmascan-api.hf.space`.

| Tool | Endpoint | Blueprint step |
|---|---|---|
| `route_medication_image` | `POST /v1/check-input` | ds_route, un_route (when the user hasn't said what the photo shows) |
| `detect_pills` | `POST /v1/detect` | ds_crop, scan_diagnostics, duplicate_boxes_guard |
| `embed_pill` | `POST /v1/embed` | en_embed, ds_gallery |
| `match_gallery` | `POST /v1/match` | ds_gallery |
| `identify_pill` | `POST /v1/identify` | ds_catalogue, un_loose |
| `search_by_embedding` | `POST /v1/search-by-embedding` | ds_catalogue after `match_gallery` said unknown (no second model run) |
| `scan_photo` | `POST /v1/scan` | diagnostics only |
| `extract_medication_document` | `POST /v1/read-label` | en_ocr, ds_ocr, un_print |
| `find_medication_product` | `GET /v1/pills/search` | en_dedupe (product identity) |
| `screen_medication_list` | `POST /v1/safety/medications` | en_risk |
| `evaluate_dose_reference` | `POST /v1/safety/dose` | reference for the backend's ds_policy |
| `retrieve_medication_info` | `GET /v1/safety/label` | ov_knowledge, lg_knowledge |
| `get_model_info` | `GET /v1/model-info` | version checks |

## How the agent uses them

- **Images:** the agent never handles bytes. A file input is an *image reference*; the app's tool layer looks up the
  upload and posts it as multipart form data.
- **The tools give evidence, the backend decides.** Every Stage 2 answer is `"evidence": "visual"` with a `limits`
  note. The backend's deterministic safety gate (docs/dose_rules.md) decides what to record or alert; the agent
  explains the evidence and its limits and asks the patient to confirm.
- **Store versions:** keep `model_version` and `embedding_space_id` with every embedding; send `model_version` with
  every gallery. On `409`, call `get_model_info`, re-embed the stored enrolment photos, retry.
- **Abstain properly:** `count_reliable: false`, `decision: ambiguous`, `verdict: no_confident_match` and fields that
  are `null` are answers. The next step is another photo, the label, or the pharmacist, never a guess.
- **Route first:** ask the user what the photo shows; if they don't say, `route_medication_image` and follow its
  `next_tools`. A blister pack or an unchecked photo means asking for the box, label or prescription.
- **Strength comes from the label:** when a candidate has `look_alike_strengths`, say the photo can't tell the
  strength. A `double_dose` alert with `confirm_count` means asking the patient to count before anything is recorded.
- **Pseudonymous ids only:** never put an NRIC, name or phone number in an id; such values are refused
  (`personal_id`).
- **Errors:** `{"code", "detail"}`; `not_ready` comes with `retry_after` seconds; an HTML page means the Space is
  waking up.

## Example traces

These follow the blueprint's demo scripts. The tool outputs are shortened; the field names are real.

**B · A normal dose check (one loose pill)**
```
patient: "Check tonight's pill" + photo
agent → get_active_regimens, get_gallery               (app)   lipitor_10: 3 embeddings, model_version …run4…
agent → detect_pills(image=photo_123)                  → pill_count 1, count_reliable true, boxes [[410,300,650,540]]
agent → embed_pill(image=photo_123, box=[410,300,650,540], read_imprint=true)
                                                       → embedding […256], imprint_reads ["PD 155", "", "", ""]
agent → match_gallery(queries=[…], gallery=[lipitor_10 + imprint "PD 155"], model_version=…, query_reads=[…])
                                                       → decision match, imprint_check confirmed,
                                                         message "This pill looks like your Lipitor 10 mg."
agent → evaluate_dose_policy                            (app)   scheduled 21:00, not yet taken → ok
agent:  "This pill looks like your Lipitor 10 mg, and its imprint matches. That's a visual check, not a guarantee.
         Have you taken it?"            patient: "yes" → record_dose_event(vision=match, intake=taken)   (app)
```

**C · One pill seen as two**
```
agent → detect_pills(image=photo_456)                  → pill_count 2, count_reliable false,
                                                         box_flags [[], ["small"]], flags ["count_uncertain"]
agent:  doesn't count pills, doesn't raise a double dose → "I can't count the pills reliably. Put the pill on a
         plain surface, apart from anything else, and take another photo."
(new photo) → detect_pills → pill_count 1, count_reliable true → continue as in B
```

**A · Adding a medicine from a pharmacy label**
```
agent → extract_medication_document(image=label_789, medications=[current list])
        → candidates[0]: name "Paracetamol 500 mg", rxcui "198440" (acetaminophen 500 MG Oral Tablet),
          fields.directions "Take 2 tablets 4 times a day when necessary for pain or fever" (line given),
          fields.quantity "20 tab", list_check not_on_list
agent:  shows the fields with their source lines: "Is this what the label says?"   patient confirms
agent → screen_medication_list(medications=[… + {id, name, rxcui}])
        → alerts [], coverage {screened 4/4, pairs_checked 6}, note "No alert doesn't mean …"
agent → activate_regimen(source=label_789, directions as printed, status confirmed)   (app)
```
