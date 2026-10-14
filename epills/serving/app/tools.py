"""
Typed tool definitions for the Medication Care Agent (blueprint job B14), generated from the API's own OpenAPI
document so they can't drift from it. Served at GET /v1/tools; usage notes and example traces in docs/agent_tools.md.

Each tool: name (the blueprint's where it names one), description written for the agent (when to call it, what it
must not conclude), input_schema (JSON Schema; an uploaded file becomes an image reference the app's tool layer
resolves), output_schema ($refs into `components`), and how to call it (method, path, content type).
"""

TOOLS = [
    ("route_medication_image", "post", "/v1/check-input",
     "Before any other photo tool when you don't know what the photo shows: loose pills, blister pack, box or bottle, "
     "printed label or document, and which tools fit (next_tools). Prefer asking the user; when type is unchecked or "
     "next_tools is empty, ask for the box, label or prescription, or suggest the pharmacist."),
    ("detect_pills", "post", "/v1/detect",
     "Find each pill in a photo and say whether the count can be trusted. Call before counting pills or embedding "
     "several pills. If count_reliable is false, don't count: ask for a new photo (pills apart, plain surface)."),
    ("embed_pill", "post", "/v1/embed",
     "Turn one loose-pill photo (or a photo + box) into a 256-number embedding. Use at enrolment (store it with "
     "model_version and embedding_space_id) and before match_gallery. Never embed boxes or blister packs."),
    ("match_gallery", "post", "/v1/match",
     "Compare pill embeddings with the patient's gallery: match / unknown / ambiguous per pill. Visual evidence only: "
     "a match means 'looks like', not 'is' or 'safe to take'. ambiguous: ask for another photo. The gallery's "
     "model_version is required; on 409 re-embed the enrolment photos."),
    ("identify_pill", "post", "/v1/identify",
     "Possible catalogue matches (US products) for a pill that isn't in the patient's list. Report the verdict and "
     "confidence, never a definite identity; no_confident_match means the pill may not be a catalogue product."),
    ("search_by_embedding", "post", "/v1/search-by-embedding",
     "identify_pill for a pill you already embedded (e.g. match_gallery said unknown): send its embeddings and "
     "imprint_reads from embed_pill. Same answer, no second model run. Possible matches only, never an identity."),
    ("scan_photo", "post", "/v1/scan",
     "Diagnostics in one call: detection plus each pill's visual decision (with a gallery) or catalogue candidates. "
     "Don't use its dose check (deprecated): apply the backend's dose rules to the pills it returns."),
    ("extract_medication_document", "post", "/v1/read-label",
     "Read a medicine box, pharmacy label or prescription (photo or PDF): medicine names, strength, RxNorm code and "
     "what is printed (directions, quantity, dates), each with its source line. Fields not printed are null; never "
     "fill them in. With the patient's medications, says same medicine / different strength / not on the list. "
     "Always have the patient confirm."),
    ("find_medication_product", "get", "/v1/pills/search",
     "Products and pictures for a medicine name (brand or generic), with strength, dose form and release as fields. "
     "Never treat products of another strength or form as the same."),
    ("screen_medication_list", "post", "/v1/safety/medications",
     "Interaction and duplicate-ingredient alerts for the whole medication list (DDInter 2.0). Read coverage: no "
     "alert does not mean safe. Advisory; defer to the pharmacist."),
    ("evaluate_dose_reference", "post", "/v1/safety/dose",
     "The reference dose rules on visual decisions (stateless). The backend's own dose policy is authoritative."),
    ("retrieve_medication_info", "get", "/v1/safety/label",
     "Warning sections of the official US label, with the label's date and DailyMed link for citation. General "
     "information, not the patient's instructions."),
    ("get_model_info", "get", "/v1/model-info",
     "The served model: version, embedding space, decision rule and known limits. Check after a 409."),
]
FILE_NOTE = "An image reference: the app's tool layer uploads the file (photo, or PDF for documents)"


def _resolve(schema, components):
    ref = schema.get("$ref", "") if isinstance(schema, dict) else ""
    return components.get(ref.rsplit("/", 1)[-1], {}) if ref else schema


def _input_schema(op, components):
    props, required = {}, []
    for p in op.get("parameters", []):
        props[p["name"]] = {**p.get("schema", {}), "description": p.get("description", "")}
        if p.get("required"):
            required.append(p["name"])
    content = op.get("requestBody", {}).get("content", {})
    for kind in ("application/json", "multipart/form-data"):
        if kind in content:
            body = _resolve(content[kind]["schema"], components)
            for name, s in body.get("properties", {}).items():
                is_file = s.get("format") == "binary" or s.get("contentMediaType") or \
                    (s.get("type") == "array" and (s.get("items", {}).get("format") == "binary"
                                                   or s.get("items", {}).get("contentMediaType")))
                props[name] = ({"type": "array", "items": {"type": "string"}, "description": FILE_NOTE}
                               if is_file and s.get("type") == "array" else
                               {"type": "string", "description": FILE_NOTE} if is_file else s)
            required += body.get("required", [])
    return {"type": "object", "properties": props, "required": sorted(set(required))}


def build_tools(openapi):
    components = openapi.get("components", {}).get("schemas", {})
    tools = []
    for name, method, path, description in TOOLS:
        op = openapi["paths"][path][method]
        ok = op.get("responses", {}).get("200", {}).get("content", {}).get("application/json", {})
        kind = next(iter(op.get("requestBody", {}).get("content", {"none": None})))
        tools.append({"name": name, "description": description, "input_schema": _input_schema(op, components),
                      "output_schema": ok.get("schema", {}),
                      "call": {"method": method.upper(), "path": path,
                               "content_type": None if kind == "none" else kind}})
    return {"tools": tools, "components": {"schemas": components},
            "notes": "Stage 2 answers are visual evidence only. The backend's safety policy decides actions; the "
                     "agent reports evidence with its limits and asks the patient to confirm."}
