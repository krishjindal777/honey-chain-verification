"""
Honey Chain Verification Backend — SDG 12
FastAPI application implementing 4-stage AI document verification pipeline
with ChromaDB live RAG compliance layer and multi-format file ingestion.

Pipeline stages
---------------
1. Extraction    — regex-based structured field parsing
2. RAG Retrieval — semantic search against honey_regulatory_standards ChromaDB collection
3. Rules Engine  — deterministic Codex / ISO checks informed by retrieved clause
4. Audit Assembly — signed, timestamped JSON response

File ingestion (POST /upload)
------------------------------
• .pdf  — text layer extracted with pypdf (PdfReader); falls back to pypdf page text
• .png / .jpg / .jpeg — OCR via pytesseract + Pillow; graceful error if Tesseract not on PATH
• .txt  — UTF-8 decode (latin-1 fallback)
Returns extracted_text + full VerificationResult in a single response.
"""

import io
import re
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Honey Chain Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# ChromaDB — in-memory vector store seeded with regulatory corpus
# ---------------------------------------------------------------------------

_REGULATORY_CORPUS = [
    {
        "id": "codex-cxs-12-1981-hmf-standard",
        "standard": "Codex CXS 12-1981",
        "clause": (
            "Hydroxymethylfurfural (HMF) maximum permissible threshold is 40 mg/kg "
            "for honey intended for direct human consumption. For honey originating from "
            "designated tropical regions, the permissible threshold is extended to 80 mg/kg "
            "due to accelerated HMF formation at elevated ambient temperatures."
        ),
    },
    {
        "id": "codex-cxs-12-1981-moisture",
        "standard": "Codex CXS 12-1981",
        "clause": (
            "Moisture content of honey shall not exceed 20 g/100 g. "
            "Fermented honey is permitted provided the ethanol content does not exceed 0.5%. "
            "Diastase activity, after processing and blending, shall not be less than 8 "
            "on the Schade scale, except for honey with a low natural enzyme content."
        ),
    },
    {
        "id": "iso-iec-17025-lab-accreditation",
        "standard": "ISO/IEC 17025:2017",
        "clause": (
            "ISO/IEC 17025 mandates that testing and calibration laboratories demonstrate "
            "technical competence and produce valid results through formal accreditation "
            "by a recognised national accreditation body. Chemical analysis results submitted "
            "as part of food safety or trade certification must originate from an ISO 17025 "
            "accredited facility to be internationally recognised under ILAC MRA."
        ),
    },
    {
        "id": "national-organic-apiculture-certification",
        "standard": "National Organic Apiculture Standards",
        "clause": (
            "Organic honey certification requires that apiaries comply with the National Organic "
            "Program (NOP) or equivalent national regulation. Certification status must be active "
            "and within the harvest shelf-life window (certificate validity ≥ harvest date). "
            "Apiaries must be located at least 3 km from non-organic agricultural land, "
            "prohibited substance use, or industrial pollution sources."
        ),
    },
    {
        "id": "codex-cxs-12-1981-adulteration",
        "standard": "Codex CXS 12-1981",
        "clause": (
            "Honey shall not contain any food additive. No substance shall be added to honey "
            "to alter its natural composition, taste, or shelf life. The addition of sugars, "
            "syrups, or any other carbohydrate sweetener constitutes adulteration and "
            "disqualifies the product from honey classification under Codex standards."
        ),
    },
]

# Initialise an ephemeral ChromaDB client (in-memory, no persistence required)
_chroma_client = chromadb.Client()
_embed_fn = DefaultEmbeddingFunction()

_collection = _chroma_client.get_or_create_collection(
    name="honey_regulatory_standards",
    embedding_function=_embed_fn,
    metadata={"hnsw:space": "cosine"},
)

# Seed only once on startup
if _collection.count() == 0:
    _collection.add(
        ids=[doc["id"] for doc in _REGULATORY_CORPUS],
        documents=[doc["clause"] for doc in _REGULATORY_CORPUS],
        metadatas=[{"standard": doc["standard"], "id": doc["id"]} for doc in _REGULATORY_CORPUS],
    )

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DocumentPayload(BaseModel):
    text: str


class ExtractedData(BaseModel):
    batch_id: str | None
    producer_name: str | None
    harvest_date: str | None
    organic_status: bool
    hmf_level: float | None       # mg/kg
    lab_accreditation: str | None


class RagMatch(BaseModel):
    standard: str
    clause: str
    query_used: str
    distance: float


class VerificationResult(BaseModel):
    audit_id: str
    timestamp: str
    status: str                   # "VERIFIED" | "FLAGGED"
    extracted: ExtractedData
    rag_matches: list[RagMatch]   # top-3 regulatory references from ChromaDB
    checklist: dict
    issues: list[str]
    document_hash: str


class UploadResult(BaseModel):
    """Response from POST /upload — extracted text plus full audit payload."""
    filename: str
    file_type: str                # "pdf" | "image" | "text"
    extracted_text: str
    ocr_warning: str | None       # set when Tesseract is unavailable
    verification: VerificationResult


# ---------------------------------------------------------------------------
# Stage 1 — Field Extraction
# ---------------------------------------------------------------------------

_PATTERNS = {
    "batch_id":          r"batch[_\s-]?id[:\s]+([A-Z0-9\-]+)",
    "producer_name":     r"producer[:\s]+([^\n,;]+)",
    "harvest_date":      r"harvest[_\s]?date[:\s]+(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})",
    "organic_status":    r"organic[_\s]?status[:\s]+(true|false|yes|no|active|inactive|certified|uncertified)",
    "hmf_level":         r"hmf[_\s]?level[:\s]+([\d.]+)\s*(?:mg/kg)?",
    "lab_accreditation": r"lab[_\s]?accreditation[:\s]+([^\n,;]+)",
}


def _extract(text: str) -> ExtractedData:
    t = text.lower()

    def find(key) -> str | None:
        m = re.search(_PATTERNS[key], t, re.IGNORECASE)
        return m.group(1).strip() if m else None

    organic_raw = find("organic_status")
    organic_bool = organic_raw in ("true", "yes", "active", "certified") if organic_raw else False

    hmf_raw = find("hmf_level")
    try:
        hmf_float = float(hmf_raw) if hmf_raw else None
    except ValueError:
        hmf_float = None

    def find_orig(key) -> str | None:
        m = re.search(_PATTERNS[key], text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    return ExtractedData(
        batch_id=find_orig("batch_id"),
        producer_name=find_orig("producer_name"),
        harvest_date=find_orig("harvest_date"),
        organic_status=organic_bool,
        hmf_level=hmf_float,
        lab_accreditation=find_orig("lab_accreditation"),
    )


# ---------------------------------------------------------------------------
# Stage 2 — ChromaDB RAG Retrieval
# ---------------------------------------------------------------------------

def _rag_query(text: str, extracted: ExtractedData) -> list[RagMatch]:
    """
    Build a focused semantic query from extracted certificate fields and
    retrieve the top-3 most relevant regulatory clauses from ChromaDB.
    """
    parts = ["honey quality compliance certificate verification"]
    if extracted.hmf_level is not None:
        parts.append(f"HMF hydroxymethylfurfural level {extracted.hmf_level} mg/kg threshold limit")
    if extracted.lab_accreditation:
        parts.append(f"laboratory accreditation {extracted.lab_accreditation} ISO 17025")
    if extracted.organic_status:
        parts.append("organic apiculture certification active status")
    else:
        parts.append("organic certification inactive status")

    query = " | ".join(parts)

    results = _collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    matches: list[RagMatch] = []
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        matches.append(RagMatch(
            standard=meta.get("standard", "Unknown"),
            clause=doc,
            query_used=query,
            distance=round(float(dist), 6),
        ))

    return matches


# ---------------------------------------------------------------------------
# Stage 3 — Rules Engine (Codex Alimentarius + ISO 17025, RAG-informed)
# ---------------------------------------------------------------------------

HMF_LIMIT = 40.0  # mg/kg — Codex Alimentarius CXS 12-1981


def _run_rules(data: ExtractedData, rag_matches: list[RagMatch]) -> tuple[dict, list[str]]:
    """
    Deterministic compliance checks. Rule logic always uses the hard-coded Codex
    constant so results are reproducible regardless of embedding drift.
    """
    issues: list[str] = []
    top_standard = rag_matches[0].standard if rag_matches else "Codex CXS 12-1981"

    # Rule 1 — HMF threshold
    if data.hmf_level is None:
        hmf_pass = False
        issues.append("HMF level could not be extracted from document.")
    elif data.hmf_level > HMF_LIMIT:
        hmf_pass = False
        issues.append(
            f"HMF level {data.hmf_level} mg/kg exceeds {top_standard} "
            f"permissible limit of {HMF_LIMIT} mg/kg."
        )
    else:
        hmf_pass = True

    # Rule 2 — ISO 17025 lab accreditation
    accred = (data.lab_accreditation or "").lower()
    iso_pass = "iso 17025" in accred or "iso17025" in accred
    if not iso_pass:
        issues.append(
            "Laboratory is not ISO/IEC 17025 accredited — results may not be "
            "internationally recognised under ILAC MRA."
        )

    # Rule 3 — Organic certification active
    organic_pass = data.organic_status
    if not organic_pass:
        issues.append(
            "Organic status is inactive or missing — certificate does not meet "
            "National Organic Apiculture Standards for active apiary certification."
        )

    checklist = {
        "hmf_within_codex_limit": hmf_pass,
        "iso_17025_accredited_lab": iso_pass,
        "organic_certification_active": organic_pass,
    }
    return checklist, issues


# ---------------------------------------------------------------------------
# Stage 4 — Audit Assembly
# ---------------------------------------------------------------------------

def _build_audit(
    text: str,
    data: ExtractedData,
    rag_matches: list[RagMatch],
    checklist: dict,
    issues: list[str],
) -> VerificationResult:
    status = "VERIFIED" if all(checklist.values()) else "FLAGGED"
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = f"{text.strip()}|{ts_str}"
    doc_hash = hashlib.sha256(raw.encode()).hexdigest()
    audit_id = f"AUD-{doc_hash[:12].upper()}-{uuid.uuid4().hex[:6].upper()}"

    return VerificationResult(
        audit_id=audit_id,
        timestamp=ts_str,
        status=status,
        extracted=data,
        rag_matches=rag_matches,
        checklist=checklist,
        issues=issues,
        document_hash=doc_hash,
    )


# ---------------------------------------------------------------------------
# Shared pipeline helper
# ---------------------------------------------------------------------------

def _run_pipeline(text: str) -> VerificationResult:
    extracted   = _extract(text)
    rag_matches = _rag_query(text, extracted)
    checklist, issues = _run_rules(extracted, rag_matches)
    return _build_audit(text, extracted, rag_matches, checklist, issues)


# ---------------------------------------------------------------------------
# File ingestion helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
_PDF_EXTS   = {".pdf"}
_TEXT_EXTS  = {".txt"}


def _ingest_pdf(raw: bytes) -> str:
    """
    Extract text from PDF bytes using pypdf (PdfReader).
    Joins text from all pages; strips excessive whitespace.
    """
    from pypdf import PdfReader  # installed: pypdf

    reader = PdfReader(io.BytesIO(raw))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages).strip()


# Common Windows install locations for Tesseract — checked in order when
# the binary is not on PATH.
_TESSERACT_WIN_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\tools\tesseract\tesseract.exe",
]


def _configure_tesseract() -> None:
    """
    Point pytesseract at the Tesseract binary if it is installed in a
    well-known Windows directory but not on the system PATH.
    No-op on non-Windows or when tesseract is already on PATH.
    """
    import shutil
    if shutil.which("tesseract"):
        return  # already on PATH — nothing to do
    import sys
    if sys.platform != "win32":
        return
    import pytesseract
    for candidate in _TESSERACT_WIN_PATHS:
        if Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


def _ingest_image(raw: bytes) -> tuple[str, str | None]:
    """
    Run OCR on image bytes using pytesseract + Pillow.
    Returns (extracted_text, warning_or_None).

    • Tries to auto-locate Tesseract in common Windows install paths when
      it is not on PATH — so the user does not need to touch environment
      variables after a standard installer run.
    • If Tesseract is genuinely absent, returns a clear warning string
      and an empty text; the caller decides what HTTP response to issue.
    """
    from PIL import Image  # installed: pillow

    try:
        import pytesseract  # installed: pytesseract

        _configure_tesseract()          # auto-locate binary if needed

        image = Image.open(io.BytesIO(raw))
        text  = pytesseract.image_to_string(image)
        return text.strip(), None

    except pytesseract.TesseractNotFoundError:
        return "", (
            "Tesseract OCR engine was not found. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "and restart the server. "
            "Until then, use PDF or TXT files, or paste text directly."
        )
    except Exception as exc:  # noqa: BLE001
        return "", f"OCR error: {exc}"


def _ingest_text(raw: bytes) -> str:
    """Decode plain-text bytes — UTF-8 with latin-1 fallback."""
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw.decode("latin-1").strip()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResult)
async def upload_file(file: UploadFile = File(...)):
    """
    Multi-format ingestion endpoint.

    Accepts .pdf, .png, .jpg, .jpeg, .txt files via multipart/form-data.
    Returns:
      - extracted_text   — the raw string passed into the pipeline
      - ocr_warning      — non-null only when Tesseract is missing
      - verification     — full 4-stage audit payload
    """
    raw_bytes = await file.read()
    fname     = (file.filename or "").strip()
    ext       = Path(fname).suffix.lower()

    ocr_warning: str | None = None
    file_type: str

    # ── Dispatch by extension ────────────────────────────────────────────
    if ext in _PDF_EXTS:
        file_type = "pdf"
        try:
            text = _ingest_pdf(raw_bytes)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=422,
                content={"detail": f"PDF parsing error: {exc}"},
            )

    elif ext in _IMAGE_EXTS:
        file_type = "image"
        text, ocr_warning = _ingest_image(raw_bytes)
        # Do NOT hard-error when Tesseract is missing.
        # Return a 200 UploadResult with ocr_warning set so the UI shows
        # the amber warning banner without a blocking error dialog.
        # The "no text extracted" guard below still fires if text is empty.

    elif ext in _TEXT_EXTS:
        file_type = "text"
        text = _ingest_text(raw_bytes)

    else:
        return JSONResponse(
            status_code=415,
            content={
                "detail": (
                    f"Unsupported file type '{ext}'. "
                    "Accepted formats: .pdf, .png, .jpg, .jpeg, .txt"
                )
            },
        )

    # ── Guard: nothing extracted ──────────────────────────────────────────
    # For images without Tesseract: return a 200 with the warning rather
    # than a hard error, so the frontend can show the install prompt inline.
    if not text:
        if ocr_warning:
            # Run pipeline on empty string so we still get a valid schema
            # back — the UI will show the warning banner above the results.
            placeholder = "[No text extracted — OCR unavailable]"
            verification = _run_pipeline(placeholder)
            return UploadResult(
                filename=fname or "unnamed",
                file_type=file_type,
                extracted_text="",
                ocr_warning=ocr_warning,
                verification=verification,
            )
        return JSONResponse(
            status_code=422,
            content={"detail": "No text could be extracted from the uploaded file."},
        )

    # ── Run 4-stage pipeline ──────────────────────────────────────────────
    verification = _run_pipeline(text)

    return UploadResult(
        filename=fname or "unnamed",
        file_type=file_type,
        extracted_text=text,
        ocr_warning=ocr_warning,
        verification=verification,
    )


@app.post("/api/verify", response_model=VerificationResult)
async def verify_document(payload: DocumentPayload):
    """JSON text body — Stage 1-4 pipeline."""
    text = payload.text.strip()
    if not text:
        return JSONResponse(status_code=422, content={"detail": "Document text is empty."})
    return _run_pipeline(text)


@app.post("/api/verify-file", response_model=VerificationResult)
async def verify_file(file: UploadFile = File(...)):
    """
    Legacy multipart endpoint retained for backward compatibility.
    Delegates to /upload internally.
    """
    raw_bytes = await file.read()
    fname = (file.filename or "").strip()
    ext   = Path(fname).suffix.lower()

    if ext in _PDF_EXTS:
        try:
            text = _ingest_pdf(raw_bytes)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=422, content={"detail": f"PDF parsing error: {exc}"})
    elif ext in _IMAGE_EXTS:
        text, warning = _ingest_image(raw_bytes)
        if warning and not text:
            # Legacy endpoint: surface warning as a 200 plain-text response
            text = f"[OCR unavailable] {warning}"
    else:
        text = _ingest_text(raw_bytes)

    text = text.strip()
    if not text:
        return JSONResponse(status_code=422, content={"detail": "Uploaded file contains no extractable text."})
    return _run_pipeline(text)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found. Place index.html in static/</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
