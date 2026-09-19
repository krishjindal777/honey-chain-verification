# 🍯 HoneyGuard AI
### Deterministic Document Compliance & Purity Verification Portal

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-8B5CF6)](https://www.trychroma.com)
[![SDG 12](https://img.shields.io/badge/UN%20SDG-12%20Responsible%20Consumption-BF8B2E)](https://sdgs.un.org/goals/goal12)
[![Powered by IBM Bob](https://img.shields.io/badge/Agentic%20AI-IBM%20Bob-0530AD)](https://www.ibm.com)

---

## Problem Statement & SDG 12 Alignment

Global honey fraud costs the industry an estimated **$700M+ annually**. The three primary attack vectors are:

| Fraud Type | Mechanism | Detection Challenge |
|---|---|---|
| **Thermal degradation** | Overheating raises HMF above safe thresholds | Requires quantitative lab comparison |
| **C4 invert syrup adulteration** | Cheap corn/cane syrup blended into product | Needs certified isotopic analysis |
| **Fraudulent certificates** | Forged lab reports, expired organic status | Requires cross-referencing regulatory standards |

Manual certificate review is slow, inconsistent, and scales poorly across supply chains. **HoneyGuard AI** automates the full audit pipeline — from raw document ingestion to a cryptographically signed compliance receipt — enforcing **UN SDG Goal 12: Responsible Consumption and Production** at the point of trade.

---

## Key Architectural Highlights

### 1 · Multi-Modal Document Ingestion
Certificates arrive in multiple formats. The ingestion layer handles all of them transparently:

- **PDF** — binary stream parsing via `pypdf` (`PdfReader`), iterating all pages and joining text layers
- **Images (PNG / JPG / JPEG)** — OCR character recognition via `pytesseract` + `Pillow`; auto-locates the Tesseract binary in common Windows install paths without requiring PATH changes
- **Plain Text (.txt)** — UTF-8 decode with latin-1 fallback for legacy lab reports

### 2 · Regulatory RAG (Retrieval-Augmented Generation)
Rather than relying on a generalist LLM, the system performs **grounded semantic retrieval** against a curated regulatory corpus stored in an in-memory ChromaDB vector collection (`honey_regulatory_standards`).

Seeded standards:
| ID | Standard | Coverage |
|---|---|---|
| `codex-cxs-12-1981-hmf-standard` | Codex Alimentarius CXS 12-1981 | HMF 40 / 80 mg/kg threshold |
| `codex-cxs-12-1981-moisture` | Codex Alimentarius CXS 12-1981 | Moisture & diastase activity |
| `iso-iec-17025-lab-accreditation` | ISO/IEC 17025:2017 | Lab accreditation & ILAC MRA |
| `national-organic-apiculture-certification` | National Organic Program (NOP) | Apiary certification & shelf-life |
| `codex-cxs-12-1981-adulteration` | Codex Alimentarius CXS 12-1981 | Adulteration prohibition |

Each verification run builds a focused semantic query from extracted fields (HMF level, accreditation string, organic status) and retrieves the **top-3 governing clauses** with cosine similarity scores surfaced directly in the audit report.

### 3 · Deterministic Zero-Hallucination Rule Engine
After RAG retrieval, compliance is evaluated by **hard-coded deterministic rules** — never by asking a language model to decide:

```
Rule 1 — HMF ≤ 40 mg/kg          (Codex Alimentarius CXS 12-1981)
Rule 2 — ISO/IEC 17025 accredited lab  (ILAC MRA recognised)
Rule 3 — Organic certification active  (NOP / national equivalent)
```

This ensures results are **fully reproducible** regardless of model updates or embedding drift. The RAG layer contributes the governing clause *citation* to the audit response; the pass/fail decision is always deterministic.

### 4 · Cryptographic Audit Trail
Every verification produces an immutable audit record:

```
audit_id      = AUD-{SHA256[:12].upper()}-{UUID4[:6].upper()}
document_hash = SHA-256(document_text | UTC_timestamp)
timestamp     = ISO 8601 UTC
```

The SHA-256 hash binds the exact document content to the audit outcome — any tampering with the certificate after verification is detectable.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HoneyGuard AI Pipeline                   │
└─────────────────────────────────────────────────────────────────┘

  Raw Certificate
  (.pdf / .png / .jpg / .txt)
         │
         ▼
  ┌─────────────────┐
  │  Ingestion Layer │  pypdf  ·  pytesseract + Pillow  ·  UTF-8 decode
  └────────┬────────┘
           │  plain text
           ▼
  ┌─────────────────┐
  │ Entity Extractor │  regex patterns → batch_id · producer · harvest_date
  └────────┬────────┘                    organic_status · hmf_level · lab_accreditation
           │  ExtractedData
           ▼
  ┌──────────────────────────┐
  │   ChromaDB RAG Retrieval  │  semantic query → top-3 regulatory clauses
  │  honey_regulatory_standards│  (Codex CXS 12-1981 · ISO/IEC 17025 · NOP)
  └────────┬─────────────────┘
           │  RagMatch[]  (standard · clause · distance)
           ▼
  ┌─────────────────┐
  │  Rules Engine    │  HMF threshold · ISO accreditation · organic status
  └────────┬────────┘  → checklist{pass/fail} · issues[]
           │
           ▼
  ┌──────────────────────────┐
  │   Audit Assembly (Stage 4)│  SHA-256 hash · UUID audit_id · UTC timestamp
  └────────┬─────────────────┘
           │
           ▼
     JSON Audit Response
     {status: VERIFIED | FLAGGED,
      extracted, checklist, rag_matches, issues, audit_id, document_hash}
```

---

## Prerequisites & System Binaries

### Python
```
Python 3.10 or higher
```

### Tesseract OCR (required for image uploads)

**Windows — recommended (winget):**
```powershell
winget install UB-Mannheim.TesseractOCR
```
Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

> **No PATH configuration needed.** HoneyGuard AI auto-detects Tesseract at the standard Windows install location via `_configure_tesseract()`.

**Windows — manual installer:**
Download from https://github.com/UB-Mannheim/tesseract/wiki and run the `.exe` installer.

**Ubuntu / Debian:**
```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Python Dependencies
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
chromadb>=0.5.0
python-multipart>=0.0.9
pypdf>=4.0.0
Pillow>=10.0.0
pytesseract>=0.3.10
```

---

## Quickstart — Local Deployment

```bash
# 1. Clone the repository
git clone https://github.com/krishjindal777/honey-chain-verification.git
cd honey-chain-verification

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start the server
python app.py
```

Open **http://localhost:8000** in your browser.

> ChromaDB runs fully **in-memory** — no database setup or configuration required. The regulatory corpus is seeded automatically on first startup.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend dashboard |
| `POST` | `/upload` | **Primary endpoint** — multipart file upload (.pdf/.png/.jpg/.jpeg/.txt), returns `UploadResult` with `extracted_text` + full audit |
| `POST` | `/api/verify` | JSON text body `{"text": "..."}` → `VerificationResult` |
| `POST` | `/api/verify-file` | Legacy multipart endpoint → `VerificationResult` |

### `UploadResult` schema
```json
{
  "filename": "certificate.pdf",
  "file_type": "pdf",
  "extracted_text": "Batch ID: HNY-2024-...",
  "ocr_warning": null,
  "verification": {
    "audit_id": "AUD-3F9A2C1B04E7-8A3D1F",
    "timestamp": "2024-06-01T10:23:45Z",
    "status": "VERIFIED",
    "extracted": { "batch_id": "...", "hmf_level": 18.0, "organic_status": true, ... },
    "rag_matches": [{ "standard": "Codex CXS 12-1981", "clause": "...", "distance": 0.286 }],
    "checklist": {
      "hmf_within_codex_limit": true,
      "iso_17025_accredited_lab": true,
      "organic_certification_active": true
    },
    "issues": [],
    "document_hash": "e3b0c44298fc..."
  }
}
```

---

## Test Fixtures & Samples

Two quick-fill samples are built into the dashboard. You can also paste the text directly into the textarea.

### ✅ Valid Certificate (PASS)
```
HONEY QUALITY CERTIFICATE
Batch ID: HNY-2024-PK-0091
Producer: Sohna Organic Honey Co., Punjab
Harvest Date: 2024-03-15
Organic Status: Active
Lab Accreditation: ISO 17025 Certified (Accred. No. PNAC-1234)
HMF Level: 18 mg/kg
```
**Expected:** `status: VERIFIED` · all 3 checklist items PASS

### ❌ Adulterated Sample (FAIL / FLAGGED)
```
HONEY ANALYSIS REPORT
Batch ID: HNY-2024-XX-9988
Producer: Unknown Wholesale Supplier
Harvest Date: 2023-11-01
Organic Status: Inactive
Lab Accreditation: None (Unaccredited facility)
HMF Level: 68 mg/kg
```
**Expected:** `status: FLAGGED` · 3 issues detected · all checklist items FAIL

### Image / PDF Upload
Drop any scanned honey certificate image (`.png`, `.jpg`) or PDF onto the upload zone. The system will:
1. OCR / parse the document
2. Auto-populate the text area with extracted content
3. Automatically run verification and render the full audit report

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | HTML5, Tailwind CSS (CDN) | Responsive dashboard, drag-and-drop upload, live audit cards |
| **Backend** | Python, FastAPI, Uvicorn | REST API, request routing, static file serving |
| **Extraction — OCR** | pytesseract, Pillow | Image → text via Tesseract v5 engine |
| **Extraction — PDF** | pypdf (PdfReader) | PDF text layer extraction, multi-page joining |
| **Vector Database** | ChromaDB (in-memory) | Semantic regulatory corpus, cosine similarity retrieval |
| **Embedding Model** | `all-MiniLM-L6-v2` (ONNX) | ChromaDB default embedding function |
| **Compliance Rules** | Python (deterministic) | Codex Alimentarius + ISO/IEC 17025 hard-coded checks |
| **Cryptography** | Python `hashlib` SHA-256 | Immutable document hash + audit ID generation |
| **Agent Orchestrator** | IBM Bob (Agentic AI) | End-to-end development orchestration |

---

## Engineering Note

This project was architected and built with **agentic AI orchestration via [IBM Bob](https://www.ibm.com)**, an enterprise AI software engineering assistant. The full-stack prototype — spanning multi-modal ingestion, ChromaDB RAG integration, deterministic compliance rules, cryptographic audit trail, and the Tailwind CSS dashboard — was developed through **human-in-the-loop verification engineering**: every generated artefact was reviewed, tested, and validated against live execution before being accepted.

This workflow demonstrates that production-grade AI-powered compliance tooling can be designed, implemented, and deployed at significantly accelerated timelines without sacrificing correctness or auditability — a core principle of responsible AI-assisted software development.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built for <strong>UN SDG 12</strong> · Responsible Consumption &amp; Production · 🍯</p>
