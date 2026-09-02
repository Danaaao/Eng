# Healthcare Clinical Trial Matcher
### مُطابِق التجارب السريرية

**Capstone project — Day 5**
Course: *Modern Data Engineering for AI Systems* · هندسة البيانات الحديثة لأنظمة الذكاء الاصطناعي
Programme: **[SDAIA — Saudi Data & AI Authority](https://github.com/SDAIA)** · الهيئة السعودية للبيانات والذكاء الاصطناعي
Course reference: `SDAIA-F-CRS-100-01-V1`

An integrated, production-simulated AI data platform that ingests real clinical trial
records from **ClinicalTrials.gov**, governs them under **HIPAA** rules, stores them in a
**Delta-style Lakehouse**, indexes them in a **vector database**, and answers researcher
questions through an **advanced RAG pipeline** — with an evidence trail, contradiction
alerts, and an audit entry for every single query.

The entire platform is one Python file: [`clinical_trial_rag.py`](clinical_trial_rag.py).

---

## نبذة عن المشروع

نظام بيانات متكامل يدمج كل ما تم تدريسه في الأيام الخمسة من الدورة في معمارية واحدة:

| اليوم | المفهوم | ما ينفّذه المشروع |
|---|---|---|
| ١ | معماريات البيانات الحديثة | مستودع بحيرة (Lakehouse) بخصائص ACID، وفرض المخطط، والسفر عبر الزمن |
| ٢ | خطوط البيانات اللحظية | معمارية مدفوعة بالأحداث: منتجون ← وسيط ← مستهلكون، غير متزامنة |
| ٣ | قواعد البيانات المتجهية و RAG المتقدم | تقطيع، تضمينات، بحث هجين + إعادة ترتيب، تقييم ثلاثي |
| ٤ | جودة البيانات والحوكمة والنَّسَب | بوابة جودة، حماية البيانات الصحية (HIPAA)، تتبع النَّسَب، سجل تدقيق |
| ٥ | التكامل المعماري | تنسيق كل ما سبق في مخطط تنفيذ واحد (DAG) |

**الفكرة:** الباحث الطبي يسأل سؤالاً بلغة طبيعية عن التجارب السريرية المناسبة لحالة مريض،
فيجيبه النظام بإجابة **مؤسَّسة على بيانات حقيقية موثَّقة**، مع **مسار أدلة** يربط كل معلومة
بمصدرها، و**تنبيهات تعارض** عند اختلاف البيانات المهيكلة عن النص، **دون أن تخرج أي بيانات
مريض محمية (PHI) من النطاق الآمن**، ومع تسجيل كل استعلام في سجل تدقيق غير قابل للتعديل.

المشروع كله في **ملف بايثون واحد** كما يقتضي تكليف اليوم الخامس، ويعتمد على **بيانات حقيقية**
من السجل الأمريكي العام للتجارب السريرية `ClinicalTrials.gov`.

---

## 1. Objective

A clinical researcher needs to know which trials a patient might qualify for. Doing this
by hand means reading hundreds of registry entries, and asking a general-purpose LLM means
getting confidently invented NCT numbers and eligibility thresholds.

This platform answers the question in natural language, and every sentence of the answer is
traceable back to a specific governed source passage. Three properties are non-negotiable:

| Property | How it is guaranteed |
|---|---|
| **Grounded** — no invented trials or thresholds | RAG over a curated corpus + a system prompt that forbids outside knowledge + a measured Groundedness score |
| **Private** — no PHI leaves the safe zone | PHI is stripped before embedding, and the vector database physically refuses any chunk that still carries an identifier |
| **Auditable** — every access is on record | Append-only audit trail plus end-to-end data lineage |

---

## 2. Architecture

```
                    +----------------------------------+
                    |          DATA SOURCES            |
                    |  ClinicalTrials.gov API v2       |
                    |  Internal EHR (synthetic, PHI)   |
                    +----------------------------------+
                                     |
                                     v
                    +----------------------------------+
                    |  REAL-TIME INGESTION (Day 2)     |
                    |  MockEventBroker (Kafka pattern) |
                    |  async Producers -> topics       |
                    +----------------------------------+
                                     |
                                     v
                    +----------------------------------+
                    |  DATA QUALITY GATE (Day 4)       |
                    |  6 dimensions | quarantine zone  |
                    +----------------------------------+
                                     |
                                     v
                    +----------------------------------+
                    |  GOVERNANCE LAYER (Day 4)        |
                    |  PHI guard | ABAC | lineage      |
                    |  audit trail | classification    |
                    +----------------------------------+
                                     |
                                     v
                    +----------------------------------+
                    |  LAKEHOUSE (Day 1)               |
                    |  ACID | schema enforcement       |
                    |  _delta_log | time travel        |
                    +----------------------------------+
                                     |
                                     v
                    +----------------------------------+
                    |  CHUNKING + EMBEDDING (Day 3)    |
                    |  recursive chunking w/ overlap   |
                    |  sentence-transformers, 384-dim  |
                    +----------------------------------+
                                     |
                                     v
                    +----------------------------------+
                    |  VECTOR DATABASE (Day 3)         |
                    |  ChromaDB | HNSW | cosine        |
                    |  PHI gate at the insert door     |
                    +----------------------------------+
                                     |
                                     v
                    +----------------------------------+
                    |  ADVANCED RAG (Day 3)            |
                    |  hybrid dense+BM25 -> RRF        |
                    |  cross-encoder rerank -> top 5   |
                    |  contradiction detection         |
                    +----------------------------------+
                                     |
                                     v
                    +----------------------------------+
                    |  GENERATION + EVALUATION         |
                    |  LLM answer | evidence trail     |
                    |  RAG Triad | audit entry         |
                    +----------------------------------+

        All six pipeline stages are scheduled by the ORCHESTRATION DAG (Day 5)
        with dependency resolution, retries, exponential backoff, and fallbacks.
```

---

## 3. Where each course concept lives in the code

This is the mapping between the five training days and the implementation.

| Day | Concept | Class / function in `clinical_trial_rag.py` |
|---|---|---|
| 1 | Lakehouse over object storage | `DeltaStyleLakehouse` |
| 1 | ACID atomicity (all-or-nothing) | `DeltaStyleLakehouse.write` — parquet written first, commit appended only on success |
| 1 | Schema enforcement | `SchemaEnforcementError`, blueprint pinned at version 0 |
| 1 | Time travel / transaction log | `DeltaStyleLakehouse.read(version=...)`, `.history()`, `_delta_log/` |
| 1 | ELT — preserve raw forever | Raw API response cached to `data/landing_zone/ctgov_raw.json` before any transformation |
| 2 | Event-driven architecture | `MockEventBroker` with `asyncio.Queue` topics |
| 2 | Producers / Broker / Consumers | `ClinicalTrialsProducer`, `InternalEHRProducer`, `MockEventBroker.consume` |
| 2 | Decoupling & async ingestion | Producers publish without waiting for downstream consumers |
| 2 | Ingestion → Transformation → Orchestration | The `ingest` → `quality` → … DAG in `build_pipeline` |
| 3 | Chunking (recursive + overlap) | `RecursiveChunker` |
| 3 | Embeddings | `EmbeddingModel`, `HashingEmbedder` (offline fallback) |
| 3 | Vector DB with HNSW / cosine | `VectorStore` (ChromaDB, `hnsw:space=cosine`) |
| 3 | Hybrid retrieval + Reciprocal Rank Fusion | `HybridRetriever` (dense + BM25, `RRF_K=60`) |
| 3 | Two-stage cross-encoder reranking | `CrossEncoderReranker`, `LexicalReranker` |
| 3 | Incremental sync | Lakehouse initial load + delta append in the `lakehouse` task |
| 3 | RAG Triad evaluation | `RAGTriadEvaluator` |
| 4 | Six data quality dimensions | `DataQualityEngine.run_all_checks` |
| 4 | Quality gate + quarantine | `quarantine_batch`, `halt_pipeline` threshold, FATAL vs ADVISORY severity |
| 4 | Data classification taxonomy | `DATA_CLASSIFICATION` |
| 4 | ABAC / principle of least privilege | `ROLE_POLICIES`, `GovernanceEngine.apply_access_policy` |
| 4 | HIPAA PHI detection & de-identification | `PHIGuard` |
| 4 | Automated data lineage | `LineageTracker` |
| 4 | Audit trail | `AuditTrail` |
| 5 | Integrating everything into one platform | `build_pipeline` |
| 5 | AI infrastructure orchestration | `PipelineOrchestrator` — DAG, retries, backoff, run report |
| 5 | Failure recovery + fallbacks | Three-tier ingestion ladder, model fallbacks, task retries |

---

## 4. Tools used, and why

| Layer | Tool | Why this one |
|---|---|---|
| Ingestion | `httpx` + ClinicalTrials.gov **API v2** | Real, public, free, no API key. The registry is the authoritative source for trial data. |
| Streaming | `asyncio.Queue` broker | Models Kafka/Redpanda semantics (topics, immutable append, decoupled consumers) without needing a cluster on a laptop. |
| Quality | `pandas` + custom `DataQualityEngine` | Vectorised checks over the batch; the same shape as a Great Expectations suite, but readable in one screen. |
| Storage | Parquet + JSON transaction log | Delta Lake semantics (ACID, schema enforcement, time travel) with no JVM dependency, so the demo never fails on a Java version mismatch. See the note below. |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` | The model recommended in the Day 5 tech stack. 384 dimensions is the right trade-off between quality and speed for a corpus of this size. |
| Vector DB | **ChromaDB** with HNSW + cosine | Day 3's decision tree: high recall needed and RAM is available → HNSW. Cosine, because document length must not dominate the score. |
| Sparse retrieval | `rank-bm25` | Vector search alone is blind to exact tokens like `NCT04223752`, `HbA1c`, `SGLT2`. BM25 covers exactly that gap. |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Required by the Day 5 brief for Idea 2. Mitigates the "lost-in-the-middle" problem by cutting 50 candidates down to the best 5. |
| LLM | OpenRouter (OpenAI-compatible) | Same interface used in the Day 3 lab, so the model is swappable via one environment variable. |
| Logging | `loguru` | Same logger used in the Day 4 lab; writes both to console and to `pipeline_execution.log`. |

**Two deliberate substitutions, and the reasoning:**

- **Delta Lake → a Delta-style log implemented in-file.** The Day 1 lab uses `delta-spark`,
  which needs Spark and a matching JVM. Keeping the capstone a single dependency-light file
  matters more than using the exact library, so the *semantics* were reimplemented: ordered
  immutable commits in `_delta_log/`, a schema blueprint pinned at version 0, write-then-commit
  atomicity, and version-addressed reads. The behaviour a reviewer checks — a blocked schema
  drift and a working time travel query — is identical.
- **Qdrant → ChromaDB.** The Day 5 brief suggests Qdrant for Idea 2; the Day 5 recommended
  stack table also lists ChromaDB, and Chroma is what Day 3 taught. Both use HNSW, so the
  retrieval architecture is unchanged. Chroma runs embedded with no server to start.

---

## 5. Internal working — what happens when a researcher asks a question

```
Researcher question
  |
  1. ACCESS CONTROL      role checked against ROLE_POLICIES (ABAC)
  |
  2. PHI GUARD           the inbound query is scanned and redacted.
  |                      A researcher pasting a patient name into the search
  |                      box is the most common PHI leak; it is stopped here.
  |
  3. EMBED               query -> 384-dim vector, same model that built the index
  |
  4. HYBRID RETRIEVAL    dense top-25 (HNSW/cosine)  +  sparse top-25 (BM25)
  |                      fused with Reciprocal Rank Fusion, RRF_K = 60
  |
  5. RERANK              cross-encoder rescores the fused candidates -> top 5
  |
  6. CONTRADICTION       structured metadata compared against the free text of
  |     DETECTION        the same trial; conflicts are surfaced, not hidden
  |
  7. PROMPT BUILD        system rules + numbered context passages + question
  |
  8. GENERATION          LLM answers, citing passages inline as [1], [2], ...
  |
  9. EVIDENCE TRAIL      each citation resolves to nct_id + chunk_id +
  |                      lineage_id + registry URL
  |
 10. RAG TRIAD           Context Relevance / Groundedness / Answer Relevance
  |
 11. AUDIT               actor, role, redacted query, NCT IDs returned, scores,
                         latency — appended to the immutable audit log
```

### How HIPAA is actually enforced

ClinicalTrials.gov data is public and contains no PHI. The PHI in this system comes from the
**internal EHR source** and from **researchers' own queries** — which is exactly where it
comes from in a real hospital. Three controls apply:

1. **De-identification at ingestion.** `PHIGuard.deidentify_patient` drops every field
   classified `RESTRICTED` (name, MRN, date of birth, phone, e-mail, address) and replaces the
   patient's identity with a deterministic HMAC-derived pseudonym such as `PATIENT_6168A565FC`.
   The same patient always maps to the same token, and the token cannot be reversed.
2. **A hard gate at the vector database door.** `GovernanceEngine.assert_vector_db_safe`
   re-scans every chunk immediately before insertion and raises `PermissionError` if an
   identifier survived. *No PHI in the vector DB* is enforced by code, not by a policy document.
3. **Query-time redaction plus an audit entry.** Inbound questions are scrubbed, and the audit
   log records which identifier families were stripped.

---

## 6. How to run it

### Setup

```bash
git clone https://github.com/Danaaao/Eng.git
cd Eng

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env              # then add your OPENROUTER_API_KEY
```

The platform runs without an API key — it falls back to a deterministic extractive answerer
that quotes the retrieved evidence verbatim. The RAG Triad is still measured.

### Build the platform (downloads real data from ClinicalTrials.gov)

```bash
python clinical_trial_rag.py --stage pipeline
```

This runs the full DAG: ingest → quality → governance → lakehouse → chunk_embed → vector_index.
Look for this line — it confirms the data came from the live registry:

```
📥 [PRODUCER: ClinicalTrials.gov] Ingestion mode = LIVE | 60 studies
```

### Where the dataset lives — أين تُحفظ البيانات

The pipeline writes the untouched registry response to:

```
data/landing_zone/ctgov_raw.json
```

**You do not create this file or place it anywhere by hand.** It is written automatically the
first time `--stage pipeline` runs against the live registry. That path is the platform's
*landing zone*, and the file is the dataset: the raw JSON exactly as ClinicalTrials.gov
returned it, before any cleaning, validation, or transformation.

Everything else under `data/` is derived from it and is regenerated on every run — the
lakehouse commits, the vector index, the quarantine zone, the lineage and audit logs. Only the
raw file is kept under version control, so commit it after your first live run:

```bash
git add data/landing_zone/ctgov_raw.json
git commit -m "Add real ClinicalTrials.gov dataset"
git push origin main
```

Once it is committed, anyone can rebuild the entire platform on your exact data with no
network access at all:

```bash
python clinical_trial_rag.py --stage pipeline --offline    # Ingestion mode = REPLAY
```

This is the **ELT principle** from Day 1 made concrete: raw data is preserved permanently, and
every transformation is re-run against it rather than replacing it. It is also what makes the
project reproducible for whoever evaluates it.

### Ask a question

```bash
python clinical_trial_rag.py --stage query \
  --q "Which trials are recruiting adults over 60 with type 2 diabetes, and what are the age limits?"
```

### Full walkthrough (use this for the presentation)

```bash
python clinical_trial_rag.py --stage demo
```

Builds the platform, prints the transaction log and time travel, shows the lineage graph, runs
two researcher queries with evidence trails and contradiction alerts, demonstrates ABAC masking
across three roles, proves the PHI gate blocks a poisoned chunk, then prints the audit trail.

### Inspect the governance artefacts

```bash
python clinical_trial_rag.py --stage history    # Lakehouse commits + time travel
python clinical_trial_rag.py --stage lineage    # end-to-end data genealogy
python clinical_trial_rag.py --stage audit      # who queried what, and when
```

### Other options

| Flag | Meaning |
|---|---|
| `--offline` | Skip the live registry call and replay the cached raw JSON |
| `--role` | ABAC role: `clinical_researcher`, `treating_physician`, `data_engineer`, `public` |
| `--actor` | Identity recorded in the audit trail |
| `--limit` | Number of audit entries to display |

You can also run it with `uv`, which reads the inline dependency header and needs no virtualenv:

```bash
uv run clinical_trial_rag.py --stage demo
```

---

## 7. Sample output

```
▶️  [DAG] Task 'lakehouse'  (depends on: ['governance'])
17:10:36 | SUCCESS  | Delta commit v0 | WRITE | 4 records
17:10:36 | SUCCESS  | Delta commit v1 | APPEND | 2 records

  Attempting to append a batch carrying an unannounced 'investigator_notes' column...
  ❌ Transaction blocked by the Lakehouse! Schema mismatch. Unexpected columns: ['investigator_notes']
  💡 This is what stops a data lake from degrading into a data swamp.
```

```
  EVIDENCE TRAIL  (every claim traceable to a governed source)
----------------------------------------------------------------------
  [1] NCT03015220 | section=eligibility | via=hybrid | rerank=0.4773
       chunk   : NCT03015220::eligibility::0
       lineage : LIN_0003_LAKEHOUSE_WRITE
       source  : https://clinicaltrials.gov/study/NCT03015220

  ⚠️  CONTRADICTION ALERTS
  • Trial NCT03015220 declares a minimum age of 60 in its structured eligibility
    field, but its criteria text states 65 years. Verify against the registry
    before screening.

  RAG TRIAD EVALUATION
  Context Relevance : 0.356   (did retrieval pull the right passages?)
  Groundedness      : 0.885   (is the answer supported by those passages?)
  Answer Relevance  : 0.346   (does it address the question asked?)
```

```
  ❌ PHI detected in chunk DEMO::phi_leak::0 (['EMAIL', 'MRN']).
     Refusing to write it to the vector database.
  💡 'No PHI in the vector DB' is enforced by code, not by policy documents.
```

---

## 8. How this was built — the development journey

1. **Started from the source of truth.** Rather than mocking a dataset, the pipeline calls the
   real ClinicalTrials.gov API v2 and caches the untouched response in a landing zone — the ELT
   principle that raw data is preserved forever so any transformation can be replayed.
2. **Added the quality gate before anything downstream.** Real registry data has missing
   summaries, stale entries, and inconsistent status vocabularies. Six dimensions are checked.
   The first version quarantined every violation equally — and testing it against a realistic
   registry response showed why that was wrong: more than half the batch was rejected, mostly
   for being *old*, and the pipeline halted. But a completed trial from 2017 is still valid
   evidence; it is simply stale. Violations were therefore split by severity. **FATAL** findings
   (no identifier, malformed NCT ID, duplicate, impossible ages) mean the record cannot be
   trusted or cited, so it is quarantined. **ADVISORY** findings (no brief summary, unrecognised
   status value, not updated in five years) travel with the record as a flag and surface in the
   evidence trail beside the passage they qualify. Only fatal violations count towards the 30%
   threshold that halts the pipeline on the assumption that the upstream system is broken.
3. **Layered governance on top.** Every field was given a classification, PHI detection was
   written against the HIPAA identifier families, and every pipeline operation started emitting
   a lineage node and an audit entry.
4. **Built the Lakehouse.** Schema enforcement was verified by deliberately trying to append a
   rogue column, and time travel by splitting the load into an initial commit plus an
   incremental delta.
5. **Made retrieval production-grade.** The first version was naive RAG — dense search only.
   It failed on exact tokens and on questions about age limits, because the ages lived in a
   metadata column and never appeared in the chunk text. Two fixes followed: BM25 fused via RRF
   to cover exact-token queries, and structured eligibility facts inlined into the eligibility
   chunks so those chunks became genuinely self-contained. Cross-encoder reranking was then
   added as a second stage.
6. **Wired it all into an orchestration DAG** with dependency resolution, retries with
   exponential backoff, and per-component fallbacks, then added the RAG Triad so any future
   change to the architecture can be measured rather than guessed at.

---

## 9. Notes on data and resilience

**Ingestion has three tiers.** The platform tries the live registry first; if the network is
unavailable it replays the cached raw JSON from the landing zone; if there is no cache either,
it falls back to six clearly-labelled synthetic records whose IDs are prefixed `SYNTH` so they
can never be mistaken for real registry data. The active tier is printed at the top of every
run (`Ingestion mode = LIVE | REPLAY | SYNTHETIC`).

**Models degrade the same way.** If `sentence-transformers` cannot download its weights, the
embedder falls back to a deterministic hashed bag-of-tokens vector space and the reranker to
IDF-weighted lexical scoring. Retrieval quality drops, but the architecture — hybrid retrieval,
two-stage reranking, triad evaluation — is unchanged and still demonstrable. The active backend
is printed with every answer.

**The patient records are synthetic.** They deliberately contain PHI-shaped values so the
governance layer has something real to detect and block. No real person is represented.

---

## 10. Repository layout

```
.
├── clinical_trial_rag.py   # the entire platform, one file
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

data/
├── landing_zone/
│   └── ctgov_raw.json      # THE DATASET — raw registry response, version-controlled
├── quarantine_zone/        # records rejected by the quality gate      (regenerated)
├── lakehouse/                                                          (regenerated)
│   └── clinical_trials/
│       ├── _delta_log/     # immutable numbered commits
│       └── data/           # parquet files, one per version
├── governance/                                                         (regenerated)
│   ├── lineage.jsonl       # end-to-end data genealogy
│   └── audit_trail.jsonl   # every query, append-only
└── chroma_db/              # persistent vector index                   (regenerated)
```

Only `data/landing_zone/ctgov_raw.json` is committed. Everything marked *(regenerated)* is
derived from it and rebuilt by `--stage pipeline`, so it is deliberately excluded from version
control — a derived artefact in a repository goes stale and starts lying about what the code
produces.
