# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "pandas>=2.2",
#     "pyarrow>=16.0",
#     "loguru>=0.7",
#     "chromadb>=0.5",
#     "rank-bm25>=0.2.2",
#     "openai>=1.40",
#     "python-dotenv>=1.0",
#     "numpy>=1.26",
# ]
# ///
"""
Healthcare Clinical Trial Matcher
==================================
SDAIA - Modern Data Engineering for AI Systems | Day 5 Capstone Project

A single-file, production-simulated AI Data Platform that integrates every
architectural component taught across the five days of the course:

    Day 1  Modern Data Architectures ....... Delta-style Lakehouse, ACID, Time Travel
    Day 2  Real-Time Data Pipelines ........ Event-Driven Broker, Producers, Consumers
    Day 3  Vector DBs & Advanced RAG ....... Chunking, Embeddings, Hybrid + Reranking
    Day 4  Quality, Governance, Lineage .... Quality Gates, HIPAA/PHI, Audit, Lineage
    Day 5  Architecture Integration ........ Orchestration DAG over all of the above

Domain
------
Researchers ask natural-language questions about clinical trials. The system
ingests REAL trial records from ClinicalTrials.gov (public API v2) plus an
internal synthetic EHR source that deliberately contains PHI, then answers
questions with an evidence trail, contradiction alerts, and a full audit log -
while guaranteeing that NO PHI ever reaches the vector database.

Run it directly with uv:

    uv run clinical_trial_rag.py --stage demo

or with a normal virtualenv:

    pip install -r requirements.txt
    python clinical_trial_rag.py --stage pipeline
    python clinical_trial_rag.py --stage query --q "trials for type 2 diabetes in adults over 60"

Environment variables (a .env file is loaded automatically):

    OPENROUTER_API_KEY   - API key for OpenRouter. Without it the system falls
                           back to a deterministic extractive answerer so the
                           whole pipeline still runs end-to-end.
Optional:
    OPENROUTER_MODEL     - chat model used for answer generation
    EMBED_MODEL_NAME     - sentence-transformers model for embeddings
    RERANK_MODEL_NAME    - cross-encoder model used by the reranker
    CTGOV_CONDITION      - medical condition to ingest from ClinicalTrials.gov
    CTGOV_TARGET_RECORDS - how many trials to pull
"""

import os
import re
import sys
import json
import time
import math
import shutil
import hashlib
import asyncio
import argparse
import datetime as dt
from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger

# =====================================================================
# 0. ARCHITECTURAL COMPONENT: CONFIGURATION & GOVERNANCE POLICY REGISTRY
# =====================================================================

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # keeps the file runnable with a bare interpreter
    pass

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("pipeline_execution.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class Settings:
    """Centralized, validated configuration read once at startup."""

    def __init__(self) -> None:
        # --- LLM / model layer -------------------------------------------------
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder:free")
        self.embed_model_name = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.rerank_model_name = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        # --- Source system -----------------------------------------------------
        self.ctgov_api = "https://clinicaltrials.gov/api/v2/studies"
        self.ctgov_condition = os.getenv("CTGOV_CONDITION", "type 2 diabetes")
        self.ctgov_target_records = int(os.getenv("CTGOV_TARGET_RECORDS", "60"))
        self.ctgov_page_size = 50

        # --- Storage paths (the local "cloud object storage") ------------------
        self.landing_zone = "data/landing_zone"
        self.quarantine_zone = "data/quarantine_zone"
        self.lakehouse_path = "data/lakehouse/clinical_trials"
        self.governance_dir = "data/governance"
        self.chroma_db_dir = os.getenv("CHROMA_DB_DIR", "data/chroma_db")

        # --- Pipeline thresholds ----------------------------------------------
        self.quality_halt_threshold = 0.30   # >30% bad records halts the pipeline
        self.chunk_size = 900                # characters per chunk
        self.chunk_overlap = 150             # ~17% overlap (Day 3 recommends 10-20%)
        self.dense_top_k = 25                # stage-1 dense candidates
        self.sparse_top_k = 25               # stage-1 sparse candidates
        self.rerank_top_n = 5                # stage-2 survivors handed to the LLM

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.ctgov_condition,
            "target_records": self.ctgov_target_records,
            "embed_model": self.embed_model_name,
            "rerank_model": self.rerank_model_name,
            "llm_model": self.openrouter_model,
            "llm_configured": bool(self.openrouter_api_key),
        }


settings = Settings()


# --- Enterprise data classification taxonomy (Day 4, Cornerstone 2) ------------
# Every field the platform touches is labelled. The label drives masking,
# access control, and whether the field is allowed into the vector database.
DATA_CLASSIFICATION: dict[str, str] = {
    # ClinicalTrials.gov is a public registry - safe for the vector DB.
    "nct_id": "PUBLIC",
    "brief_title": "PUBLIC",
    "brief_summary": "PUBLIC",
    "conditions": "PUBLIC",
    "eligibility_criteria": "PUBLIC",
    "interventions": "PUBLIC",
    "overall_status": "PUBLIC",
    "phase": "PUBLIC",
    "sponsor": "PUBLIC",
    "locations": "PUBLIC",
    "min_age_years": "PUBLIC",
    "max_age_years": "PUBLIC",
    "sex": "PUBLIC",
    "enrollment": "PUBLIC",
    "last_update": "PUBLIC",
    # Internal EHR fields - PHI under HIPAA, must never leave the safe zone.
    "patient_name": "RESTRICTED",
    "medical_record_number": "RESTRICTED",
    "date_of_birth": "RESTRICTED",
    "phone": "RESTRICTED",
    "email": "RESTRICTED",
    "address": "RESTRICTED",
    # De-identified clinical facts - usable for matching.
    "patient_pseudonym": "INTERNAL",
    "age_years": "INTERNAL",
    "diagnosis": "INTERNAL",
    "hba1c": "INTERNAL",
    "comorbidities": "INTERNAL",
}

# Attribute-Based Access Control matrix (Day 4, Cornerstone 2 - PoLP).
ROLE_POLICIES: dict[str, set[str]] = {
    "clinical_researcher": {"PUBLIC", "INTERNAL"},          # cannot see PHI
    "treating_physician": {"PUBLIC", "INTERNAL", "RESTRICTED"},
    "data_engineer": {"PUBLIC"},                            # pipelines see public only
    "public": {"PUBLIC"},
}

# Fields that are categorically forbidden from the vector database (HIPAA).
VECTOR_DB_FORBIDDEN_CLASSES = {"RESTRICTED"}


def banner(title: str) -> None:
    """Prints the standard console section header used across the platform."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def ensure_dirs() -> None:
    for path in (
        settings.landing_zone,
        settings.quarantine_zone,
        settings.governance_dir,
        os.path.dirname(settings.lakehouse_path),
    ):
        os.makedirs(path, exist_ok=True)


# =====================================================================
# 1. ARCHITECTURAL COMPONENT: INGESTION LAYER (EVENT-DRIVEN PIPELINE)
# =====================================================================

class MockEventBroker:
    """Simulates a highly resilient message broker like Apache Kafka or Redpanda."""

    def __init__(self) -> None:
        self.topics: dict[str, asyncio.Queue] = {
            "raw_clinical_trials": asyncio.Queue(),
            "raw_patient_records": asyncio.Queue(),
            "validated_records": asyncio.Queue(),
        }
        self.published_counts: dict[str, int] = {topic: 0 for topic in self.topics}

    async def publish(self, topic: str, payload: dict, source: str) -> None:
        """Asynchronously writes an immutable event envelope to a topic channel."""
        envelope = {
            "event_id": f"EVT_{hashlib.md5(f'{topic}{self.published_counts[topic]}{source}'.encode()).hexdigest()[:10].upper()}",
            "event_time": utcnow(),
            "topic": topic,
            "source_system": source,
            "payload": payload,
        }
        await self.topics[topic].put(envelope)
        self.published_counts[topic] += 1

    async def consume(self, topic: str) -> dict:
        """Blocks asynchronously until a new event becomes available on the channel."""
        return await self.topics[topic].get()

    def drain(self, topic: str) -> list[dict]:
        """Synchronously pulls every buffered event out of a topic channel."""
        events = []
        queue = self.topics[topic]
        while not queue.empty():
            events.append(queue.get_nowait())
        return events


def parse_age_to_years(raw: str | None) -> float | None:
    """Converts ClinicalTrials.gov age strings ('18 Years', '6 Months') to years."""
    if not raw or not isinstance(raw, str):
        return None
    match = re.match(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)", raw.strip(), re.IGNORECASE)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2).lower()
    factor = {"year": 1.0, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365}[unit]
    return round(value * factor, 2)


def flatten_ctgov_study(study: dict) -> dict[str, Any]:
    """Maps a nested ClinicalTrials.gov API v2 study into a flat tabular record.

    Every lookup is defensive: the registry omits whole modules for some trials,
    and a KeyError in ingestion would take down the entire streaming pipeline.
    """
    section = study.get("protocolSection", {}) or {}
    ident = section.get("identificationModule", {}) or {}
    status = section.get("statusModule", {}) or {}
    desc = section.get("descriptionModule", {}) or {}
    conds = section.get("conditionsModule", {}) or {}
    design = section.get("designModule", {}) or {}
    elig = section.get("eligibilityModule", {}) or {}
    arms = section.get("armsInterventionsModule", {}) or {}
    contacts = section.get("contactsLocationsModule", {}) or {}
    sponsors = section.get("sponsorCollaboratorsModule", {}) or {}

    interventions = [
        f"{i.get('type', 'OTHER')}: {i.get('name', '')}".strip()
        for i in (arms.get("interventions") or [])
    ]
    locations = [
        ", ".join(filter(None, [loc.get("city"), loc.get("state"), loc.get("country")]))
        for loc in (contacts.get("locations") or [])
    ]

    return {
        "nct_id": ident.get("nctId"),
        "brief_title": ident.get("briefTitle"),
        "overall_status": status.get("overallStatus"),
        "last_update": (status.get("lastUpdatePostDateStruct") or {}).get("date"),
        "brief_summary": desc.get("briefSummary"),
        "conditions": "; ".join(conds.get("conditions") or []),
        "study_type": design.get("studyType"),
        "phase": "; ".join(design.get("phases") or []),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "eligibility_criteria": elig.get("eligibilityCriteria"),
        "min_age_years": parse_age_to_years(elig.get("minimumAge")),
        "max_age_years": parse_age_to_years(elig.get("maximumAge")),
        "sex": elig.get("sex"),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "interventions": "; ".join(interventions[:6]),
        "locations": "; ".join(dict.fromkeys(locations))[:400],
        "sponsor": (sponsors.get("leadSponsor") or {}).get("name"),
        "source_url": f"https://clinicaltrials.gov/study/{ident.get('nctId')}",
    }


class ClinicalTrialsProducer:
    """Ingestion producer for the public ClinicalTrials.gov registry (API v2).

    Implements a three-tier resilience strategy, exactly the 'automatic retries
    and fallbacks' behaviour the orchestration layer is supposed to provide:

        1. LIVE    - call the real registry API and cache the raw response
        2. REPLAY  - re-read the cached raw JSON from the landing zone (ELT:
                     raw data is preserved forever, so the pipeline is
                     reproducible offline)
        3. SYNTHETIC - clearly-labelled fallback records so a demo never dies
    """

    def __init__(self, broker: MockEventBroker) -> None:
        self.broker = broker
        self.raw_cache = os.path.join(settings.landing_zone, "ctgov_raw.json")
        self.mode = "UNKNOWN"

    # -- tier 1 ---------------------------------------------------------------
    def _fetch_live(self) -> list[dict]:
        import httpx

        studies: list[dict] = []
        page_token: str | None = None
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            while len(studies) < settings.ctgov_target_records:
                params: dict[str, Any] = {
                    "query.cond": settings.ctgov_condition,
                    "pageSize": min(settings.ctgov_page_size, settings.ctgov_target_records - len(studies)),
                    "format": "json",
                    "countTotal": "true",
                }
                if page_token:
                    params["pageToken"] = page_token

                response = client.get(settings.ctgov_api, params=params)
                response.raise_for_status()
                body = response.json()

                page = body.get("studies") or []
                if not page:
                    break
                studies.extend(page)
                logger.info(f"Fetched {len(page)} studies (running total: {len(studies)})")

                page_token = body.get("nextPageToken")
                if not page_token:
                    break

        if not studies:
            raise RuntimeError("ClinicalTrials.gov returned zero studies for this query.")

        os.makedirs(settings.landing_zone, exist_ok=True)
        with open(self.raw_cache, "w", encoding="utf-8") as handle:
            json.dump(studies, handle, indent=2)
        return studies

    # -- tier 2 ---------------------------------------------------------------
    def _fetch_cached(self) -> list[dict]:
        with open(self.raw_cache, encoding="utf-8") as handle:
            return json.load(handle)

    # -- tier 3 ---------------------------------------------------------------
    def _fetch_synthetic(self) -> list[dict]:
        """Schema-faithful stand-in records, prefixed SYNTH so they are never
        mistaken for real registry data during a review."""
        blueprint = [
            ("Metformin Versus Placebo for Glycaemic Control in Adults With Type 2 Diabetes",
             "RECRUITING", "PHASE3", "18 Years", "75 Years", 420,
             "Inclusion Criteria:\n* Adults aged 18 years or older\n* Documented HbA1c between 7.0% and 10.0%\n"
             "* Body mass index between 25 and 40 kg/m2\n\nExclusion Criteria:\n* Type 1 diabetes mellitus\n"
             "* Estimated glomerular filtration rate below 45 mL/min\n* Pregnancy or breastfeeding"),
            ("Continuous Glucose Monitoring to Reduce Hypoglycaemia in Older Adults",
             "RECRUITING", "PHASE2", "60 Years", "90 Years", 180,
             "Inclusion Criteria:\n* Participants must be at least 65 years of age\n"
             "* Insulin-treated type 2 diabetes for at least one year\n\nExclusion Criteria:\n"
             "* Severe cognitive impairment\n* Active malignancy"),
            ("Structured Exercise and Dietary Counselling in Newly Diagnosed Type 2 Diabetes",
             "ACTIVE_NOT_RECRUITING", "NA", "21 Years", "65 Years", 240,
             "Inclusion Criteria:\n* Diagnosis of type 2 diabetes within the previous 12 months\n"
             "* Able to walk unaided for 10 minutes\n\nExclusion Criteria:\n* Unstable angina\n"
             "* Prior bariatric surgery"),
            ("SGLT2 Inhibitor Effect on Renal Outcomes in Diabetic Kidney Disease",
             "RECRUITING", "PHASE3", "40 Years", None, 900,
             "Inclusion Criteria:\n* Type 2 diabetes with albuminuria\n* eGFR 25 to 75 mL/min/1.73m2\n\n"
             "Exclusion Criteria:\n* Dialysis or prior kidney transplant\n* Recurrent genital infection"),
            ("Telehealth-Delivered Self-Management Support for Rural Diabetes Patients",
             "COMPLETED", "NA", "18 Years", "80 Years", 310,
             "Inclusion Criteria:\n* Residence in a designated rural area\n* Reliable telephone access\n\n"
             "Exclusion Criteria:\n* Enrolled in another behavioural trial"),
            ("Fixed-Ratio Basal Insulin Combination in Insulin-Naive Participants",
             "RECRUITING", "PHASE3", "18 Years", "80 Years", 550,
             "Inclusion Criteria:\n* HbA1c between 7.5% and 11.0%\n* No prior insulin therapy\n\n"
             "Exclusion Criteria:\n* History of diabetic ketoacidosis\n* Severe hepatic impairment"),
        ]

        studies = []
        for index, (title, status, phase, min_age, max_age, enrollment, criteria) in enumerate(blueprint, start=1):
            studies.append({
                "protocolSection": {
                    "identificationModule": {"nctId": f"SYNTH{index:08d}", "briefTitle": title},
                    "statusModule": {
                        "overallStatus": status,
                        "lastUpdatePostDateStruct": {"date": "2026-01-15"},
                    },
                    "descriptionModule": {
                        "briefSummary": (
                            f"{title}. This study evaluates the intervention in participants living with "
                            "type 2 diabetes mellitus, measuring change in glycated haemoglobin and "
                            "treatment-related adverse events over the study period."
                        )
                    },
                    "conditionsModule": {"conditions": ["Type 2 Diabetes Mellitus", "Metabolic Disease"]},
                    "designModule": {
                        "studyType": "INTERVENTIONAL",
                        "phases": [phase],
                        "enrollmentInfo": {"count": enrollment},
                    },
                    "eligibilityModule": {
                        "eligibilityCriteria": criteria,
                        "minimumAge": min_age,
                        "maximumAge": max_age,
                        "sex": "ALL",
                        "healthyVolunteers": False,
                    },
                    "armsInterventionsModule": {
                        "interventions": [{"type": "DRUG", "name": title.split()[0]}]
                    },
                    "contactsLocationsModule": {
                        "locations": [{"city": "Riyadh", "country": "Saudi Arabia"}]
                    },
                    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Synthetic Research Institute"}},
                },
            })
        return studies

    async def run(self, allow_live: bool = True) -> int:
        """Executes the ingestion tier ladder and streams each trial as an event."""
        studies: list[dict] = []

        if allow_live:
            try:
                logger.info(f"Calling ClinicalTrials.gov API v2 for condition '{settings.ctgov_condition}'...")
                studies = self._fetch_live()
                self.mode = "LIVE"
            except Exception as exc:
                logger.warning(f"Live registry call failed ({type(exc).__name__}: {exc}). Falling back.")

        if not studies and os.path.exists(self.raw_cache):
            studies = self._fetch_cached()
            self.mode = "REPLAY"
            logger.info(f"Replaying {len(studies)} cached raw studies from the landing zone.")

        if not studies:
            studies = self._fetch_synthetic()
            self.mode = "SYNTHETIC"
            logger.warning("Using SYNTHETIC fallback records - NCT IDs are prefixed 'SYNTH'.")

        print(f"📥 [PRODUCER: ClinicalTrials.gov] Ingestion mode = {self.mode} | {len(studies)} studies")
        for study in studies:
            record = flatten_ctgov_study(study)
            if record.get("nct_id"):
                await self.broker.publish("raw_clinical_trials", record, source="CTGOV_API_V2")

        print(f"📡 [BROKER] Published {self.broker.published_counts['raw_clinical_trials']} events "
              f"to topic 'raw_clinical_trials'")
        return len(studies)


class InternalEHRProducer:
    """Second ingestion source: the hospital's internal patient records.

    These records intentionally carry PHI (names, MRNs, dates of birth, phone
    numbers, e-mail addresses). They exist so the governance layer has real
    protected data to detect, pseudonymise, and block - proving the HIPAA
    control rather than merely describing it. The records themselves are
    entirely synthetic; no real person is represented.
    """

    PATIENTS = [
        {
            "patient_name": "Sarah Al-Mutairi", "medical_record_number": "MRN-4471902",
            "date_of_birth": "1958-03-14", "phone": "+966-55-114-8820",
            "email": "s.almutairi@example-hospital.sa", "address": "8123 King Fahd Road, Riyadh",
            "age_years": 68, "diagnosis": "Type 2 diabetes mellitus with diabetic nephropathy",
            "hba1c": 8.4, "comorbidities": "hypertension; stage 3 chronic kidney disease",
        },
        {
            "patient_name": "Omar Haddad", "medical_record_number": "MRN-3390114",
            "date_of_birth": "1979-11-02", "phone": "+966-50-772-3391",
            "email": "o.haddad@example-hospital.sa", "address": "44 Prince Sultan Street, Jeddah",
            "age_years": 46, "diagnosis": "Newly diagnosed type 2 diabetes mellitus",
            "hba1c": 7.2, "comorbidities": "obesity",
        },
        {
            "patient_name": "Layla Ibrahim", "medical_record_number": "MRN-9920township",
            "date_of_birth": "2009-06-21", "phone": "+966-53-441-0092",
            "email": "not-an-email", "address": "17 Olaya District, Riyadh",
            "age_years": 16, "diagnosis": "Type 1 diabetes mellitus",
            "hba1c": 9.1, "comorbidities": "",
        },
    ]

    def __init__(self, broker: MockEventBroker) -> None:
        self.broker = broker

    async def run(self) -> int:
        print(f"📥 [PRODUCER: Internal EHR] Streaming {len(self.PATIENTS)} patient records "
              f"(SYNTHETIC, PHI-bearing by design)")
        for patient in self.PATIENTS:
            await self.broker.publish("raw_patient_records", dict(patient), source="INTERNAL_EHR")
        return len(self.PATIENTS)


# =====================================================================
# 2. ARCHITECTURAL COMPONENT: AUTOMATED DATA QUALITY GATE
# =====================================================================

class DataQualityEngine:
    """Executes automated quality assertions across the six quality dimensions.

    The engine grades every record individually AND produces a batch verdict.
    Individual failures are quarantined; a batch whose failure ratio exceeds the
    configured threshold halts the pipeline entirely, exactly as a production
    gatekeeper would when it suspects an upstream system is broken.
    """

    REQUIRED_FIELDS = ["nct_id", "brief_title", "brief_summary", "overall_status"]
    VALID_STATUSES = {
        "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED",
        "ENROLLING_BY_INVITATION", "SUSPENDED", "TERMINATED", "WITHDRAWN",
        "UNKNOWN", "AVAILABLE", "NO_LONGER_AVAILABLE", "APPROVED_FOR_MARKETING",
        "TEMPORARILY_NOT_AVAILABLE", "WITHHELD",
    }
    NCT_PATTERN = re.compile(r"^(NCT\d{8}|SYNTH\d{8})$")

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe.copy()
        self.results: dict[str, Any] = {
            "timestamp": utcnow(),
            "records_evaluated": int(len(dataframe)),
            "metrics": {},
            "passed_all_gates": True,
        }

    def run_all_checks(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Returns (clean_frame, rejected_frame, quality_report)."""
        logger.info("Initiating automated data quality scan across 6 dimensions...")
        df = self.df
        failures = pd.Series([""] * len(df), index=df.index)

        def record_metric(name: str, mask: pd.Series, reason: str) -> None:
            """Registers one dimension and tags every offending row with a reason."""
            count = int(mask.sum())
            self.results["metrics"][name] = {
                "violations": count,
                "status": "PASSED" if count == 0 else "FAILED",
            }
            failures.loc[mask] = failures.loc[mask].str.cat([reason] * count, sep="|").str.strip("|")

        # 1. COMPLETENESS - no critical identifier may be missing.
        missing = df[self.REQUIRED_FIELDS].isnull().any(axis=1) | (
            df["brief_title"].fillna("").str.strip() == ""
        )
        record_metric("completeness_required_fields", missing, "MISSING_REQUIRED_FIELD")

        # 2. VALIDITY - identifiers must match the registry format.
        invalid_id = ~df["nct_id"].fillna("").str.match(self.NCT_PATTERN)
        record_metric("validity_nct_id_format", invalid_id, "INVALID_NCT_ID")

        # 3. UNIQUENESS - the registry must never return the same trial twice.
        duplicated = df.duplicated(subset=["nct_id"], keep="first")
        record_metric("uniqueness_nct_id", duplicated, "DUPLICATE_NCT_ID")

        # 4. ACCURACY - ages and enrollment must sit inside physically possible ranges.
        ages = pd.to_numeric(df["min_age_years"], errors="coerce")
        max_ages = pd.to_numeric(df["max_age_years"], errors="coerce")
        enrollment = pd.to_numeric(df["enrollment"], errors="coerce")
        impossible = (
            (ages < 0) | (ages > 120)
            | (max_ages < 0) | (max_ages > 120)
            | ((ages.notna()) & (max_ages.notna()) & (ages > max_ages))
            | (enrollment < 0)
        ).fillna(False)
        record_metric("accuracy_numeric_ranges", impossible, "IMPOSSIBLE_NUMERIC_VALUE")

        # 5. CONSISTENCY - status vocabulary must match the registry's enumeration.
        inconsistent = ~df["overall_status"].fillna("").str.upper().isin(self.VALID_STATUSES)
        record_metric("consistency_status_vocabulary", inconsistent, "UNKNOWN_STATUS_VALUE")

        # 6. TIMELINESS - a trial not updated in over five years is stale evidence.
        cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=365 * 5)
        updated = pd.to_datetime(df["last_update"], errors="coerce", utc=True)
        stale = (updated.notna() & (updated < cutoff))
        record_metric("timeliness_last_update", stale, "STALE_RECORD")

        for check in self.results["metrics"].values():
            if check["status"] == "FAILED":
                self.results["passed_all_gates"] = False

        df = df.assign(quality_failures=failures)
        clean = df[df["quality_failures"] == ""].drop(columns=["quality_failures"])
        rejected = df[df["quality_failures"] != ""]

        ratio = len(rejected) / len(df) if len(df) else 0.0
        self.results["rejected_records"] = int(len(rejected))
        self.results["clean_records"] = int(len(clean))
        self.results["failure_ratio"] = round(ratio, 4)
        self.results["halt_pipeline"] = ratio > settings.quality_halt_threshold

        return clean, rejected, self.results


def quarantine_batch(rejected: pd.DataFrame, label: str) -> str | None:
    """Routes corrupt records to the quarantine zone for diagnostics."""
    if rejected.empty:
        return None
    os.makedirs(settings.quarantine_zone, exist_ok=True)
    path = os.path.join(
        settings.quarantine_zone,
        f"corrupt_{label}_{int(dt.datetime.now(dt.UTC).timestamp())}.csv",
    )
    rejected.to_csv(path, index=False)
    logger.warning(f"ACTION REQUIRED: {len(rejected)} corrupt records isolated to {path}")
    return path


# =====================================================================
# 3. ARCHITECTURAL COMPONENT: GOVERNANCE, PHI PROTECTION, LINEAGE & AUDIT
# =====================================================================

class PHIGuard:
    """Detects and removes HIPAA Protected Health Information from any text.

    Covers the identifier families that actually appear in clinical free text:
    names, medical record numbers, national IDs, dates of birth, telephone
    numbers, e-mail addresses, and street addresses. Detected identities are
    replaced by a deterministic HMAC-derived pseudonym so the same patient maps
    to the same token across runs without the token ever being reversible.
    """

    PATTERNS: dict[str, re.Pattern] = {
        "MRN": re.compile(r"\bMRN[-\s]?[A-Z0-9]{4,12}\b", re.IGNORECASE),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "NATIONAL_ID": re.compile(r"\b[12]\d{9}\b"),
        "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
        "PHONE": re.compile(r"(?:\+\d{1,3}[-\s]?)?(?:\d{2,4}[-\s]?){2,4}\d{2,4}\b"),
        "DATE_OF_BIRTH": re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),
        "STREET_ADDRESS": re.compile(
            r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
            r"(?:Street|St|Road|Rd|Avenue|Ave|District|Boulevard|Blvd)\b"
        ),
    }

    # Salt would live in a KMS in production; kept in-file for reproducibility.
    PSEUDONYM_SALT = b"sdaia-capstone-deid-salt"

    @classmethod
    def pseudonymise(cls, value: str) -> str:
        digest = hashlib.sha256(cls.PSEUDONYM_SALT + value.encode("utf-8")).hexdigest()
        return f"PATIENT_{digest[:10].upper()}"

    @classmethod
    def scan(cls, text: str) -> list[dict[str, str]]:
        """Returns every PHI hit found in the text, without modifying it."""
        if not isinstance(text, str) or not text:
            return []
        findings = []
        for kind, pattern in cls.PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({"type": kind, "value": match.group(0)})
        return findings

    @classmethod
    def redact(cls, text: str) -> tuple[str, list[dict[str, str]]]:
        """Returns (de-identified text, findings)."""
        if not isinstance(text, str) or not text:
            return text, []
        findings = cls.scan(text)
        clean = text
        for kind, pattern in cls.PATTERNS.items():
            clean = pattern.sub(f"[REDACTED_{kind}]", clean)
        return clean, findings

    @classmethod
    def deidentify_patient(cls, record: dict) -> tuple[dict, list[dict[str, str]]]:
        """Splits a patient record into a safe clinical profile and a PHI report."""
        findings: list[dict[str, str]] = []
        safe: dict[str, Any] = {}

        for field, value in record.items():
            classification = DATA_CLASSIFICATION.get(field, "CONFIDENTIAL")
            if classification == "RESTRICTED":
                findings.append({"type": f"FIELD:{field}", "value": str(value)})
                continue
            if isinstance(value, str):
                cleaned, hits = cls.redact(value)
                findings.extend(hits)
                safe[field] = cleaned
            else:
                safe[field] = value

        safe["patient_pseudonym"] = cls.pseudonymise(
            str(record.get("medical_record_number", "")) or str(record.get("patient_name", ""))
        )
        return safe, findings


class LineageTracker:
    """Records the end-to-end genealogy of every dataset the platform touches.

    Each pipeline operation appends a node describing what went in, what came
    out, and how many records survived. Tracing a record backwards therefore
    answers the executive's question: where did this number come from?
    """

    def __init__(self) -> None:
        self.path = os.path.join(settings.governance_dir, "lineage.jsonl")
        self.nodes: list[dict[str, Any]] = []

    def record(self, operation: str, inputs: list[str], outputs: list[str],
               record_count: int, details: dict | None = None) -> str:
        node_id = f"LIN_{len(self.nodes):04d}_{operation.upper()}"
        node = {
            "lineage_id": node_id,
            "timestamp": utcnow(),
            "operation": operation,
            "inputs": inputs,
            "outputs": outputs,
            "record_count": record_count,
            "details": details or {},
        }
        self.nodes.append(node)
        os.makedirs(settings.governance_dir, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(node) + "\n")
        return node_id

    def load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def render_graph(self) -> str:
        """Renders the lineage chain as an indented dependency map."""
        nodes = self.nodes or self.load()
        if not nodes:
            return "(no lineage recorded yet - run the pipeline first)"
        lines = []
        for node in nodes:
            lines.append(
                f"  {node['lineage_id']}\n"
                f"     operation : {node['operation']}\n"
                f"     inputs    : {', '.join(node['inputs']) or '-'}\n"
                f"     outputs   : {', '.join(node['outputs']) or '-'}\n"
                f"     records   : {node['record_count']}\n"
                f"     at        : {node['timestamp']}"
            )
        return "\n  |\n  v\n".join(lines)


class AuditTrail:
    """Append-only audit log: every query against protected data is recorded.

    Idea 2 in the Day 5 brief requires 'an audit trail for every query'. This is
    the enforcement point: no retrieval happens without a corresponding entry.
    """

    def __init__(self) -> None:
        self.path = os.path.join(settings.governance_dir, "audit_trail.jsonl")

    def log(self, actor: str, role: str, action: str, payload: dict) -> str:
        entry_id = f"AUD_{hashlib.sha256((actor + utcnow() + action).encode()).hexdigest()[:12].upper()}"
        entry = {
            "audit_id": entry_id,
            "timestamp": utcnow(),
            "actor": actor,
            "role": role,
            "action": action,
            **payload,
        }
        os.makedirs(settings.governance_dir, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        return entry_id

    def tail(self, limit: int = 10) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as handle:
            entries = [json.loads(line) for line in handle if line.strip()]
        return entries[-limit:]


class GovernanceEngine:
    """Enforces classification, attribute-based access control, and PHI policy."""

    def __init__(self, lineage: LineageTracker, audit: AuditTrail) -> None:
        self.lineage = lineage
        self.audit = audit

    @staticmethod
    def classify(field: str) -> str:
        return DATA_CLASSIFICATION.get(field, "CONFIDENTIAL")

    @staticmethod
    def can_access(role: str, classification: str) -> bool:
        return classification in ROLE_POLICIES.get(role, set())

    def apply_access_policy(self, record: dict, role: str) -> dict:
        """Masks every field the requesting role is not cleared to read."""
        governed = {}
        for field, value in record.items():
            classification = self.classify(field)
            if self.can_access(role, classification):
                governed[field] = value
            else:
                governed[field] = f"***MASKED[{classification}]***"
        return governed

    def assert_vector_db_safe(self, text: str, chunk_id: str) -> None:
        """Hard gate: refuses to index any chunk that still carries PHI.

        This is the literal implementation of the Day 5 requirement
        'Apply HIPAA governance (no PHI in vector DB)'.
        """
        findings = PHIGuard.scan(text)
        # Registry text legitimately contains numeric ranges that the broad
        # phone pattern can catch; only identity-bearing families are fatal.
        fatal = [f for f in findings if f["type"] in {"MRN", "SSN", "EMAIL", "NATIONAL_ID", "STREET_ADDRESS"}]
        if fatal:
            self.audit.log(
                actor="pipeline", role="data_engineer", action="VECTOR_INSERT_BLOCKED",
                payload={"chunk_id": chunk_id, "phi_types": sorted({f['type'] for f in fatal})},
            )
            raise PermissionError(
                f"PHI detected in chunk {chunk_id} ({sorted({f['type'] for f in fatal})}). "
                "Refusing to write it to the vector database."
            )


# =====================================================================
# 4. ARCHITECTURAL COMPONENT: DELTA-STYLE LAKEHOUSE STORAGE LAYER
# =====================================================================

class SchemaEnforcementError(Exception):
    """Raised when an incoming batch violates the registered table blueprint."""


class DeltaStyleLakehouse:
    """A Delta Lake-style transactional storage layer over local object storage.

    Implements the three guardrails taught on Day 1 without requiring a JVM:

        ACID transactions   - the parquet file is written first and the commit
                              entry appended only on success, so a crash mid-write
                              leaves an orphan file that no reader can ever see
                              (all-or-nothing atomicity).
        Schema enforcement  - the blueprint is pinned at version 0; a batch with
                              extra, missing, or retyped columns is rejected.
        Time travel         - every commit is an immutable numbered entry in
                              _delta_log, so any past version can be re-read.
    """

    def __init__(self, table_path: str) -> None:
        self.table_path = table_path
        self.log_dir = os.path.join(table_path, "_delta_log")
        self.data_dir = os.path.join(table_path, "data")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    # -- transaction log ------------------------------------------------------
    def _commits(self) -> list[dict]:
        commits = []
        for name in sorted(os.listdir(self.log_dir)):
            if name.endswith(".json"):
                with open(os.path.join(self.log_dir, name), encoding="utf-8") as handle:
                    commits.append(json.load(handle))
        return commits

    def _next_version(self) -> int:
        return len(self._commits())

    def registered_schema(self) -> dict[str, str] | None:
        commits = self._commits()
        return commits[0]["schema"] if commits else None

    # -- writes ---------------------------------------------------------------
    def write(self, df: pd.DataFrame, mode: str = "append", operation: str = "WRITE") -> int:
        """Commits a dataframe to the table under ACID + schema-enforcement rules."""
        if mode not in {"append", "overwrite"}:
            raise ValueError("mode must be 'append' or 'overwrite'")

        incoming_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

        if mode == "overwrite":
            shutil.rmtree(self.log_dir, ignore_errors=True)
            shutil.rmtree(self.data_dir, ignore_errors=True)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)

        blueprint = self.registered_schema()
        if blueprint is not None and set(blueprint) != set(incoming_schema):
            extra = sorted(set(incoming_schema) - set(blueprint))
            missing = sorted(set(blueprint) - set(incoming_schema))
            raise SchemaEnforcementError(
                f"Schema mismatch. Unexpected columns: {extra or 'none'}; "
                f"missing columns: {missing or 'none'}."
            )

        version = self._next_version()
        data_file = os.path.join(self.data_dir, f"part-{version:05d}.parquet")

        # --- ACID: write data first, commit second -----------------------------
        try:
            df.to_parquet(data_file, index=False)
        except Exception:
            if os.path.exists(data_file):
                os.remove(data_file)
            raise

        commit = {
            "version": version,
            "timestamp": utcnow(),
            "operation": operation,
            "operationParameters": {"mode": mode, "numRecords": int(len(df))},
            "schema": blueprint or incoming_schema,
            "files": [os.path.relpath(data_file, self.table_path)],
        }
        with open(os.path.join(self.log_dir, f"{version:020d}.json"), "w", encoding="utf-8") as handle:
            json.dump(commit, handle, indent=2)

        logger.success(f"Delta commit v{version} | {operation} | {len(df)} records")
        return version

    # -- reads ----------------------------------------------------------------
    def read(self, version: int | None = None) -> pd.DataFrame:
        """Reads the table, optionally travelling back to an earlier version."""
        commits = self._commits()
        if not commits:
            return pd.DataFrame()
        if version is None:
            version = commits[-1]["version"]
        if version >= len(commits):
            raise ValueError(f"Version {version} does not exist (latest is {len(commits) - 1}).")

        frames = []
        for commit in commits[: version + 1]:
            for relative in commit["files"]:
                frames.append(pd.read_parquet(os.path.join(self.table_path, relative)))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def history(self) -> pd.DataFrame:
        """Returns the transaction log ledger, mirroring DeltaTable.history()."""
        return pd.DataFrame([
            {
                "version": c["version"],
                "timestamp": c["timestamp"],
                "operation": c["operation"],
                "numRecords": c["operationParameters"]["numRecords"],
                "mode": c["operationParameters"]["mode"],
            }
            for c in self._commits()
        ])


# =====================================================================
# 5. ARCHITECTURAL COMPONENT: CHUNKING & EMBEDDING LAYER
# =====================================================================

class RecursiveChunker:
    """Recursive/hierarchical chunking (Day 3, Strategy 3) with overlap.

    Splits on the largest natural boundary first (blank line), then single
    newline, then sentence, then word - so a chunk is a unit of meaning rather
    than an arbitrary slice of bytes. Overlap is applied so a fact split across
    a boundary survives in at least one chunk.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, chunk_size: int, overlap: int) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split(self, text: str, depth: int = 0) -> list[str]:
        if len(text) <= self.chunk_size or depth >= len(self.SEPARATORS):
            return [text]

        separator = self.SEPARATORS[depth]
        pieces, buffer = [], ""
        for part in text.split(separator):
            candidate = f"{buffer}{separator}{part}" if buffer else part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    pieces.append(buffer)
                buffer = part
        if buffer:
            pieces.append(buffer)

        resolved: list[str] = []
        for piece in pieces:
            resolved.extend(self._split(piece, depth + 1) if len(piece) > self.chunk_size else [piece])
        return resolved

    def _apply_overlap(self, pieces: list[str]) -> list[str]:
        if self.overlap <= 0 or len(pieces) < 2:
            return pieces
        overlapped = [pieces[0]]
        for previous, current in zip(pieces, pieces[1:]):
            tail = previous[-self.overlap:]
            overlapped.append(f"{tail} {current}".strip())
        return overlapped

    def chunk_record(self, record: dict) -> list[dict]:
        """Turns one trial record into retrievable, self-contained chunks.

        Each chunk is prefixed with the trial identity so it stays meaningful on
        its own once the retriever pulls it out of context.
        """
        nct_id = record["nct_id"]
        header = (
            f"Trial {nct_id} - {record.get('brief_title', '')}\n"
            f"Status: {record.get('overall_status')} | Phase: {record.get('phase') or 'N/A'} | "
            f"Conditions: {record.get('conditions')}\n"
        )

        # Eligibility chunks carry the structured age/sex facts inline. Without
        # them the chunk is not self-contained: a question about age limits would
        # never retrieve the passage that actually answers it, because the numbers
        # live in a metadata column rather than in the free text.
        min_age, max_age = record.get("min_age_years"), record.get("max_age_years")
        eligibility_facts = (
            f"Eligible ages: {min_age if min_age is not None else 'unspecified'} to "
            f"{max_age if max_age is not None else 'no upper limit'} years. "
            f"Sex eligible: {record.get('sex') or 'unspecified'}. "
            f"Enrollment target: {record.get('enrollment') or 'unspecified'}.\n"
        )

        sections = {
            "summary": record.get("brief_summary") or "",
            "eligibility": eligibility_facts + str(record.get("eligibility_criteria") or ""),
            "interventions": record.get("interventions") or "",
        }

        chunks: list[dict] = []
        for section_name, body in sections.items():
            if not str(body).strip():
                continue
            pieces = self._apply_overlap(self._split(str(body)))
            for index, piece in enumerate(pieces):
                chunks.append({
                    "chunk_id": f"{nct_id}::{section_name}::{index}",
                    "nct_id": nct_id,
                    "section": section_name,
                    "chunk_index": index,
                    "text": f"{header}[{section_name.upper()}] {piece}".strip(),
                })
        return chunks


class HashingEmbedder:
    """Deterministic offline fallback embedder (hashed bag-of-tokens, 384-dim).

    Used only when the sentence-transformers model cannot be loaded (no network
    or no local cache). It is a genuine vector space - sublinear term frequency,
    hashed into fixed dimensions, L2-normalised - so cosine similarity remains
    meaningful and the whole pipeline stays demonstrable offline.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.backend = "hashing-fallback"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in self._tokenize(text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
            for bucket, count in counts.items():
                matrix[row, bucket] = 1.0 + math.log(count)   # sublinear term frequency
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix


class EmbeddingModel:
    """Generates text embeddings, preferring the real model and degrading safely."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.backend = "sentence-transformers"
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding backend: {model_name} ({self.dimensions}-dim)")
        except Exception as exc:
            logger.warning(f"Could not load '{model_name}' ({type(exc).__name__}). Using hashing fallback.")
            self.model = HashingEmbedder()
            self.backend = self.model.backend
            self.dimensions = self.model.dimensions

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encodes in batches - the resource-management concern that compute
        orchestration exists to solve."""
        vectors = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()


# =====================================================================
# 6. ARCHITECTURAL COMPONENT: VECTOR DATABASE LAYER
# =====================================================================

class VectorStore:
    """Thin wrapper around a persistent Chroma collection indexed with HNSW.

    HNSW is chosen deliberately: Day 3 recommends it for datasets that need
    >99% recall with ample RAM, which is exactly a clinical-evidence corpus of
    this size. Cosine is the space, since document length must not dominate.
    """

    COLLECTION = "clinical_trials"

    def __init__(self, persist_dir: str) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _sanitize(metadata: dict) -> dict:
        """Chroma metadata accepts only scalars - nulls and lists are coerced."""
        clean = {}
        for key, value in metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.COLLECTION)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[dict], embeddings: np.ndarray, batch_size: int = 200) -> int:
        for start in range(0, len(chunks), batch_size):
            window = chunks[start:start + batch_size]
            self.collection.add(
                ids=[c["chunk_id"] for c in window],
                documents=[c["text"] for c in window],
                embeddings=embeddings[start:start + batch_size].tolist(),
                metadatas=[self._sanitize({k: v for k, v in c.items() if k not in ("text",)}) for c in window],
            )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(self.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, doc, meta, dist in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            hits.append({
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                "dense_score": 1.0 - float(dist),   # cosine distance -> similarity
            })
        return hits

    def all_documents(self) -> tuple[list[str], list[str], list[dict]]:
        """Pulls the full corpus back out, used to build the sparse BM25 index."""
        payload = self.collection.get(include=["documents", "metadatas"])
        return payload.get("ids", []), payload.get("documents", []), payload.get("metadatas", [])


# =====================================================================
# 7. ARCHITECTURAL COMPONENT: ADVANCED RAG RETRIEVAL
# =====================================================================

class LexicalReranker:
    """Offline fallback reranker: IDF-weighted token overlap between query and chunk.

    It is not a cross-encoder, but it performs the same architectural job -
    a second, more expensive scoring pass over the stage-one candidates.
    """

    def __init__(self, corpus: list[str]) -> None:
        self.backend = "lexical-fallback"
        documents = [set(re.findall(r"[a-z0-9]+", d.lower())) for d in corpus] or [set()]
        total = len(documents)
        frequency: dict[str, int] = {}
        for tokens in documents:
            for token in tokens:
                frequency[token] = frequency.get(token, 0) + 1
        self.idf = {t: math.log(1 + total / (1 + c)) for t, c in frequency.items()}

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for query, document in pairs:
            q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
            d_tokens = set(re.findall(r"[a-z0-9]+", document.lower()))
            shared = q_tokens & d_tokens
            numerator = sum(self.idf.get(t, 1.0) for t in shared)
            denominator = sum(self.idf.get(t, 1.0) for t in q_tokens) or 1.0
            scores.append(numerator / denominator)
        return scores


class CrossEncoderReranker:
    """Stage-two reranker that deeply rescores stage-one candidates."""

    def __init__(self, model_name: str, corpus: list[str]) -> None:
        self.backend = "cross-encoder"
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            logger.info(f"Reranker backend: {model_name}")
        except Exception as exc:
            logger.warning(f"Could not load reranker '{model_name}' ({type(exc).__name__}). Using lexical fallback.")
            self.model = LexicalReranker(corpus)
            self.backend = self.model.backend

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c["text"]) for c in candidates])
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]


class HybridRetriever:
    """Dense + sparse retrieval fused with Reciprocal Rank Fusion (Day 3).

    Pure vector search is a classic prototype mistake: it is excellent on
    abstract concepts and weak on exact tokens such as 'NCT04223752', 'HbA1c',
    or 'SGLT2'. BM25 covers exactly that blind spot; RRF merges the two ranked
    lists without needing their scores to be on a comparable scale.
    """

    RRF_K = 60

    def __init__(self, store: VectorStore, embedder: EmbeddingModel) -> None:
        from rank_bm25 import BM25Okapi

        self.store = store
        self.embedder = embedder

        ids, documents, metadatas = store.all_documents()
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.bm25 = BM25Okapi([re.findall(r"[a-z0-9]+", d.lower()) for d in documents]) if documents else None

    def _dense(self, query: str, top_k: int) -> list[dict]:
        return self.store.query(self.embedder.encode_one(query), top_k)

    def _sparse(self, query: str, top_k: int) -> list[dict]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(re.findall(r"[a-z0-9]+", query.lower()))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "chunk_id": self.ids[i],
                "text": self.documents[i],
                "metadata": self.metadatas[i],
                "sparse_score": float(scores[i]),
            }
            for i in order if scores[i] > 0
        ]

    def retrieve(self, query: str) -> list[dict]:
        """Returns stage-one fused candidates, each carrying its provenance."""
        dense = self._dense(query, settings.dense_top_k)
        sparse = self._sparse(query, settings.sparse_top_k)

        fused: dict[str, dict] = {}
        for rank, hit in enumerate(dense):
            entry = fused.setdefault(hit["chunk_id"], dict(hit))
            entry["rrf_score"] = entry.get("rrf_score", 0.0) + 1.0 / (self.RRF_K + rank + 1)
            entry["retrieved_by"] = "dense"

        for rank, hit in enumerate(sparse):
            entry = fused.setdefault(hit["chunk_id"], dict(hit))
            entry["rrf_score"] = entry.get("rrf_score", 0.0) + 1.0 / (self.RRF_K + rank + 1)
            entry["sparse_score"] = hit["sparse_score"]
            entry["retrieved_by"] = "hybrid" if entry.get("retrieved_by") == "dense" else "sparse"

        return sorted(fused.values(), key=lambda c: c["rrf_score"], reverse=True)


class ContradictionDetector:
    """Flags trials whose structured metadata disagrees with their own free text.

    This is a real defect class in the registry: the eligibility module may
    declare a minimum age of 18 while the criteria text says 'at least 65 years
    of age'. A researcher acting on the wrong one screens the wrong patients, so
    the platform surfaces the conflict instead of silently picking a side.
    """

    AGE_IN_TEXT = re.compile(
        r"(?:at least|aged|age of|older than|minimum age of|>=?\s*)\s*(\d{1,3})\s*(?:years|yrs|year)",
        re.IGNORECASE,
    )

    @classmethod
    def check(cls, candidates: list[dict]) -> list[dict]:
        alerts = []
        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            declared = metadata.get("min_age_years")
            try:
                declared = float(declared)
            except (TypeError, ValueError):
                continue

            for match in cls.AGE_IN_TEXT.finditer(candidate.get("text", "")):
                stated = float(match.group(1))
                if abs(stated - declared) >= 3:
                    alerts.append({
                        "type": "ELIGIBILITY_AGE_CONFLICT",
                        "nct_id": metadata.get("nct_id"),
                        "chunk_id": candidate.get("chunk_id"),
                        "structured_min_age": declared,
                        "stated_in_text": stated,
                        "message": (
                            f"Trial {metadata.get('nct_id')} declares a minimum age of {declared:.0f} "
                            f"in its structured eligibility field, but its criteria text states "
                            f"{stated:.0f} years. Verify against the registry before screening."
                        ),
                    })
                    break
        return alerts


# =====================================================================
# 8. ARCHITECTURAL COMPONENT: GENERATION, EVIDENCE TRAIL & RAG TRIAD
# =====================================================================

SYSTEM_PROMPT = """You are a clinical research assistant supporting trial matching.

Rules you must obey:
1. Answer ONLY from the numbered context passages provided. Never use outside knowledge.
2. Cite the source of every claim inline using its number, for example [1] or [2].
3. If the context does not contain the answer, say so plainly. Do not speculate.
4. Never invent NCT identifiers, eligibility thresholds, or enrollment figures.
5. Be concise and clinical in tone.
"""

ANSWER_PROMPT = """{system}

Context passages:
{context}

Researcher question: {question}

Answer:"""


class LLMClient:
    """OpenAI-compatible chat client pointed at OpenRouter."""

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self.backend = "openrouter"
        self.client = None
        if api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            except Exception as exc:
                logger.warning(f"Could not initialise OpenRouter client ({exc}).")
        if self.client is None:
            self.backend = "extractive-fallback"
            logger.warning("No LLM configured - answers will be extractive (grounded, not generated).")

    def generate(self, prompt: str, context_blocks: list[str]) -> str:
        if self.client is None:
            return self._extractive(context_blocks)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            return content or self._extractive(context_blocks)
        except Exception as exc:
            logger.warning(f"LLM call failed ({type(exc).__name__}: {exc}). Falling back to extractive answer.")
            return self._extractive(context_blocks)

    @staticmethod
    def _extractive(context_blocks: list[str]) -> str:
        """Deterministic grounded answer used when no LLM is available.

        It quotes the retrieved evidence verbatim rather than generating prose,
        so the answer remains 100% grounded and the RAG Triad stays measurable.
        """
        lines = ["[EXTRACTIVE MODE - no generative model configured; evidence quoted verbatim]", ""]
        for index, block in enumerate(context_blocks, start=1):
            snippet = " ".join(block.split())[:400]
            lines.append(f"{snippet} [{index}]")
            lines.append("")
        return "\n".join(lines).strip()


class RAGTriadEvaluator:
    """Scores every answer on the three production RAG metrics (Day 3).

    Context Relevance - did the retriever pull the right passages?
    Groundedness      - is the answer actually supported by those passages?
    Answer Relevance  - does the answer address the question that was asked?
    """

    STOPWORDS = {
        "the", "a", "an", "of", "for", "in", "on", "to", "and", "or", "is", "are",
        "with", "that", "this", "it", "as", "be", "by", "at", "from", "was", "were",
    }

    def __init__(self, embedder: EmbeddingModel) -> None:
        self.embedder = embedder

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in cls.STOPWORDS and len(t) > 2}

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))

    def evaluate(self, query: str, answer: str, contexts: list[str]) -> dict[str, float]:
        if not contexts:
            return {"context_relevance": 0.0, "groundedness": 0.0, "answer_relevance": 0.0}

        vectors = self.embedder.encode([query, answer] + contexts)
        query_vec, answer_vec, context_vecs = vectors[0], vectors[1], vectors[2:]

        context_relevance = float(np.mean([self._cosine(query_vec, c) for c in context_vecs]))

        answer_tokens = self._content_tokens(answer)
        context_tokens = self._content_tokens(" ".join(contexts))
        groundedness = (
            len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
        )

        answer_relevance = self._cosine(query_vec, answer_vec)

        return {
            "context_relevance": round(max(context_relevance, 0.0), 4),
            "groundedness": round(groundedness, 4),
            "answer_relevance": round(max(answer_relevance, 0.0), 4),
        }


class ClinicalTrialRAG:
    """The query-time service: the eight-step path from a question to an answer."""

    def __init__(self, embedder: EmbeddingModel, store: VectorStore,
                 governance: GovernanceEngine, audit: AuditTrail) -> None:
        self.embedder = embedder
        self.store = store
        self.governance = governance
        self.audit = audit
        self.retriever = HybridRetriever(store, embedder)
        _, documents, _ = store.all_documents()
        self.reranker = CrossEncoderReranker(settings.rerank_model_name, documents)
        self.llm = LLMClient(settings.openrouter_api_key, settings.openrouter_model)
        self.evaluator = RAGTriadEvaluator(embedder)

    def answer(self, question: str, actor: str = "researcher@hospital.sa",
               role: str = "clinical_researcher") -> dict[str, Any]:
        started = time.perf_counter()

        # 1. Access control -----------------------------------------------------
        if role not in ROLE_POLICIES:
            raise PermissionError(f"Unknown role '{role}'.")

        # 2. PHI guard on the INBOUND query -------------------------------------
        # A researcher pasting a patient's name into the search box is the single
        # most common way PHI leaks into an AI system. It is scrubbed here.
        safe_question, phi_findings = PHIGuard.redact(question)
        if phi_findings:
            logger.warning(f"PHI stripped from inbound query: {sorted({f['type'] for f in phi_findings})}")

        # 3. Hybrid retrieval ----------------------------------------------------
        candidates = self.retriever.retrieve(safe_question)

        # 4. Two-stage reranking -------------------------------------------------
        top = self.reranker.rerank(safe_question, candidates, settings.rerank_top_n)

        # 5. Contradiction detection --------------------------------------------
        alerts = ContradictionDetector.check(top)

        # 6. Prompt construction -------------------------------------------------
        context_blocks = [c["text"] for c in top]
        numbered = "\n\n".join(f"[{i}] {block}" for i, block in enumerate(context_blocks, start=1))
        prompt = ANSWER_PROMPT.format(system=SYSTEM_PROMPT, context=numbered, question=safe_question)

        # 7. Generation ----------------------------------------------------------
        answer_text = self.llm.generate(prompt, context_blocks)

        # 8. Evidence trail + evaluation + audit ---------------------------------
        evidence = [
            {
                "citation": index,
                "nct_id": candidate["metadata"].get("nct_id"),
                "chunk_id": candidate["chunk_id"],
                "section": candidate["metadata"].get("section"),
                "lineage_id": candidate["metadata"].get("lineage_id"),
                "retrieved_by": candidate.get("retrieved_by", "dense"),
                "rrf_score": round(candidate.get("rrf_score", 0.0), 6),
                "rerank_score": round(candidate.get("rerank_score", 0.0), 4),
                "source_url": candidate["metadata"].get("source_url"),
            }
            for index, candidate in enumerate(top, start=1)
        ]

        triad = self.evaluator.evaluate(safe_question, answer_text, context_blocks)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        audit_id = self.audit.log(
            actor=actor, role=role, action="RAG_QUERY",
            payload={
                "query_redacted": safe_question,
                "query_hash": hashlib.sha256(question.encode()).hexdigest()[:16],
                "phi_stripped_from_query": sorted({f["type"] for f in phi_findings}),
                "nct_ids_returned": [e["nct_id"] for e in evidence],
                "chunks_returned": [e["chunk_id"] for e in evidence],
                "rag_triad": triad,
                "contradiction_alerts": len(alerts),
                "llm_backend": self.llm.backend,
                "latency_ms": latency_ms,
            },
        )

        return {
            "question": question,
            "question_used": safe_question,
            "answer": answer_text,
            "evidence": evidence,
            "contradiction_alerts": alerts,
            "rag_triad": triad,
            "audit_id": audit_id,
            "latency_ms": latency_ms,
            "candidates_stage_one": len(candidates),
            "backends": {
                "embedding": self.embedder.backend,
                "reranker": self.reranker.backend,
                "llm": self.llm.backend,
            },
        }


# =====================================================================
# 9. ARCHITECTURAL COMPONENT: AI INFRASTRUCTURE ORCHESTRATION (THE DAG)
# =====================================================================

class PipelineOrchestrator:
    """A minimal DAG scheduler standing in for Airflow / Prefect / Dagster.

    It provides the four orchestration guarantees Day 5 calls for: dependency
    resolution so no task starts before its inputs exist, automatic retries with
    exponential backoff, per-task fallbacks so one failure does not crash the
    whole pipeline, and a run report for observability.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.state: dict[str, Any] = {}
        self.report: list[dict[str, Any]] = []

    def task(self, name: str, depends_on: list[str] | None = None, retries: int = 2):
        """Registers a node in the DAG."""
        def decorator(function):
            self.tasks[name] = {"fn": function, "depends_on": depends_on or [], "retries": retries}
            self.order.append(name)
            return function
        return decorator

    def _resolve(self) -> list[str]:
        """Topological sort - a task never runs before its dependencies."""
        resolved, visiting = [], set()

        def visit(name: str) -> None:
            if name in resolved:
                return
            if name in visiting:
                raise RuntimeError(f"Cycle detected in the pipeline DAG at '{name}'.")
            visiting.add(name)
            for dependency in self.tasks[name]["depends_on"]:
                visit(dependency)
            visiting.discard(name)
            resolved.append(name)

        for name in self.order:
            visit(name)
        return resolved

    def run(self) -> list[dict]:
        banner("AI INFRASTRUCTURE ORCHESTRATION - EXECUTING PIPELINE DAG")
        for name in self._resolve():
            spec = self.tasks[name]
            print(f"\n▶️  [DAG] Task '{name}'  (depends on: {spec['depends_on'] or 'none'})")
            started = time.perf_counter()
            last_error: Exception | None = None

            for attempt in range(1, spec["retries"] + 2):
                try:
                    spec["fn"](self.state)
                    elapsed = round(time.perf_counter() - started, 2)
                    self.report.append({"task": name, "status": "SUCCESS", "attempts": attempt, "seconds": elapsed})
                    logger.success(f"Task '{name}' completed in {elapsed}s (attempt {attempt})")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    backoff = 2 ** (attempt - 1)
                    logger.error(f"Task '{name}' failed on attempt {attempt}: {type(exc).__name__}: {exc}")
                    if attempt <= spec["retries"]:
                        logger.info(f"Retrying '{name}' in {backoff}s (exponential backoff)...")
                        time.sleep(backoff)

            if last_error is not None:
                elapsed = round(time.perf_counter() - started, 2)
                self.report.append({
                    "task": name, "status": "FAILED", "attempts": spec["retries"] + 1,
                    "seconds": elapsed, "error": f"{type(last_error).__name__}: {last_error}",
                })
                logger.critical(f"DAG halted: task '{name}' exhausted all retries.")
                break

        return self.report

    def print_report(self) -> None:
        banner("ORCHESTRATION RUN REPORT")
        print(pd.DataFrame(self.report).to_string(index=False) if self.report else "(no tasks executed)")


def build_pipeline(allow_live: bool = True, reset_vectors: bool = True) -> PipelineOrchestrator:
    """Wires the six pipeline stages into a dependency-ordered DAG."""
    orchestrator = PipelineOrchestrator()
    lineage = LineageTracker()
    audit = AuditTrail()
    governance = GovernanceEngine(lineage, audit)

    # ---- Task 1: INGEST ------------------------------------------------------
    @orchestrator.task("ingest", retries=1)
    def _ingest(state: dict) -> None:
        ensure_dirs()

        async def stream() -> tuple[list[dict], list[dict], str]:
            broker = MockEventBroker()
            trials_producer = ClinicalTrialsProducer(broker)
            ehr_producer = InternalEHRProducer(broker)
            await trials_producer.run(allow_live=allow_live)
            await ehr_producer.run()
            trial_events = broker.drain("raw_clinical_trials")
            patient_events = broker.drain("raw_patient_records")
            return trial_events, patient_events, trials_producer.mode

        trial_events, patient_events, mode = asyncio.run(stream())

        state["ingestion_mode"] = mode
        state["trials_raw"] = pd.DataFrame([e["payload"] for e in trial_events])
        state["patients_raw"] = [e["payload"] for e in patient_events]
        state["lineage_ingest"] = lineage.record(
            operation="ingest",
            inputs=["clinicaltrials.gov/api/v2/studies", "internal_ehr"],
            outputs=["topic:raw_clinical_trials", "topic:raw_patient_records"],
            record_count=len(trial_events) + len(patient_events),
            details={"mode": mode, "condition": settings.ctgov_condition},
        )
        print(f"✅ [INGEST] {len(trial_events)} trials + {len(patient_events)} patient records on the bus")

    # ---- Task 2: QUALITY -----------------------------------------------------
    @orchestrator.task("quality", depends_on=["ingest"])
    def _quality(state: dict) -> None:
        engine = DataQualityEngine(state["trials_raw"])
        clean, rejected, report = engine.run_all_checks()

        banner("DATA QUALITY VALIDATION REPORT")
        print(json.dumps(report, indent=4))

        quarantine_path = quarantine_batch(rejected, "clinical_trials")

        if report["halt_pipeline"]:
            raise RuntimeError(
                f"CRITICAL: {report['failure_ratio']:.0%} of the batch failed quality gates "
                f"(threshold {settings.quality_halt_threshold:.0%}). Pipeline halted; batch quarantined."
            )

        state["trials_clean"] = clean
        state["quality_report"] = report
        state["lineage_quality"] = lineage.record(
            operation="quality_gate",
            inputs=[state["lineage_ingest"]],
            outputs=["clean_batch", "quarantine_zone"],
            record_count=len(clean),
            details={"rejected": len(rejected), "quarantine_path": quarantine_path},
        )
        print(f"✅ [QUALITY] {len(clean)} records passed | {len(rejected)} quarantined")

    # ---- Task 3: GOVERNANCE --------------------------------------------------
    @orchestrator.task("governance", depends_on=["quality"])
    def _governance(state: dict) -> None:
        safe_patients, all_findings = [], []
        for patient in state["patients_raw"]:
            safe, findings = PHIGuard.deidentify_patient(patient)
            safe_patients.append(safe)
            all_findings.extend(findings)

        banner("HIPAA GOVERNANCE - PHI DE-IDENTIFICATION")
        print(f"  Patient records processed ......... {len(safe_patients)}")
        print(f"  PHI identifiers detected/removed .. {len(all_findings)}")
        print(f"  Identifier families ............... {sorted({f['type'] for f in all_findings})}")
        print(f"  Fields entering the safe zone ..... {sorted(safe_patients[0].keys()) if safe_patients else []}")
        print("\n  Example de-identified profile:")
        if safe_patients:
            print("   ", json.dumps(safe_patients[0], indent=6, default=str))

        audit.log(
            actor="pipeline", role="data_engineer", action="PHI_DEIDENTIFICATION",
            payload={
                "records": len(safe_patients),
                "phi_removed": len(all_findings),
                "phi_types": sorted({f["type"] for f in all_findings}),
            },
        )

        state["patients_safe"] = safe_patients
        state["lineage_governance"] = lineage.record(
            operation="governance_deidentification",
            inputs=[state["lineage_quality"]],
            outputs=["deidentified_patient_profiles"],
            record_count=len(safe_patients),
            details={"phi_removed": len(all_findings)},
        )
        print(f"\n✅ [GOVERNANCE] {len(all_findings)} PHI identifiers stripped before any downstream use")

    # ---- Task 4: LAKEHOUSE ---------------------------------------------------
    @orchestrator.task("lakehouse", depends_on=["governance"])
    def _lakehouse(state: dict) -> None:
        lake = DeltaStyleLakehouse(settings.lakehouse_path)
        frame = state["trials_clean"].copy()

        # Normalise dtypes so the registered blueprint stays stable across runs.
        for column in ("min_age_years", "max_age_years", "enrollment"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].astype("string")

        # Split the batch to model the incremental sync strategy from Day 3: an
        # initial full load, then a smaller delta that arrives later. Each lands
        # as its own immutable commit, which is what makes time travel possible.
        split = max(1, int(len(frame) * 0.75))
        initial_load, incremental_delta = frame.iloc[:split], frame.iloc[split:]

        version = lake.write(initial_load, mode="overwrite", operation="WRITE")
        if not incremental_delta.empty:
            version = lake.write(incremental_delta, mode="append", operation="APPEND")

        banner("LAKEHOUSE - SCHEMA ENFORCEMENT GUARDRAIL")
        print("  Attempting to append a batch carrying an unannounced 'investigator_notes' column...")
        rogue = frame.head(2).copy()
        rogue["investigator_notes"] = "free text added by an upstream developer"
        try:
            lake.write(rogue, mode="append", operation="APPEND")
            print("  ⚠️  Schema drift was NOT blocked - guardrail misconfigured.")
        except SchemaEnforcementError as exc:
            print(f"  ❌ Transaction blocked by the Lakehouse! {exc}")
            print("  💡 This is what stops a data lake from degrading into a data swamp.")

        state["lakehouse"] = lake
        state["lakehouse_version"] = version
        state["lineage_lakehouse"] = lineage.record(
            operation="lakehouse_write",
            inputs=[state["lineage_governance"]],
            outputs=[f"{settings.lakehouse_path}@v{version}"],
            record_count=len(frame),
            details={"mode": "overwrite", "columns": len(frame.columns)},
        )
        print(f"\n✅ [LAKEHOUSE] Committed version {version} with {len(frame)} governed records")

    # ---- Task 5: CHUNK + EMBED ----------------------------------------------
    @orchestrator.task("chunk_embed", depends_on=["lakehouse"])
    def _chunk_embed(state: dict) -> None:
        records = state["lakehouse"].read().to_dict(orient="records")
        chunker = RecursiveChunker(settings.chunk_size, settings.chunk_overlap)

        chunks: list[dict] = []
        for record in records:
            for chunk in chunker.chunk_record(record):
                chunk["lineage_id"] = state["lineage_lakehouse"]
                chunk["source_url"] = record.get("source_url")
                chunk["overall_status"] = record.get("overall_status")
                chunk["phase"] = record.get("phase")
                chunk["min_age_years"] = record.get("min_age_years")
                chunk["max_age_years"] = record.get("max_age_years")
                chunk["classification"] = "PUBLIC"
                chunks.append(chunk)

        embedder = EmbeddingModel(settings.embed_model_name)
        vectors = embedder.encode([c["text"] for c in chunks])

        state["chunks"] = chunks
        state["vectors"] = vectors
        state["embedder"] = embedder
        state["lineage_chunks"] = lineage.record(
            operation="chunk_and_embed",
            inputs=[state["lineage_lakehouse"]],
            outputs=["chunk_vectors"],
            record_count=len(chunks),
            details={
                "chunk_size": settings.chunk_size,
                "overlap": settings.chunk_overlap,
                "backend": embedder.backend,
                "dimensions": embedder.dimensions,
            },
        )
        print(f"✅ [CHUNK+EMBED] {len(records)} trials -> {len(chunks)} chunks -> "
              f"{vectors.shape[1]}-dim vectors via {embedder.backend}")

    # ---- Task 6: VECTOR INDEX -----------------------------------------------
    @orchestrator.task("vector_index", depends_on=["chunk_embed"])
    def _vector_index(state: dict) -> None:
        store = VectorStore(settings.chroma_db_dir)
        if reset_vectors:
            store.reset()

        # HIPAA hard gate: every chunk is re-scanned immediately before insert.
        blocked = 0
        admitted_chunks, admitted_rows = [], []
        for position, chunk in enumerate(state["chunks"]):
            try:
                governance.assert_vector_db_safe(chunk["text"], chunk["chunk_id"])
                admitted_chunks.append(chunk)
                admitted_rows.append(position)
            except PermissionError as exc:
                blocked += 1
                logger.error(str(exc))

        vectors = state["vectors"][admitted_rows] if admitted_rows else np.empty((0, 1))
        store.add(admitted_chunks, vectors)

        state["store"] = store
        state["lineage_index"] = lineage.record(
            operation="vector_index",
            inputs=[state["lineage_chunks"]],
            outputs=[f"chroma:{VectorStore.COLLECTION}"],
            record_count=len(admitted_chunks),
            details={"blocked_for_phi": blocked, "index": "HNSW", "space": "cosine"},
        )
        print(f"✅ [VECTOR INDEX] {len(admitted_chunks)} chunks indexed (HNSW/cosine) | "
              f"{blocked} blocked by the PHI gate | collection size = {store.count()}")

    return orchestrator


# =====================================================================
# 10. COMMAND-LINE ENTRYPOINT
# =====================================================================

def print_query_result(result: dict) -> None:
    banner("ANSWER")
    print(result["answer"])

    print("\n" + "-" * 70)
    print("  EVIDENCE TRAIL  (every claim traceable to a governed source)")
    print("-" * 70)
    for item in result["evidence"]:
        print(f"  [{item['citation']}] {item['nct_id']} | section={item['section']} | "
              f"via={item['retrieved_by']} | rerank={item['rerank_score']}")
        print(f"       chunk   : {item['chunk_id']}")
        print(f"       lineage : {item['lineage_id']}")
        print(f"       source  : {item['source_url']}")

    if result["contradiction_alerts"]:
        print("\n" + "-" * 70)
        print("  ⚠️  CONTRADICTION ALERTS")
        print("-" * 70)
        for alert in result["contradiction_alerts"]:
            print(f"  • {alert['message']}")

    triad = result["rag_triad"]
    print("\n" + "-" * 70)
    print("  RAG TRIAD EVALUATION")
    print("-" * 70)
    print(f"  Context Relevance : {triad['context_relevance']:.3f}   (did retrieval pull the right passages?)")
    print(f"  Groundedness      : {triad['groundedness']:.3f}   (is the answer supported by those passages?)")
    print(f"  Answer Relevance  : {triad['answer_relevance']:.3f}   (does it address the question asked?)")

    print("\n" + "-" * 70)
    print(f"  stage-1 candidates : {result['candidates_stage_one']}  ->  stage-2 survivors : {len(result['evidence'])}")
    print(f"  backends           : {result['backends']}")
    print(f"  latency            : {result['latency_ms']} ms")
    print(f"  audit entry        : {result['audit_id']}")
    print("-" * 70)


def open_query_service() -> ClinicalTrialRAG:
    """Attaches to the already-built platform for query-only operation."""
    store = VectorStore(settings.chroma_db_dir)
    if store.count() == 0:
        raise RuntimeError(
            "The vector database is empty. Build the platform first:\n"
            "    python clinical_trial_rag.py --stage pipeline"
        )
    embedder = EmbeddingModel(settings.embed_model_name)
    lineage = LineageTracker()
    audit = AuditTrail()
    governance = GovernanceEngine(lineage, audit)
    return ClinicalTrialRAG(embedder, store, governance, audit)


def stage_pipeline(args: argparse.Namespace) -> None:
    banner("HEALTHCARE CLINICAL TRIAL MATCHER - INTEGRATED AI DATA PLATFORM")
    print(f"  Configuration: {json.dumps(settings.as_dict(), indent=2)}")
    orchestrator = build_pipeline(allow_live=not args.offline, reset_vectors=True)
    orchestrator.run()
    orchestrator.print_report()


def stage_query(args: argparse.Namespace) -> None:
    service = open_query_service()
    result = service.answer(args.q, actor=args.actor, role=args.role)
    print_query_result(result)


def stage_history(_: argparse.Namespace) -> None:
    lake = DeltaStyleLakehouse(settings.lakehouse_path)
    history = lake.history()
    banner("LAKEHOUSE TRANSACTION LOG (TIME TRAVEL LEDGER)")
    if history.empty:
        print("  (no commits yet - run --stage pipeline first)")
        return
    print(history.to_string(index=False))
    print("\n  ⏪ Reading version 0 (the original committed batch):")
    print(f"     {len(lake.read(version=0))} records")
    print("  ⏩ Reading the latest version:")
    print(f"     {len(lake.read())} records")


def stage_lineage(_: argparse.Namespace) -> None:
    banner("DATA LINEAGE - END-TO-END GENEALOGY")
    print(LineageTracker().render_graph())


def stage_audit(args: argparse.Namespace) -> None:
    banner("AUDIT TRAIL (most recent entries)")
    entries = AuditTrail().tail(args.limit)
    if not entries:
        print("  (no audit entries yet)")
        return
    for entry in entries:
        print(json.dumps(entry, indent=2))
        print("-" * 70)


def stage_demo(args: argparse.Namespace) -> None:
    """Walks the full architecture end to end - the presentation script."""
    stage_pipeline(args)

    stage_history(args)
    stage_lineage(args)

    service = open_query_service()
    questions = [
        "Which trials are recruiting adults over 60 with type 2 diabetes, and what are the age limits?",
        "What are the common exclusion criteria related to kidney function?",
    ]
    for question in questions:
        banner(f"RESEARCHER QUERY: {question}")
        print_query_result(service.answer(question))

    banner("GOVERNANCE DEMONSTRATION - ATTRIBUTE-BASED ACCESS CONTROL")
    lineage, audit = LineageTracker(), AuditTrail()
    governance = GovernanceEngine(lineage, audit)
    patient = InternalEHRProducer.PATIENTS[0]
    for role in ("treating_physician", "clinical_researcher", "data_engineer"):
        governed = governance.apply_access_policy(patient, role)
        print(f"\n  Role: {role}")
        print(f"    patient_name          -> {governed['patient_name']}")
        print(f"    medical_record_number -> {governed['medical_record_number']}")
        print(f"    diagnosis             -> {governed['diagnosis']}")

    banner("GOVERNANCE DEMONSTRATION - PHI BLOCKED AT THE VECTOR DATABASE DOOR")
    poisoned = "Patient Sarah Al-Mutairi, MRN-4471902, s.almutairi@example-hospital.sa, meets inclusion criteria."
    print(f"  Attempting to index: {poisoned}")
    try:
        governance.assert_vector_db_safe(poisoned, "DEMO::phi_leak::0")
        print("  ⚠️  PHI was NOT blocked - governance misconfigured.")
    except PermissionError as exc:
        print(f"  ❌ {exc}")
        print("  💡 'No PHI in the vector DB' is enforced by code, not by policy documents.")

    stage_audit(argparse.Namespace(limit=3))
    banner("🏁 CAPSTONE DEMONSTRATION COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Healthcare Clinical Trial Matcher - SDAIA Day 5 Capstone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["pipeline", "query", "history", "lineage", "audit", "demo"],
        default="demo",
        help="pipeline: build the platform | query: ask a question | demo: full walkthrough",
    )
    parser.add_argument("--q", default="Which trials are recruiting adults with type 2 diabetes?",
                        help="the researcher's question (used with --stage query)")
    parser.add_argument("--actor", default="researcher@hospital.sa", help="identity recorded in the audit trail")
    parser.add_argument("--role", default="clinical_researcher", choices=sorted(ROLE_POLICIES),
                        help="ABAC role of the requesting user")
    parser.add_argument("--offline", action="store_true",
                        help="skip the live ClinicalTrials.gov call and replay cached raw data")
    parser.add_argument("--limit", type=int, default=5, help="how many audit entries to show")
    args = parser.parse_args()

    dispatch = {
        "pipeline": stage_pipeline,
        "query": stage_query,
        "history": stage_history,
        "lineage": stage_lineage,
        "audit": stage_audit,
        "demo": stage_demo,
    }
    dispatch[args.stage](args)


if __name__ == "__main__":
    main()
