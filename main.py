"""
Email Classifier API
Stack: FastAPI + Pinecone (v3) + Qwen3-Embedding-0.6B
     + cross-encoder/nli-MiniLM2-L6-H768 (reranker)
     + GPT-4o-mini fallback

Classification pipeline:
  1. Bi-encoder retrieval      — Qwen3 embeddings vs Pinecone prototypes
  2. Structural features       — zero-ML regex patterns (unsubscribe, currency, etc.)
  3. Sender reputation         — Bayesian-smoothed global cross-user domain signal
  4. Per-user sender affinity   — this user's personal domain history
  5. Cross-encoder reranking   — top-5 candidates rescored with joint attention
  6. Confidence routing        — high confidence → return, low → GPT-4o-mini fallback
  7. Feedback loop             — LLM-confirmed results stored back into Pinecone

Label storage — single namespace, two tiers via metadata:
  • System labels  — scope field absent OR scope="system"
                     Existing DB vectors are automatically treated as system labels.
                     Shared globally, admin-managed via POST /system/label.
  • User labels    — scope="user" + user_id field
                     Per-user custom labels. Same name can mean different things
                     for different users. Managed via POST /user/label.
"""

import os
import re
import uuid
import json
import math
import hashlib
import asyncio
from datetime import datetime, timezone
import redis
from typing import Optional
from contextlib import asynccontextmanager
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from openai import OpenAI
from structural_patterns import STRUCTURAL_PATTERNS, SIGNAL_TO_CATEGORY, CATEGORY_KEYWORDS

# Only load .env locally, not on Modal
import os
if not os.environ.get("MODAL_ENVIRONMENT"):
    from dotenv import load_dotenv
    load_dotenv()


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX_NAME"]   # dim=1024, metric=cosine
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
UPSTASH_REDIS_URL = os.environ.get("UPSTASH_REDIS_URL", "")  # redis://...:port

GLOBAL_NAMESPACE = "labels"
PROTOTYPES_PER_LABEL = 10


SCOPE_SYSTEM = "system"
SCOPE_USER = "user"

# Confidence thresholds
CONFIDENCE_MARGIN        = 0.12   # send to LLM if margin below this (wider — let LLM arbitrate close calls)
LOW_ABSOLUTE_SCORE       = 0.45   # top score must be strong to trust (lowered for blended score scale)

TOP_K_PER_LABEL          = 15     # vectors fetched PER LABEL (per-label querying ensures equal retrieval budget)
TOPK_MEAN_K              = 3      # average top-k matches per label (more robust than max)
TOP_K                    = TOP_K_PER_LABEL  # kept for delete endpoints / admin queries
SIMILARITY_THRESHOLD     = 0.85   # store more examples (less strict dedup)
LLM_CONFIDENCE_THRESHOLD = 0.80   # accept slightly less confident LLM results
SENDER_AFFINITY_WEIGHT   = 0.08   # blend weight for per-user sender affinity (reduced; reputation takes rest)

# Cross-encoder reranker
RERANKER_MODEL_NAME      = "cross-encoder/nli-MiniLM2-L6-H768"
RERANKER_TOP_N           = 5      # rerank top-N label candidates
RERANKER_BLEND_WEIGHT    = 0.30   # cross-encoder vs embedding weight (reduced — preserve user-label signals)

# Structural feature prior boost (zero-ML)
STRUCTURAL_BOOST_WEIGHT  = 0.25   # blend weight for structural signal prior (raised — strong zero-ML signal)

# User label priority — user-defined labels get a score multiplier so they aren't
# drowned out by system labels with many more prototypes.
# e.g. user's "Payments" label beats system "Automated alerts" when appropriate.
USER_LABEL_PRIORITY_BOOST = 1.50  # multiply final score for user-scoped labels (applied post-rerank)

# Structural signal amplification for user-custom labels
USER_STRUCTURAL_AMPLIFIER = 1.5   # multiply structural boost when it matches a user-custom label

# User-label conflict — force LLM when user-custom label competes with system label
USER_LABEL_CONFLICT_MARGIN = 0.15 # force LLM if user-label is within this margin of top

# Sender reputation (global Bayesian-smoothed, collaborative filtering)
REPUTATION_PRIOR_STRENGTH = 10.0  # Bayesian pseudo-count for smoothing
REPUTATION_MIN_OBSERVATIONS = 30  # observations needed for full confidence
SENDER_REPUTATION_WEIGHT = 0.12   # blend weight for global reputation


QUERY_INSTRUCTION = "Instruct: Given an email, identify which category it belongs to.\nQuery: "





@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, pinecone_index, openai_client, redis_client, reranker_model

    if hasattr(app.state, "embedding_model"):
        # ── Running on Modal ──────────────────────────────────────────
        # Models already loaded by @enter() before this runs.
        # Just read from app.state — no blocking work here.
        embedding_model = app.state.embedding_model
        pinecone_index = app.state.pinecone_index
        openai_client = app.state.openai_client
        reranker_model = app.state.reranker_model
        print("✅ Clients ready (restored from Modal snapshot).")
    else:
        # ── Local development fallback ────────────────────────────────
        print("Loading Qwen3 embedding model...")
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        print("Loading cross-encoder reranker...")
        reranker_model = CrossEncoder(RERANKER_MODEL_NAME)



        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX)

        print("Connecting to OpenAI...")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

        print("✅ All clients ready.")

    # ── Redis (Upstash) for sender affinity ───────────────────────
    if UPSTASH_REDIS_URL:
        redis_client = redis.from_url(UPSTASH_REDIS_URL, decode_responses=True)
        redis_client.ping()
        print("✅ Redis (Upstash) connected.")
    else:
        redis_client = None
        print("⚠️  UPSTASH_REDIS_URL not set — sender affinity disabled.")

    yield


app = FastAPI(
    title="Email Classifier API",
    description="Hybrid embedding + LLM email classifier with system + per-user label tiers.",
    lifespan=lifespan
)


class AddSystemLabelRequest(BaseModel):
    label_name: str
    description: Optional[str] = None


class AddUserLabelRequest(BaseModel):
    user_id: str
    label_name: str
    # shapes prototypes to match THIS user's meaning
    description: Optional[str] = None


class DeleteSystemLabelRequest(BaseModel):
    label_name: str


class DeleteUserLabelRequest(BaseModel):
    user_id: str
    label_name: str


class ClassifyRequest(BaseModel):
    user_id: str
    subject: str
    sender: str
    body: str
    labels: list[str]
    use_llm: Optional[bool] = True

class Email(BaseModel):
    subject: str
    sender: str
    body: str

class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    margin: float
    method: str
    all_scores: dict[str, float]


   


def embed_document(text: str) -> list[float]:
    """Embed label prototype — no instruction prefix."""
    return embedding_model.encode(
        text,
        prompt_name="document",
        normalize_embeddings=True,
        convert_to_tensor=False
    ).tolist()


def embed_query(text: str) -> list[float]:
    """Embed email query — with task instruction prefix."""
    return embedding_model.encode(
        text,
        prompt=QUERY_INSTRUCTION,
        normalize_embeddings=True,
        convert_to_tensor=False
    ).tolist()


# ─────────────────────────────────────────────
# Structural Feature Extraction  (zero-ML prior boost)
# ─────────────────────────────────────────────
# Pattern definitions live in structural_patterns.py — imported above.


def extract_structural_signals(subject: str, sender: str, body: str) -> dict[str, bool]:
    """Detect structural patterns in raw email text. Returns {signal_name: detected}."""
    text = f"{subject}\n{sender}\n{body}"
    return {
        signal: bool(pattern.search(text))
        for signal, pattern in STRUCTURAL_PATTERNS.items()
    }


def _category_matches_label(category: str, label: str) -> bool:
    """Check if a broad category matches a user's label via keyword stems."""
    label_lower = label.lower()
    if category in label_lower or label_lower in category:
        return True
    keywords = CATEGORY_KEYWORDS.get(category, [])
    return any(kw in label_lower for kw in keywords)


def compute_structural_boost(
    signals: dict[str, bool], label_names: list[str]
) -> dict[str, float]:
    """
    Convert detected structural signals into per-label boost scores.
    Maps broad categories to user's actual labels via fuzzy keyword matching.
    Returns {label: 0.0–1.0} — max signal confidence per label.
    """
    boosts: dict[str, float] = {lbl: 0.0 for lbl in label_names}

    for signal, detected in signals.items():
        if not detected:
            continue
        category_hints = SIGNAL_TO_CATEGORY.get(signal, {})
        for category, confidence in category_hints.items():
            for label in label_names:
                if _category_matches_label(category, label):
                    boosts[label] = max(boosts[label], confidence)

    return boosts


# ─────────────────────────────────────────────
# Cross-Encoder Reranker
# ─────────────────────────────────────────────
# After bi-encoder retrieval, the top-N label candidates are rescored
# using a cross-encoder that jointly attends to (email, prototype).
# This dramatically improves discrimination between close categories
# (e.g. "Finance" vs "Automated alerts") at ~10ms cost per call.


def rerank_top_labels(
    email_text: str,
    label_prototypes: dict[str, list[tuple[float, str]]],
    embedding_scores: dict[str, float],
    top_n: int = RERANKER_TOP_N,
) -> dict[str, float]:
    """
    Rerank top-N labels using the cross-encoder model.

    For each candidate label, takes the best bi-encoder prototype
    and cross-encodes it with the email text. The cross-encoder score
    is blended with the original embedding score.

    Returns updated {label: blended_score} dict for ALL labels
    (non-top-N labels keep their original embedding score).
    """
    sorted_labels = sorted(
        embedding_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_labels = [lbl for lbl, sc in sorted_labels[:top_n] if sc > 0]

    if not top_labels:
        return dict(embedding_scores)

    # Build (email, best_prototype) pairs for each top candidate
    pairs = []
    pair_labels = []
    for lbl in top_labels:
        protos = label_prototypes.get(lbl, [])
        if protos:
            # Instead of 1 best prototype, cross-encode top-3 and take max
            top_protos = sorted(protos, key=lambda x: x[0], reverse=True)[:3]
            for proto_score, proto_text in top_protos:
                pairs.append((email_text, proto_text))
                pair_labels.append(lbl)

    if not pairs:
        return dict(embedding_scores)

    # Batch cross-encode — single forward pass
    # nli-MiniLM2-L6-H768 outputs 3 logits per pair:
    #   [contradiction, neutral, entailment]
    # We apply softmax and take the entailment probability as relevance score.
    raw_scores = reranker_model.predict(pairs, apply_softmax=False)

    def _nli_entailment_prob(logits) -> float:
        """Softmax over 3 NLI logits → entailment (index 2) probability."""
        import numpy as np
        logits = np.asarray(logits, dtype=float)
        e = np.exp(logits - logits.max())
        return float(e[2] / e.sum())

    sigmoid_scores = [_nli_entailment_prob(s) for s in raw_scores]

    # Aggregate cross-encoder scores per label (take max across prototypes)
    ce_per_label: dict[str, float] = {}
    for lbl, ce_score in zip(pair_labels, sigmoid_scores):
        if lbl not in ce_per_label or ce_score > ce_per_label[lbl]:
            ce_per_label[lbl] = ce_score

    # Blend with original scores for reranked labels
    updated = dict(embedding_scores)
    for lbl, ce_score in ce_per_label.items():
        updated[lbl] = (
            RERANKER_BLEND_WEIGHT * ce_score
            + (1 - RERANKER_BLEND_WEIGHT) * embedding_scores[lbl]
        )

    return updated


# ─────────────────────────────────────────────
# Sender Affinity  (Redis-backed via Upstash)
# ─────────────────────────────────────────────
# Keys:
#   affinity:sys:{domain}          → hash {label: count}  (global)
#   affinity:u:{user_id}:{domain}  → hash {label: count}  (per-user)
#   rep:users:{domain}:{label}     → set of user_ids     (unique user tracking)
#
# The unique-user sets power Bayesian-smoothed sender reputation:
# domains classified consistently across many users get higher
# confidence, acting as collaborative filtering for new users.
# ~4 commands per classification → well within Upstash free tier.


def extract_domain(sender: str) -> str:
    """Extract domain from sender email. 'noreply@alerts.hdfcbank.com' → 'alerts.hdfcbank.com'"""
    match = re.search(r'@([\w.-]+)', sender)
    return match.group(1).lower() if match else ""


def update_sender_affinity(domain: str, label: str, user_id: str, is_user_label: bool = False):
    """
    Record a classification result in Redis.
    Updates global counts, per-user counts, AND the unique-user set
    for this domain-label pair (powers the sender reputation graph).

    is_user_label=True: the winning label is a user-custom label.
    In that case we skip the global affinity:sys update so that this
    user's personal vocabulary doesn't pollute the shared reputation
    signal seen by other users (who don't have that custom label).
    """
    if not domain or not redis_client:
        return
    pipe = redis_client.pipeline(transaction=False)
    if not is_user_label:
        # Only record in global reputation when the label is a system label.
        # User-custom label wins must not teach other users' classifiers.
        pipe.hincrby(f"affinity:sys:{domain}", label, 1)
        pipe.sadd(f"rep:users:{domain}:{label}", user_id)   # unique user tracking
    pipe.hincrby(f"affinity:u:{user_id}:{domain}", label, 1)
    pipe.execute()


def get_sender_affinity_scores(
    domain: str, user_id: str, label_names: list[str]
) -> dict[str, float]:
    """
    Per-user sender affinity — this user's personal history with a domain.
    Returns normalized scores {label: 0.0–1.0} summing to 1.0,
    or all 0s if no affinity data exists.
    """
    if not domain or not redis_client:
        return {label: 0.0 for label in label_names}

    user_counts = redis_client.hgetall(f"affinity:u:{user_id}:{domain}")
    if not user_counts:
        return {label: 0.0 for label in label_names}

    filtered = {lbl: int(user_counts.get(lbl, 0)) for lbl in label_names}
    total = sum(filtered.values())

    if total == 0:
        return {label: 0.0 for label in label_names}

    return {lbl: count / total for lbl, count in filtered.items()}


def get_sender_reputation_scores(
    domain: str, label_names: list[str]
) -> dict[str, float]:
    """
    Global sender reputation with Bayesian smoothing.

    Uses classification history across ALL users for this sender domain.
    Applies Laplace-style smoothing so domains with few observations
    produce weak signals, while heavily-observed domains produce strong ones.

    Confidence ramps linearly from 0 → 1 as observation count reaches
    REPUTATION_MIN_OBSERVATIONS (default 30).  This means a brand-new user
    immediately benefits from the collective knowledge of all other users
    who received mail from the same domain.

    Returns {label: 0.0–1.0} — NOT normalised to sum to 1 because the
    confidence scaling already tempers the magnitudes.
    """
    if not domain or not redis_client:
        return {label: 0.0 for label in label_names}

    sys_counts = redis_client.hgetall(f"affinity:sys:{domain}")
    if not sys_counts:
        return {label: 0.0 for label in label_names}

    total = sum(int(c) for c in sys_counts.values())
    n_labels = max(len(label_names), 1)

    # Bayesian smoothing: (count + prior) / (total + prior × categories)
    prior = REPUTATION_PRIOR_STRENGTH / n_labels
    denominator = total + REPUTATION_PRIOR_STRENGTH

    smoothed = {}
    for lbl in label_names:
        count = int(sys_counts.get(lbl, 0))
        smoothed[lbl] = (count + prior) / denominator

    # Confidence ramp — don't trust few observations
    confidence = min(1.0, total / REPUTATION_MIN_OBSERVATIONS)

    return {lbl: s * confidence for lbl, s in smoothed.items()}


def generate_prototypes(label_name: str, description: Optional[str]) -> list[str]:
    """
    Ask GPT-4o-mini to generate N prototype emails for the label.
    Each prototype covers a distinct sub-type for better vector space coverage.
    """
    desc_line = f"User description: {description}" if description else ""

    prompt = f"""
You are helping build an email classifier. Generate {PROTOTYPES_PER_LABEL} distinct prototype emails 
for the label "{label_name}". {desc_line}

Each prototype should:
- Cover a DIFFERENT sub-type of "{label_name}" emails
- Read like a real email with subject and body
- Be 4-6 sentences long
- Cover different types of scenerios
- NOT use placeholder text like [Company Name]

Return ONLY a JSON object with key "prototypes" containing an array of {PROTOTYPES_PER_LABEL} strings.
Format: {{"prototypes": ["prototype 1 text", "prototype 2 text", ...]}}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )

    parsed = json.loads(response.choices[0].message.content)

    if "prototypes" in parsed and isinstance(parsed["prototypes"], list):
        return parsed["prototypes"]

    # Fallback: grab first list value found
    for val in parsed.values():
        if isinstance(val, list):
            return val

    raise ValueError(f"Unexpected LLM response: {parsed}")


def llm_classify(subject: str, sender: str, body: str, label_names: list[str]) -> tuple[str, float]:
    """
    GPT-4o-mini fallback when embedding confidence is low.
    Returns (label, confidence) — confidence is used to decide whether
    to store the result back into Pinecone as a learned example.
    """
    labels_str = "\n- ".join(label_names)

    system_prompt = f"""You are an email classification system. Your ONLY job is to return a valid JSON object with "label" and "confidence" fields.

ALLOWED CATEGORIES (case-sensitive, exact match required):
- {labels_str}

RULES (highest priority first):
1. SPECIFICITY: Always prefer the MOST SPECIFIC matching category.
   - If "Payments" exists, use it for payment/transaction emails instead of generic "Automated alerts"
   - If "Orders" exists, use it for shipping/delivery emails instead of "Updates"
   - Specific user-defined labels ALWAYS beat generic catch-all labels
2. PURPOSE: Classify by the email's PRIMARY PURPOSE, not surface keywords:
   - Payment/transaction/invoice/billing/UPI/debit/credit → payment-related category
   - Calendar/meeting/RSVP/invite → event-related category
   - Marketing/promotions/deals/offers → marketing-related category
   - System notifications with no specific purpose → alerts/notification category
3. DOMAIN: Match sender domain patterns to category purpose
4. CONFIDENCE: If < 85% confident → return empty string for label

OUTPUT FORMAT (strict JSON):
{{"label": "exact_category_name", "confidence": 0.95}}
OR if uncertain:
{{"label": "", "confidence": 0.0}}"""

    user_prompt = f"""Classify this email into ONE category or return empty label if uncertain:

Subject: {subject}
From: {sender}
Body: {body[:1000]}

Available categories:
- {labels_str}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=50,
        seed=42,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("No response from OpenAI")

    parsed = json.loads(content)
    predicted = parsed.get("label", "").strip()
    llm_confidence = float(parsed.get("confidence", 0.0))

    # Match back to known label names (handles minor LLM deviations)
    for name in label_names:
        if name.lower() in predicted.lower():
            return name, llm_confidence

    return predicted, llm_confidence


def clean_email_for_storage(subject: str, sender: str, body: str, return_structured: bool = False) -> str | dict:
    """
    Produce a privacy-safe, semantically rich email representation for Pinecone.

    GPT-4o-mini acts as a single-step anonymizer:
      - Detects and replaces all PII (names, emails, URLs, phone numbers, etc.)
        with natural, role-based descriptions (e.g. "a colleague", "an invoice link").
      - Preserves tone, urgency, formality, topic, and intent so that the stored
        embedding captures the email's real category signal.

    On failure (network / quota) the raw truncated text is stored as a fallback
    so the feedback loop never hard-crashes.
    """
    raw_text = f"Subject: {subject}\nSender: {sender}\nBody: {body[:500]}"

    anonymize_prompt = f"""You are an email anonymizer preparing training data for an email classifier.

Your task:
- Identify and replace ALL personal information (full names, email addresses, URLs/links,
  phone numbers, account numbers, company names, physical addresses) with natural,
  role-based descriptions that preserve context.
  Good replacement examples:
    Person name      → "a colleague", "a client", "the support agent", "a vendor"
    URL / link       → "a product link", "a tracking URL", "a login link", "an invoice link"
    Email address    → "their email address"
    Phone number     → "their contact number"
    Company name     → "the company", "the service provider", "the sender's organization"
- Preserve EXACTLY: tone, urgency, formality level, topic/category, and intent.
- Do NOT summarize, shorten, or add new information.
- Keep the Subject / Sender / Body structure intact.
- Output ONLY the rewritten email — nothing else.

Email:
{raw_text}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": anonymize_prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        cleaned = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  LLM anonymization failed, storing raw truncated text: {e}")
        cleaned = raw_text

    if not return_structured:
        return cleaned

    # Parse back into subject, sender, body
    lines = cleaned.split("\n", 2)
    return {
        "subject": lines[0].removeprefix("Subject: ").strip() if len(lines) > 0 else "",
        "sender":  lines[1].removeprefix("Sender: ").strip()  if len(lines) > 1 else "",
        "body":    lines[2].removeprefix("Body: ").strip()    if len(lines) > 2 else "",
    }


def is_already_covered(vector: list[float], label: str, extra_filter: dict) -> bool:
    """
    Check if Pinecone already has a near-identical vector for this label.
    extra_filter scopes the check to either system or a specific user's vectors.
    """
    results = pinecone_index.query(
        vector=vector,
        top_k=1,
        namespace=GLOBAL_NAMESPACE,
        filter={"label": {"$eq": label}, **extra_filter},
        include_metadata=False
    )

    if results.matches and results.matches[0].score >= SIMILARITY_THRESHOLD:
        return True

    return False


def store_learned_example(
    subject: str,
    sender: str,
    body: str,
    label: str,
    query_vector: list[float],
    user_id: str,
    scope: str = SCOPE_USER,
):
    """
    Store an LLM-confirmed email as a learned vector in Pinecone.

    - scope="user"  : stored under this user only (custom labels)
    - scope="system": stored globally, improves classification for all users (system labels)

    Only called when:
    - LLM confidence >= LLM_CONFIDENCE_THRESHOLD
    - No near-identical vector already exists for the target scope+label
    """
    if scope == SCOPE_SYSTEM:
        dedup_filter = {"scope": {"$ne": SCOPE_USER}}
        vector_id = f"{label}_sys_learned_{uuid.uuid4().hex[:8]}"
        scope_tag = "system"
    else:
        dedup_filter = {"scope": {"$eq": SCOPE_USER},
                        "user_id": {"$eq": user_id}}
        vector_id = f"{label}_user_{user_id}_{uuid.uuid4().hex[:8]}"
        scope_tag = f"user={user_id}"

    if is_already_covered(query_vector, label, dedup_filter):
        print(
            f"⏭️  Skipping storage — pattern already covered for '{label}' ({scope_tag})")
        return

    cleaned_text = clean_email_for_storage(subject, sender, body)

    email_hash = hashlib.md5(
        f"{subject}{sender}{body[:200]}".encode()
    ).hexdigest()

    sender_domain = extract_domain(sender)
    metadata = {
        "label":         label,
        "prototype":     cleaned_text,
        "scope":         scope,
        "source":        "llm_fallback",
        "email_hash":    email_hash,
        "sender_domain": sender_domain,
        "llm_model":     "gpt-4o-mini",
        "created_at":    datetime.now(timezone.utc).isoformat()
    }
    if scope == SCOPE_USER:
        # user_id only on user-scoped vectors
        metadata["user_id"] = user_id

    pinecone_index.upsert(
        vectors=[{
            "id":     vector_id,
            "values": embed_document(cleaned_text),
            "metadata": metadata
        }],
        namespace=GLOBAL_NAMESPACE
    )

    print(f"✅ Learned new example for '{label}' → {vector_id} ({scope_tag})")


@app.post("/system/label", summary="[Admin] Add a system label (shared by all users)")
async def add_system_label(request: AddSystemLabelRequest):
    """
    Generates prototypes and stores them with scope="system".
    Existing DB vectors without a scope field are already treated as system labels
    by classify — so this is safe to run alongside legacy data.
    """
    label_name = request.label_name.strip()
    validation_vector = [0.01 if i % 3 == 0 else 0.0 for i in range(1024)]

    # Check both old (no scope) and new (scope=system) vectors
    existing = pinecone_index.query(
        vector=validation_vector,
        top_k=1,
        namespace=GLOBAL_NAMESPACE,
        filter={"label": {"$eq": label_name}, "scope": {"$ne": SCOPE_USER}},
        include_metadata=False
    )
    if existing.matches:
        raise HTTPException(
            status_code=409,
            detail=f"System label '{label_name}' already exists."
        )

    try:
        prototypes = generate_prototypes(label_name, request.description)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prototype generation failed: {str(e)}")

    vectors = []
    for i, prototype_text in enumerate(prototypes):
        vectors.append({
            "id": f"{label_name}_sys_{i}_{uuid.uuid4().hex[:8]}",
            "values": embed_document(prototype_text),
            "metadata": {
                "label":     label_name,
                "prototype": prototype_text,
                "scope":     SCOPE_SYSTEM,
                "sub_index": i
            }
        })

    pinecone_index.upsert(vectors=vectors, namespace=GLOBAL_NAMESPACE)

    return {
        "status":            "ok",
        "label":             label_name,
        "scope":             SCOPE_SYSTEM,
        "prototypes_stored": len(vectors),
        "prototypes":        prototypes
    }


@app.delete("/system/label", summary="[Admin] Delete a system label")
async def delete_system_label(request: DeleteSystemLabelRequest):
    """
    Deletes all system-scoped vectors for the label.
    Does NOT touch any user's custom label of the same name.
    """
    label_name = request.label_name.strip()
    validation_vector = [0.01 if i % 3 == 0 else 0.0 for i in range(1024)]

    results = pinecone_index.query(
        vector=validation_vector,
        top_k=TOP_K,
        namespace=GLOBAL_NAMESPACE,
        filter={"label": {"$eq": label_name}, "scope": {"$ne": SCOPE_USER}},
        include_metadata=False
    )

    if not results.matches:
        raise HTTPException(
            status_code=404,
            detail=f"System label '{label_name}' not found."
        )

    pinecone_index.delete(
        ids=[m.id for m in results.matches],
        namespace=GLOBAL_NAMESPACE
    )

    return {"status": "ok", "label": label_name, "deleted": len(results.matches)}


# ─────────────────────────────────────────────
# Endpoints — User Labels  (per-user custom)
# ─────────────────────────────────────────────

@app.post("/user/label", summary="Add a custom label for a specific user")
async def add_user_label(request: AddUserLabelRequest):
    """
    Stores prototypes with scope="user" + user_id in metadata.
    The description shapes prototypes to match what THIS user means by the label,
    so "Finance" can mean personal budgeting for one user and investor relations for another.
    """
    user_id = request.user_id.strip()
    label_name = request.label_name.strip()
    validation_vector = [0.01 if i % 3 == 0 else 0.0 for i in range(1024)]

    existing = pinecone_index.query(
        vector=validation_vector,
        top_k=1,
        namespace=GLOBAL_NAMESPACE,
        filter={"label": {"$eq": label_name}, "scope": {
            "$eq": SCOPE_USER}, "user_id": {"$eq": user_id}},
        include_metadata=False
    )
    if existing.matches:
        raise HTTPException(
            status_code=409,
            detail=f"Label '{label_name}' already exists for user '{user_id}'."
        )

    try:
        prototypes = generate_prototypes(label_name, request.description)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prototype generation failed: {str(e)}")

    vectors = []
    for i, prototype_text in enumerate(prototypes):
        vectors.append({
            "id": f"{label_name}_user_{user_id}_{i}_{uuid.uuid4().hex[:8]}",
            "values": embed_document(prototype_text),
            "metadata": {
                "label":     label_name,
                "prototype": prototype_text,
                "scope":     SCOPE_USER,
                "user_id":   user_id,
                "sub_index": i
            }
        })

    pinecone_index.upsert(vectors=vectors, namespace=GLOBAL_NAMESPACE)

    return {
        "status":            "ok",
        "label":             label_name,
        "user_id":           user_id,
    }


@app.delete("/user/label", summary="Delete a custom label for a specific user")
async def delete_user_label(request: DeleteUserLabelRequest):
    """
    Deletes only this user's vectors for the label.
    System labels of the same name are completely unaffected.
    """
    user_id = request.user_id.strip()
    label_name = request.label_name.strip()
    validation_vector = [0.01 if i % 3 == 0 else 0.0 for i in range(1024)]

    results = pinecone_index.query(
        vector=validation_vector,
        top_k=TOP_K,
        namespace=GLOBAL_NAMESPACE,
        filter={"label": {"$eq": label_name}, "scope": {
            "$eq": SCOPE_USER}, "user_id": {"$eq": user_id}},
        include_metadata=False
    )

    if not results.matches:
        raise HTTPException(
            status_code=404,
            detail=f"Label '{label_name}' not found for user '{user_id}'."
        )

    pinecone_index.delete(
        ids=[m.id for m in results.matches],
        namespace=GLOBAL_NAMESPACE
    )

    return {"status": "ok", "label": label_name, "user_id": user_id, "deleted": len(results.matches)}


# ─────────────────────────────────────────────
# Classify
# ─────────────────────────────────────────────

@app.post("/classify", response_model=ClassifyResponse, summary="Classify an email")
async def classify_email(request: ClassifyRequest):
    """
    Classifies email against the user's chosen labels.

    Pipeline (executed in order):
      1. Validate labels exist (system + user tiers)
      2. Embed email → bi-encoder retrieval from Pinecone (both tiers)
      3. Structural feature extraction — zero-ML pattern matching
      4. Sender reputation — Bayesian-smoothed global cross-user signal
      5. Per-user sender affinity — this user's history with the domain
      6. Score blending — weighted combination of all signals
      7. Cross-encoder reranking — top-5 candidates rescored jointly
      8. Confidence routing → embedding result OR GPT-4o-mini fallback
      9. Feedback loop — learned examples stored scoped to label tier
    """
    if not request.labels:
        raise HTTPException(
            status_code=400, detail="labels array cannot be empty.")

    user_id = request.user_id.strip()
    label_names = [l.strip() for l in request.labels]
    validation_vector = [0.01 if i % 3 == 0 else 0.0 for i in range(1024)]

    # ── Step 1: Validate each label exists in at least one tier ──
    missing = []
    user_scoped_labels: set[str] = set()   # labels that are user-defined for this user

    def check_label(label: str) -> tuple[str, bool, str]:
        """
        Returns (label, exists, scope) where scope is SCOPE_SYSTEM or SCOPE_USER.
        System tier is checked first; if found there the label is treated as system
        even if the user also has a copy.  Only labels that exist EXCLUSIVELY in the
        user tier (i.e. not in the system tier) are marked as user-scoped — this is
        what gives them the priority boost.
        """
        # Check system tier
        in_system = pinecone_index.query(
            vector=validation_vector, top_k=1, namespace=GLOBAL_NAMESPACE,
            filter={"label": {"$eq": label}, "scope": {"$ne": SCOPE_USER}},
            include_metadata=False
        ).matches
        if in_system:
            return label, True, SCOPE_SYSTEM

        # Check user tier
        in_user = pinecone_index.query(
            vector=validation_vector, top_k=1, namespace=GLOBAL_NAMESPACE,
            filter={"label": {"$eq": label}, "scope": {
                "$eq": SCOPE_USER}, "user_id": {"$eq": user_id}},
            include_metadata=False
        ).matches
        return label, bool(in_user), SCOPE_USER if in_user else ""

    check_tasks = [asyncio.to_thread(check_label, label) for label in label_names]
    check_results = await asyncio.gather(*check_tasks)

    for label, exists, scope in check_results:
        if not exists:
            missing.append(label)
        elif scope == SCOPE_USER:
            user_scoped_labels.add(label)

    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Labels not found (neither system nor user): {missing}"
        )

    # ── Step 2: Embed the email ───────────────────────────────────
    body = request.body
    third = len(body) // 3

    body_snippet = (
        body[:200] + " ... " + body[third:third+200] + " ... " + body[-200:]
        if len(body) > 500
        else body
    )

    email_text = f"Subject: {request.subject}\nSender: {request.sender}\nBody: {body_snippet}"
    query_vector = embed_query(email_text)

    # ── Step 3: Per-label querying — equal retrieval budget for every label ──
    #
    # WHY PER-LABEL: A single global top_k pool is unfair when labels have
    # different numbers of stored vectors. "Marketing" accumulates more learned
    # examples over time (high email volume), so it would crowd out "Finance" or
    # "Action Needed" in a shared top_k=100 pool. By querying each label with
    # a fixed TOP_K_PER_LABEL budget, every label competes on a level playing field.
    #
    label_hits: dict[str, list[float]] = {label: [] for label in label_names}
    label_prototypes: dict[str, list[tuple[float, str]]] = {
        label: [] for label in label_names
    }

    def query_label(label: str, scope_filter: dict) -> list:
        """Query Pinecone for a single label with a fixed per-label budget."""
        return pinecone_index.query(
            vector=query_vector,
            top_k=TOP_K_PER_LABEL,
            namespace=GLOBAL_NAMESPACE,
            filter={**scope_filter, "label": {"$eq": label}},
            include_metadata=True
        ).matches

    # Build per-label tasks for both system and user tiers — all run concurrently
    system_filter = {"scope": {"$ne": SCOPE_USER}}
    user_filter   = {"scope": {"$eq": SCOPE_USER}, "user_id": {"$eq": user_id}}

    per_label_tasks = [
        asyncio.to_thread(query_label, lbl, scope_filter)
        for lbl in label_names
        for scope_filter in (system_filter, user_filter)
    ]
    per_label_results = await asyncio.gather(*per_label_tasks)

    # Interleave results: [lbl0_sys, lbl0_user, lbl1_sys, lbl1_user, ...]
    for i, lbl in enumerate(label_names):
        for matches in per_label_results[i * 2 : i * 2 + 2]:   # sys + user pair
            for match in matches:
                label_hits[lbl].append(match.score)
                proto = match.metadata.get("prototype", "")
                if proto:
                    label_prototypes[lbl].append((match.score, proto))

    # Top-k mean: average the best k matches per label
    embedding_scores: dict[str, float] = {}
    for label, scores in label_hits.items():
        if not scores:
            embedding_scores[label] = 0.0
        else:
            top_k_scores = sorted(scores, reverse=True)[:TOPK_MEAN_K]
            embedding_scores[label] = sum(top_k_scores) / len(top_k_scores)

    # (User-label priority boost applied post-rerank — see Step 7.5)

    # ── Step 4: Structural feature extraction ─────────────────────
    structural_signals = extract_structural_signals(
        request.subject, request.sender, request.body
    )
    structural_boost = compute_structural_boost(structural_signals, label_names)

    # Amplify structural boost for user-custom labels — the user explicitly
    # created this label AND the email has structural evidence for it.
    for lbl in user_scoped_labels:
        if lbl in structural_boost and structural_boost[lbl] > 0:
            structural_boost[lbl] = min(structural_boost[lbl] * USER_STRUCTURAL_AMPLIFIER, 1.0)

    has_structural = any(v > 0 for v in structural_boost.values())

    # ── Step 5: Sender reputation + per-user affinity ─────────────
    domain = extract_domain(request.sender)
    reputation_scores = get_sender_reputation_scores(domain, label_names)
    has_reputation = any(v > 0 for v in reputation_scores.values())

    affinity_scores = get_sender_affinity_scores(domain, user_id, label_names)
    has_affinity = any(v > 0 for v in affinity_scores.values())

    # ── Guard: protect custom labels from cross-user reputation pollution ─────
    # Reputation is built by users who may NOT have this user's custom labels.
    # Their fallback votes for system labels (e.g. "action needed") must not
    # override an explicit custom label intent (e.g. "university").
    # When the user has any user-scoped label, zero out reputation for all
    # system labels — the crowd signal is irrelevant for this user's taxonomy.
    if user_scoped_labels:
        for lbl in label_names:
            if lbl not in user_scoped_labels:
                reputation_scores[lbl] = 0.0
        has_reputation = any(v > 0 for v in reputation_scores.values())

    # ── Step 6: Blend all signals ─────────────────────────────────
    # Dynamic weight allocation: when a signal is inactive, its weight
    # is redistributed proportionally to the active signals.
    raw_weights = {"embedding": 1.0}    # always active
    if has_structural:
        raw_weights["structural"] = STRUCTURAL_BOOST_WEIGHT
    if has_reputation:
        raw_weights["reputation"] = SENDER_REPUTATION_WEIGHT
    if has_affinity:
        raw_weights["affinity"] = SENDER_AFFINITY_WEIGHT

    total_weight = sum(raw_weights.values())
    w = {k: v / total_weight for k, v in raw_weights.items()}

    label_scores: dict[str, float] = {}
    for lbl in label_names:
        score = w["embedding"] * embedding_scores[lbl]
        if has_structural:
            score += w["structural"] * structural_boost[lbl]
        if has_reputation:
            score += w["reputation"] * reputation_scores[lbl]
        if has_affinity:
            score += w["affinity"] * affinity_scores[lbl]
        label_scores[lbl] = score

    # ── Step 7: Cross-encoder reranking ───────────────────────────
    label_scores = rerank_top_labels(
        email_text, label_prototypes, label_scores
    )

    # ── Step 7.5: User-label priority boost (post-rerank) ────────
    # Applied AFTER reranking so the cross-encoder can't undo it.
    # User creating a custom label is an explicit signal of intent.
    for label in user_scoped_labels:
        if label in label_scores:
            label_scores[label] = min(
                label_scores[label] * USER_LABEL_PRIORITY_BOOST, 1.0
            )

    # ── Step 8: Rank and check confidence ────────────────────────
    sorted_labels = sorted(label_scores.items(),
                           key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_labels[0]
    second_score = sorted_labels[1][1] if len(sorted_labels) > 1 else 0.0
    margin = top_score - second_score

    # Force LLM when a user-custom label competes with a system label.
    # The embedding model can't understand user intent; the LLM can arbitrate.
    user_label_conflict = False
    if user_scoped_labels and len(sorted_labels) >= 2:
        top_lbl = sorted_labels[0][0]
        second_lbl = sorted_labels[1][0]
        # If system label won but user-label is close behind (or vice versa)
        if ((top_lbl not in user_scoped_labels and second_lbl in user_scoped_labels) or
            (top_lbl in user_scoped_labels and second_lbl not in user_scoped_labels)):
            if margin < USER_LABEL_CONFLICT_MARGIN:
                user_label_conflict = True

    use_llm = ((margin < CONFIDENCE_MARGIN) or (
        top_score < LOW_ABSOLUTE_SCORE) or user_label_conflict) and request.use_llm

    # Build method string showing which signals contributed
    method_parts = ["embedding"]
    if has_structural:
        method_parts.append("structural")
    if has_reputation:
        method_parts.append("reputation")
    if has_affinity:
        method_parts.append("affinity")
    method_parts.append("reranker")
    if user_label_conflict:
        method_parts.append("user_conflict")

    # ── Step 9: LLM fallback if needed ───────────────────────────
    if use_llm:
        llm_label, llm_confidence = llm_classify(
            subject=request.subject,
            sender=request.sender,
            body=request.body,
            label_names=label_names
        )

        # Feedback loop — scope depends on whether the label is system or user-custom.
        if llm_confidence >= LLM_CONFIDENCE_THRESHOLD:
            is_system_label = pinecone_index.query(
                vector=query_vector,
                top_k=1,
                namespace=GLOBAL_NAMESPACE,
                filter={"label": {"$eq": llm_label},
                        "scope": {"$ne": SCOPE_USER}},
                include_metadata=False
            ).matches

            feedback_scope = SCOPE_SYSTEM if is_system_label else SCOPE_USER

            store_learned_example(
                subject=request.subject,
                sender=request.sender,
                body=request.body,
                label=llm_label,
                query_vector=query_vector,
                user_id=user_id,
                scope=feedback_scope
            )
        else:
            print(
                f"⚠️  LLM confidence too low ({llm_confidence:.2f}) — skipping storage")

        update_sender_affinity(domain, llm_label, user_id, is_user_label=(llm_label in user_scoped_labels))
        return ClassifyResponse(
            label=llm_label,
            confidence=round(top_score, 4),
            margin=round(margin, 4),
            method="llm_fallback+" + "+".join(method_parts),
            all_scores={k: round(v, 4) for k, v in label_scores.items()},
            
        )

    update_sender_affinity(domain, top_label, user_id, is_user_label=(top_label in user_scoped_labels))
    return ClassifyResponse(
        label=top_label,
        confidence=round(top_score, 4),
        margin=round(margin, 4),
        method="+".join(method_parts),
        all_scores={k: round(v, 4) for k, v in label_scores.items()},
        
    )


if __name__ == "__main__":
    uvicorn.run("main:app")
