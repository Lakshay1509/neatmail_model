"""
Email Classifier API
Stack: FastAPI + Pinecone (v3) + Qwen3-Embedding-0.6B + GPT-4o-mini fallback

Label storage — single namespace, two tiers via metadata:
  • System labels  — scope field absent OR scope="system"
                     Existing DB vectors are automatically treated as system labels.
                     Shared globally, admin-managed via POST /system/label.
  • User labels    — scope="user" + user_id field
                     Per-user custom labels. Same name can mean different things
                     for different users. Managed via POST /user/label.

Classification queries both tiers separately and merges scores (max per label).
Feedback loop scope mirrors the label tier:
  • System label → learned example stored as scope="system" (improves globally for all users)
  • User label   → learned example stored as scope="user" (stays private to that user)
"""

import os
import uuid
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

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

GLOBAL_NAMESPACE = "labels"
PROTOTYPES_PER_LABEL = 5


SCOPE_SYSTEM = "system"
SCOPE_USER = "user"

# Confidence thresholds
CONFIDENCE_MARGIN        = 0.08   # send to LLM if margin below this
LOW_ABSOLUTE_SCORE       = 0.50   # top score must be strong to trust

TOP_K                    = 100    # unchanged
SIMILARITY_THRESHOLD     = 0.85   # store more examples (less strict dedup)
LLM_CONFIDENCE_THRESHOLD = 0.80   # accept slightly less confident LLM results

QUERY_INSTRUCTION = "Instruct: Given an email, identify which category it belongs to.\nQuery: "





@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, pinecone_index, openai_client

    if hasattr(app.state, "embedding_model"):
        # ── Running on Modal ──────────────────────────────────────────
        # Models already loaded by @enter() before this runs.
        # Just read from app.state — no blocking work here.
        embedding_model = app.state.embedding_model
        pinecone_index = app.state.pinecone_index
        openai_client = app.state.openai_client
        print("✅ Clients ready (restored from Modal snapshot).")
    else:
        # ── Local development fallback ────────────────────────────────
        print("Loading Qwen3 embedding model...")
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX)

        print("Connecting to OpenAI...")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

        print("✅ All clients ready.")

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
    labels_str = "\n".join(f"- {name}" for name in label_names)

    prompt = f"""
You are an email classification system.
ALLOWED CATEGORIES (case-sensitive, exact match required):
{labels_str}

CLASSIFICATION RULES (apply in order, highest priority first):
1. FINANCE/PAYMENT: If email contains transactions, payments, UPI, bank alerts, invoices, money (₹/$) → use "Finance" if available, else use "Automated alerts" as fallback
2. DOMAIN-SPECIFIC: Match sender domain to category (bank → Finance/Automated alerts, calendar → Event update)
3. SEMANTIC CONTEXT: Analyze PURPOSE, not keywords
   - Financial transactions → Finance (or Automated alerts if Finance unavailable)
   - Calendar invites → Event update
   - Marketing → Marketing
4. KEYWORD MATCHING: Use for unclear cases


Email:
Subject: {subject}
Sender: {sender}
Body: {body[:1000]}

Return ONLY a JSON object with two fields:
- "label": the category name (must be one of the options above)
- "confidence": your confidence score as a float between 0.0 and 1.0

Format: {{"label": "Marketing", "confidence": 0.95}}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    parsed = json.loads(response.choices[0].message.content)
    predicted = parsed.get("label", "").strip()
    llm_confidence = float(parsed.get("confidence", 0.0))

    # Match back to known label names (handles extra punctuation from LLM)
    for name in label_names:
        if name.lower() in predicted.lower():
            return name, llm_confidence

    return predicted, llm_confidence


def clean_email_for_storage(subject: str, sender: str, body: str) -> str:
    """
    Strip personal info before storing as a learned example.
    Keeps the pattern and intent, removes user-specific noise.
    """
    prompt = f"""
Clean the following email for use as a training example in an email classifier.
Remove: names, order numbers, specific dates, account numbers, personal details, phone numbers.
Keep: the subject pattern, intent, key phrases, general content structure.
Return ONLY the cleaned email text, no explanation.

Subject: {subject}
Sender: {sender}
Body: {body[:500]}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


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

    metadata = {
        "label":      label,
        "prototype":  cleaned_text,
        "scope":      scope,
        "source":     "llm_fallback",
        "email_hash": email_hash,
        "llm_model":  "gpt-4o-mini",
        "created_at": datetime.now(timezone.utc).isoformat()
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
        "scope":             SCOPE_USER,
        "prototypes_stored": len(vectors),
        "prototypes":        prototypes
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

    For each label, we check both tiers:
      - System tier:  scope != "user"  (catches legacy DB vectors + new scope="system" ones)
      - User tier:    scope = "user" AND user_id = <this user>

    Scores from both tiers are merged (max per label), then confidence routing decides
    whether to use embedding result or fall back to GPT-4o-mini.

    Feedback loop scope follows the label tier:
      - System label → stored as scope="system" so ALL users benefit from the learned example.
      - User label   → stored as scope="user" + user_id, stays private to this user.
    """
    if not request.labels:
        raise HTTPException(
            status_code=400, detail="labels array cannot be empty.")

    user_id = request.user_id.strip()
    label_names = [l.strip() for l in request.labels]
    validation_vector = [0.01 if i % 3 == 0 else 0.0 for i in range(1024)]

    # ── Step 1: Validate each label exists in at least one tier ──
    missing = []
    for label in label_names:
        in_system = pinecone_index.query(
            vector=validation_vector, top_k=1, namespace=GLOBAL_NAMESPACE,
            filter={"label": {"$eq": label}, "scope": {"$ne": SCOPE_USER}},
            include_metadata=False
        ).matches

        in_user = pinecone_index.query(
            vector=validation_vector, top_k=1, namespace=GLOBAL_NAMESPACE,
            filter={"label": {"$eq": label}, "scope": {
                "$eq": SCOPE_USER}, "user_id": {"$eq": user_id}},
            include_metadata=False
        ).matches

        if not in_system and not in_user:
            missing.append(label)

    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Labels not found (neither system nor user): {missing}"
        )

    # ── Step 2: Embed the email ───────────────────────────────────
    body_snippet = (
        request.body[:300] + " ... " + request.body[-200:]
        if len(request.body) > 500
        else request.body
    )

    query_vector = embed_query(
        f"Subject: {request.subject}\nSender: {request.sender}\nBody: {body_snippet}"
    )

    # ── Step 3: Query both tiers, merge max score per label ───────
    label_scores: dict[str, float] = {label: 0.0 for label in label_names}

    def query_and_merge(filter_dict: dict):
        results = pinecone_index.query(
            vector=query_vector,
            top_k=TOP_K,
            namespace=GLOBAL_NAMESPACE,
            filter={**filter_dict, "label": {"$in": label_names}},
            include_metadata=True
        )
        for match in results.matches:
            lbl = match.metadata["label"]
            if match.score > label_scores.get(lbl, 0.0):
                label_scores[lbl] = match.score

    # System tier — catches legacy vectors (no scope field) and scope="system"
    query_and_merge({"scope": {"$ne": SCOPE_USER}})

    # User tier — only this user's custom vectors
    query_and_merge({"scope": {"$eq": SCOPE_USER},
                    "user_id": {"$eq": user_id}})

    # ── Step 4: Rank and check confidence ────────────────────────
    sorted_labels = sorted(label_scores.items(),
                           key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_labels[0]
    second_score = sorted_labels[1][1] if len(sorted_labels) > 1 else 0.0
    margin = top_score - second_score

    use_llm = (margin < CONFIDENCE_MARGIN) or (
        top_score < LOW_ABSOLUTE_SCORE)

    # ── Step 5: LLM fallback if needed ───────────────────────────
    if use_llm:
        llm_label, llm_confidence = llm_classify(
            subject=request.subject,
            sender=request.sender,
            body=request.body,
            label_names=label_names          # only THIS user's requested labels, not all labels
        )

        # Feedback loop — scope depends on whether the label is system or user-custom.
        # System labels (e.g. "action needed") improve globally; user labels stay private.
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

        return ClassifyResponse(
            label=llm_label,
            confidence=round(top_score, 4),
            margin=round(margin, 4),
            method="llm_fallback",
            all_scores={k: round(v, 4) for k, v in label_scores.items()}
        )

    return ClassifyResponse(
        label=top_label,
        confidence=round(top_score, 4),
        margin=round(margin, 4),
        method="embedding",
        all_scores={k: round(v, 4) for k, v in label_scores.items()}
    )


if __name__ == "__main__":
    uvicorn.run("main:app")
