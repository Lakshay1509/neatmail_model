"""
Structural pattern definitions for zero-ML email feature extraction.

Three dictionaries are exported:
  STRUCTURAL_PATTERNS  — compiled regex patterns keyed by signal name
  SIGNAL_TO_CATEGORY   — maps signal names to broad semantic categories + confidence
  CATEGORY_KEYWORDS    — keyword stems for fuzzy-matching categories to user label names
"""

import re

# ─────────────────────────────────────────────
# Structural Patterns
# ─────────────────────────────────────────────
# Detects structural patterns in raw email text (unsubscribe links,
# currency symbols, calendar/.ics references, tracking numbers, OTP
# codes, etc.) and maps them to broad semantic categories.  These are
# then fuzzy-matched to the user's actual label names and blended as
# a lightweight prior signal — no model inference required.

STRUCTURAL_PATTERNS: dict[str, re.Pattern] = {
    "unsubscribe": re.compile(
        r'(?i)(unsubscribe|opt[\s-]?out|email[\s-]?preferences'
        r'|manage[\s-]?subscriptions?|view\s+in\s+browser|email\s+settings)'
    ),
    "currency": re.compile(
        r'[₹$€£¥]\s?\d[\d,.]*|\d[\d,.]*\s?(?:USD|INR|EUR|GBP|Rs\.?)'
    ),
    "calendar": re.compile(
        r'(?i)(\.ics|calendar\s?invite|event\s?invitation|add\s?to\s?calendar'
        r'|when:.*where:|rsvp|google\s?calendar|starts?\s?at\s?\d)'
    ),
    "tracking": re.compile(
        r'(?i)(tracking\s?(?:number|id|#|code)|shipment\s|shipped|'
        r'delivered|out\s?for\s?delivery|in\s?transit|dispatch)'
    ),
    "otp": re.compile(
        r'(?i)(otp|one[\s-]?time[\s-]?password|verification\s?code'
        r'|security\s?code|(?:code|pin)\s*[:=]\s*\d{4,8})'
    ),
    "action_required": re.compile(
        r'(?i)(action\s?required|urgent|immediate\s?attention|deadline'
        r'|due\s?(?:date|by)|expires?\s?(?:on|in|soon)|respond\s?by)'
    ),
    "social": re.compile(
        r'(?i)(liked?\s+your|commented?\s+on|mentioned?\s+you'
        r'|tagged?\s+you|new\s?follower|friend\s?request|connection\s?request)'
    ),
    "newsletter": re.compile(
        r'(?i)(weekly\s?digest|daily\s?roundup|newsletter'
        r'|read\s?more\s?articles?|top\s?stories|in\s?this\s?issue)'
    ),
}

# ─────────────────────────────────────────────
# Signal → Category mapping
# ─────────────────────────────────────────────
# Each structural signal maps to broad semantic categories with confidence

SIGNAL_TO_CATEGORY: dict[str, dict[str, float]] = {
    "unsubscribe":     {"marketing": 0.70, "newsletter": 0.50, "promotions": 0.60},
    "currency":        {"finance": 0.60, "orders": 0.30, "payments": 0.55},
    "calendar":        {"events": 0.80, "meetings": 0.65},
    "tracking":        {"orders": 0.70, "updates": 0.30, "shipping": 0.65},
    "otp":             {"security": 0.70, "alerts": 0.50, "automated": 0.40},
    "action_required": {"urgent": 0.60, "important": 0.50, "action": 0.55},
    "social":          {"social": 0.70, "notifications": 0.40},
    "newsletter":      {"newsletter": 0.60, "marketing": 0.30},
}

# ─────────────────────────────────────────────
# Category → keyword stems
# ─────────────────────────────────────────────
# Keyword stems used to fuzzy-match broad categories → user label names

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "marketing":     ["market", "promot", "advert", "deal", "offer", "sale", "discount", "promo"],
    "newsletter":    ["newsletter", "digest", "roundup", "weekly", "daily", "bulletin"],
    "finance":       ["financ", "payment", "bank", "transaction", "money", "bill", "invoice", "upi"],
    "orders":        ["order", "ship", "deliver", "package", "track", "purchase", "cart"],
    "events":        ["event", "calendar", "meeting", "invit", "rsvp", "webinar", "conference"],
    "security":      ["secur", "otp", "verif", "auth", "password", "2fa", "login"],
    "alerts":        ["alert", "automat", "notif", "system", "warning"],
    "urgent":        ["urgent", "action", "deadline", "critical", "asap"],
    "important":     ["important", "priority", "flag", "starred"],
    "social":        ["social", "facebook", "twitter", "linkedin", "instagram", "follower"],
    "updates":       ["update", "status", "change", "news", "progress"],
    "promotions":    ["promot", "deal", "offer", "discount", "coupon", "sale", "clearance"],
    "notifications": ["notif", "ping", "mention", "activity"],
    "payments":      ["payment", "pay", "invoice", "receipt", "billing"],
    "shipping":      ["ship", "deliver", "courier", "logistics", "freight"],
    "meetings":      ["meeting", "standup", "sync", "call", "huddle"],
    "automated":     ["automat", "bot", "noreply", "system", "generated"],
    "action":        ["action", "todo", "task", "assign", "require"],
}