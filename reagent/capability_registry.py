"""
reagent/capability_registry.py — Phase 6 Generic Capability Resolution

Replaces the closed keyword-match classifier (_classify_and_validate_task) with a
scored, deterministic capability resolver.  No live LLM required.

Design principles:
  - Natural paraphrases of supported tasks are accepted, not only exact fixture phrases.
  - Unsupported requests receive honest rejection explaining available capabilities.
  - Fixture-bound execution is disclosed, not hidden.
  - Ambiguous input produces a clarification question, not a silent pick.
  - All resolution is deterministic and offline.

Public surface:
  CapabilityDescriptor  — dataclass describing one capability
  CapabilityRegistry    — register() + resolve() → ResolutionResult
  ResolutionResult      — match outcome + entities + disclosure fields
  CAPABILITY_REGISTRY   — module-level singleton with the three built-in adapters
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, FrozenSet


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CapabilityDescriptor:
    """
    Describes one registered capability — a task family that has a real,
    locally-testable adapter and simulator.
    """
    name: str                           # e.g. "hotel_booking"
    label: str                          # e.g. "Hotel / Seat Booking"
    intent_verbs: FrozenSet[str]        # words that suggest this action
    entity_keywords: FrozenSet[str]     # domain-specific entity signals
    fixture_description: str            # honest description of what the fixture does
    fixture_entities: Dict[str, str]    # canonical entity values in the local fixture
    fixture_notice: str                 # human-readable notice when entity values differ
    supported_examples: List[str]       # example prompts to surface in rejection msg
    unsupported_signals: FrozenSet[str] = field(default_factory=frozenset)
    # Keywords that are in the broad category but NOT supported (e.g. "flight" under booking)


@dataclass
class ResolutionResult:
    """
    Outcome of CapabilityRegistry.resolve().

    Invariant: if matched is not None, rejection_reason is None and vice-versa.
    """
    matched: Optional[str]              # capability name if resolved, else None
    confidence: float                   # 0.0–1.0
    entities: Dict[str, str]           # extracted entity values from user text
    ambiguous: bool                     # True if 2+ capabilities scored within threshold
    clarification_question: Optional[str]  # set when ambiguous=True
    rejection_reason: Optional[str]        # set when matched is None
    fixture_override: Dict[str, str]    # canonical fixture values (may differ from entities)
    fixture_notice: Optional[str]       # disclosure when user values ≠ fixture values
    competing_capabilities: List[str]   # names of other high-scoring capabilities

    @property
    def resolved(self) -> bool:
        """True if unambiguously matched to a capability."""
        return self.matched is not None and not self.ambiguous


# ─── Registry ─────────────────────────────────────────────────────────────────

# How much higher the top score must be over the second-best to be unambiguous
_AMBIGUITY_MARGIN = 0.15
# Minimum score to be considered a match at all
_MIN_MATCH_SCORE = 0.20


class CapabilityRegistry:
    """
    Deterministic, offline capability resolver.

    Scoring formula per capability:
        verb_score   = matching intent_verbs / total intent_verbs   (weight 0.5)
        entity_score = matching entity_keywords / total entity_keywords (weight 0.5)
        raw_score    = verb_score * 0.5 + entity_score * 0.5

    Special case: if any unsupported_signal is matched and the capability-specific
    entity_keywords don't clearly confirm it, the score is zeroed and the caller
    receives a targeted rejection explaining the unsupported sub-case.
    """

    def __init__(self) -> None:
        self._descriptors: Dict[str, CapabilityDescriptor] = {}
        self._registration_order: List[str] = []

    def register(self, desc: CapabilityDescriptor) -> None:
        """Register a capability descriptor. Overwrites any prior entry with same name."""
        self._descriptors[desc.name] = desc
        if desc.name not in self._registration_order:
            self._registration_order.append(desc.name)

    def unregister(self, name: str) -> None:
        """Unregister a capability descriptor by name."""
        self._descriptors.pop(name, None)
        if name in self._registration_order:
            self._registration_order.remove(name)

    def resolve(self, raw: str) -> ResolutionResult:
        """
        Resolve a raw user request string to zero or one registered capabilities.

        Returns a ResolutionResult.  Never raises — all error conditions are
        encoded in the result fields.
        """
        if not raw or not raw.strip():
            return ResolutionResult(
                matched=None, confidence=0.0, entities={},
                ambiguous=False, clarification_question=None,
                rejection_reason="Task input is empty. Enter a request or select an example chip.",
                fixture_override={}, fixture_notice=None, competing_capabilities=[],
            )

        if len(raw) > 300:
            return ResolutionResult(
                matched=None, confidence=0.0, entities={},
                ambiguous=False, clarification_question=None,
                rejection_reason="Task input exceeds maximum length (300 characters).",
                fixture_override={}, fixture_notice=None, competing_capabilities=[],
            )

        normalized = raw.strip().lower()
        tokens: FrozenSet[str] = frozenset(re.findall(r"[a-zA-Z0-9₹$\-]+", normalized))

        scores: Dict[str, float] = {}
        unsupported_hits: Dict[str, str] = {}  # name → message

        for name in self._registration_order:
            desc = self._descriptors[name]

            # Check unsupported sub-signals first (e.g. "flight" under booking)
            for sig in desc.unsupported_signals:
                if sig in normalized:
                    # Only disqualify if capability-specific entity_keywords are NOT present
                    entity_hits = sum(1 for kw in desc.entity_keywords if kw in normalized)
                    if entity_hits == 0:
                        unsupported_hits[name] = sig
                        scores[name] = 0.0
                        break
            else:
                # Use word-boundary token matching to avoid accidental substring matches
                verb_hits = sum(1 for v in desc.intent_verbs if (v in tokens or re.search(r'\b' + re.escape(v) + r'\b', normalized)))
                entity_hits = sum(1 for kw in desc.entity_keywords if (kw in tokens or re.search(r'\b' + re.escape(kw) + r'\b', normalized)))

                # Check for strong specific domain identifiers (e.g. 14b, ord-8821, acct-3390, etc.)
                strong_id_hits = any(
                    sig in normalized for sig in ("14b", "fl-2047", "ord-", "cust-", "ref-", "txn-", "acct-", "asha", "maya")
                )

                # An execution match requires both an action verb and domain entity,
                # unless strong specific identifiers are explicitly present.
                if (verb_hits > 0 and entity_hits > 0) or strong_id_hits:
                    verb_score = min(1.0, verb_hits / 1.0) if verb_hits > 0 else 0.4
                    entity_score = min(1.0, entity_hits / 2.0) if entity_hits > 0 else 0.4
                    scores[name] = verb_score * 0.5 + entity_score * 0.5
                else:
                    scores[name] = 0.0

        if not scores:
            return self._no_match(raw)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_name, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Below minimum threshold → no match
        if top_score < _MIN_MATCH_SCORE:
            # If there was an unsupported signal hit that caused zero score, give targeted message
            if unsupported_hits:
                sig = next(iter(unsupported_hits.values()))
                return self._unsupported_signal_rejection(raw, sig, normalized)
            return self._no_match(raw)

        # Ambiguous: two capabilities scored within the margin of each other
        if (top_score - second_score) < _AMBIGUITY_MARGIN and second_score >= _MIN_MATCH_SCORE:
            competing = [name for name, score in ranked if score >= _MIN_MATCH_SCORE]
            competing_labels = [self._descriptors[n].label for n in competing]
            clarification = (
                f"Your request could match multiple task families: "
                + ", ".join(f"'{lbl}'" for lbl in competing_labels)
                + ". Please select the correct task chip or be more specific."
            )
            return ResolutionResult(
                matched=None, confidence=top_score, entities={},
                ambiguous=True, clarification_question=clarification,
                rejection_reason=None,
                fixture_override={}, fixture_notice=None,
                competing_capabilities=competing,
            )

        # Unambiguous match
        desc = self._descriptors[top_name]
        extracted = self._extract_entities(normalized, desc)

        # ── Capability-specific fixture boundary enforcement ─────────────────
        if top_name == "hotel_booking":
            if not any(k in normalized for k in ("14b", "fl-2047", "maya", "grand hotel")):
                return ResolutionResult(
                    matched=None, confidence=0.0, entities=extracted,
                    ambiguous=False, clarification_question=None,
                    rejection_reason=(
                        "Unsupported booking target. In this offline evaluation environment, only the pre-configured "
                        "Hotel Booking simulator fixture ('Book Room 14B at Grand Hotel for Guest Maya') is available. "
                        "Select the Hotel Booking example chip to use this fixture."
                    ),
                    fixture_override={}, fixture_notice=None, competing_capabilities=[],
                )

        if top_name == "wallet_transfer":
            _wallet_verbs = {"send", "transfer", "pay", "wire", "remit"}
            _has_wallet_verb = any(v in normalized for v in _wallet_verbs)
            if _has_wallet_verb and not any(c.isdigit() for c in normalized):
                return ResolutionResult(
                    matched=None, confidence=0.0, entities=extracted,
                    ambiguous=False, clarification_question=None,
                    rejection_reason=(
                        "Unsupported task: no numeric transfer amount detected. "
                        "In this sandbox environment only pre-configured fixtures are available. "
                        "Example: 'Send ₹500 to Asha for lunch'."
                    ),
                    fixture_override={}, fixture_notice=None, competing_capabilities=[],
                )
            if "to " in normalized and not any(r in normalized for r in ("asha", "acct-3390")):
                return ResolutionResult(
                    matched=None, confidence=0.0, entities=extracted,
                    ambiguous=False, clarification_question=None,
                    rejection_reason=(
                        "Unsupported recipient. In this sandbox environment, the only allowed "
                        "recipient is 'Asha' (ACCT-3390). Example: 'Send ₹500 to Asha for lunch'."
                    ),
                    fixture_override={}, fixture_notice=None, competing_capabilities=[],
                )

        fixture_notice, fixture_override = self._check_fixture_binding(extracted, desc, raw)

        return ResolutionResult(
            matched=top_name, confidence=top_score, entities=extracted,
            ambiguous=False, clarification_question=None, rejection_reason=None,
            fixture_override=fixture_override, fixture_notice=fixture_notice,
            competing_capabilities=[],
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _no_match(self, raw: str) -> ResolutionResult:
        """Generate a no-match result with an honest, capability-listing message."""
        available = []
        for name in self._registration_order:
            desc = self._descriptors[name]
            examples = desc.supported_examples[:2]
            available.append(
                f"• {desc.label}: e.g. \"{examples[0]}\"" if examples else f"• {desc.label}"
            )
        rejection = (
            f"Unsupported task: '{raw}'. "
            "ReAgent executes tasks backed by local simulators with independent verification. "
            "Available capabilities:\n" + "\n".join(available) + "\n"
            "Select an example chip or rephrase your request using one of the available task types."
        )
        return ResolutionResult(
            matched=None, confidence=0.0, entities={},
            ambiguous=False, clarification_question=None,
            rejection_reason=rejection,
            fixture_override={}, fixture_notice=None, competing_capabilities=[],
        )

    def _unsupported_signal_rejection(self, raw: str, signal: str, normalized: str) -> ResolutionResult:
        """
        Generate a targeted rejection for a keyword that is in the broad
        category but not supported (e.g. arbitrary flight booking).
        """
        # Find which capability this signal blocked
        for name in self._registration_order:
            desc = self._descriptors[name]
            if signal in desc.unsupported_signals:
                if signal in ("flight", "flights", "airline", "airlines", "plane", "airplane", "fly", "airport"):
                    rejection = (
                        "Unsupported task: Flights and arbitrary airline bookings are not supported by the local simulator. "
                        "Supported tasks are: (1) Hotel Booking (Room 14B at Grand Hotel), (2) Wallet Transfer (Send ₹500 to Asha), "
                        "and (3) Database Refund (Order ORD-8821). Select an example chip to test a supported task."
                    )
                else:
                    available_examples = "; ".join(
                        f'"{ex}"' for ex in desc.supported_examples[:2]
                    )
                    rejection = (
                        f"Unsupported task variant: '{raw}'. "
                        f"The signal '{signal}' is not backed by a local simulator. "
                        f"For {desc.label}, only these fixtures are available: {available_examples}. "
                        "Select an example chip to run a supported task."
                    )
                return ResolutionResult(
                    matched=None, confidence=0.0, entities={},
                    ambiguous=False, clarification_question=None,
                    rejection_reason=rejection,
                    fixture_override={}, fixture_notice=None, competing_capabilities=[],
                )
        return self._no_match(raw)

    @staticmethod
    def _extract_entities(normalized: str, desc: CapabilityDescriptor) -> Dict[str, str]:
        """
        Extract simple numeric and identifier entities from the normalized prompt.
        Returned dict uses generic keys; keys vary per capability family.
        """
        entities: Dict[str, str] = {}

        # Amount: ₹NNN, $NNN, or bare NNN preceded by send/transfer/pay
        amount_match = re.search(r"[₹$](\d+(?:[.,]\d+)?)", normalized)
        if not amount_match:
            amount_match = re.search(r"\b(\d+(?:\.\d+)?)\b", normalized)
        if amount_match:
            entities["amount"] = amount_match.group(1).replace(",", "")

        # Room / seat number patterns like 14B, 12C, etc.
        room_match = re.search(r"\b(\d+[A-Za-z])\b", normalized)
        if room_match:
            entities["room"] = room_match.group(1).upper()

        # Known IDs from fixture: ORD-NNNNN, CUST-NNNN, REF-NNNNN, TXN-NNNNN, BKG-NNNNN
        id_match = re.search(r"\b(ORD-\d+|CUST-\d+|REF-\d+|TXN-\d+|BKG-\d+|ACCT-\d+)\b",
                             normalized.upper())
        if id_match:
            entities["fixture_id"] = id_match.group(1)

        # Named recipient or guest: "asha", "maya"
        if "asha" in normalized:
            entities["recipient"] = "asha"
        if "maya" in normalized or "maya lin" in normalized:
            entities["guest"] = "maya"

        return entities

    @staticmethod
    def _check_fixture_binding(
        extracted: Dict[str, str],
        desc: CapabilityDescriptor,
        raw: str,
    ) -> tuple[Optional[str], Dict[str, str]]:
        """
        Compare extracted entities against fixture_entities.
        Returns (fixture_notice, fixture_override).

        If all user-specified entities match the fixture, notice is None.
        If user specified values that differ from the fixture, notice discloses this.
        """
        fixture = desc.fixture_entities
        mismatches: List[str] = []

        for key, fixture_value in fixture.items():
            user_value = extracted.get(key)
            if user_value is not None and user_value.lower() != fixture_value.lower():
                mismatches.append(f"{key}={user_value!r} (fixture uses {fixture_value!r})")

        if not mismatches:
            return None, fixture.copy()

        notice = (
            desc.fixture_notice
            + " [Note: "
            + ", ".join(mismatches)
            + "]"
        )
        return notice, fixture.copy()

    def get_all_examples(self) -> List[str]:
        """Return all supported example phrases across all registered capabilities."""
        examples = []
        for name in self._registration_order:
            examples.extend(self._descriptors[name].supported_examples)
        return examples

    def get_descriptor(self, name: str) -> Optional[CapabilityDescriptor]:
        """Return the descriptor for a registered capability name, or None."""
        return self._descriptors.get(name)

    def list_capabilities(self) -> List[str]:
        """Return registered capability names in registration order."""
        return list(self._registration_order)


# ─── Built-in capability descriptors ──────────────────────────────────────────

_HOTEL_BOOKING = CapabilityDescriptor(
    name="hotel_booking",
    label="Hotel / Seat Booking",
    intent_verbs=frozenset({
        "book", "reserve", "booking", "reservation", "check", "room",
        "stay", "hotel", "seat", "accommodation",
    }),
    entity_keywords=frozenset({
        "room", "hotel", "14b", "maya", "grand", "guest", "booking",
        "seat", "fl-2047", "bkg-003341", "grand hotel",
    }),
    fixture_description=(
        "Books Room 14B at Grand Hotel for Guest Maya Lin. "
        "Uses the local HTTP hotel simulator with isolated SQLite state."
    ),
    fixture_entities={"room": "14B", "guest": "maya"},
    fixture_notice=(
        "Running the pre-configured hotel fixture (Room 14B / Guest Maya Lin). "
        "This simulator does not support arbitrary room selection or guest names."
    ),
    supported_examples=[
        "Book Room 14B at Grand Hotel for Guest Maya",
        "Reserve room 14B for Maya",
        "Book a hotel room for Maya",
        "Reserve a seat at the Grand Hotel",
    ],
    unsupported_signals=frozenset({
        "flight", "flights", "airline", "airlines", "airplane", "airport",
        "fly", "cab", "taxi", "train", "bus",
    }),
)

_WALLET_TRANSFER = CapabilityDescriptor(
    name="wallet_transfer",
    label="Wallet Transfer",
    intent_verbs=frozenset({
        "send", "transfer", "pay", "wire", "remit", "payment", "money",
        "wallet", "debit", "credit",
    }),
    entity_keywords=frozenset({
        "asha", "₹", "500", "acct-3390", "acct-2201", "txn-779043",
        "wallet", "lunch", "transfer",
    }),
    fixture_description=(
        "Transfers ₹500 from ACCT-2201 to ACCT-3390 (Asha). "
        "Uses the DbopsStore wallet_transfer_ledger with isolated SQLite state."
    ),
    fixture_entities={"amount": "500", "recipient": "asha"},
    fixture_notice=(
        "Running the pre-configured wallet-transfer fixture (₹500 → Asha / ACCT-3390). "
        "This simulator does not support arbitrary amounts or recipients."
    ),
    supported_examples=[
        "Send ₹500 to Asha for lunch",
        "Transfer $500 to Asha",
        "Pay Asha ₹500",
        "Send 500 to Asha",
    ],
    unsupported_signals=frozenset({
        "bitcoin", "crypto", "eth", "nft", "stock",
    }),
)

_DATABASE_UPDATE = CapabilityDescriptor(
    name="database_update",
    label="Database Update / Refund",
    intent_verbs=frozenset({
        "refund", "update", "issue", "process", "credit", "reimburse",
        "rollback", "reverse", "cancel", "database",
    }),
    entity_keywords=frozenset({
        "refund", "ord-8821", "cust-1042", "ref-550291", "order", "250",
        "database", "db", "audit", "record",
    }),
    fixture_description=(
        "Issues refund REF-550291 ($250.00) for Order ORD-8821 / Customer CUST-1042. "
        "Uses the DbopsStore refunds table with isolated SQLite state."
    ),
    fixture_entities={"fixture_id": "ORD-8821", "amount": "250"},
    fixture_notice=(
        "Running the pre-configured database-refund fixture (ORD-8821 / $250.00 / CUST-1042). "
        "This simulator does not support arbitrary order IDs or refund amounts."
    ),
    supported_examples=[
        "Update Order ORD-8821: Issue $250.00 refund for CUST-1042",
        "Issue refund for order ORD-8821",
        "Refund $250 for customer CUST-1042",
        "Process refund for order 8821",
    ],
    unsupported_signals=frozenset(),
)


# ─── Module-level singleton ────────────────────────────────────────────────────

CAPABILITY_REGISTRY = CapabilityRegistry()
CAPABILITY_REGISTRY.register(_HOTEL_BOOKING)
CAPABILITY_REGISTRY.register(_WALLET_TRANSFER)
CAPABILITY_REGISTRY.register(_DATABASE_UPDATE)
