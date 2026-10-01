"""Stage 5 -- relation extraction rules (specification sections 10 and 11).

Each rule is a named, testable unit answering one question: *given this dependency
pattern, which canonical entities are related, and by what relation?* Rules never invent
an endpoint. A nominal the coreference stage did not resolve to a canonical entity
produces a recorded skip, never a guess -- which is how "She leads the entity resolution
work" correctly yields nothing while "She leads Project Aurora" yields a triple.
"""

from .base import (
    REPORTING_VERBS,
    Argument,
    RuleContext,
    build_candidate,
    confidence_for,
    find_predicates,
)
from .extraction import (
    RULES,
    ContractRenewalInferenceRule,
    ModalXcompIntegrationRule,
    ParticipialPartnershipRule,
    PassiveAgentRule,
    PossessiveContainmentRule,
    PrepositionalContainmentRule,
    RelationExtractor,
    RelationRule,
    TransitiveVerbRule,
)

__all__ = [
    "RULES",
    "REPORTING_VERBS",
    "Argument",
    "RuleContext",
    "RelationExtractor",
    "RelationRule",
    "TransitiveVerbRule",
    "PassiveAgentRule",
    "ParticipialPartnershipRule",
    "ModalXcompIntegrationRule",
    "ContractRenewalInferenceRule",
    "PrepositionalContainmentRule",
    "PossessiveContainmentRule",
    "build_candidate",
    "confidence_for",
    "find_predicates",
]
