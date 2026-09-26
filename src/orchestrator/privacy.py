"""PII detection for LGPD compliance.

Two consumers in this codebase:

- A guardrail test (tests/test_privacy.py) that scans every file in
  knowledge_base/ for CPF numbers, emails, phone numbers, and card numbers,
  so a real customer record can never be committed into what is supposed to
  be generic policy content. This is the actual failure mode this module
  defends against: someone pastes a real support ticket into the knowledge
  base while writing a new policy doc, and it ships.
- Anything that needs to redact personal data before it leaves the system
  boundary (a log line, an export, a third-party integration). The
  Helicone integration in claude_client.py does NOT use this module,
  deliberately: redacting a copy of the prompt before it goes to Helicone
  would still mean personal data crossed into Helicone's systems on its way
  to Claude, since Helicone sits in the request path, not beside it.
  Excluding the body from being logged at all (Helicone-Omit-Request /
  Helicone-Omit-Response) is the correct fix for that specific leak -- see
  the module docstring there.

Every match is validated, not just pattern-matched: a CPF is checked
against its real check-digit algorithm and a card number against Luhn, so
an 11-digit order number or a made-up "4111-1111-1111-1234" placeholder
does not get flagged just for having the right shape. That validation is
what keeps this usable against real knowledge-base prose instead of
flagging every phone-number-shaped string in it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CPF_RE = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE_BR_RE = re.compile(r"(?<!\d)(?:\+?55\s?)?\(?\d{2}\)?[\s.-]?9?\d{4}[\s.-]?\d{4}(?!\d)")


@dataclass
class PIIMatch:
    kind: str  # "cpf" | "email" | "phone" | "credit_card"
    value: str
    start: int
    end: int


def _only_digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _cpf_check_digit(digits: str) -> str:
    weight = len(digits) + 1
    total = sum(int(d) * (weight - i) for i, d in enumerate(digits))
    remainder = (total * 10) % 11
    return "0" if remainder == 10 else str(remainder)


def is_valid_cpf(digits: str) -> bool:
    """Validate a CPF's two check digits. Rejects the classic false
    positive of 11 repeated digits ("000.000.000-00", "111.111.111-11", ...
    all of which are common when someone types a fake placeholder).
    """
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    d1 = _cpf_check_digit(digits[:9])
    d2 = _cpf_check_digit(digits[:9] + d1)
    return digits[9] == d1 and digits[10] == d2


def is_valid_luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return len(digits) >= 13 and total % 10 == 0


def find_pii(text: str) -> list[PIIMatch]:
    matches: list[PIIMatch] = []

    for m in _EMAIL_RE.finditer(text):
        matches.append(PIIMatch("email", m.group(), *m.span()))

    for m in _CPF_RE.finditer(text):
        digits = _only_digits(m.group())
        if is_valid_cpf(digits):
            matches.append(PIIMatch("cpf", m.group(), *m.span()))

    for m in _CARD_RE.finditer(text):
        digits = _only_digits(m.group())
        if is_valid_luhn(digits):
            matches.append(PIIMatch("credit_card", m.group(), *m.span()))

    def _overlaps_existing(start: int, end: int) -> bool:
        return any(start < other.end and end > other.start for other in matches)

    for m in _PHONE_BR_RE.finditer(text):
        start, end = m.span()
        if _overlaps_existing(start, end):
            continue
        digits = _only_digits(m.group())
        if 10 <= len(digits) <= 13:
            matches.append(PIIMatch("phone", m.group(), start, end))

    return sorted(matches, key=lambda m: m.start)


def redact_pii(text: str, placeholder_fmt: str = "[{kind}]") -> str:
    """Return `text` with every detected PII span replaced by a tag."""
    out = text
    for match in sorted(find_pii(text), key=lambda m: m.start, reverse=True):
        tag = placeholder_fmt.format(kind=match.kind.upper())
        out = out[: match.start] + tag + out[match.end :]
    return out
