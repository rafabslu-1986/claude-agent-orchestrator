"""Etapa 11 (LGPD): PII detection, and the knowledge-base guardrail it
exists for.
"""

from pathlib import Path

from orchestrator.privacy import find_pii, is_valid_cpf, redact_pii


def test_detects_a_real_email():
    matches = find_pii("please reach me at joao.silva@example.com about this")
    assert [m.kind for m in matches] == ["email"]


def test_detects_a_valid_cpf_and_rejects_common_fakes():
    assert is_valid_cpf("11144477735")  # a commonly-used valid test CPF
    assert not is_valid_cpf("11111111111")  # 11 repeated digits, a frequent placeholder
    assert not is_valid_cpf("00000000000")

    matches = find_pii("meu CPF e 111.444.777-35, pode confirmar meus dados?")
    assert [m.kind for m in matches] == ["cpf"]

    matches = find_pii("cpf de teste invalido: 111.111.111-11")
    assert matches == []


def test_detects_a_valid_card_number_via_luhn_and_rejects_a_fake_one():
    matches = find_pii("o cartao 4532015112830366 foi recusado")
    assert [m.kind for m in matches] == ["credit_card"]

    matches = find_pii("cartao de exemplo 4111111111111234")
    assert matches == []


def test_detects_a_brazilian_phone_number():
    matches = find_pii("pode me ligar no (65) 99123-4567 por favor")
    assert [m.kind for m in matches] == ["phone"]


def test_does_not_flag_ordinary_business_text():
    ordinary = [
        "o pedido 482910 ainda nao chegou",
        "a sessao eval-refund_timing rodou ok",
        "o plano Growth custa 15% menos no anual",
        "ligar as 14h30 ou 09h00 amanha",
    ]
    for text in ordinary:
        assert find_pii(text) == [], text


def test_redact_pii_replaces_every_match_with_a_tag():
    redacted = redact_pii("Meu CPF e 111.444.777-35 e meu email e joao@example.com")
    assert redacted == "Meu CPF e [CPF] e meu email e [EMAIL]"


def test_knowledge_base_contains_no_pii():
    """Guardrail: knowledge_base/ is meant to hold generic policy prose, never
    a real customer record. If this ever fails, someone pasted something
    they shouldn't have into a doc that ships in this public repo.
    """
    kb_dir = Path(__file__).parent.parent / "src" / "orchestrator" / "knowledge_base"
    offenders = []
    for path in sorted(kb_dir.glob("*.md")):
        matches = find_pii(path.read_text(encoding="utf-8"))
        if matches:
            offenders.append((path.name, matches))

    assert offenders == [], f"PII found in knowledge base files: {offenders}"
