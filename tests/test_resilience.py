"""Etapa 13: retry com backoff exponencial + circuit breaker.

Todos os testes sao deterministicos e offline -- nenhum `time.sleep` de
verdade acontece (o retry recebe uma funcao `sleep` fake que so registra as
chamadas) e o circuit breaker recebe um relogio fake (uma lista mutavel que
avancamos manualmente), entao "o circuito reabre depois de 30s" e testado
sem esperar 30 segundos de verdade.
"""

from __future__ import annotations

import anthropic
import pytest

from orchestrator.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    ResilientClaudeClient,
    RetryConfig,
    is_transient,
)

from conftest import FakeClaudeClient, fake_api_error, fake_connection_error, text_response


class FakeSleep:
    """Registra os delays pedidos sem realmente esperar."""

    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, delay: float) -> None:
        self.calls.append(delay)


class FakeClock:
    """Relogio controlavel manualmente, para testar o cooldown do circuit
    breaker sem esperar o tempo real passar."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# is_transient
# ---------------------------------------------------------------------------


def test_rate_limit_and_5xx_and_connection_errors_are_transient():
    assert is_transient(fake_api_error(429))
    assert is_transient(fake_api_error(500))
    assert is_transient(fake_api_error(503))
    assert is_transient(fake_connection_error())
    assert is_transient(fake_connection_error(timeout=True))


def test_4xx_and_unrelated_exceptions_are_not_transient():
    assert not is_transient(fake_api_error(400))
    assert not is_transient(fake_api_error(401))
    assert not is_transient(fake_api_error(404))
    assert not is_transient(ValueError("not an anthropic error at all"))


# ---------------------------------------------------------------------------
# ResilientClaudeClient: retry
# ---------------------------------------------------------------------------


def test_retries_a_transient_failure_and_returns_the_eventual_success():
    inner = FakeClaudeClient([fake_api_error(429), text_response("ok on the second try")])
    sleep = FakeSleep()
    client = ResilientClaudeClient(inner, retry_config=RetryConfig(max_attempts=3), sleep=sleep)

    result = client.send(system="s", messages=[{"role": "user", "content": "hi"}])

    assert result.text == "ok on the second try"
    assert client.attempts_on_last_call == 2
    assert len(sleep.calls) == 1  # one wait, between attempt 1 and attempt 2


def test_does_not_retry_a_non_transient_error():
    inner = FakeClaudeClient([fake_api_error(400), text_response("never reached")])
    sleep = FakeSleep()
    client = ResilientClaudeClient(inner, sleep=sleep)

    with pytest.raises(anthropic.BadRequestError):
        client.send(system="s", messages=[])

    assert client.attempts_on_last_call == 1
    assert sleep.calls == []  # a 400 is our bug -- retrying it wastes an attempt for nothing


def test_gives_up_after_max_attempts_and_raises_the_last_transient_error():
    always_fails = FakeClaudeClient([fake_api_error(503)] * 5)
    sleep = FakeSleep()
    client = ResilientClaudeClient(
        always_fails, retry_config=RetryConfig(max_attempts=3), sleep=sleep
    )

    with pytest.raises(anthropic.InternalServerError):
        client.send(system="s", messages=[])

    assert client.attempts_on_last_call == 3
    assert len(sleep.calls) == 2  # waited between 1->2 and 2->3, gave up after 3


def test_backoff_delay_grows_exponentially_and_respects_the_cap():
    always_fails = FakeClaudeClient([fake_api_error(503)] * 5)
    sleep = FakeSleep()
    config = RetryConfig(max_attempts=4, base_delay=1.0, max_delay=3.0, jitter=False)
    client = ResilientClaudeClient(always_fails, retry_config=config, sleep=sleep)

    with pytest.raises(anthropic.InternalServerError):
        client.send(system="s", messages=[])

    # 1.0, 2.0, then capped at max_delay=3.0 instead of 4.0
    assert sleep.calls == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


def test_circuit_stays_closed_below_the_failure_threshold():
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()

    breaker.before_call()  # does not raise: only 2 of 3 failures so far
    assert breaker.state == "closed"


def test_circuit_opens_and_fails_fast_after_the_threshold():
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_circuit_half_opens_after_the_recovery_timeout_and_closes_on_success():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0, clock=clock)

    breaker.record_failure()
    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        breaker.before_call()

    clock.advance(29.0)
    assert breaker.state == "open"  # not yet -- one second short of the cooldown

    clock.advance(1.0)
    assert breaker.state == "half_open"
    breaker.before_call()  # half-open lets the trial call through

    breaker.record_success()
    assert breaker.state == "closed"


def test_circuit_reopens_if_the_half_open_trial_call_also_fails():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, clock=clock)

    breaker.record_failure()
    clock.advance(10.0)
    assert breaker.state == "half_open"

    breaker.record_failure()  # the trial call failed too
    assert breaker.state == "open"


# ---------------------------------------------------------------------------
# ResilientClaudeClient + CircuitBreaker wired together
# ---------------------------------------------------------------------------


def test_resilient_client_opens_the_circuit_across_separate_calls_then_fails_fast():
    """Three separate .send() calls (not retries within one call) each fail
    once; the third failure crosses failure_threshold=3, so a fourth call
    never even reaches the inner client.
    """
    inner = FakeClaudeClient([fake_api_error(503)] * 3)
    client = ResilientClaudeClient(
        inner,
        retry_config=RetryConfig(max_attempts=1),  # no retries: isolate the breaker's own count
        circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_timeout=30.0),
        sleep=FakeSleep(),
    )

    for _ in range(3):
        with pytest.raises(anthropic.InternalServerError):
            client.send(system="s", messages=[])

    calls_before = len(inner.calls)
    with pytest.raises(CircuitOpenError):
        client.send(system="s", messages=[])

    assert len(inner.calls) == calls_before  # the 4th call never reached the inner client
