"""Etapa 13: resiliencia contra falhas transitorias da API do Claude.

Sem isto, um 429 (rate limit) ou um timeout momentaneo da rede vira uma
excecao que sobe direto ate quem chamou `Orchestrator.handle_message` e
derruba o turno inteiro do cliente -- mesmo quando a mesma chamada, tentada
de novo meio segundo depois, teria funcionado. Isto nao e sobre esconder
erros reais: um 400 (payload nosso esta errado) ou 401 (chave invalida)
continuam subindo na hora, porque tentar de novo um request que esta
errado nao resolve nada, so atrasa o erro de verdade.

Duas pecas, compostas, nunca substituindo `ClaudeClient`:

- `is_transient` classifica a excecao: rate limit, timeout, erro de conexao
  e 5xx sao transitorios (vale a pena tentar de novo); qualquer outra coisa
  (400, 401, 404, ...) nao e -- e sobe imediatamente, sem retry.
- `CircuitBreaker` para de tentar de vez quando as falhas se acumulam (ex.:
  a API inteira esta fora do ar): depois de `failure_threshold` falhas
  transitorias consecutivas, `before_call()` passa a levantar
  `CircuitOpenError` na hora, sem nem tentar a chamada -- poupa o cliente
  de esperar o timeout de rede de novo a cada mensagem enquanto o servico
  esta fora do ar. Depois de `recovery_timeout` segundos, libera uma
  tentativa de teste (half-open): sucesso fecha o circuito novamente,
  falha reabre.

`ResilientClaudeClient` combina os dois atras da mesma interface
`.send(...)` que `ClaudeClient` (e o `FakeClaudeClient` dos testes) ja
usam. E composicao (decorator), nunca uma subclasse -- nenhum outro modulo
(router, agents, orchestrator) precisa saber que ela existe; e por isso
que ela funciona com qualquer objeto que tenha um metodo `.send()` igual.

Nao e thread-safe (o projeto inteiro e sincrono, uma chamada por vez) --
uso concorrente do mesmo CircuitBreaker exigiria um lock em `before_call`
e `record_*`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import anthropic


@dataclass
class RetryConfig:
    """max_attempts conta a tentativa original; max_attempts=3 significa
    a chamada original + ate 2 retries. base_delay/max_delay sao segundos;
    o delay real e `min(base_delay * 2**(tentativa-1), max_delay)`, com
    jitter uniforme em [0, delay] quando jitter=True (evita que varias
    chamadas que falharam juntas tentem de novo no mesmo instante -- o
    "thundering herd").
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: bool = True


def is_transient(exc: BaseException) -> bool:
    """True para erros que valem retry: rate limit, timeout, erro de
    conexao e qualquer 5xx. False para tudo que um retry identico nao
    resolve -- 4xx (request nosso esta errado) e qualquer excecao que nao
    seja da API do Claude.
    """
    if isinstance(
        exc,
        (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return 500 <= (exc.status_code or 0) < 600
    return False


class CircuitOpenError(RuntimeError):
    """Levantado no lugar da chamada de verdade enquanto o circuito esta
    aberto -- falha rapido, sem esperar rede."""


class _State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._state = _State.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        """'closed' | 'open' | 'half_open'. half_open e derivado (nao
        armazenado): o circuito abriu, e ja se passou recovery_timeout
        desde entao, entao a proxima chamada pode tentar de novo."""
        if self._state is _State.OPEN and self._opened_at is not None:
            if self._clock() - self._opened_at >= self.recovery_timeout:
                return _State.HALF_OPEN.value
        return self._state.value

    def before_call(self) -> None:
        if self.state == _State.OPEN.value:
            raise CircuitOpenError(
                f"circuit open after {self._consecutive_failures} consecutive "
                f"transient failures; try again after the {self.recovery_timeout}s "
                "cooldown"
            )

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = _State.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = _State.OPEN
            self._opened_at = self._clock()


class ResilientClaudeClient:
    """Envolve qualquer objeto com `.send(...)` (ClaudeClient real,
    FakeClaudeClient de teste, ou ate outro ResilientClaudeClient) com
    retry + circuit breaker. Uso: `ResilientClaudeClient(ClaudeClient())`,
    ou `Orchestrator(resilient=True)` para o caso comum.
    """

    def __init__(
        self,
        inner: Any,
        retry_config: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._inner = inner
        self._retry = retry_config or RetryConfig()
        self._circuit = circuit_breaker or CircuitBreaker()
        self._sleep = sleep
        self.attempts_on_last_call = 0  # inspecionavel nos testes

    def send(self, *args: Any, **kwargs: Any) -> Any:
        self._circuit.before_call()
        last_exc: BaseException | None = None

        for attempt in range(1, self._retry.max_attempts + 1):
            self.attempts_on_last_call = attempt
            try:
                result = self._inner.send(*args, **kwargs)
            except Exception as exc:
                if not is_transient(exc):
                    raise  # bug nosso -- retry nao ajuda, e nao conta pro breaker
                last_exc = exc
                self._circuit.record_failure()
                if attempt == self._retry.max_attempts:
                    raise
                delay = min(
                    self._retry.base_delay * (2 ** (attempt - 1)), self._retry.max_delay
                )
                if self._retry.jitter:
                    delay = random.uniform(0, delay)
                self._sleep(delay)
            else:
                self._circuit.record_success()
                return result

        raise last_exc  # pragma: no cover -- loop sempre retorna ou levanta acima
