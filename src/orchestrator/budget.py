"""Etapa 14: orcamento de custo por sessao.

`MAX_TOOL_ITERATIONS` (`agents.py`) limita quantas chamadas o tool-use loop
faz DENTRO de um unico turno. Nada, ate a Etapa 13, limitava o custo
ACUMULADO de uma sessao inteira ao longo de VARIOS turnos -- uma conversa
longa, ou um cliente preso repetindo a mesma duvida de jeitos diferentes,
gera chamada atras de chamada sem que nada avise ou interrompa. E o mesmo
tipo de problema que a Etapa 8 (observabilidade) resolve depois do fato,
mostrando o custo no dashboard -- aqui a ideia e agir ANTES do fato: um
teto configuravel e opcional que escalona pra humano assim que uma sessao
cruza o limite, em vez de deixar a conversa (e a fatura) continuar sem
controle.

`SessionBudget` acumula uma estimativa de custo em USD por `session_id`,
a partir dos tokens de entrada/saida que cada resposta real da API do
Claude ja carrega (`ClaudeResponse.usage`, adicionado em `claude_client.py`
pra isso). Desligado por padrao (`max_cost_usd=None`): sem configurar
nada, nenhuma sessao jamais "estoura" o orcamento, e o comportamento de
hoje (sem limite) continua identico.

A estimativa e de ordem de grandeza, nao uma fatura exata -- nao inclui
desconto de prompt caching nem os precos separados de cache
read/write, so o preco base de input/output por modelo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD por 1 milhao de tokens -- tabela publica da Anthropic (Claude Sonnet
# 4.5), valida em Jan/2026. Atualizar aqui se os precos mudarem; nao ha
# chamada de rede nenhuma envolvida em manter isso atualizado.
_PRICING_PER_MILLION_TOKENS_USD: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimativa de custo em USD para uma chamada com esses tokens. Cai no
    preco padrao (Sonnet) se o modelo nao estiver na tabela -- uma
    estimativa aproximada e melhor que nao ter guardrail nenhum."""
    pricing = _PRICING_PER_MILLION_TOKENS_USD.get(model, _DEFAULT_PRICING)
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


@dataclass
class SessionBudget:
    """Acumula custo estimado por session_id. `max_cost_usd=None` desliga o
    guardrail por completo (comportamento de hoje: sem limite)."""

    max_cost_usd: float | None = None
    model: str = "claude-sonnet-4-5"
    _spent: dict[str, float] = field(default_factory=dict)

    def record(self, session_id: str, input_tokens: int, output_tokens: int) -> float:
        """Registra o custo de uma chamada real e devolve o total
        acumulado da sessao ate agora."""
        cost = estimate_cost_usd(self.model, input_tokens, output_tokens)
        self._spent[session_id] = self._spent.get(session_id, 0.0) + cost
        return self._spent[session_id]

    def spent(self, session_id: str) -> float:
        return self._spent.get(session_id, 0.0)

    def exceeded(self, session_id: str) -> bool:
        """True quando a sessao ja cruzou o teto -- o chamador deve
        escalonar pra humano em vez de gastar mais nela. Sempre False com
        o guardrail desligado (`max_cost_usd=None`)."""
        if self.max_cost_usd is None:
            return False
        return self._spent.get(session_id, 0.0) >= self.max_cost_usd

    def reset(self, session_id: str) -> None:
        self._spent.pop(session_id, None)
