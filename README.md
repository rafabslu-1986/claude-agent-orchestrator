# Claude Agent Orchestrator

A multi-agent orchestration framework built directly on the Claude API — no no-code layer in between. It routes an inbound message to the right specialist, grounds every policy-level answer in a real knowledge base through native tool use, and hands off to a human the moment an agent isn't confident.

This generalizes the routing/handoff pattern from a production WhatsApp + Instagram customer service system I built and run for a travel agency, rewritten here as a reusable, client-agnostic framework so the architecture itself — not one company's data — is what's on display.

## What it does

1. **Routes** an inbound message to one of three specialists (`sales`, `support`, `billing`) using a single, cheap Claude classification call. Anything it can't confidently classify goes straight to `unknown` instead of being guessed at.
2. **Answers** using a specialist agent configured with the **FPCL prompt framework** — Role, Context, Limits, Format — so every agent's behavior is defined by four short, auditable sections instead of one long free-form prompt.
3. **Grounds** policy answers (prices, refund timelines, shipping windows) in a local knowledge base via Claude's native tool use, instead of letting the model answer from assumptions.
4. **Escalates** to a human, with a logged reason, whenever a specialist calls the `escalate_to_human` tool — the same fallback the production system uses on the human-handoff step.
5. **Remembers** the conversation across turns per session, so a specialist sees the full thread, not just the latest message.

## Architecture

```
inbound message
│
▼
Router (1 Claude call, no tools)
│
├── sales / support / billing ──▶ Specialist agent
│ │
│ ┌────────────┴────────────┐
│ ▼ ▼
│ search_knowledge_base escalate_to_human
│ (BM25 over local docs) (flags for a human)
│ │
│ ▼
│ grounded final answer
│
└── unknown ──▶ escalate_to_human directly
```

Every arrow above is a real function call in this repo, not a diagram aspiration — see `src/orchestrator/orchestrator.py` for the whole pipeline in about 40 lines.

## Why BM25 instead of an embeddings API

The retrieval layer (`rag.py`) uses BM25 (lexical ranking, via the tiny `rank_bm25` package) over local markdown files, not an embeddings API. That's a scope decision, not an architectural limitation: `KnowledgeBase.search(query, top_k)` is the only interface the rest of the code depends on, so swapping in a real vector database (pgvector, Pinecone, etc.) with embeddings is a one-file change — nothing in the router, agents, or orchestrator needs to know. Keeping it lexical means the whole project installs, runs, and tests fully offline with no external API cost. It started as TF-IDF + cosine similarity; see "RAG avançado (Etapa 10)" below for why it's BM25 now, and for the honest limits of lexical retrieval that swapping the ranking algorithm alone doesn't fix.

## Key results

- **18 automated test scenarios, all passing** — routing (including the unknown-intent fallback), RAG retrieval accuracy (including the precision-gate behavior added in Etapa 10), both tools, session memory isolation, and 6 full end-to-end pipeline scenarios (grounded answer, direct answer, escalation, multi-turn memory, and the tool-loop giving up gracefully after too many iterations).
- Tests run **fully offline** against a scripted fake Claude client (`tests/conftest.py`) — no API key or network access needed to verify the logic. A separate `examples/demo_conversation.py` script is provided for a live run against the real API.
- The tool-use loop has a hard iteration cap with a graceful escalation fallback, so a specialist that can't converge on an answer degrades to a human handoff instead of hanging or erroring.

## Project layout

```
src/orchestrator/
claude_client.py thin wrapper around the Anthropic SDK (the only file that imports it)
prompts.py FPCL system-prompt builder
router.py intent classification
agents.py specialist agents + the shared tool-use loop
rag.py BM25 knowledge base retrieval
tools.py tool schemas + local tool implementations
memory.py per-session conversation state
orchestrator.py ties it all together
knowledge_base/ sample markdown docs the specialists search against
tests/ 18 scenarios, run offline against a fake client
examples/ live demo script (needs a real API key)
```

## Installation

```bash
git clone https://github.com/rafabslu-1986/claude-agent-orchestrator
cd claude-agent-orchestrator
pip install -r requirements.txt
```

## Running the tests

```bash
pytest tests/ -v
```

No API key needed — every test runs against a scripted fake Claude client.

## Running the live demo

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/demo_conversation.py
```

## Extending it

- **New specialist**: write one new `FPCLPrompt` in `agents.py`, add it to `Orchestrator.specialists`, add its name to `router.INTENTS`.
- **New tool**: add a schema to `TOOL_SCHEMAS` in `tools.py` and a handler method on `ToolExecutor`.
- **Real vector DB**: implement the same `search(query, top_k)` interface as `KnowledgeBase` and swap it in — no other file changes.
- **Persistent memory**: swap the dict in `SessionMemory` for Redis or a database table behind the same `get` / `append` / `clear` interface.

## Observabilidade: custo e latencia (Helicone)

### O problema

Rodar agentes em producao sem visibilidade de custo por conversa e latencia
por chamada e operar no escuro. O primeiro sinal de problema costuma ser a
fatura do fim do mes ou o cliente reclamando de demora, nao um alerta.

### A solucao

Instrumentacao opcional via Helicone, ativada so por uma variavel de
ambiente (`HELICONE_API_KEY`). Sem a variavel, o sistema funciona
exatamente como antes: zero mudanca de comportamento, zero dependencia
nova obrigatoria. Com ela, toda chamada ao Claude passa a ser rastreada:
prompt, resposta, tokens, custo e latencia, por conversa.

A mudanca ficou isolada em um unico arquivo, `claude_client.py`, o unico
ponto do projeto que importa o SDK da Anthropic, sem tocar em router,
agentes, RAG ou memoria.

### Stack tecnica

Helicone (proxy de observabilidade, plano Hobby gratuito) configurado via
parametros padrao do SDK, sem biblioteca adicional.

### Resultado

Testado em uma conversa real de 3 turnos (roteamento sales/billing e
escalonamento humano no caso "unknown"). As 8 chamadas subjacentes
(router, especialistas e tool use) apareceram no dashboard da Helicone com
custo e latencia individuais, sem quebrar nenhum dos 16 testes automatizados
existentes.

![Dashboard Helicone mostrando requisicoes capturadas](docs/helicone-dashboard-cropped.png)

## Evals: taxa de acerto do roteamento (Etapa 9)

### O problema

Testes automatizados provam que o codigo funciona (dado input X, a funcao Y
retorna Z), mas nao provam que o agente toma a decisao certa quando a
mensagem e ambigua, mal escrita ou em outro idioma. Sem isso, a unica forma
de saber se o roteador esta bom e observar producao e torcer.

### A solucao

Um eval set com 20 cenarios reais de conversa (`evals/scenarios.py`),
cobrindo os tres intents (`sales`, `support`, `billing`), casos que devem
escalar para humano (`unknown`) e casos de borda deliberados: intent
misturado, mensagem curta, portugues em vez de ingles, e cliente frustrado.
Um runner (`evals/run_evals.py`) manda cada cenario pro orchestrator de
verdade, contra a API real do Claude, e compara o intent e o escalonamento
retornados com o esperado.

Diferente dos 16 testes offline (que usam um cliente Claude falso e
travado), esse eval roda contra o modelo real: mede o comportamento do
sistema em producao, nao so a logica do codigo.

### Stack tecnica

Python puro (`dataclasses`), sem framework de eval. Resultado de cada
rodada salvo em JSON (`evals/results/`, ignorado no git) para comparar
rodadas ao longo do tempo.

### Resultado

20/20 cenarios passaram (100%) na primeira rodada real, incluindo os
quatro casos de borda e o cenario em portugues.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python evals/run_evals.py
```

## RAG avancado (Etapa 10)

### O problema

O roteamento (Etapa 9) mede se o agente conversa direito. Nada, ate aqui,
media se o proprio retrieval (`rag.py`) busca o pedaco certo da base de
conhecimento -- ou, mais importante, se ele confiantemente devolve um
pedaco errado para uma pergunta que a base simplesmente nao responde.
Retrieval "bom o suficiente" na demonstracao mas silenciosamente errado em
producao e como um RAG alimenta o modelo com contexto que parece relevante
e nao e, o que vira resposta inventada em vez de escalonamento.

### A solucao

Duas mudancas, medidas uma contra a outra em `evals/rag_eval.py`, offline,
sem chave de API:

Primeiro, troquei o ranking de TF-IDF + similaridade de cosseno por BM25
(via `rank_bm25`), que soma saturacao de frequencia de termo e normalizacao
por tamanho do documento -- e removeu duas dependencias pesadas
(scikit-learn, numpy) do projeto. So essa troca, medida, nao resolveu o
problema real: com um eval de 15 perguntas respondiveis + 6 perguntas
propositalmente nao respondiveis pela base (ex.: "voces aceitam quais
moedas de pagamento?", que a base nunca cobre), tanto o TF-IDF antigo
quanto o BM25 novo devolviam, com confianca, algum pedaco da base para
100% das perguntas nao respondiveis -- porque uma unica palavra em comum
("pagamento", "envio") ja basta para pontuar bem em qualquer ranking
lexical, bounded ou nao.

A correcao real foi um segundo gate, opcional e explicito:
`min_token_overlap`, que exige um numero minimo de palavras distintas (nao
apenas uma pontuacao alta) em comum entre a pergunta e o pedaco antes de
confiar nele. Ligado (`min_token_overlap=2`), a taxa de falso-retrieval cai
de 100% para 17%, ao custo de perder 2 das 15 perguntas respondiveis (Hit
Rate@3 cai de 100% para 87%). E uma troca real de precisao por cobertura,
nao uma correcao gratuita -- por isso o gate fica desligado por padrao
(comportamento de hoje preservado) e vira uma escolha explicita de quem
chama `search()`.

### Stack tecnica

`rank_bm25` (puro Python, substitui scikit-learn + numpy). Nenhuma
dependencia nova para o gate de precisao -- e so contagem de conjuntos de
tokens.

### Resultado

| Metrica | Antes (score apenas) | Depois (+ min_token_overlap=2) |
|---|---|---|
| Hit Rate@3 (15 perguntas respondiveis) | 100% | 87% |
| MRR@3 | 1.000 | 0.867 |
| False-Retrieval Rate (6 perguntas nao respondiveis) | 100% | 17% |

```bash
python evals/rag_eval.py
```

## License

MIT
