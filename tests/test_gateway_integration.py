"""Integration tests against the live Mangaba Gateway.

Everything else in this suite runs on stubs. These tests point the framework at
a real gateway (https://mangaba.ngrok.app) serving local GGUF models, so a
regression that only shows up against an actual model still gets caught.

The gateway speaks its own multipart API rather than the OpenAI wire format, so
the adapters below bridge it to the interfaces the framework expects. They are
deliberately kept in the test file: promoting them to a first-class provider is
a separate decision.

Skipped automatically when the gateway is unreachable, so offline runs and CI
stay green::

    pytest tests/test_gateway_integration.py -v          # run them
    pytest -m "not network"                              # skip them
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import pytest

from mangaba.core.types import FinishReason, LLMResponse, TokenUsage
from mangaba.embeddings.base import BaseEmbedding

pytestmark = [pytest.mark.integration, pytest.mark.network]

GATEWAY = "https://mangaba.ngrok.app"
HEADERS = {"ngrok-skip-browser-warning": "true"}
CHAT_TIMEOUT = 180
EMBED_TIMEOUT = 90


def _gateway_is_up() -> bool:
    """Probe the gateway once so the whole module can skip cleanly."""
    try:
        import requests

        resp = requests.get(f"{GATEWAY}/health", headers=HEADERS, timeout=15)
        return resp.status_code == 200 and resp.json().get("models_backend_reachable") is True
    except Exception:
        return False


pytest.importorskip("requests", reason="the gateway adapters need requests")

if not _gateway_is_up():  # pragma: no cover - depends on the environment
    pytest.skip(
        f"Mangaba Gateway at {GATEWAY} is unreachable — skipping live tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class GatewayLLM:
    """Bridge the gateway's ``POST /chat`` to the framework's LLM interface.

    The gateway persists history per ``session_id``, so every call gets a fresh
    one: an LLM client is expected to be stateless, and a shared session would
    leak one test's turns into the next.
    """

    def __init__(self, base_url: str = GATEWAY, timeout: int = CHAT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.calls = 0

    # -- the interface Agent/ReActEngine/guardrails rely on -----------------

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        text = self._chat(prompt)
        return LLMResponse(
            content=text,
            usage=TokenUsage(total_tokens=len(text.split())),
            model="mangaba-chat",
            finish_reason=FinishReason.STOP,
        )

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        return self._chat(prompt)

    def generate_with_tools(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> LLMResponse:
        # The gateway has no function-calling surface; flatten and answer as text.
        flattened = "\n\n".join(
            str(m.get("content") or "") for m in messages if m.get("content")
        )
        return self.generate(flattened)

    # -- transport ---------------------------------------------------------

    def _chat(self, prompt: str) -> str:
        import requests

        self.calls += 1
        resp = requests.post(
            f"{self.base_url}/chat",
            headers=HEADERS,
            files={
                "session_id": (None, f"test-{uuid.uuid4().hex[:12]}"),
                "message": (None, prompt),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]


class GatewayEmbedding(BaseEmbedding):
    """Bridge the gateway's ``POST /embeddings`` to ``BaseEmbedding``."""

    def __init__(self, base_url: str = GATEWAY, timeout: int = EMBED_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dimension = 768  # mangaba-embed; confirmed by test_embedding_dimensions

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        import requests

        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers=HEADERS,
            files={"text": (None, text)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._dimension = payload["dimensions"]
        return payload["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)


@pytest.fixture(scope="module")
def llm() -> GatewayLLM:
    return GatewayLLM()


@pytest.fixture(scope="module")
def embedding() -> GatewayEmbedding:
    return GatewayEmbedding()


def _agent(llm: GatewayLLM, **kwargs: Any):
    from mangaba import Agent

    kwargs.setdefault("role", "Assistente")
    kwargs.setdefault("goal", "Responder de forma direta e correta")
    kwargs.setdefault("backstory", "Você responde em português, sempre conciso.")
    return Agent(llm=llm, **kwargs)


# ---------------------------------------------------------------------------
# Gateway surface
# ---------------------------------------------------------------------------

class TestGatewaySurface:
    def test_health_lists_models(self) -> None:
        import requests

        payload = requests.get(f"{GATEWAY}/health", headers=HEADERS, timeout=15).json()
        assert payload["status"] == "ok"
        assert payload["models_backend_reachable"] is True
        assert any("chat" in m for m in payload["models_available"])

    def test_embedding_dimensions(self, embedding: GatewayEmbedding) -> None:
        vector = embedding.embed_query("Maceió é a capital de Alagoas")
        assert len(vector) == embedding.dimension
        assert all(isinstance(x, float) for x in vector[:8])

    def test_embeddings_are_deterministic(self, embedding: GatewayEmbedding) -> None:
        a = embedding.embed_query("contrato de prestação de serviços")
        b = embedding.embed_query("contrato de prestação de serviços")
        assert a == b, "the same text should embed to the same vector"


# ---------------------------------------------------------------------------
# Agent / Task / Crew against a real model
# ---------------------------------------------------------------------------

class TestAgainstRealModel:
    def test_agent_answers(self, llm: GatewayLLM) -> None:
        answer = _agent(llm).execute_task(
            "Qual é a capital de Alagoas? Responda apenas o nome da cidade."
        )
        assert "macei" in answer.lower(), answer

    def test_task_and_crew_run_end_to_end(self, llm: GatewayLLM) -> None:
        from mangaba import Crew, Process, Task

        agent = _agent(llm, role="Geógrafo", goal="Responder sobre geografia")
        task = Task(
            description="Diga em qual estado brasileiro fica a cidade de Maceió.",
            expected_output="O nome do estado",
            agent=agent,
        )
        result = Crew(agents=[agent], tasks=[task], process=Process.SEQUENTIAL).kickoff()
        assert "alagoas" in result.final_output.lower(), result.final_output

    def test_system_prompt_actually_reaches_the_model(self, llm: GatewayLLM) -> None:
        """Regression guard for the ReAct bug that dropped the system prompt.

        A tool-less agent used to send only the last message, so its role and
        backstory never arrived. If that regresses, the model will not obey an
        instruction that lives solely in the backstory.
        """
        agent = _agent(
            llm,
            role="Papagaio",
            goal="Obedecer à regra do seu histórico",
            backstory="Regra absoluta: termine toda resposta com a palavra BANANA.",
        )
        answer = agent.execute_task("Diga olá em uma frase curta.")
        assert "banana" in answer.lower(), (
            f"the backstory never reached the model — got: {answer!r}"
        )

    def test_context_injection_reaches_the_model(self, llm: GatewayLLM) -> None:
        """Injected context must survive into the prompt, not just memory."""
        answer = _agent(llm).execute_task(
            "Qual é o código do projeto? Responda apenas o código.",
            context="O código interno do projeto é XK-4417.",
        )
        assert "xk-4417" in answer.lower().replace(" ", ""), answer


# ---------------------------------------------------------------------------
# Subsystems wired to real models
# ---------------------------------------------------------------------------

class TestSubsystems:
    def test_knowledge_grounds_the_answer(
        self, llm: GatewayLLM, embedding: GatewayEmbedding
    ) -> None:
        from mangaba import Knowledge, StringKnowledgeSource

        knowledge = Knowledge(
            embedding=embedding,
            sources=[
                StringKnowledgeSource(
                    content="A política de férias da empresa concede 32 dias corridos.",
                    name="rh",
                ),
                StringKnowledgeSource(
                    content="O estacionamento fica no subsolo 2.", name="predial"
                ),
            ],
        )
        hits = knowledge.query("quantos dias de férias?")
        assert hits, "knowledge returned nothing"
        assert "32" in hits[0].content, hits[0].content

        answer = _agent(llm, knowledge=knowledge).execute_task(
            "Quantos dias de férias a empresa concede? Responda apenas o número."
        )
        assert "32" in answer, answer

    def test_memory_recall_with_real_embeddings(
        self, embedding: GatewayEmbedding
    ) -> None:
        from mangaba import Memory

        memory = Memory(embedding=embedding)
        memory.add("O cliente prefere ser chamado de Dr. Dheiver")
        memory.add("O estacionamento fica no subsolo 2")
        memory.flush()

        recall = memory.get_relevant("como chamar o cliente?", max_results=1)
        assert "dheiver" in recall.lower(), recall

    def test_llm_guardrail_judges_with_a_real_model(self, llm: GatewayLLM) -> None:
        from mangaba import LLMGuardrail
        from mangaba.core.guardrails import GuardrailValidationError

        guardrail = LLMGuardrail(
            "A resposta deve conter um número.", llm=llm, fail_open=False
        )

        # The guardrail deliberately fails closed when it cannot reach the
        # judge, so a gateway blip looks identical to a rejection. Retry only
        # that case — a verdict of "invalid" is still a hard failure.
        def approves(text: str) -> bool:
            for attempt in range(3):
                try:
                    return bool(guardrail.validate(text))
                except GuardrailValidationError as exc:
                    unreachable = "judge unavailable" in str(exc).lower()
                    if unreachable and attempt < 2:
                        continue
                    return False
            return False

        assert approves("O total foi de 42 unidades."), "judge rejected a valid answer"

        with pytest.raises(GuardrailValidationError):
            guardrail.validate("Não sei dizer.")

    def test_human_review_can_replace_the_output(self, llm: GatewayLLM) -> None:
        from mangaba import CallbackHumanInput, HumanFeedback, Task

        task = Task(
            description="Escreva uma saudação curta.",
            expected_output="Uma saudação",
            agent=_agent(llm),
            human_input=True,
            human_input_handler=CallbackHumanInput(
                lambda desc, out, role: HumanFeedback(
                    approved=True, revised_output="revisado pelo humano"
                )
            ),
        )
        assert task.execute().result == "revisado pelo humano"
        assert task.human_feedback[0].approved

    def test_trace_id_spans_the_whole_run(self, llm: GatewayLLM) -> None:
        from mangaba import Crew, EventBus, Process, Task

        seen: List[Optional[str]] = []
        handler = EventBus.register(lambda e: seen.append(e.trace_id))

        agent = _agent(llm)
        task = Task(
            description="Diga apenas: ok", expected_output="ok", agent=agent
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.SEQUENTIAL)
        crew.kickoff()

        traces = {t for t in seen if t is not None}
        assert traces == {crew.trace_id}, traces
