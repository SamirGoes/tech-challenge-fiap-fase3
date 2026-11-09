"""Agente médico: pipeline LangChain (LCEL) de ponta a ponta.

    entrada (guardrail) -> prontuário (SQL) -> RAG (base vetorial) -> LLM fine-tuned -> guardrail de saída -> saída

Cada etapa é um Runnable nomeado (aparece assim no LangSmith) e grava um evento na auditoria com o mesmo trace_id.
Se a entrada for bloqueada, o pipeline pula direto para a saída.
"""

from langchain_core.runnables import RunnableBranch, RunnableLambda

from medassist.alertas import Notificador
from medassist.auditoria import Auditoria, novo_id
from medassist.config import ativar_langsmith
from medassist.guardrails import verificar_entrada, verificar_resposta
from medassist.llm import criar_modelo
from medassist.prompt import SEM_PACIENTE, formatar_protocolos, montar_prompt
from medassist.prontuario import buscar_paciente, resumo_clinico
from medassist.rag import IndiceProtocolos, RetrieverProtocolos


class AgenteMedico:
    def __init__(self, llm=None, indice=None, auditoria=None, notificador=None, k: int = 3) -> None:
        ativar_langsmith()
        self.llm = llm or criar_modelo()
        if indice is None:
            indice = IndiceProtocolos()
            indice.garantir()
        self.indice = indice
        self.retriever = RetrieverProtocolos(indice=indice, k=k)
        self.auditoria = auditoria or Auditoria()
        self.notificador = notificador or Notificador(self.auditoria)
        self.cadeia = self._montar_cadeia()

    # --- etapas -----------------------------------------------------------------------------------------
    def _entrada(self, e: dict) -> dict:
        with self.auditoria.etapa(e["trace_id"], "entrada", usuario=e["usuario"], paciente_id=e["paciente_id"]) as reg:
            r = verificar_entrada(e["pergunta"])
            reg.update(pergunta=r["pergunta"], pii_removida=r["pii"], permitida=r["permitida"],
                       pede_prescricao=r["pede_prescricao"], motivo_bloqueio=r["motivo"])
        return {**e, "pergunta": r["pergunta"], "permitida": r["permitida"], "pede_prescricao": r["pede_prescricao"],
                "motivo_bloqueio": r["motivo"]}

    def _bloqueada(self, e: dict) -> dict:
        return {**e, "texto": f"Pedido fora do escopo do assistente: {e['motivo_bloqueio']}.", "fontes": [],
                "violacoes": ["fora_do_escopo"], "revisao_humana": True}

    def _prontuario(self, e: dict) -> dict:
        with self.auditoria.etapa(e["trace_id"], "prontuario", paciente_id=e["paciente_id"]) as reg:
            if e["paciente_id"] is None:
                reg["consultado"] = False
                return {**e, "paciente": SEM_PACIENTE, "paciente_ref": "-", "motivo": ""}
            p = buscar_paciente(e["paciente_id"])
            if p is None:
                raise ValueError(f"paciente {e['paciente_id']} não encontrado")
            reg.update(consultado=True, exames_pendentes=[x["nome"] for x in p["exames"] if x["status"] == "pendente"])
        return {**e, "paciente": resumo_clinico(p), "paciente_ref": e["paciente_id"], "motivo": p["motivo"]}

    def _rag(self, e: dict, config) -> dict:
        with self.auditoria.etapa(e["trace_id"], "rag") as reg:
            consulta = f"{e['pergunta']} {e['motivo']}".strip()
            docs = self.retriever.invoke(consulta, config=config)
            reg.update(consulta=consulta, fontes=[{"fonte": d.metadata["fonte"], "score": d.metadata["score"]}
                                                  for d in docs])
        return {**e, "protocolos": formatar_protocolos([(d.metadata["fonte"], d.page_content) for d in docs])}

    def _llm(self, e: dict, config) -> dict:
        prompt = montar_prompt(e["protocolos"], e["paciente"], e["pergunta"], e["paciente_ref"])
        with self.auditoria.etapa(e["trace_id"], "llm", modelo=self.llm._llm_type) as reg:
            bruta = self.llm.invoke(prompt, config=config)
            reg.update(prompt=prompt, resposta_bruta=bruta)
        return {**e, "bruta": bruta}

    def _guardrail_saida(self, e: dict) -> dict:
        with self.auditoria.etapa(e["trace_id"], "guardrail_saida") as reg:
            r = verificar_resposta(e["bruta"], e["pede_prescricao"])
            reg.update(violacoes=r["violacoes"], revisao_humana=r["revisao_humana"], fontes=r["fontes"])
        return {**e, **r}

    def _saida(self, e: dict) -> dict:
        resultado = {"trace_id": e["trace_id"], "texto": e["texto"], "fontes": e["fontes"],
                     "violacoes": e["violacoes"], "revisao_humana": e["revisao_humana"],
                     "bloqueado": "fora_do_escopo" in e["violacoes"]}
        with self.auditoria.etapa(e["trace_id"], "saida") as reg:
            reg.update(resposta_final=resultado["texto"], fontes=resultado["fontes"],
                       violacoes=resultado["violacoes"], revisao_humana=resultado["revisao_humana"])
        return resultado

    # --- composição ---------------------------------------------------------------------------------------
    def _montar_cadeia(self):
        def etapa(fn, nome):
            return RunnableLambda(fn).with_config(run_name=nome)

        saida = etapa(self._saida, "saida")
        gerar = (etapa(self._prontuario, "prontuario") | etapa(self._rag, "rag_protocolos")
                 | etapa(self._llm, "llm") | etapa(self._guardrail_saida, "guardrail_saida") | saida)
        bloquear = etapa(self._bloqueada, "bloqueio") | saida
        return etapa(self._entrada, "entrada") | RunnableBranch((lambda e: not e["permitida"], bloquear), gerar)

    def perguntar(self, pergunta: str, paciente_id: int | None = None, usuario: str = "medico",
                  trace_id: str | None = None) -> dict:
        trace_id = trace_id or novo_id()
        config = {"run_name": "medassist.perguntar", "tags": ["medassist"],
                  "metadata": {"trace_id": trace_id, "paciente_id": paciente_id}}
        return self.cadeia.invoke({"trace_id": trace_id, "pergunta": pergunta, "paciente_id": paciente_id,
                                   "usuario": usuario}, config)
