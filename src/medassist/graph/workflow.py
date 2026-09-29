"""Fluxo automatizado de atendimento com LangGraph.

    carregar_prontuario
        ├── verificar_exames_pendentes ─┐
        └── avaliar_valores_criticos ───┴── gerar_sugestao ── validar ──┬── revisao_humana ── emitir_alertas ── FIM
                                                                        └──────────────────── emitir_alertas ── FIM

A revisão humana usa `interrupt` do LangGraph: o fluxo pausa até um médico
aprovar ou rejeitar a sugestão, e a decisão fica registrada na auditoria.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from medassist.audit.logger import AuditLogger, new_trace_id
from medassist.chains.assistant import AssistenteMedico

FONTE_CRITICOS = "PROT-006 v3.0"

# Limites de valores críticos de laboratório segundo PROT-006
_LAB_CRITICOS = {
    "potássio": lambda v: v < 2.5 or v > 6.0,
    "sódio": lambda v: v < 120 or v > 160,
    "glicemia": lambda v: v < 50 or v > 400,
    "lactato": lambda v: v > 4,
    "hemoglobina": lambda v: v < 7,
}


class EstadoFluxo(TypedDict, total=False):
    trace_id: str
    paciente_id: int
    pergunta: str
    paciente: dict
    exames_pendentes: list[str]
    alertas: Annotated[list[dict], operator.add]
    sugestao: str
    fontes: list[str]
    requer_revisao_humana: bool
    revisao: dict
    notificacoes: list[str]


def avaliar_sinais(sv: dict) -> list[str]:
    achados = []
    if sv.get("spo2") is not None and sv["spo2"] < 92:
        achados.append(f"SpO2 {sv['spo2']}%")
    if sv.get("pas") is not None and sv["pas"] < 90:
        achados.append(f"PAS {sv['pas']} mmHg (hipotensão)")
    if sv.get("pas", 0) >= 180 or sv.get("pad", 0) >= 120:
        achados.append(f"PA {sv['pas']}/{sv['pad']} mmHg (crise hipertensiva)")
    if sv.get("fc", 0) > 130:
        achados.append(f"FC {sv['fc']} bpm")
    if sv.get("fr", 0) >= 30:
        achados.append(f"FR {sv['fr']} irpm")
    if sv.get("temp", 0) >= 39:
        achados.append(f"T {sv['temp']} °C")
    return achados


def avaliar_exames(exames: list[dict]) -> list[str]:
    achados = []
    for e in exames:
        regra = _LAB_CRITICOS.get(e["nome"].lower())
        if regra and e["status"] == "resultado" and e["valor"] is not None and regra(e["valor"]):
            achados.append(f"{e['nome']} {e['valor']:g} {e['unidade']}")
    return achados


def construir_fluxo(assistente: AssistenteMedico | None = None, audit: AuditLogger | None = None,
                    checkpointer=None):
    assistente = assistente or AssistenteMedico()
    audit = audit or assistente.audit

    def carregar_prontuario(state: EstadoFluxo) -> dict:
        pac = assistente.repo.paciente(state["paciente_id"])
        if pac is None:
            raise ValueError(f"paciente {state['paciente_id']} não encontrado")
        audit.log(state["trace_id"], "fluxo_iniciado", paciente_id=state["paciente_id"])
        return {"paciente": pac}

    def verificar_exames_pendentes(state: EstadoFluxo) -> dict:
        pendentes = [e["nome"] for e in state["paciente"]["exames"] if e["status"] == "pendente"]
        audit.log(state["trace_id"], "exames_pendentes", exames=pendentes)
        alertas = [{"nivel": "info", "mensagem": f"Exames pendentes: {', '.join(pendentes)}",
                    "fonte": f"Prontuário #{state['paciente_id']}"}] if pendentes else []
        return {"exames_pendentes": pendentes, "alertas": alertas}

    def avaliar_valores_criticos(state: EstadoFluxo) -> dict:
        pac = state["paciente"]
        achados = avaliar_exames(pac["exames"]) + avaliar_sinais(pac["sinais_vitais"])
        audit.log(state["trace_id"], "valores_criticos", achados=achados)
        return {"alertas": [{"nivel": "critico", "mensagem": f"Valor crítico: {a}. Comunicar médico em até 30 min.",
                             "fonte": FONTE_CRITICOS} for a in achados]}

    def gerar_sugestao(state: EstadoFluxo) -> dict:
        pergunta = state.get("pergunta") or (
            f"Paciente com {state['paciente']['motivo_internacao']}: quais protocolos se aplicam e quais os próximos passos?")
        r = assistente.perguntar(pergunta, state["paciente_id"], usuario="fluxo_automatico",
                                 trace_id=state["trace_id"])
        return {"sugestao": r.resposta, "fontes": r.fontes, "requer_revisao_humana": r.requer_revisao_humana}

    def validar(state: EstadoFluxo) -> str:
        critico = any(a["nivel"] == "critico" for a in state.get("alertas", []))
        return "revisao_humana" if critico or state.get("requer_revisao_humana") else "emitir_alertas"

    def revisao_humana(state: EstadoFluxo) -> dict:
        decisao = interrupt({"paciente_id": state["paciente_id"], "sugestao": state["sugestao"],
                             "alertas": state.get("alertas", [])})
        audit.log(state["trace_id"], "revisao_humana", **decisao)
        return {"revisao": decisao}

    def emitir_alertas(state: EstadoFluxo) -> dict:
        notificacoes = []
        for a in sorted(state.get("alertas", []), key=lambda a: a["nivel"] != "critico"):
            destino = "equipe médica (urgente)" if a["nivel"] == "critico" else "enfermagem"
            notificacoes.append(f"[{a['nivel'].upper()}] Paciente #{state['paciente_id']} -> {destino}: "
                                f"{a['mensagem']} [{a['fonte']}]")
        rev = state.get("revisao")
        if rev and not rev.get("aprovado"):
            notificacoes.append(f"Sugestão rejeitada por {rev.get('medico')}: {rev.get('comentario', '')}")
        audit.log(state["trace_id"], "alertas_emitidos", notificacoes=notificacoes)
        return {"notificacoes": notificacoes}

    g = StateGraph(EstadoFluxo)
    for fn in (carregar_prontuario, verificar_exames_pendentes, avaliar_valores_criticos, gerar_sugestao,
               revisao_humana, emitir_alertas):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "carregar_prontuario")
    g.add_edge("carregar_prontuario", "verificar_exames_pendentes")
    g.add_edge("carregar_prontuario", "avaliar_valores_criticos")
    g.add_edge(["verificar_exames_pendentes", "avaliar_valores_criticos"], "gerar_sugestao")
    g.add_conditional_edges("gerar_sugestao", validar, ["revisao_humana", "emitir_alertas"])
    g.add_edge("revisao_humana", "emitir_alertas")
    g.add_edge("emitir_alertas", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def executar_fluxo(paciente_id: int, pergunta: str | None = None,
                   revisor: Callable[[dict], dict] | None = None, fluxo=None) -> dict[str, Any]:
    """Executa o fluxo; se pausar para revisão, chama `revisor(payload)` e retoma com a decisão."""
    fluxo = fluxo or construir_fluxo()
    trace_id = new_trace_id()
    config = {"configurable": {"thread_id": trace_id}}
    estado = fluxo.invoke({"trace_id": trace_id, "paciente_id": paciente_id, "pergunta": pergunta or "",
                           "alertas": []}, config)
    if "__interrupt__" in estado:
        payload = estado["__interrupt__"][0].value
        if revisor is None:
            return {**estado, "pendente_revisao": payload}
        estado = fluxo.invoke(Command(resume=revisor(payload)), config)
    return estado
