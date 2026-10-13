import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from medassist import seguranca
from medassist.assistente import perguntar
from medassist.prontuario import buscar_paciente

LIMITES_EXAMES = {
    "potássio": lambda v: v < 2.5 or v > 6.0,
    "sódio": lambda v: v < 120 or v > 160,
    "glicemia": lambda v: v < 50 or v > 400,
    "lactato": lambda v: v > 4,
    "hemoglobina": lambda v: v < 7,
}


class Estado(TypedDict, total=False):
    trace_id: str
    paciente_id: int
    paciente: dict
    alertas: Annotated[list, operator.add]
    sugestao: str
    revisao_humana: bool
    revisao: dict
    notificacoes: list


def sinais_criticos(s):
    achados = []
    if s["spo2"] < 92:
        achados.append(f"SpO2 {s['spo2']}%")
    if s["pas"] < 90:
        achados.append(f"PAS {s['pas']} mmHg")
    if s["pas"] >= 180 or s["pad"] >= 120:
        achados.append(f"PA {s['pas']}/{s['pad']} mmHg")
    if s["fc"] > 130:
        achados.append(f"FC {s['fc']} bpm")
    if s["fr"] >= 30:
        achados.append(f"FR {s['fr']} irpm")
    if s["temp"] >= 39:
        achados.append(f"T {s['temp']} °C")
    return achados


def exames_criticos(exames):
    return [f"{e['nome']} {e['valor']:g} {e['unidade']}" for e in exames
            if e["status"] == "resultado" and e["nome"] in LIMITES_EXAMES and LIMITES_EXAMES[e["nome"]](e["valor"])]


def carregar_paciente(estado):
    seguranca.registrar(estado["trace_id"], "fluxo_iniciado", paciente_id=estado["paciente_id"])
    return {"paciente": buscar_paciente(estado["paciente_id"])}


def verificar_pendentes(estado):
    pendentes = [e["nome"] for e in estado["paciente"]["exames"] if e["status"] == "pendente"]
    if not pendentes:
        return {"alertas": []}
    return {"alertas": [{"nivel": "info", "texto": f"Exames pendentes: {', '.join(pendentes)}"}]}


def verificar_criticos(estado):
    p = estado["paciente"]
    achados = exames_criticos(p["exames"]) + sinais_criticos(p["sinais"])
    return {"alertas": [{"nivel": "critico", "texto": f"Valor crítico: {a} [PROT-006 §comunicacao]"} for a in achados]}


def gerar_sugestao(estado):
    pergunta = f"Paciente com {estado['paciente']['motivo']}: quais os próximos passos?"
    r = perguntar(pergunta, estado["paciente_id"], trace_id=estado["trace_id"], usuario="fluxo")
    return {"sugestao": r["texto"], "revisao_humana": r["revisao_humana"]}


def precisa_revisao(estado):
    critico = any(a["nivel"] == "critico" for a in estado["alertas"])
    return "revisao_medica" if critico or estado["revisao_humana"] else "emitir_alertas"


def revisao_medica(estado):
    decisao = interrupt({"paciente_id": estado["paciente_id"], "sugestao": estado["sugestao"],
                         "alertas": estado["alertas"]})
    seguranca.registrar(estado["trace_id"], "revisao_medica", **decisao)
    return {"revisao": decisao}


def emitir_alertas(estado):
    notificacoes = []
    for a in sorted(estado["alertas"], key=lambda a: a["nivel"] != "critico"):
        destino = "equipe médica" if a["nivel"] == "critico" else "enfermagem"
        notificacoes.append(f"[{a['nivel'].upper()}] Paciente #{estado['paciente_id']} para {destino}: {a['texto']}")
    revisao = estado.get("revisao")
    if revisao and not revisao.get("aprovado"):
        notificacoes.append(f"Sugestão rejeitada por {revisao.get('medico')}")
    seguranca.registrar(estado["trace_id"], "alertas_emitidos", notificacoes=notificacoes)
    return {"notificacoes": notificacoes}


def criar_fluxo():
    g = StateGraph(Estado)
    for etapa in (carregar_paciente, verificar_pendentes, verificar_criticos, gerar_sugestao, revisao_medica,
                  emitir_alertas):
        g.add_node(etapa.__name__, etapa)
    g.add_edge(START, "carregar_paciente")
    g.add_edge("carregar_paciente", "verificar_pendentes")
    g.add_edge("carregar_paciente", "verificar_criticos")
    g.add_edge(["verificar_pendentes", "verificar_criticos"], "gerar_sugestao")
    g.add_conditional_edges("gerar_sugestao", precisa_revisao, ["revisao_medica", "emitir_alertas"])
    g.add_edge("revisao_medica", "emitir_alertas")
    g.add_edge("emitir_alertas", END)
    return g.compile(checkpointer=MemorySaver())


def executar(paciente_id, revisor=None):
    fluxo = criar_fluxo()
    trace_id = seguranca.novo_id()
    config = {"configurable": {"thread_id": trace_id}}
    estado = fluxo.invoke({"trace_id": trace_id, "paciente_id": paciente_id, "alertas": []}, config)
    if "__interrupt__" in estado:
        pedido = estado["__interrupt__"][0].value
        if revisor is None:
            return {**estado, "aguardando_revisao": pedido}
        estado = fluxo.invoke(Command(resume=revisor(pedido)), config)
    return estado
