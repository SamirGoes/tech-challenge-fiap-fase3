"""Fluxo automatizado por paciente, orquestrado só com LangChain (LCEL):

    exames pendentes ‖ valores críticos -> sugestão do agente -> [revisão humana] -> alertas (e-mail)

A revisão humana é uma função `revisor(payload) -> {"aprovado": bool, "medico": str, "comentario": str}`.
Os alertas de valor crítico saem sempre; a sugestão da IA só segue no alerta se não exigir revisão ou se for aprovada.
"""

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from medassist.agente import AgenteMedico
from medassist.auditoria import novo_id
from medassist.clinica import FONTE_COMUNICACAO
from medassist.config import config
from medassist.ferramentas import exames_pendentes, valores_criticos
from medassist.prontuario import buscar_paciente

PERGUNTA_PADRAO = "Quais os próximos passos para este paciente?"


def executar_fluxo(paciente_id: int, revisor=None, agente: AgenteMedico | None = None,
                   pergunta: str | None = None) -> dict:
    agente = agente or AgenteMedico()
    auditoria, trace_id = agente.auditoria, novo_id()
    entrada = {"trace_id": trace_id, "paciente_id": paciente_id}

    def iniciar(e: dict) -> dict:
        with auditoria.etapa(trace_id, "fluxo_iniciado", paciente_id=paciente_id):
            if buscar_paciente(paciente_id) is None:
                raise ValueError(f"paciente {paciente_id} não encontrado")
        return e

    def ver_pendentes(e: dict, config) -> list[str]:
        with auditoria.etapa(trace_id, "exames_pendentes") as reg:
            reg["exames"] = exames_pendentes.invoke({"paciente_id": paciente_id}, config)
        return reg["exames"]

    def ver_criticos(e: dict, config) -> list[str]:
        with auditoria.etapa(trace_id, "valores_criticos") as reg:
            reg["achados"] = valores_criticos.invoke({"paciente_id": paciente_id}, config)
        return reg["achados"]

    def sugerir(e: dict) -> dict:
        r = agente.perguntar(pergunta or PERGUNTA_PADRAO, paciente_id, usuario="fluxo_automatico", trace_id=trace_id)
        return {**e, "sugestao": r["texto"], "fontes": r["fontes"], "violacoes": r["violacoes"],
                "revisao_exigida": bool(e["criticos"]) or r["revisao_humana"]}

    def revisar(e: dict) -> dict:
        payload = {"paciente_id": paciente_id, "sugestao": e["sugestao"], "criticos": e["criticos"],
                   "violacoes": e["violacoes"]}
        with auditoria.etapa(trace_id, "revisao_humana") as reg:
            decisao = revisor(payload) if revisor else None
            status = "aguardando_revisao" if decisao is None else "aprovada" if decisao["aprovado"] else "rejeitada"
            reg.update(status_revisao=status, **(decisao or {}))
        return {**e, "status_revisao": status, "revisao": decisao}

    def sem_revisao(e: dict) -> dict:
        return {**e, "status_revisao": "nao_requer_revisao", "revisao": None}

    def emitir(e: dict) -> dict:
        c = config()
        notificacoes = []
        with auditoria.etapa(trace_id, "alertas_emitidos") as reg:
            sugestao_liberada = e["status_revisao"] in ("nao_requer_revisao", "aprovada")
            rodape = (f"Sugestão da IA ({e['status_revisao']}):\n{e['sugestao']}" if sugestao_liberada
                      else f"Sugestão da IA: {e['status_revisao']}; não liberada.")
            if e["criticos"]:
                corpo = "\n".join(f"- Valor crítico: {a} [{FONTE_COMUNICACAO}]" for a in e["criticos"])
                notificacoes += [f"[CRITICO] Paciente #{paciente_id} -> equipe médica: {a} [{FONTE_COMUNICACAO}]"
                                 for a in e["criticos"]]
                agente.notificador.enviar(trace_id, c.email_equipe_medica,
                                          f"[CRÍTICO] Paciente #{paciente_id}: comunicar em até 30 min",
                                          f"{corpo}\n\n{rodape}")
            if e["pendentes"]:
                texto = f"Exames pendentes: {', '.join(e['pendentes'])} [Prontuário #{paciente_id}]"
                notificacoes.append(f"[INFO] Paciente #{paciente_id} -> enfermagem: {texto}")
                agente.notificador.enviar(trace_id, c.email_enfermagem, f"Exames pendentes - paciente #{paciente_id}",
                                          texto)
            reg["notificacoes"] = notificacoes
        return {**e, "notificacoes": notificacoes}

    cadeia = (
        RunnableLambda(iniciar).with_config(run_name="fluxo_iniciado")
        | RunnablePassthrough.assign(pendentes=RunnableLambda(ver_pendentes).with_config(run_name="exames_pendentes"),
                                     criticos=RunnableLambda(ver_criticos).with_config(run_name="valores_criticos"))
        | RunnableLambda(sugerir).with_config(run_name="sugestao_agente")
        | RunnableBranch((lambda e: e["revisao_exigida"], RunnableLambda(revisar).with_config(run_name="revisao_humana")),
                         RunnableLambda(sem_revisao))
        | RunnableLambda(emitir).with_config(run_name="alertas")
    )
    return cadeia.invoke(entrada, {"run_name": "medassist.fluxo", "tags": ["medassist", "fluxo"],
                                   "metadata": {"trace_id": trace_id, "paciente_id": paciente_id}})
