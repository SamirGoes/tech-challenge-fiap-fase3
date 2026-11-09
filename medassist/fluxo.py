"""Fluxo automatizado por paciente, orquestrado com LangChain:

    exames pendentes ‖ valores críticos -> sugestão do assistente -> revisão humana (se preciso) -> alertas

Os alertas de valor crítico saem sempre. A sugestão da IA só vai junto se não exigir revisão ou se o médico aprovar.
"""

import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from medassist import dados
from medassist.assistente import novo_id, registrar
from medassist.dados import FONTE_COMUNICACAO, buscar_paciente, exames_pendentes, valores_criticos

EMAIL_EQUIPE = os.environ.get("MEDASSIST_EMAIL_EQUIPE_MEDICA", "equipe.medica@hospital.exemplo")
EMAIL_ENFERMAGEM = os.environ.get("MEDASSIST_EMAIL_ENFERMAGEM", "enfermagem@hospital.exemplo")


def enviar_alerta(trace_id, destino, assunto, corpo):
    """Envia e-mail se MEDASSIST_SMTP_HOST estiver definido; senão grava em logs/alertas.txt (e-mail simulado)."""
    host = os.environ.get("MEDASSIST_SMTP_HOST")
    if host:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = os.environ.get("MEDASSIST_EMAIL_ORIGEM", "medassist@hospital.exemplo"), destino, assunto
        msg.set_content(corpo)
        with smtplib.SMTP(host, int(os.environ.get("MEDASSIST_SMTP_PORTA", "587")), timeout=15) as smtp:
            smtp.starttls()
            if os.environ.get("MEDASSIST_SMTP_USUARIO"):
                smtp.login(os.environ["MEDASSIST_SMTP_USUARIO"], os.environ.get("MEDASSIST_SMTP_SENHA", ""))
            smtp.send_message(msg)
        canal = "email"
    else:
        dados.LOGS.mkdir(parents=True, exist_ok=True)
        with open(dados.LOGS / "alertas.txt", "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now():%d/%m/%Y %H:%M:%S} | trace {trace_id}\nPara: {destino}\nAssunto: {assunto}\n{corpo}\n\n")
        canal = "arquivo logs/alertas.txt"
    registrar(trace_id, "alerta_enviado", canal=canal, destino=destino, assunto=assunto)


def executar_fluxo(paciente_id, assistente, revisor=None, pergunta=None):
    """revisor(payload) -> {"aprovado": bool, "medico": str, "comentario": str}. Sem revisor, a sugestão fica aguardando."""
    trace_id = novo_id()

    def iniciar(e):
        if buscar_paciente(paciente_id) is None:
            raise ValueError(f"paciente {paciente_id} não encontrado")
        registrar(trace_id, "fluxo_iniciado", paciente_id=paciente_id)
        return e

    def ver_pendentes(e):
        pendentes = exames_pendentes(buscar_paciente(paciente_id))
        registrar(trace_id, "exames_pendentes", exames=pendentes)
        return pendentes

    def ver_criticos(e):
        achados = valores_criticos(buscar_paciente(paciente_id))
        registrar(trace_id, "valores_criticos", achados=achados)
        return achados

    def sugerir(e):
        r = assistente.perguntar(pergunta or "Quais os próximos passos para este paciente?", paciente_id,
                                 usuario="fluxo_automatico", trace_id=trace_id)
        return {**e, "sugestao": r["texto"], "fontes": r["fontes"], "exige_revisao": bool(e["criticos"]) or r["revisao_humana"]}

    def revisar(e):
        if not e["exige_revisao"]:
            return {**e, "status_revisao": "nao_requer_revisao", "revisao": None}
        decisao = revisor({"paciente_id": paciente_id, "sugestao": e["sugestao"], "criticos": e["criticos"]}) if revisor else None
        status = "aguardando_revisao" if decisao is None else "aprovada" if decisao["aprovado"] else "rejeitada"
        registrar(trace_id, "revisao_humana", status=status, **(decisao or {}))
        return {**e, "status_revisao": status, "revisao": decisao}

    def alertar(e):
        liberada = e["status_revisao"] in ("nao_requer_revisao", "aprovada")
        rodape = f"Sugestão da IA ({e['status_revisao']}):\n{e['sugestao']}" if liberada else \
            f"Sugestão da IA: {e['status_revisao']}; não liberada."
        notificacoes = []
        if e["criticos"]:
            notificacoes += [f"[CRITICO] Paciente #{paciente_id} -> equipe médica: {a} [{FONTE_COMUNICACAO}]" for a in e["criticos"]]
            enviar_alerta(trace_id, EMAIL_EQUIPE, f"[CRÍTICO] Paciente #{paciente_id}: comunicar em até 30 min",
                          "\n".join(f"- Valor crítico: {a} [{FONTE_COMUNICACAO}]" for a in e["criticos"]) + f"\n\n{rodape}")
        if e["pendentes"]:
            texto = f"Exames pendentes: {', '.join(e['pendentes'])} [Prontuário #{paciente_id}]"
            notificacoes.append(f"[INFO] Paciente #{paciente_id} -> enfermagem: {texto}")
            enviar_alerta(trace_id, EMAIL_ENFERMAGEM, f"Exames pendentes - paciente #{paciente_id}", texto)
        registrar(trace_id, "alertas_emitidos", notificacoes=notificacoes)
        return {**e, "notificacoes": notificacoes}

    cadeia = (RunnableLambda(iniciar)
              | RunnablePassthrough.assign(pendentes=RunnableLambda(ver_pendentes), criticos=RunnableLambda(ver_criticos))
              | RunnableLambda(sugerir) | RunnableLambda(revisar) | RunnableLambda(alertar))
    return cadeia.invoke({"trace_id": trace_id, "paciente_id": paciente_id},
                         {"run_name": "medassist.fluxo", "metadata": {"trace_id": trace_id, "paciente_id": paciente_id}})
