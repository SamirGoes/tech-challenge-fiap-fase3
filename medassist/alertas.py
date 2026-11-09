"""Envio de alertas à equipe: e-mail por SMTP quando configurado; caso contrário, caixa de saída em arquivo."""

import smtplib
from datetime import datetime
from email.message import EmailMessage

from medassist.auditoria import Auditoria
from medassist.config import config


class Notificador:
    def __init__(self, auditoria: Auditoria | None = None) -> None:
        self.auditoria = auditoria or Auditoria()

    def enviar(self, trace_id: str, destinatario: str, assunto: str, corpo: str) -> dict:
        c = config()
        if c.smtp_host:
            msg = EmailMessage()
            msg["From"], msg["To"], msg["Subject"] = c.email_origem, destinatario, assunto
            msg.set_content(corpo)
            with smtplib.SMTP(c.smtp_host, c.smtp_porta, timeout=15) as smtp:
                smtp.starttls()
                if c.smtp_usuario:
                    smtp.login(c.smtp_usuario, c.smtp_senha)
                smtp.send_message(msg)
            canal, destino = "email", destinatario
        else:
            pasta = c.pasta_logs / "outbox"
            pasta.mkdir(parents=True, exist_ok=True)
            nome = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{trace_id}_{destinatario.split('@')[0]}.txt"
            (pasta / nome).write_text(f"Para: {destinatario}\nAssunto: {assunto}\n\n{corpo}\n", encoding="utf-8")
            canal, destino = "arquivo", str(pasta / nome)
        return self.auditoria.registrar(trace_id, "alerta_enviado", canal=canal, destinatario=destinatario,
                                        assunto=assunto, destino=destino)
