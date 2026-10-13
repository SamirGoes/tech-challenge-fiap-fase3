import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG = Path("logs/auditoria.jsonl")
AVISO = "Apoio à decisão, não é prescrição: toda conduta deve ser validada pelo médico responsável."

PEDIDO_PRESCRICAO = re.compile(r"\b(prescrev\w*|receit\w*|dose exata|qual a dose|posologia)\b", re.I)
FORA_DO_ESCOPO = re.compile(r"\b(ignore (as|suas) instru\w+|atestado falso|burlar)\b", re.I)
PRESCRICAO_DIRETA = re.compile(
    r"\b(prescrevo|administre|tome|aplique)\b|\b\d+(?:[.,]\d+)?\s?(mg|mcg|g|ml|UI)\b.{0,30}\b(a cada \d+ ?h|\d+/\d+ ?h|\dx ao dia)",
    re.I)
FONTE = re.compile(r"\[((?:PROT-\d{3}|Prontuário|Política|LAUDO|RECEITA)[^\]]*)\]")


def novo_id():
    return uuid.uuid4().hex[:12]


def registrar(trace_id, evento, **dados):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    linha = {"ts": datetime.now(timezone.utc).isoformat(), "trace_id": trace_id, "evento": evento, **dados}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(linha, ensure_ascii=False, default=str) + "\n")


def ler_log(trace_id=None):
    if not LOG.exists():
        return []
    linhas = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l]
    return [l for l in linhas if trace_id in (None, l["trace_id"])]


def verificar_pergunta(pergunta):
    return {"permitida": not FORA_DO_ESCOPO.search(pergunta),
            "pede_prescricao": bool(PEDIDO_PRESCRICAO.search(pergunta))}


def verificar_resposta(texto, pede_prescricao=False):
    fontes = sorted(set(FONTE.findall(texto)))
    violacoes = []
    if PRESCRICAO_DIRETA.search(texto):
        violacoes.append("prescricao_direta")
        texto = "Resposta retida: continha prescrição direta. Consulte o protocolo e o médico responsável."
    if not fontes:
        violacoes.append("sem_fonte")
    return {"texto": f"{texto}\n\n{AVISO}", "fontes": fontes, "violacoes": violacoes,
            "revisao_humana": pede_prescricao or bool(violacoes)}
