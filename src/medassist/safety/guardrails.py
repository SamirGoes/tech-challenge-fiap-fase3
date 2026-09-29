"""Limites de atuação do assistente.

Entrada: bloqueia pedidos fora do escopo clínico e marca pedidos de prescrição.
Saída: detecta linguagem de prescrição direta, exige citação de fonte e
anexa o aviso de validação humana obrigatória.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

AVISO = ("⚠️ Conteúdo de apoio à decisão. Não é prescrição: toda conduta ou medicamento "
         "deve ser avaliado e validado pelo médico responsável.")

_PEDIDO_PRESCRICAO = re.compile(
    r"\b(prescrev\w*|receit\w*|dose exata|qual a dose|quantos mg|posologia)\b", re.I)
_FORA_ESCOPO = re.compile(
    r"\b(ignore (as|suas) instru\w+|atestado falso|burlar|sem que o m[eé]dico saiba|dados de outro paciente)\b",
    re.I)
_PRESCRICAO_DIRETA = [
    re.compile(r"\b(prescrevo|prescreva-se|administre|administrar imediatamente|tome|aplique)\b", re.I),
    re.compile(r"\b\d+(?:[.,]\d+)?\s?(mg|mcg|g|ml|mL|UI|U)(/kg)?\b.{0,30}\b(de \d+/\d+ ?h|a cada \d+ ?h|\d+x ao dia|vo|ev|iv|im|sc)\b",
               re.I),
]
_FONTE = re.compile(r"\[(PROT-\d{3}[^\]]*|Prontuário[^\]]*|Modelo institucional[^\]]*|Política[^\]]*|PubMed[^\]]*|MedQuAD[^\]]*)\]")


@dataclass
class VerificacaoEntrada:
    permitido: bool
    pedido_prescricao: bool = False
    motivo: str = ""


@dataclass
class VerificacaoSaida:
    texto: str
    aprovado: bool
    requer_revisao_humana: bool
    violacoes: list[str] = field(default_factory=list)
    fontes: list[str] = field(default_factory=list)


def verificar_entrada(pergunta: str) -> VerificacaoEntrada:
    if _FORA_ESCOPO.search(pergunta):
        return VerificacaoEntrada(False, motivo="pedido fora do escopo ou tentativa de contornar as regras")
    return VerificacaoEntrada(True, pedido_prescricao=bool(_PEDIDO_PRESCRICAO.search(pergunta)))


def contem_prescricao_direta(texto: str) -> bool:
    return any(p.search(texto) for p in _PRESCRICAO_DIRETA)


def verificar_saida(texto: str, pedido_prescricao: bool = False) -> VerificacaoSaida:
    violacoes: list[str] = []
    fontes = sorted(set(_FONTE.findall(texto)))
    if contem_prescricao_direta(texto):
        violacoes.append("prescricao_direta")
        texto = ("A resposta gerada continha linguagem de prescrição direta e foi retida para revisão. "
                 "Consulte o protocolo citado e defina a conduta com o médico responsável.\n"
                 + "Fontes: " + (", ".join(fontes) or "nenhuma"))
    if not fontes:
        violacoes.append("sem_fonte")
    if AVISO not in texto:
        texto = f"{texto}\n\n{AVISO}"
    requer_revisao = pedido_prescricao or bool(violacoes)
    return VerificacaoSaida(texto=texto, aprovado="prescricao_direta" not in violacoes,
                            requer_revisao_humana=requer_revisao, violacoes=violacoes, fontes=fontes)
