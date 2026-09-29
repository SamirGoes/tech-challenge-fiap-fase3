"""Métricas de avaliação sem dependências externas."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from medassist.safety.guardrails import contem_prescricao_direta

_FONTE = re.compile(r"\[([^\[\]]+)\]")


def _tokens(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.findall(r"\w+", texto)


def token_f1(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    comum = sum((Counter(p) & Counter(r)).values())
    if not p or not r or not comum:
        return 0.0
    prec, rec = comum / len(p), comum / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    prec, rec = lcs / len(p), lcs / len(r)
    return 0.0 if not lcs else 2 * prec * rec / (prec + rec)


def fontes(texto: str) -> set[str]:
    return {f.strip() for f in _FONTE.findall(texto) if not f.isupper()}  # ignora marcadores como [NOME]


def citou_fonte_correta(pred: str, ref: str) -> bool:
    ids_ref = {f.split()[0] for f in fontes(ref)}
    return bool(ids_ref & {f.split()[0] for f in fontes(pred)})


def avaliar(preds: list[str], refs: list[str]) -> dict[str, float]:
    n = len(refs) or 1
    return {
        "n": len(refs),
        "token_f1": sum(map(token_f1, preds, refs)) / n,
        "rouge_l": sum(map(rouge_l, preds, refs)) / n,
        "citacao_fonte_correta": sum(map(citou_fonte_correta, preds, refs)) / n,
        "taxa_prescricao_direta": sum(contem_prescricao_direta(p) for p in preds) / n,
    }
