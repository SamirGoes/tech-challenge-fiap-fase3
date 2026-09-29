"""Pré-processamento e curadoria do dataset de fine-tuning.

Etapas: normalização de texto -> anonimização -> filtros de qualidade ->
deduplicação -> divisão treino/validação/teste -> formato de chat.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from medassist.data.anonymize import anonymize
from medassist.data.synthetic import SYSTEM_PROMPT


@dataclass
class CurationReport:
    entrada: int = 0
    removidos_curtos: int = 0
    removidos_sem_fonte: int = 0
    removidos_duplicados: int = 0
    anonimizacoes: dict[str, int] = field(default_factory=dict)
    saida: dict[str, int] = field(default_factory=dict)


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def para_chat(exemplo: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": exemplo["pergunta"]},
            {"role": "assistant", "content": exemplo["resposta"]},
        ],
        "tipo": exemplo.get("tipo", "faq"),
        "fonte": exemplo.get("fonte", ""),
    }


def curar(exemplos: list[dict], known_names: list[str] | None = None,
          min_chars: int = 15) -> tuple[list[dict], CurationReport]:
    rep = CurationReport(entrada=len(exemplos))
    vistos: set[str] = set()
    saida: list[dict] = []
    for ex in exemplos:
        pergunta, resposta = (_anonimizar(normalizar(ex[c]), known_names, rep) for c in ("pergunta", "resposta"))
        if len(pergunta) < min_chars or len(resposta) < min_chars:
            rep.removidos_curtos += 1
            continue
        if "[" not in resposta:  # toda resposta precisa citar uma fonte
            rep.removidos_sem_fonte += 1
            continue
        chave = hashlib.sha1(pergunta.lower().encode()).hexdigest()
        if chave in vistos:
            rep.removidos_duplicados += 1
            continue
        vistos.add(chave)
        saida.append({**ex, "pergunta": pergunta, "resposta": resposta})
    return saida, rep


def _anonimizar(texto: str, known_names: list[str] | None, rep: CurationReport) -> str:
    res = anonymize(texto, known_names)
    for k, v in res.replacements.items():
        rep.anonimizacoes[k] = rep.anonimizacoes.get(k, 0) + v
    return res.text


def dividir(exemplos: list[dict], seed: int = 42, val: float = 0.1, test: float = 0.15) -> dict[str, list[dict]]:
    """Divisão estratificada por tipo. Paráfrases da mesma pergunta (mesmo `grupo`) ficam no mesmo
    split, para que o teste não contenha variações de perguntas vistas no treino."""
    rng = random.Random(seed)
    por_tipo: dict[str, dict[str, list[dict]]] = {}
    for ex in exemplos:
        grupos = por_tipo.setdefault(ex.get("tipo", "faq"), {})
        grupos.setdefault(ex.get("grupo") or ex["pergunta"], []).append(ex)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for grupos in por_tipo.values():
        chaves = sorted(grupos)
        rng.shuffle(chaves)
        n = len(chaves)
        n_test = max(1, round(n * test)) if n >= 3 else 0
        n_val = max(1, round(n * val)) if n >= 6 else 0
        for i, chave in enumerate(chaves):
            destino = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            splits[destino] += grupos[chave]
    return splits


def preparar(raw_path: Path, out_dir: Path, known_names: list[str] | None = None, seed: int = 42,
             extras: list[dict] | None = None) -> CurationReport:
    exemplos = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
    exemplos += extras or []
    curados, rep = curar(exemplos, known_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    for nome, itens in dividir(curados, seed).items():
        with (out_dir / f"{nome}.jsonl").open("w", encoding="utf-8") as fh:
            for ex in itens:
                fh.write(json.dumps(para_chat(ex), ensure_ascii=False) + "\n")
        rep.saida[nome] = len(itens)
    (out_dir / "curadoria.json").write_text(json.dumps(rep.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep
