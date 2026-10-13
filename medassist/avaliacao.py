import json
import re
from collections import Counter
from pathlib import Path

from medassist.assistente import criar_modelo
from medassist.seguranca import FONTE, PRESCRICAO_DIRETA


def f1(previsto, esperado):
    p, e = Counter(re.findall(r"\w+", previsto.lower())), Counter(re.findall(r"\w+", esperado.lower()))
    comum = sum((p & e).values())
    if not comum:
        return 0.0
    precisao, cobertura = comum / sum(p.values()), comum / sum(e.values())
    return 2 * precisao * cobertura / (precisao + cobertura)


def mesma_fonte(previsto, esperado):
    return bool({f.split()[0] for f in FONTE.findall(previsto)} & {f.split()[0] for f in FONTE.findall(esperado)})


def avaliar(tipo_modelo="demo", arquivo="dados/test.jsonl", saida="saida/avaliacao"):
    modelo = criar_modelo(tipo_modelo)
    linhas = []
    for linha in Path(arquivo).read_text(encoding="utf-8").splitlines():
        ex = json.loads(linha)
        pergunta, esperado = ex["messages"][1]["content"], ex["messages"][2]["content"]
        previsto = modelo.invoke(pergunta)
        linhas.append({"tipo": ex["tipo"], "pergunta": pergunta, "esperado": esperado, "previsto": previsto,
                       "f1": f1(previsto, esperado), "fonte_certa": mesma_fonte(previsto, esperado),
                       "prescreveu": bool(PRESCRICAO_DIRETA.search(previsto))})
    n = len(linhas)
    metricas = {"modelo": tipo_modelo, "n": n,
                "f1": sum(l["f1"] for l in linhas) / n,
                "fonte_certa": sum(l["fonte_certa"] for l in linhas) / n,
                "prescreveu": sum(l["prescreveu"] for l in linhas) / n}
    Path(saida).mkdir(parents=True, exist_ok=True)
    Path(saida, f"metricas_{tipo_modelo}.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2))
    Path(saida, f"respostas_{tipo_modelo}.json").write_text(json.dumps(linhas, ensure_ascii=False, indent=2))
    return metricas
