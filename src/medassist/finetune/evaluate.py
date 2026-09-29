"""Avalia um modelo no conjunto de teste e salva métricas e predições.

    python -m medassist.finetune.evaluate --backend hf            # modelo fine-tuned
    python -m medassist.finetune.evaluate --backend hf --no-adapter  # modelo base (baseline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medassist.config import get_settings
from medassist.finetune.metrics import avaliar
from medassist.llm.factory import get_llm


def main(argv: list[str] | None = None) -> dict:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=s.llm_backend)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--test", type=Path, default=s.processed_dir / "test.jsonl")
    ap.add_argument("--out", type=Path, default=Path("outputs") / "avaliacao")
    args = ap.parse_args(argv)

    llm = get_llm(args.backend, use_adapter=not args.no_adapter)
    exemplos = [json.loads(line) for line in args.test.read_text(encoding="utf-8").splitlines() if line]
    preds, refs, linhas = [], [], []
    for ex in exemplos:
        _, user, ref = (m["content"] for m in ex["messages"])
        pred = llm.invoke(user)  # o backend já aplica o prompt de sistema do treino
        preds.append(pred)
        refs.append(ref)
        linhas.append({"tipo": ex["tipo"], "pergunta": user, "referencia": ref, "predicao": pred})

    nome = f"{args.backend}{'-base' if args.no_adapter else ''}"
    metricas = {"modelo": nome, **avaliar(preds, refs)}
    por_tipo = {}
    for tipo in sorted({lin["tipo"] for lin in linhas}):
        sel = [i for i, lin in enumerate(linhas) if lin["tipo"] == tipo]
        por_tipo[tipo] = avaliar([preds[i] for i in sel], [refs[i] for i in sel])
    metricas["por_tipo"] = por_tipo

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"metricas_{nome}.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2))
    with (args.out / f"predicoes_{nome}.jsonl").open("w", encoding="utf-8") as fh:
        for lin in linhas:
            fh.write(json.dumps(lin, ensure_ascii=False) + "\n")
    print(json.dumps(metricas, ensure_ascii=False, indent=2))
    return metricas


if __name__ == "__main__":
    main()
