"""Linha de comando do MedAssist.

    medassist preparar                 # gera dados sintéticos, anonimiza, divide e cria o banco
    medassist perguntar "..." -p 1     # pergunta contextualizada com o prontuário do paciente 1
    medassist fluxo -p 1 --aprovar     # executa o fluxo LangGraph com revisão humana
    medassist auditoria <trace_id>     # mostra os eventos de uma interação
    medassist diagrama                 # imprime o diagrama Mermaid do fluxo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medassist.config import get_settings


def cmd_preparar(args) -> None:
    from medassist.data import preprocess, public_datasets, synthetic
    from medassist.db.prontuario import inicializar

    s = get_settings()
    paths = synthetic.salvar(s.raw_dir, seed=args.seed)
    extras = []
    if args.pubmedqa:
        extras += public_datasets.carregar_pubmedqa(Path(args.pubmedqa), args.limite)
    if args.medquad:
        extras += public_datasets.carregar_medquad(Path(args.medquad), args.limite)
    nomes = [p["nome"] for p in synthetic.PACIENTES]
    rep = preprocess.preparar(paths["instrucoes"], s.processed_dir, known_names=nomes, seed=args.seed, extras=extras)
    db = inicializar(s.db_path)
    print(json.dumps(rep.__dict__, ensure_ascii=False, indent=2))
    print(f"Dataset em {s.processed_dir} | banco de prontuários em {db}")


def cmd_perguntar(args) -> None:
    from medassist.chains.assistant import AssistenteMedico

    r = AssistenteMedico().perguntar(args.pergunta, args.paciente, usuario=args.usuario)
    print(r.resposta)
    print(f"\nFontes: {', '.join(r.fontes) or 'nenhuma'} | revisão humana: {r.requer_revisao_humana} "
          f"| trace_id: {r.trace_id}")


def cmd_fluxo(args) -> None:
    from medassist.graph.workflow import executar_fluxo

    def revisor(payload: dict) -> dict:
        print("\n=== REVISÃO HUMANA NECESSÁRIA ===")
        print(payload["sugestao"])
        if args.aprovar:
            return {"aprovado": True, "medico": args.medico, "comentario": "aprovado via CLI"}
        ok = input("\nAprovar a sugestão? [s/N] ").strip().lower() == "s"
        return {"aprovado": ok, "medico": args.medico, "comentario": input("Comentário: ")}

    estado = executar_fluxo(args.paciente, args.pergunta, revisor)
    print("\n=== SUGESTÃO ===\n" + estado.get("sugestao", ""))
    print("\n=== ALERTAS ===")
    for n in estado.get("notificacoes", []):
        print(n)
    print(f"\ntrace_id: {estado['trace_id']}")


def cmd_auditoria(args) -> None:
    from medassist.audit.logger import AuditLogger

    for ev in AuditLogger().read(args.trace_id):
        print(json.dumps(ev, ensure_ascii=False))


def cmd_diagrama(args) -> None:
    from medassist.graph.workflow import construir_fluxo

    print(construir_fluxo().get_graph().draw_mermaid())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="medassist", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preparar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pubmedqa", help="caminho para ori_pqal.json")
    p.add_argument("--medquad", help="pasta raiz do MedQuAD")
    p.add_argument("--limite", type=int, default=500)
    p.set_defaults(func=cmd_preparar)

    p = sub.add_parser("perguntar")
    p.add_argument("pergunta")
    p.add_argument("-p", "--paciente", type=int)
    p.add_argument("--usuario", default="medico_demo")
    p.set_defaults(func=cmd_perguntar)

    p = sub.add_parser("fluxo")
    p.add_argument("-p", "--paciente", type=int, required=True)
    p.add_argument("--pergunta")
    p.add_argument("--aprovar", action="store_true", help="aprova automaticamente a revisão humana (demo)")
    p.add_argument("--medico", default="Dr(a). Plantonista")
    p.set_defaults(func=cmd_fluxo)

    p = sub.add_parser("auditoria")
    p.add_argument("trace_id", nargs="?")
    p.set_defaults(func=cmd_auditoria)

    sub.add_parser("diagrama").set_defaults(func=cmd_diagrama)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
