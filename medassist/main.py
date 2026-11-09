"""Linha de comando do MedAssist (chat no terminal, fluxo, RAG, auditoria e avaliação)."""

import argparse
import json
import sys


def _configurar_saida() -> None:
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")


def _mostrar_resposta(r: dict) -> None:
    print("\n" + r["texto"])
    print(f"\nFontes: {', '.join(r['fontes']) or 'nenhuma'} | revisão humana: {r['revisao_humana']} | "
          f"violações: {', '.join(r['violacoes']) or 'nenhuma'} | trace_id: {r['trace_id']}")


def cmd_preparar(args) -> None:
    from medassist.config import config
    from medassist.dados import carregar_pubmedqa, preparar
    from medassist.prontuario import criar_banco
    from medassist.rag import IndiceProtocolos

    extras = carregar_pubmedqa(args.pubmedqa, args.limite) if args.pubmedqa else []
    print(json.dumps(preparar(extras=extras), ensure_ascii=False, indent=2))
    print("Banco de prontuários:", criar_banco())
    indice = IndiceProtocolos()
    print(f"Base vetorial: {indice.reindexar()} trechos indexados em {config().pasta_indice}")


def cmd_perguntar(args) -> None:
    from medassist.agente import AgenteMedico

    _mostrar_resposta(AgenteMedico().perguntar(args.pergunta, args.paciente, usuario=args.usuario))


def cmd_chat(args) -> None:
    from medassist.agente import AgenteMedico

    agente = AgenteMedico()
    print("MedAssist (apoio à decisão clínica; não prescreve). Digite 'sair' para encerrar.")
    paciente = args.paciente
    while True:
        try:
            pergunta = input(f"\n[paciente {paciente or '-'}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pergunta.lower() in ("sair", "exit", "quit"):
            break
        if pergunta.lower().startswith("/paciente"):
            partes = pergunta.split()
            paciente = int(partes[1]) if len(partes) > 1 else None
            continue
        if pergunta:
            _mostrar_resposta(agente.perguntar(pergunta, paciente, usuario="medico_chat"))


def cmd_fluxo(args) -> None:
    from medassist.fluxo import executar_fluxo

    def revisor(p: dict) -> dict:
        print("\n--- REVISÃO MÉDICA NECESSÁRIA ---\n" + p["sugestao"])
        if args.aprovar:
            return {"aprovado": True, "medico": args.medico, "comentario": "aprovado via CLI"}
        ok = input("\nAprovar a sugestão? [s/N] ").strip().lower() == "s"
        return {"aprovado": ok, "medico": args.medico, "comentario": input("Comentário: ")}

    r = executar_fluxo(args.paciente, revisor, pergunta=args.pergunta)
    print("\n--- SUGESTÃO ---\n" + r["sugestao"])
    print(f"\n--- REVISÃO: {r['status_revisao']} ---")
    print("\n--- ALERTAS ---\n" + ("\n".join(r["notificacoes"]) or "nenhum"))
    print("\ntrace_id:", r["trace_id"])


def cmd_auditoria(args) -> None:
    from medassist.auditoria import Auditoria

    a = Auditoria()
    if args.json:
        for e in a.ler(args.trace_id):
            print(json.dumps(e, ensure_ascii=False))
    elif args.trace_id:
        print(a.linha_do_tempo(args.trace_id))
    else:
        vistos = []
        for e in a.ler():
            if e["trace_id"] not in vistos:
                vistos.append(e["trace_id"])
        print("\n".join(vistos[-20:]) or "Nenhum registro.")


def cmd_avaliar(args) -> None:
    from medassist.avaliacao import avaliar

    print(json.dumps(avaliar(args.modelo), ensure_ascii=False, indent=2))


def cmd_rag(args) -> None:
    from medassist.rag import IndiceProtocolos

    indice = IndiceProtocolos()
    if args.acao == "indexar":
        print(f"{indice.reindexar()} trechos indexados.")
    elif args.acao == "adicionar":
        print(f"{indice.adicionar_arquivo(args.alvo)} trechos adicionados/atualizados.")
    elif args.acao == "remover":
        print(f"{indice.remover(args.alvo)} trechos removidos.")
    elif args.acao == "listar":
        for doc, n in indice.listar().items():
            print(f"{doc}: {n} trechos")
    elif args.acao == "buscar":
        indice.garantir()
        for r in indice.buscar(args.alvo):
            print(f"[{r['fonte']}] score {r['score']}: {r['texto']}")


def main() -> None:
    _configurar_saida()
    parser = argparse.ArgumentParser(prog="medassist", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("preparar", help="gera pacientes (CSV), dataset de treino, banco SQLite e índice vetorial")
    p.add_argument("--pubmedqa", help="caminho do ori_pqal.json (opcional)")
    p.add_argument("--limite", type=int, default=300)
    p.set_defaults(func=cmd_preparar)

    p = sub.add_parser("perguntar", help="uma pergunta clínica, com ou sem paciente")
    p.add_argument("pergunta")
    p.add_argument("-p", "--paciente", type=int)
    p.add_argument("--usuario", default="medico_demo")
    p.set_defaults(func=cmd_perguntar)

    p = sub.add_parser("chat", help="conversa no terminal (use /paciente N para trocar de paciente)")
    p.add_argument("-p", "--paciente", type=int)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("fluxo", help="fluxo automatizado: pendências, críticos, sugestão, revisão e alertas")
    p.add_argument("-p", "--paciente", type=int, required=True)
    p.add_argument("--pergunta")
    p.add_argument("--aprovar", action="store_true", help="aprova a revisão automaticamente (demonstração)")
    p.add_argument("--medico", default="Dr(a). Plantonista")
    p.set_defaults(func=cmd_fluxo)

    p = sub.add_parser("auditoria", help="linha do tempo de uma requisição (sem trace_id: lista as últimas)")
    p.add_argument("trace_id", nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_auditoria)

    p = sub.add_parser("avaliar", help="métricas no conjunto de teste")
    p.add_argument("--modelo", default="demo", choices=["demo", "base", "treinado"])
    p.set_defaults(func=cmd_avaliar)

    p = sub.add_parser("rag", help="gerencia a base vetorial: indexar | adicionar ARQ | remover DOC | listar | buscar TEXTO")
    p.add_argument("acao", choices=["indexar", "adicionar", "remover", "listar", "buscar"])
    p.add_argument("alvo", nargs="?")
    p.set_defaults(func=cmd_rag)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
