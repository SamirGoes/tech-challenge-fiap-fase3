"""Linha de comando do MedAssist: chat, pergunta única, fluxo automatizado, RAG, auditoria e avaliação."""

import argparse
import json
import sys


def mostrar(r):
    print("\n" + r["texto"])
    print(f"\nFontes: {', '.join(r['fontes']) or 'nenhuma'} | revisão humana: {r['revisao_humana']} | "
          f"violações: {', '.join(r['violacoes']) or 'nenhuma'} | trace_id: {r['trace_id']}")


def criar_assistente(modelo="rag"):
    from medassist.assistente import Assistente

    if modelo == "rag":
        return Assistente()
    from medassist.treino import criar_modelo  # fase 2: modelo com fine-tuning (GPU)

    return Assistente(criar_modelo(modelo))


def chat(modelo="rag", paciente=None):
    """Chat no console. Comandos: /paciente N (troca o paciente), /log (linha do tempo da última resposta), /sair."""
    from medassist.assistente import linha_do_tempo

    assistente = criar_assistente(modelo)
    aviso = " (sem LLM: a resposta vem dos protocolos e do prontuário)" if modelo == "rag" else ""
    print(f"MedAssist | modelo: {modelo}{aviso}\nApoio à decisão clínica: não prescreve. Comandos: /paciente N, /log, /sair")
    ultimo = None
    while True:
        try:
            pergunta = input(f"\n[paciente {paciente or '-'}] Pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pergunta in ("/sair", "sair"):
            break
        if pergunta.startswith("/paciente"):
            partes = pergunta.split()
            paciente = int(partes[1]) if len(partes) > 1 else None
        elif pergunta == "/log":
            print(linha_do_tempo(ultimo["trace_id"]) if ultimo else "Nenhuma pergunta ainda.")
        elif pergunta:
            ultimo = assistente.perguntar(pergunta, paciente, usuario="medico_chat")
            mostrar(ultimo)


def cmd_preparar(args):
    from medassist import dados
    from medassist.assistente import abrir_indice, reindexar

    extras = dados.carregar_pubmedqa(args.pubmedqa) if args.pubmedqa else []
    print(json.dumps(dados.preparar(extras=extras), ensure_ascii=False, indent=2))
    print("Banco de prontuários:", dados.criar_banco())
    print(f"Base vetorial: {reindexar(abrir_indice())} trechos indexados")


def cmd_perguntar(args):
    mostrar(criar_assistente(args.modelo).perguntar(args.pergunta, args.paciente))


def cmd_fluxo(args):
    from medassist.fluxo import executar_fluxo

    def revisor(p):
        print("\n--- REVISÃO MÉDICA NECESSÁRIA ---\n" + p["sugestao"])
        if args.aprovar:
            return {"aprovado": True, "medico": "Dr(a). Plantonista", "comentario": "aprovado via CLI"}
        ok = input("\nAprovar a sugestão? [s/N] ").strip().lower() == "s"
        return {"aprovado": ok, "medico": "Dr(a). Plantonista", "comentario": input("Comentário: ")}

    r = executar_fluxo(args.paciente, criar_assistente(args.modelo), revisor)
    print(f"\n--- SUGESTÃO ({r['status_revisao']}) ---\n{r['sugestao']}")
    print("\n--- ALERTAS ---\n" + ("\n".join(r["notificacoes"]) or "nenhum"))
    print("\ntrace_id:", r["trace_id"])


def cmd_auditoria(args):
    from medassist.assistente import ler_log, linha_do_tempo

    if args.trace_id:
        print(linha_do_tempo(args.trace_id))
    else:
        print("\n".join(dict.fromkeys(e["trace_id"] for e in ler_log())) or "Nenhum registro.")


def cmd_avaliar(args):
    from medassist.treino import avaliar

    print(json.dumps(avaliar(args.modelo), ensure_ascii=False, indent=2))


def cmd_rag(args):
    from medassist import assistente as a

    indice = a.abrir_indice()
    if args.acao == "indexar":
        print(a.reindexar(indice), "trechos indexados")
    elif args.acao == "adicionar":
        print(a.adicionar(indice, a.documentos_de_arquivo(args.alvo)), "trechos adicionados/atualizados")
    elif args.acao == "remover":
        print(a.remover(indice, args.alvo), "trechos removidos")
    elif args.acao == "listar":
        for doc, n in a.listar(indice).items():
            print(f"{doc}: {n} trechos")
    elif args.acao == "buscar":
        for r in a.buscar(indice, args.alvo):
            print(f"[{r['fonte']}] score {r['score']}: {r['texto']}")


def main():
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")
    modelo = argparse.ArgumentParser(add_help=False)
    modelo.add_argument("--modelo", default="rag", choices=["rag", "base", "treinado"],
                        help="rag (padrão, sem LLM) ou base/treinado (fase 2, GPU)")
    parser = argparse.ArgumentParser(prog="medassist", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("preparar", help="gera pacientes (CSV), dataset de treino, banco SQLite e base vetorial")
    p.add_argument("--pubmedqa", help="caminho do ori_pqal.json (opcional)")
    p.set_defaults(func=cmd_preparar)

    p = sub.add_parser("chat", parents=[modelo], help="conversa no console")
    p.add_argument("-p", "--paciente", type=int)
    p.set_defaults(func=lambda a: chat(a.modelo, a.paciente))

    p = sub.add_parser("perguntar", parents=[modelo], help="uma pergunta clínica, com ou sem paciente")
    p.add_argument("pergunta")
    p.add_argument("-p", "--paciente", type=int)
    p.set_defaults(func=cmd_perguntar)

    p = sub.add_parser("fluxo", parents=[modelo], help="fluxo automatizado: pendências, críticos, revisão e alertas")
    p.add_argument("-p", "--paciente", type=int, required=True)
    p.add_argument("--aprovar", action="store_true", help="aprova a revisão automaticamente (demonstração)")
    p.set_defaults(func=cmd_fluxo)

    p = sub.add_parser("auditoria", help="linha do tempo de uma requisição (sem trace_id: lista os ids)")
    p.add_argument("trace_id", nargs="?")
    p.set_defaults(func=cmd_auditoria)

    p = sub.add_parser("avaliar", parents=[modelo], help="métricas no conjunto de teste")
    p.set_defaults(func=cmd_avaliar)

    p = sub.add_parser("rag", help="base vetorial: indexar | adicionar ARQUIVO | remover DOC | listar | buscar TEXTO")
    p.add_argument("acao", choices=["indexar", "adicionar", "remover", "listar", "buscar"])
    p.add_argument("alvo", nargs="?")
    p.set_defaults(func=cmd_rag)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
