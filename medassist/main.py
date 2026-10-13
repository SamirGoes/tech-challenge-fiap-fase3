import argparse
import json

from medassist import seguranca


def main():
    parser = argparse.ArgumentParser(prog="medassist")
    comandos = parser.add_subparsers(dest="comando", required=True)
    comandos.add_parser("preparar")
    p = comandos.add_parser("perguntar")
    p.add_argument("pergunta")
    p.add_argument("-p", "--paciente", type=int)
    p = comandos.add_parser("fluxo")
    p.add_argument("-p", "--paciente", type=int, required=True)
    p.add_argument("--aprovar", action="store_true")
    p = comandos.add_parser("avaliar")
    p.add_argument("--modelo", default="demo", choices=["demo", "base", "treinado"])
    p = comandos.add_parser("auditoria")
    p.add_argument("trace_id", nargs="?")
    comandos.add_parser("diagrama")
    args = parser.parse_args()

    if args.comando == "preparar":
        from medassist.dados import preparar
        from medassist.prontuario import criar_banco
        print(json.dumps(preparar(), ensure_ascii=False, indent=2))
        print("Banco criado em", criar_banco())

    elif args.comando == "perguntar":
        from medassist.assistente import perguntar
        r = perguntar(args.pergunta, args.paciente)
        print(r["texto"])
        print(f"\nFontes: {', '.join(r['fontes']) or 'nenhuma'} | revisão humana: {r['revisao_humana']} | "
              f"trace_id: {r['trace_id']}")

    elif args.comando == "fluxo":
        from medassist.fluxo import executar

        def revisor(pedido):
            print("\n--- REVISÃO MÉDICA ---\n" + pedido["sugestao"])
            aprovado = args.aprovar or input("\nAprovar? [s/N] ").strip().lower() == "s"
            return {"aprovado": aprovado, "medico": "Dr(a). Plantonista"}

        estado = executar(args.paciente, revisor)
        print("\n--- SUGESTÃO ---\n" + estado["sugestao"])
        print("\n--- ALERTAS ---\n" + "\n".join(estado["notificacoes"]))
        print("\ntrace_id:", estado["trace_id"])

    elif args.comando == "avaliar":
        from medassist.avaliacao import avaliar
        print(json.dumps(avaliar(args.modelo), ensure_ascii=False, indent=2))

    elif args.comando == "auditoria":
        for linha in seguranca.ler_log(args.trace_id):
            print(json.dumps(linha, ensure_ascii=False))

    elif args.comando == "diagrama":
        from medassist.fluxo import criar_fluxo
        print(criar_fluxo().get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
