import json

from medassist import dados, prontuario
from medassist.anonimizacao import anonimizar
from medassist.config import config
from medassist.protocolos import carregar_protocolos


def _linhas(nome):
    return [json.loads(l) for l in (config().pasta_dados / f"{nome}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_anonimizacao():
    texto, contagem = anonimizar("Paciente: Maria Souza, CPF 123.456.789-09, tel (11) 91234-5678, em 12/08/2026.")
    for dado in ["Maria", "123.456.789-09", "91234-5678", "12/08/2026"]:
        assert dado not in texto
    assert contagem["CPF"] == 1


def test_dataset_sem_dados_pessoais_e_no_formato_do_assistente():
    nomes = [p["nome"] for p in dados.gerar_pacientes()]
    todos = [ex for n in ("train", "val", "test") for ex in _linhas(n)]
    texto = json.dumps(todos, ensure_ascii=False)
    assert not any(n in texto for n in nomes)
    assert "[CPF]" in texto and "[NOME]" in texto
    exemplo = next(ex for ex in todos if ex["tipo"] == "faq")
    assert [m["role"] for m in exemplo["messages"]] == ["system", "user", "assistant"]
    assert "### Protocolos" in exemplo["messages"][1]["content"]


def test_dataset_nao_usa_o_termo_diagnostico():
    for n in ("train", "val", "test"):
        assert "diagnóstic" not in (config().pasta_dados / f"{n}.jsonl").read_text(encoding="utf-8").lower()
    assert all("diagnóstic" not in str(p).lower() for p in carregar_protocolos().values())


def test_split_nao_vaza_perguntas_do_mesmo_grupo():
    splits = dados.dividir(dados.gerar_exemplos(dados.gerar_pacientes()))
    grupos = {n: {(e["tipo"], e["grupo"]) for e in itens} for n, itens in splits.items()}
    assert not grupos["train"] & grupos["test"] and not grupos["val"] & grupos["test"]
    assert all(len(itens) > 0 for itens in splits.values())


def test_cinquenta_pacientes_no_csv_e_no_banco():
    assert len(dados.gerar_pacientes()) == 50
    assert len(prontuario.listar_pacientes()) == 50
    p = prontuario.buscar_paciente(1)
    assert p["motivo"].startswith("febre") and any(e["status"] == "pendente" for e in p["exames"])
    assert prontuario.buscar_paciente(999) is None


def test_todo_quadro_clinico_aponta_para_o_protocolo_esperado():
    protocolos = carregar_protocolos()
    esperado = {"tosse": "PROT-005", "febre": "PROT-001", "torácica": "PROT-002", "cefaleia": "PROT-003",
                "cetônico": "PROT-004", "fraqueza": "PROT-006"}  # "tosse" antes de "febre" (pneumonia também tem febre)
    for p in dados.gerar_pacientes():
        chave = next(k for k in esperado if k in p["motivo"])
        assert dados.protocolo_do_quadro(p["motivo"], protocolos) == esperado[chave], p["motivo"]
