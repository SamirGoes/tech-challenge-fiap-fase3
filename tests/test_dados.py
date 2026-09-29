import json

from medassist.data.anonymize import anonymize
from medassist.data.preprocess import curar, dividir, preparar
from medassist.data.synthetic import PACIENTES, gerar_instrucoes, salvar


def test_anonimiza_identificadores():
    texto = ("Paciente: Maria Aparecida Souza, CPF 123.456.789-09, telefone (11) 91234-5678, "
             "email maria@ex.com, admitida em 12/08/2026, prontuário 482131.")
    res = anonymize(texto)
    for pii in ["Maria", "123.456.789-09", "91234-5678", "maria@ex.com", "12/08/2026", "482131"]:
        assert pii not in res.text
    assert res.replacements["CPF"] == 1


def test_anonimiza_nomes_conhecidos_sem_contexto():
    res = anonymize("Reavaliar João Carlos Pereira amanhã.", known_names=["João Carlos Pereira"])
    assert "João" not in res.text and "[NOME]" in res.text


def test_dataset_curado_sem_pii_e_com_fonte():
    nomes = [p["nome"] for p in PACIENTES]
    curados, rep = curar(gerar_instrucoes(), nomes)
    blob = json.dumps(curados, ensure_ascii=False)
    assert not any(n in blob for n in nomes)
    assert all("[" in ex["resposta"] for ex in curados)
    assert rep.anonimizacoes["CPF"] > 0


def test_parafrases_nao_vazam_entre_splits():
    curados, _ = curar(gerar_instrucoes(), [])
    splits = dividir(curados)
    grupos = {nome: {ex.get("grupo") for ex in itens if ex.get("grupo")} for nome, itens in splits.items()}
    assert not grupos["train"] & grupos["test"]
    assert not grupos["train"] & grupos["val"]


def test_preparar_gera_formato_chat(tmp_path):
    paths = salvar(tmp_path / "raw")
    rep = preparar(paths["instrucoes"], tmp_path / "proc", [p["nome"] for p in PACIENTES])
    linha = json.loads((tmp_path / "proc" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in linha["messages"]] == ["system", "user", "assistant"]
    assert rep.saida["test"] > 0
