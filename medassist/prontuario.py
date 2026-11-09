"""Base estruturada de prontuários: CSVs sintéticos carregados em SQLite e consultados com SQL."""

import csv
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from medassist.config import config

SCHEMA = """
CREATE TABLE pacientes (id INTEGER PRIMARY KEY, nome TEXT, idade INTEGER, sexo TEXT, motivo TEXT,
                        alergias TEXT, comorbidades TEXT, pas INTEGER, pad INTEGER, fc INTEGER, fr INTEGER,
                        spo2 INTEGER, temp REAL);
CREATE TABLE exames (paciente_id INTEGER, nome TEXT, valor REAL, unidade TEXT, status TEXT);
"""
CAMPOS_SINAIS = ("pas", "pad", "fc", "fr", "spo2", "temp")


def ler_csv(pasta=None) -> list[dict]:
    """Lê pacientes.csv e exames.csv e devolve pacientes no formato usado pelo assistente."""
    pasta = Path(pasta or config().pasta_dados)
    exames = {}
    with open(pasta / "exames.csv", encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            valor = float(linha["valor"]) if linha["valor"] else None
            exames.setdefault(int(linha["paciente_id"]), []).append(
                {"nome": linha["nome"], "valor": valor, "unidade": linha["unidade"], "status": linha["status"]})
    pacientes = []
    with open(pasta / "pacientes.csv", encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            pid = int(linha["id"])
            sinais = {c: (float(linha[c]) if c == "temp" else int(linha[c])) for c in CAMPOS_SINAIS}
            pacientes.append({
                "id": pid, "nome": linha["nome"], "idade": int(linha["idade"]), "sexo": linha["sexo"],
                "motivo": linha["motivo"],
                "alergias": [a for a in linha["alergias"].split(";") if a],
                "comorbidades": [c for c in linha["comorbidades"].split(";") if c],
                "sinais": sinais, "exames": exames.get(pid, [])})
    return pacientes


def criar_banco(banco=None, pasta_csv=None) -> Path:
    banco = Path(banco or config().banco)
    banco.parent.mkdir(parents=True, exist_ok=True)
    banco.unlink(missing_ok=True)
    with closing(sqlite3.connect(banco)) as conn, conn:
        conn.executescript(SCHEMA)
        for p in ler_csv(pasta_csv):
            conn.execute("INSERT INTO pacientes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                p["id"], p["nome"], p["idade"], p["sexo"], p["motivo"], json.dumps(p["alergias"], ensure_ascii=False),
                json.dumps(p["comorbidades"], ensure_ascii=False), *(p["sinais"][c] for c in CAMPOS_SINAIS)))
            conn.executemany("INSERT INTO exames VALUES (?,?,?,?,?)",
                             [(p["id"], e["nome"], e["valor"], e["unidade"], e["status"]) for e in p["exames"]])
    return banco


def buscar_paciente(paciente_id: int, banco=None) -> dict | None:
    with closing(sqlite3.connect(banco or config().banco)) as conn:
        conn.row_factory = sqlite3.Row
        linha = conn.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()
        if linha is None:
            return None
        p = dict(linha)
        p["alergias"] = json.loads(p["alergias"])
        p["comorbidades"] = json.loads(p["comorbidades"])
        p["sinais"] = {c: p.pop(c) for c in CAMPOS_SINAIS}
        p["exames"] = [dict(e) for e in conn.execute(
            "SELECT nome, valor, unidade, status FROM exames WHERE paciente_id = ?", (paciente_id,))]
        return p


def listar_pacientes(banco=None) -> list[dict]:
    with closing(sqlite3.connect(banco or config().banco)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT id, idade, sexo, motivo FROM pacientes ORDER BY id")]


def resumo_clinico(p: dict) -> str:
    """Resumo SEM nome nem documentos: é o único dado do paciente que vai para o prompt da LLM."""
    s = p["sinais"]
    resultados = [f"{e['nome']} {e['valor']:g} {e['unidade']}".strip() for e in p["exames"] if e["status"] == "resultado"]
    pendentes = [e["nome"] for e in p["exames"] if e["status"] == "pendente"]
    return (f"Paciente #{p['id']}, {p['idade']} anos, sexo {p['sexo']}, motivo: {p['motivo']}. "
            f"Comorbidades: {', '.join(p['comorbidades']) or 'nenhuma'}. Alergias: {', '.join(p['alergias']) or 'nenhuma'}. "
            f"PA {s['pas']}/{s['pad']}, FC {s['fc']}, FR {s['fr']}, SpO2 {s['spo2']}%, T {s['temp']}. "
            f"Resultados: {'; '.join(resultados) or 'nenhum'}. Pendentes: {', '.join(pendentes) or 'nenhum'}.")
