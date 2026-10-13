import json
import sqlite3
from pathlib import Path

from medassist.dados import PACIENTES

BANCO = Path("dados/hospital.db")


def conectar(banco=None):
    conn = sqlite3.connect(banco or BANCO)
    conn.row_factory = sqlite3.Row
    return conn


def criar_banco(banco=None):
    banco = Path(banco or BANCO)
    banco.parent.mkdir(parents=True, exist_ok=True)
    banco.unlink(missing_ok=True)
    with conectar(banco) as conn:
        conn.executescript("""
            CREATE TABLE pacientes (id INTEGER PRIMARY KEY, nome TEXT, idade INTEGER, sexo TEXT, motivo TEXT,
                                    alergias TEXT, comorbidades TEXT, sinais TEXT);
            CREATE TABLE exames (paciente_id INTEGER, nome TEXT, valor REAL, unidade TEXT, status TEXT);
        """)
        for p in PACIENTES:
            conn.execute("INSERT INTO pacientes VALUES (?,?,?,?,?,?,?,?)",
                         (p["id"], p["nome"], p["idade"], p["sexo"], p["motivo"], json.dumps(p["alergias"]),
                          json.dumps(p["comorbidades"]), json.dumps(p["sinais"])))
            for nome, valor, unidade in p["exames"]:
                status = "pendente" if valor is None else "resultado"
                conn.execute("INSERT INTO exames VALUES (?,?,?,?,?)", (p["id"], nome, valor, unidade, status))
    conn.close()
    return banco


def buscar_paciente(paciente_id, banco=None):
    conn = conectar(banco)
    linha = conn.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()
    if linha is None:
        conn.close()
        return None
    paciente = dict(linha)
    for campo in ("alergias", "comorbidades", "sinais"):
        paciente[campo] = json.loads(paciente[campo])
    paciente["exames"] = [dict(e) for e in conn.execute("SELECT * FROM exames WHERE paciente_id = ?", (paciente_id,))]
    conn.close()
    return paciente


def resumo_clinico(p):
    s = p["sinais"]
    resultados = [f"{e['nome']} {e['valor']:g} {e['unidade']}".strip() for e in p["exames"] if e["status"] == "resultado"]
    pendentes = [e["nome"] for e in p["exames"] if e["status"] == "pendente"]
    return (f"Paciente #{p['id']}, {p['idade']} anos, sexo {p['sexo']}, motivo: {p['motivo']}. "
            f"Comorbidades: {', '.join(p['comorbidades']) or 'nenhuma'}. Alergias: {', '.join(p['alergias']) or 'nenhuma'}. "
            f"PA {s['pas']}/{s['pad']}, FC {s['fc']}, FR {s['fr']}, SpO2 {s['spo2']}%, T {s['temp']}. "
            f"Resultados: {'; '.join(resultados) or 'nenhum'}. Pendentes: {', '.join(pendentes) or 'nenhum'}.")
