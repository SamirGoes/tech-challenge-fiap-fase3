"""Base estruturada de prontuários (SQLite) e consultas usadas pelo assistente."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from medassist.config import get_settings
from medassist.data.synthetic import PACIENTES

SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY, nome TEXT NOT NULL, idade INTEGER, sexo TEXT,
    motivo_internacao TEXT, comorbidades TEXT, alergias TEXT, medicamentos_em_uso TEXT
);
CREATE TABLE IF NOT EXISTS sinais_vitais (
    paciente_id INTEGER REFERENCES pacientes(id), registrado_em TEXT DEFAULT CURRENT_TIMESTAMP,
    pas INTEGER, pad INTEGER, fc INTEGER, fr INTEGER, spo2 INTEGER, temp REAL
);
CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY AUTOINCREMENT, paciente_id INTEGER REFERENCES pacientes(id),
    nome TEXT, valor REAL, unidade TEXT, status TEXT CHECK (status IN ('pendente', 'resultado')),
    solicitado_em TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def conectar(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or get_settings().db_path)
    conn.row_factory = sqlite3.Row
    return conn


def inicializar(path: Path | None = None, pacientes: list[dict] | None = None) -> Path:
    path = path or get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with closing(conectar(path)) as conn, conn:
        conn.executescript(SCHEMA)
        for p in pacientes or PACIENTES:
            conn.execute("INSERT INTO pacientes VALUES (?,?,?,?,?,?,?,?)", (
                p["id"], p["nome"], p["idade"], p["sexo"], p["motivo"], json.dumps(p["comorbidades"]),
                json.dumps(p["alergias"]), json.dumps(p["medicamentos"])))
            sv = p["sinais"]
            conn.execute("INSERT INTO sinais_vitais (paciente_id,pas,pad,fc,fr,spo2,temp) VALUES (?,?,?,?,?,?,?)",
                         (p["id"], sv["pas"], sv["pad"], sv["fc"], sv["fr"], sv["spo2"], sv["temp"]))
            conn.executemany("INSERT INTO exames (paciente_id,nome,valor,unidade,status) VALUES (?,?,?,?,?)",
                             [(p["id"], *e) for e in p["exames"]])
    return path


class ProntuarioRepository:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().db_path

    def paciente(self, paciente_id: int) -> dict | None:
        with closing(conectar(self.path)) as conn:
            row = conn.execute("SELECT * FROM pacientes WHERE id=?", (paciente_id,)).fetchone()
            if row is None:
                return None
            pac = dict(row)
            for campo in ("comorbidades", "alergias", "medicamentos_em_uso"):
                pac[campo] = json.loads(pac[campo])
            sv = conn.execute("SELECT * FROM sinais_vitais WHERE paciente_id=? ORDER BY registrado_em DESC LIMIT 1",
                              (paciente_id,)).fetchone()
            pac["sinais_vitais"] = dict(sv) if sv else {}
            pac["exames"] = [dict(r) for r in conn.execute(
                "SELECT nome, valor, unidade, status FROM exames WHERE paciente_id=? ORDER BY id", (paciente_id,))]
            return pac

    def exames_pendentes(self, paciente_id: int) -> list[str]:
        with closing(conectar(self.path)) as conn:
            return [r["nome"] for r in conn.execute(
                "SELECT nome FROM exames WHERE paciente_id=? AND status='pendente'", (paciente_id,))]

    def listar(self) -> list[dict]:
        with closing(conectar(self.path)) as conn:
            return [dict(r) for r in conn.execute("SELECT id, idade, sexo, motivo_internacao FROM pacientes")]


def resumo_clinico(pac: dict) -> str:
    """Resumo SEM identificadores diretos, que é o que vai para o prompt da LLM."""
    sv = pac.get("sinais_vitais", {})
    resultados = [f"{e['nome']} {e['valor']:g} {e['unidade']}".strip() for e in pac["exames"] if e["status"] == "resultado"]
    pendentes = [e["nome"] for e in pac["exames"] if e["status"] == "pendente"]
    return (
        f"Paciente #{pac['id']}, {pac['idade']} anos, sexo {pac['sexo']}. Motivo: {pac['motivo_internacao']}.\n"
        f"Comorbidades: {', '.join(pac['comorbidades']) or 'nenhuma'}. "
        f"Alergias: {', '.join(pac['alergias']) or 'nenhuma conhecida'}. "
        f"Em uso: {', '.join(pac['medicamentos_em_uso']) or 'nada'}.\n"
        f"Sinais vitais: PA {sv.get('pas')}/{sv.get('pad')} mmHg, FC {sv.get('fc')} bpm, FR {sv.get('fr')} irpm, "
        f"SpO2 {sv.get('spo2')}%, T {sv.get('temp')} °C.\n"
        f"Resultados: {'; '.join(resultados) or 'nenhum'}. Pendentes: {', '.join(pendentes) or 'nenhum'}."
    )
