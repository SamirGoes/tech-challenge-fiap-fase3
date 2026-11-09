"""Regras clínicas determinísticas usadas nos fluxos automatizados (limites do PROT-006)."""

FONTE_CRITICOS = "PROT-006 §laboratorio"
FONTE_COMUNICACAO = "PROT-006 §comunicacao"

LIMITES_EXAMES = {
    "potássio": lambda v: v < 2.5 or v > 6.0,
    "sódio": lambda v: v < 120 or v > 160,
    "glicemia": lambda v: v < 50 or v > 400,
    "lactato": lambda v: v > 4,
    "hemoglobina": lambda v: v < 7,
}


def exames_pendentes(paciente: dict) -> list[str]:
    return [e["nome"] for e in paciente["exames"] if e["status"] == "pendente"]


def sinais_criticos(s: dict) -> list[str]:
    achados = []
    if s["spo2"] < 92:
        achados.append(f"SpO2 {s['spo2']}%")
    if s["pas"] < 90:
        achados.append(f"PAS {s['pas']} mmHg")
    if s["pas"] >= 180 or s["pad"] >= 120:
        achados.append(f"PA {s['pas']}/{s['pad']} mmHg")
    if s["fc"] > 130:
        achados.append(f"FC {s['fc']} bpm")
    if s["fr"] >= 30:
        achados.append(f"FR {s['fr']} irpm")
    if s["temp"] >= 39:
        achados.append(f"T {s['temp']} °C")
    return achados


def exames_criticos(exames: list[dict]) -> list[str]:
    achados = []
    for e in exames:
        regra = LIMITES_EXAMES.get(e["nome"].strip().lower())
        if regra and e["status"] == "resultado" and e["valor"] is not None and regra(e["valor"]):
            achados.append(f"{e['nome']} {e['valor']:g} {e['unidade']}".strip())
    return achados


def valores_criticos(paciente: dict) -> list[str]:
    return exames_criticos(paciente["exames"]) + sinais_criticos(paciente["sinais"])
