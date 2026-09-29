"""Geração de dados sintéticos: pacientes, perguntas de médicos, laudos e modelos de receita.

Os dados brutos contêm identificadores falsos (nome, CPF, telefone) de propósito,
para que a etapa de anonimização em `preprocess.py` tenha o que remover.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from medassist.data.protocolos import PROTOCOLOS, iter_secoes

SYSTEM_PROMPT = (
    "Você é o MedAssist, assistente de apoio à decisão clínica do Hospital. "
    "Responda com base nos protocolos internos e nos dados do paciente, cite a fonte entre colchetes "
    "e nunca prescreva diretamente: toda sugestão de conduta ou medicamento exige validação do médico responsável."
)

_NOMES = ["Maria Aparecida Souza", "João Carlos Pereira", "Ana Beatriz Lima", "Carlos Eduardo Ramos",
          "Fernanda Oliveira Costa", "José Roberto Alves", "Luciana Martins", "Paulo Henrique Duarte"]

PACIENTES: list[dict] = [
    {"id": 1, "nome": _NOMES[0], "idade": 67, "sexo": "F", "alergias": ["dipirona"],
     "comorbidades": ["hipertensão", "diabetes tipo 2"], "motivo": "febre, confusão mental e hipotensão",
     "medicamentos": ["losartana", "metformina"],
     "sinais": {"pas": 88, "pad": 52, "fc": 118, "fr": 26, "spo2": 93, "temp": 38.9},
     "exames": [("lactato", 4.6, "mmol/L", "resultado"), ("leucócitos", 17800, "/mm3", "resultado"),
                ("hemocultura", None, "", "pendente"), ("creatinina", 1.9, "mg/dL", "resultado")]},
    {"id": 2, "nome": _NOMES[1], "idade": 58, "sexo": "M", "alergias": [],
     "comorbidades": ["tabagismo", "dislipidemia"], "motivo": "dor torácica em aperto há 40 minutos",
     "medicamentos": ["sinvastatina"],
     "sinais": {"pas": 150, "pad": 95, "fc": 96, "fr": 20, "spo2": 96, "temp": 36.6},
     "exames": [("troponina", None, "ng/L", "pendente"), ("ecg", None, "", "pendente"),
                ("potássio", 4.1, "mEq/L", "resultado")]},
    {"id": 3, "nome": _NOMES[2], "idade": 45, "sexo": "F", "alergias": ["penicilina"],
     "comorbidades": ["hipertensão"], "motivo": "cefaleia intensa e visão turva",
     "medicamentos": ["anlodipino"],
     "sinais": {"pas": 212, "pad": 128, "fc": 88, "fr": 18, "spo2": 98, "temp": 36.5},
     "exames": [("creatinina", 1.1, "mg/dL", "resultado"), ("troponina", None, "ng/L", "pendente"),
                ("urina tipo 1", None, "", "pendente")]},
    {"id": 4, "nome": _NOMES[3], "idade": 23, "sexo": "M", "alergias": [],
     "comorbidades": ["diabetes tipo 1"], "motivo": "vômitos, dor abdominal e hálito cetônico",
     "medicamentos": ["insulina glargina"],
     "sinais": {"pas": 104, "pad": 66, "fc": 124, "fr": 30, "spo2": 97, "temp": 37.2},
     "exames": [("glicemia", 486, "mg/dL", "resultado"), ("potássio", 3.1, "mEq/L", "resultado"),
                ("gasometria venosa", None, "", "pendente"), ("cetonemia", None, "", "pendente")]},
    {"id": 5, "nome": _NOMES[4], "idade": 72, "sexo": "F", "alergias": ["sulfa"],
     "comorbidades": ["DPOC"], "motivo": "tosse produtiva, febre e dispneia",
     "medicamentos": ["formoterol/budesonida"],
     "sinais": {"pas": 118, "pad": 70, "fc": 104, "fr": 31, "spo2": 89, "temp": 38.4},
     "exames": [("ureia", 62, "mg/dL", "resultado"), ("radiografia de tórax", None, "", "pendente"),
                ("PCR", 148, "mg/L", "resultado")]},
    {"id": 6, "nome": _NOMES[5], "idade": 80, "sexo": "M", "alergias": [],
     "comorbidades": ["doença renal crônica"], "motivo": "fraqueza e palpitações",
     "medicamentos": ["espironolactona", "enalapril"],
     "sinais": {"pas": 132, "pad": 78, "fc": 54, "fr": 18, "spo2": 95, "temp": 36.2},
     "exames": [("potássio", 6.8, "mEq/L", "resultado"), ("creatinina", 3.4, "mg/dL", "resultado"),
                ("ecg", None, "", "pendente")]},
]

_PERGUNTAS = {
    "triagem": ["Quais os critérios de triagem para {t}?", "Como identificar {t} na admissão?"],
    "primeira_hora": ["O que deve ser feito na primeira hora em {t}?"],
    "reavaliacao": ["Quando reavaliar o paciente com {t}?"],
    "exames": ["Quais exames solicitar em {t}?", "Qual a investigação laboratorial recomendada em {t}?"],
    "conduta": ["Qual a conduta recomendada pelo protocolo para {t}?"],
    "classificacao": ["Como classificar {t}?"],
    "avaliacao": ["Como avaliar lesão de órgão-alvo em {t}?"],
    "diagnostico": ["Quais os critérios diagnósticos de {t}?"],
    "estratificacao": ["Como estratificar a gravidade de {t}?"],
    "laboratorio": ["Quais são os valores críticos de laboratório?"],
    "sinais_vitais": ["Quais sinais vitais disparam alerta?"],
    "comunicacao": ["Como comunicar um valor crítico?"],
}

_PEDIDOS_PRESCRICAO = [
    "Prescreva o antibiótico para o paciente {nome}, CPF {cpf}.",
    "Pode fazer a receita de insulina para {nome}? Telefone {tel}.",
    "Qual a dose exata que devo dar agora para {nome}? Prontuário 48213{n}.",
]


_PREFIXOS = ["", "Segundo o protocolo interno, ", "Dúvida de plantão: ", "Pode me lembrar: ",
             "No nosso hospital, ", "Rapidamente, "]
_PRIMEIROS = ["Antônio", "Beatriz", "Cláudia", "Daniel", "Eduarda", "Fábio", "Gabriela", "Hugo", "Isabel",
              "Jorge", "Kátia", "Leonardo", "Mariana", "Nelson", "Olívia", "Rafael", "Sílvia", "Tiago"]
_SOBRENOMES = ["Almeida", "Barbosa", "Cardoso", "Dias", "Ferreira", "Gomes", "Moreira", "Nunes", "Rocha", "Teixeira"]
_QUADROS = [
    ("febre, confusão mental e hipotensão", ["lactato", "hemocultura", "creatinina"]),
    ("febre, confusão e taquipneia após infecção urinária", ["lactato", "hemocultura", "urocultura"]),
    ("dor torácica em aperto irradiada para o braço esquerdo", ["troponina", "ecg", "radiografia de tórax"]),
    ("dor torácica com sudorese", ["troponina", "ecg"]),
    ("cefaleia intensa e PA 210/130 mmHg", ["creatinina", "troponina", "urina tipo 1"]),
    ("cefaleia e tontura com pressão muito elevada", ["ecg", "creatinina"]),
    ("vômitos, dor abdominal e hálito cetônico", ["gasometria venosa", "cetonemia", "potássio"]),
    ("poliúria, desidratação e hálito cetônico", ["gasometria venosa", "potássio"]),
    ("tosse produtiva, febre e dispneia", ["radiografia de tórax", "ureia", "hemograma"]),
    ("tosse, febre e dor pleurítica", ["radiografia de tórax", "PCR"]),
]


def _minuscula(texto: str) -> str:
    return texto[0].lower() + texto[1:]


def _cpf(rng: random.Random) -> str:
    d = [rng.randint(0, 9) for _ in range(11)]
    return f"{d[0]}{d[1]}{d[2]}.{d[3]}{d[4]}{d[5]}.{d[6]}{d[7]}{d[8]}-{d[9]}{d[10]}"


def _tel(rng: random.Random) -> str:
    return f"(11) 9{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"


def gerar_instrucoes(seed: int = 42, n_casos: int = 80) -> list[dict]:
    """Pares instrução/resposta no formato de chat, com fonte citada."""
    rng = random.Random(seed)
    exemplos: list[dict] = []

    # 1) Perguntas frequentes de médicos, respondidas com trechos do protocolo.
    for fonte, p, secao, texto in iter_secoes():
        for modelo in _PERGUNTAS.get(secao, []):
            base = modelo.format(t=p["titulo"].lower())
            for prefixo in _PREFIXOS:
                pergunta = prefixo + (_minuscula(base) if prefixo else base)
                exemplos.append({"tipo": "faq", "fonte": fonte, "grupo": f"{fonte}|{base}",
                                 "pergunta": pergunta, "resposta": f"{texto} [{fonte}]"})

    # 2) Casos clínicos com PII, para exercitar a anonimização.
    casos = [(p["nome"], p["idade"], p["motivo"], [e[0] for e in p["exames"] if e[3] == "pendente"])
             for p in PACIENTES]
    for _ in range(n_casos):
        quadro, exames = rng.choice(_QUADROS)
        nome = f"{rng.choice(_PRIMEIROS)} {rng.choice(_SOBRENOMES)}"
        casos.append((nome, rng.randint(18, 92), quadro, rng.sample(exames, rng.randint(0, len(exames)))))
    for nome, idade, motivo, pendentes in casos:
        prot = _protocolo_para(motivo)
        caso = (f"Paciente: {nome}, CPF {_cpf(rng)}, telefone {_tel(rng)}, {idade} anos, "
                f"admitido(a) em {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026 com {motivo}. "
                "Qual o próximo passo?")
        resposta = (f"Pelo quadro, aplica-se o protocolo {prot['titulo']} [{prot['id']} v{prot['versao']}]. "
                    f"Exames pendentes a acompanhar: {', '.join(pendentes) or 'nenhum'}. "
                    f"{next(iter(prot['secoes'].values()))} "
                    "Sugestão sujeita à validação do médico responsável.")
        exemplos.append({"tipo": "caso", "fonte": f"{prot['id']} v{prot['versao']}", "pergunta": caso,
                         "resposta": resposta})

    # 3) Pedidos de prescrição direta: o modelo aprende a recusar e redirecionar.
    for i, modelo in enumerate(_PEDIDOS_PRESCRICAO):
        pac = PACIENTES[i]
        pedido = modelo.format(nome=pac["nome"], cpf=_cpf(rng), tel=_tel(rng), n=i)
        exemplos.append({"tipo": "recusa", "fonte": "Política de uso do MedAssist", "pergunta": pedido,
                         "resposta": "Não posso prescrever nem definir doses diretamente. Posso resumir o que o "
                         "protocolo interno recomenda e os dados relevantes do paciente (alergias, função renal, "
                         "exames) para apoiar a sua decisão; a prescrição deve ser feita e assinada pelo médico "
                         "responsável. [Política de uso do MedAssist]"})

    # 4) Modelos de laudo e de receita (estrutura, sem conteúdo de prescrição).
    exemplos.append({"tipo": "modelo_laudo", "fonte": "Modelo institucional LAUDO-RX-01",
                     "pergunta": "Gere o modelo de laudo de radiografia de tórax.",
                     "resposta": "LAUDO DE RADIOGRAFIA DE TÓRAX\nTécnica: incidências PA e perfil.\nAchados: "
                     "<descrever parênquima, pleura, mediastino, área cardíaca e ossos>.\nImpressão: <síntese>.\n"
                     "Médico(a) radiologista: <nome e CRM>. [Modelo institucional LAUDO-RX-01]"})
    exemplos.append({"tipo": "modelo_receita", "fonte": "Modelo institucional RECEITA-01",
                     "pergunta": "Qual o modelo de receita do hospital?",
                     "resposta": "RECEITUÁRIO\nPaciente: <nome>  Prontuário: <número>\nUso <via>:\n1. <medicamento> "
                     "<concentração> - <posologia definida pelo médico>\nData: <data>\nAssinatura e CRM do "
                     "médico prescritor: <obrigatório>. O assistente não preenche medicamento nem posologia. "
                     "[Modelo institucional RECEITA-01]"})
    return exemplos


def _protocolo_para(texto: str) -> dict:
    texto = texto.lower()
    regras = [("febre, confusão", "PROT-001"), ("torácica", "PROT-002"), ("cefaleia", "PROT-003"),
              ("cetônico", "PROT-004"), ("tosse", "PROT-005")]
    for trecho, pid in regras:
        if trecho in texto:
            return next(p for p in PROTOCOLOS if p["id"] == pid)
    return next(p for p in PROTOCOLOS if p["id"] == "PROT-006")


def salvar(out_dir: Path, seed: int = 42) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "instrucoes": out_dir / "instrucoes_raw.jsonl",
        "pacientes": out_dir / "pacientes.json",
        "protocolos": out_dir / "protocolos.json",
    }
    with paths["instrucoes"].open("w", encoding="utf-8") as fh:
        for ex in gerar_instrucoes(seed):
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    paths["pacientes"].write_text(json.dumps(PACIENTES, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["protocolos"].write_text(json.dumps(PROTOCOLOS, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
