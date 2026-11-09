"""Dados do projeto: protocolos, pacientes sintéticos, anonimização, banco SQLite e dataset de treino.

Tudo é fictício e gerado por script (com seed), então o resultado é sempre o mesmo.
"""

import csv
import json
import random
import re
import sqlite3
from pathlib import Path

from langchain_core.prompts import PromptTemplate

PASTA = Path("dados")
CONHECIMENTO = Path("conhecimento")
LOGS = Path("logs")

# ---------------------------------------------------------------------------------------------------------
# Prompt (o mesmo no treino, na avaliação e no assistente)
# ---------------------------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Você é o MedAssist, assistente de apoio à decisão clínica do hospital. "
    "Responda com base nos protocolos internos e nos dados do paciente, cite a fonte entre colchetes "
    "e nunca prescreva: toda conduta ou medicamento exige validação do médico responsável."
)
RECUSA = ("Não posso prescrever nem definir doses. Posso resumir o que o protocolo recomenda e os dados do paciente "
          "para apoiar a sua decisão; a prescrição deve ser feita pelo médico responsável. [Política MedAssist]")
AVISO = ("Apoio à decisão clínica: não substitui a avaliação do médico e não é prescrição. "
         "Consulte o médico responsável antes de qualquer conduta.")
SEM_PROTOCOLO = "(nenhum protocolo relevante)"
SEM_PACIENTE = "(nenhum paciente selecionado)"
NAO_APLICA = "Nenhum protocolo interno aplicável a este quadro. [Política MedAssist]"

PROMPT = PromptTemplate.from_template(
    "Use apenas as informações abaixo e cite a fonte entre colchetes. Não prescreva.\n\n"
    "### Protocolos\n{protocolos}\n\n### Paciente [Prontuário #{paciente_id}]\n{paciente}\n\n### Pergunta\n{pergunta}"
)


def montar_prompt(trechos, paciente, pergunta, paciente_id="-"):
    """trechos: lista de (fonte, texto)."""
    protocolos = "\n".join(f"[{fonte}] {texto}" for fonte, texto in trechos) or SEM_PROTOCOLO
    return PROMPT.format(protocolos=protocolos, paciente=paciente, paciente_id=paciente_id, pergunta=pergunta)


# ---------------------------------------------------------------------------------------------------------
# Protocolos (arquivos .md em conhecimento/)
# ---------------------------------------------------------------------------------------------------------
def ler_protocolos(pasta=None):
    """Lê conhecimento/PROT-*.md. Formato: '# Título', 'id:', 'palavras:' e uma seção '## nome' por trecho."""
    protocolos = {}
    for arq in sorted(Path(pasta or CONHECIMENTO).glob("PROT-*.md")):
        p = {"id": arq.stem, "titulo": arq.stem, "palavras": "", "secoes": {}}
        secao = None
        for linha in arq.read_text(encoding="utf-8-sig").splitlines():
            linha = linha.strip()
            if linha.startswith("## "):
                secao = linha[3:].strip()
                p["secoes"][secao] = ""
            elif linha.startswith("# "):
                p["titulo"] = linha[2:].strip()
            elif secao is None and linha.startswith("id:"):
                p["id"] = linha[3:].strip()
            elif secao is None and linha.startswith("palavras:"):
                p["palavras"] = linha[9:].strip()
            elif secao and linha:
                p["secoes"][secao] = (p["secoes"][secao] + " " + linha).strip()
        protocolos[p["id"]] = p
    return protocolos


def secoes():
    """Gera (fonte, protocolo, nome_da_secao, texto). A fonte é citável, ex.: 'PROT-001 §primeira_hora'."""
    for pid, p in ler_protocolos().items():
        for nome, texto in p["secoes"].items():
            yield f"{pid} §{nome}", p, nome, texto


# ---------------------------------------------------------------------------------------------------------
# Anonimização
# ---------------------------------------------------------------------------------------------------------
REGRAS_PII = [
    ("CPF", r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    ("TELEFONE", r"\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b"),
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("DATA", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    ("PRONTUARIO", r"\bprontu[aá]rio\s*:?\s*\d{4,}\b"),
]
MAIUSC = "A-ZÁÉÍÓÚÂÊÔÃÕÇ"
NOME_COM_CONTEXTO = (rf"(\b(?:Paciente|Sr\.|Sra\.|Dr\.|Dra\.)\s*:?\s+)[{MAIUSC}][\wáéíóúâêôãõç]+"
                     rf"(?:\s+[{MAIUSC}][\wáéíóúâêôãõç]+)*")


def anonimizar(texto, nomes=()):
    """Troca dados pessoais por [NOME], [CPF]... Devolve (texto, {tipo: quantidade})."""
    contagem = {}
    for nome in nomes:
        texto, n = re.subn(re.escape(nome), "[NOME]", texto, flags=re.I)
        contagem["NOME"] = contagem.get("NOME", 0) + n
    texto, n = re.subn(NOME_COM_CONTEXTO, r"\1[NOME]", texto)
    contagem["NOME"] = contagem.get("NOME", 0) + n
    for rotulo, regra in REGRAS_PII:
        texto, n = re.subn(regra, f"[{rotulo}]", texto, flags=re.I)
        contagem[rotulo] = contagem.get(rotulo, 0) + n
    return texto, {k: v for k, v in contagem.items() if v}


# ---------------------------------------------------------------------------------------------------------
# Pacientes sintéticos
# ---------------------------------------------------------------------------------------------------------
def _ex(nome, valor=None, unidade=""):
    return {"nome": nome, "valor": valor, "unidade": unidade, "status": "pendente" if valor is None else "resultado"}


def _sv(pas, pad, fc, fr, spo2, temp):
    return {"pas": pas, "pad": pad, "fc": fc, "fr": fr, "spo2": spo2, "temp": temp}


def _n(r, a, b, casas=0):
    v = r.uniform(a, b)
    return round(v, casas) if casas else int(round(v))


def _talvez(r, nome, valor, unidade="", p=0.3):
    return _ex(nome, None if r.random() < p else valor, unidade)


# Seis pacientes escritos à mão (ids 1 a 6), com valores críticos de propósito para a demonstração.
CURADOS = [
    {"id": 1, "nome": "Maria Aparecida Souza", "idade": 67, "sexo": "F", "protocolo": "PROT-001",
     "motivo": "febre, confusão mental e hipotensão", "alergias": ["dipirona"],
     "comorbidades": ["hipertensão", "diabetes tipo 2"], "medicamentos": ["losartana", "metformina"],
     "sinais": _sv(88, 52, 118, 26, 93, 38.9),
     "exames": [_ex("lactato", 4.6, "mmol/L"), _ex("leucócitos", 17800, "/mm3"), _ex("hemocultura")]},
    {"id": 2, "nome": "João Carlos Pereira", "idade": 58, "sexo": "M", "protocolo": "PROT-002",
     "motivo": "dor torácica em aperto há 40 minutos", "alergias": [], "comorbidades": ["tabagismo"],
     "medicamentos": ["sinvastatina"], "sinais": _sv(150, 95, 96, 20, 96, 36.6),
     "exames": [_ex("troponina", None, "ng/L"), _ex("ecg"), _ex("potássio", 4.1, "mEq/L")]},
    {"id": 3, "nome": "Ana Beatriz Lima", "idade": 45, "sexo": "F", "protocolo": "PROT-003",
     "motivo": "cefaleia intensa e visão turva", "alergias": ["penicilina"], "comorbidades": ["hipertensão"],
     "medicamentos": ["anlodipino"], "sinais": _sv(212, 128, 88, 18, 98, 36.5),
     "exames": [_ex("creatinina", 1.1, "mg/dL"), _ex("troponina", None, "ng/L")]},
    {"id": 4, "nome": "Carlos Eduardo Ramos", "idade": 23, "sexo": "M", "protocolo": "PROT-004",
     "motivo": "vômitos, dor abdominal e hálito cetônico", "alergias": [], "comorbidades": ["diabetes tipo 1"],
     "medicamentos": ["insulina glargina"], "sinais": _sv(104, 66, 124, 30, 97, 37.2),
     "exames": [_ex("glicemia", 486, "mg/dL"), _ex("potássio", 3.1, "mEq/L"), _ex("gasometria venosa")]},
    {"id": 5, "nome": "Fernanda Oliveira Costa", "idade": 72, "sexo": "F", "protocolo": "PROT-005",
     "motivo": "tosse produtiva, febre e dispneia", "alergias": ["sulfa"], "comorbidades": ["DPOC"],
     "medicamentos": ["formoterol/budesonida"], "sinais": _sv(118, 70, 104, 31, 89, 38.4),
     "exames": [_ex("ureia", 62, "mg/dL"), _ex("radiografia de tórax")]},
    {"id": 6, "nome": "José Roberto Alves", "idade": 80, "sexo": "M", "protocolo": "PROT-006",
     "motivo": "fraqueza e palpitações", "alergias": [], "comorbidades": ["doença renal crônica"],
     "medicamentos": ["espironolactona", "enalapril"], "sinais": _sv(132, 78, 54, 18, 95, 36.2),
     "exames": [_ex("potássio", 6.8, "mEq/L"), _ex("creatinina", 3.4, "mg/dL"), _ex("ecg")]},
]

_NORMAIS = lambda r: _sv(_n(r, 110, 135), _n(r, 65, 85), _n(r, 62, 95), _n(r, 14, 19), _n(r, 96, 99), _n(r, 36.0, 37.2, 1))

# Cada cenário gera um tipo de paciente. "protocolo" None = quadro de rotina, sem protocolo interno.
CENARIOS = [
    dict(protocolo="PROT-001", idade=(40, 90), minimo=1,
         motivos=["febre, confusão mental e hipotensão", "febre, confusão e taquipneia após infecção urinária",
                  "febre alta, calafrios e hipotensão"],
         comorb=["hipertensão", "diabetes tipo 2", "doença renal crônica", "câncer em quimioterapia"],
         sinais=lambda r: _sv(_n(r, 76, 98), _n(r, 40, 60), _n(r, 105, 132), _n(r, 22, 31), _n(r, 89, 95), _n(r, 38.2, 39.6, 1)),
         exames=lambda r: [_talvez(r, "lactato", _n(r, 1.8, 6.0, 1), "mmol/L", 0.1), _ex("leucócitos", _n(r, 12000, 21000), "/mm3"),
                           _ex("creatinina", _n(r, 0.9, 2.6, 1), "mg/dL"), _ex("hemocultura")]),
    dict(protocolo="PROT-002", idade=(40, 85),
         motivos=["dor torácica em aperto há 40 minutos", "dor torácica com sudorese irradiada para o braço esquerdo",
                  "dor torácica e aperto no peito após esforço"],
         comorb=["tabagismo", "dislipidemia", "hipertensão", "diabetes tipo 2"],
         sinais=lambda r: _sv(_n(r, 120, 165), _n(r, 70, 100), _n(r, 70, 110), _n(r, 16, 22), _n(r, 94, 99), _n(r, 36.0, 37.0, 1)),
         exames=lambda r: [_talvez(r, "troponina", _n(r, 5, 80), "ng/L", 0.6), _ex("ecg"),
                           _ex("potássio", _n(r, 3.6, 4.8, 1), "mEq/L")]),
    dict(protocolo="PROT-003", idade=(35, 85),
         motivos=["cefaleia intensa e visão turva", "cefaleia e tontura com pressão muito elevada"],
         comorb=["hipertensão", "doença renal crônica", "obesidade"],
         sinais=lambda r: _sv(_n(r, 182, 230), _n(r, 110, 135), _n(r, 70, 100), _n(r, 16, 20), _n(r, 96, 99), _n(r, 36.0, 37.0, 1)),
         exames=lambda r: [_ex("creatinina", _n(r, 0.9, 2.2, 1), "mg/dL"), _talvez(r, "troponina", _n(r, 4, 30), "ng/L", 0.6),
                           _ex("urina tipo 1")]),
    dict(protocolo="PROT-004", idade=(18, 45), minimo=1,
         motivos=["vômitos, dor abdominal e hálito cetônico", "poliúria, desidratação e hálito cetônico"],
         comorb=["diabetes tipo 1", "diabetes tipo 2"],
         sinais=lambda r: _sv(_n(r, 95, 115), _n(r, 55, 72), _n(r, 110, 135), _n(r, 24, 32), _n(r, 95, 99), _n(r, 36.5, 37.8, 1)),
         exames=lambda r: [_ex("glicemia", _n(r, 280, 560), "mg/dL"), _ex("potássio", _n(r, 2.8, 5.4, 1), "mEq/L"),
                           _ex("gasometria venosa"), _ex("cetonemia")]),
    dict(protocolo="PROT-005", idade=(50, 90),
         motivos=["tosse produtiva, febre e dispneia", "tosse, febre e dor pleurítica"],
         comorb=["DPOC", "tabagismo", "insuficiência cardíaca", "diabetes tipo 2"],
         sinais=lambda r: _sv(_n(r, 100, 135), _n(r, 60, 80), _n(r, 95, 115), _n(r, 24, 33), _n(r, 86, 94), _n(r, 37.8, 39.2, 1)),
         exames=lambda r: [_ex("ureia", _n(r, 30, 80), "mg/dL"), _ex("PCR", _n(r, 60, 220), "mg/L"),
                           _ex("radiografia de tórax"), _ex("hemocultura")]),
    dict(protocolo="PROT-006", idade=(55, 90), minimo=1,
         motivos=["fraqueza e palpitações", "fraqueza muscular e palpitações"],
         comorb=["doença renal crônica", "insuficiência cardíaca", "diabetes tipo 2"],
         sinais=lambda r: _sv(_n(r, 110, 145), _n(r, 65, 85), _n(r, 45, 70), _n(r, 16, 20), _n(r, 93, 97), _n(r, 36.0, 37.0, 1)),
         exames=lambda r: [_ex("potássio", _n(r, 5.2, 7.4, 1), "mEq/L"), _ex("creatinina", _n(r, 1.8, 4.0, 1), "mg/dL"), _ex("ecg")]),
    dict(protocolo="PROT-007", idade=(50, 90), minimo=1,
         motivos=["fraqueza súbita no braço direito e dificuldade para falar", "desvio da face e fala arrastada"],
         comorb=["hipertensão", "fibrilação atrial", "diabetes tipo 2", "dislipidemia"],
         sinais=lambda r: _sv(_n(r, 150, 200), _n(r, 85, 115), _n(r, 70, 100), _n(r, 16, 20), _n(r, 94, 99), _n(r, 36.0, 37.5, 1)),
         exames=lambda r: [_ex("glicemia", _n(r, 90, 250), "mg/dL"), _ex("tomografia de crânio"), _ex("ecg"), _ex("coagulograma")]),
    dict(protocolo="PROT-008", idade=(18, 80), minimo=1,
         motivos=["falta de ar e chiado no peito", "tosse noturna, chiado e dispneia"],
         comorb=["asma", "rinite alérgica", "tabagismo"],
         sinais=lambda r: _sv(_n(r, 110, 140), _n(r, 65, 85), _n(r, 100, 130), _n(r, 24, 34), _n(r, 88, 95), _n(r, 36.0, 37.2, 1)),
         exames=lambda r: [_ex("gasometria arterial"), _ex("radiografia de tórax")]),
    dict(protocolo="PROT-009", idade=(18, 90),
         motivos=["ardência ao urinar, polaciúria e dor lombar", "disúria e febre"],
         comorb=["diabetes tipo 2", "litíase renal", "hipertensão"],
         sinais=lambda r: _sv(_n(r, 105, 135), _n(r, 65, 85), _n(r, 80, 105), _n(r, 16, 22), _n(r, 95, 99), _n(r, 37.2, 39.0, 1)),
         exames=lambda r: [_ex("urina tipo 1"), _ex("urocultura"), _ex("leucócitos", _n(r, 9000, 18000), "/mm3"),
                           _ex("creatinina", _n(r, 0.7, 1.8, 1), "mg/dL")]),
    dict(protocolo="PROT-010", idade=(55, 92), minimo=1,
         motivos=["falta de ar, inchaço nas pernas e ganho de peso"],
         comorb=["insuficiência cardíaca", "hipertensão", "fibrilação atrial"],
         sinais=lambda r: _sv(_n(r, 100, 150), _n(r, 60, 90), _n(r, 80, 115), _n(r, 20, 28), _n(r, 88, 95), _n(r, 36.0, 37.0, 1)),
         exames=lambda r: [_ex("BNP", _n(r, 400, 2500), "pg/mL"), _ex("creatinina", _n(r, 1.0, 2.5, 1), "mg/dL"),
                           _ex("potássio", _n(r, 3.4, 5.2, 1), "mEq/L"), _ex("radiografia de tórax")]),
    dict(protocolo=None, idade=(20, 70), motivos=["dor lombar após esforço, sem febre"], comorb=["obesidade", "tabagismo"],
         sinais=_NORMAIS, exames=lambda r: [_talvez(r, "PCR", _n(r, 1, 8), "mg/L", 0.5), _ex("radiografia de coluna")]),
    dict(protocolo=None, idade=(18, 80), motivos=["diarreia e vômitos há 2 dias"], comorb=["hipertensão"],
         sinais=_NORMAIS, exames=lambda r: [_ex("potássio", _n(r, 3.5, 4.8, 1), "mEq/L"),
                                            _ex("creatinina", _n(r, 0.7, 1.2, 1), "mg/dL"), _ex("exame de fezes")]),
    dict(protocolo=None, idade=(18, 60), motivos=["coriza, dor de garganta e febre baixa há 3 dias"], comorb=[],
         sinais=_NORMAIS, exames=lambda r: [_ex("PCR", _n(r, 2, 12), "mg/L")]),
    dict(protocolo=None, idade=(15, 65), motivos=["dor e inchaço no tornozelo após torção"], comorb=[],
         sinais=_NORMAIS, exames=lambda r: [_ex("radiografia de tornozelo")]),
]

MEDICAMENTOS = {
    "hipertensão": ["losartana", "anlodipino", "hidroclorotiazida"], "diabetes tipo 2": ["metformina"],
    "diabetes tipo 1": ["insulina glargina"], "dislipidemia": ["sinvastatina"], "DPOC": ["formoterol/budesonida"],
    "asma": ["salbutamol", "budesonida"], "insuficiência cardíaca": ["furosemida", "carvedilol"],
    "fibrilação atrial": ["varfarina"], "doença renal crônica": ["enalapril"], "rinite alérgica": ["loratadina"],
}
FEMININOS = ["Ana", "Beatriz", "Cláudia", "Daniela", "Eduarda", "Fernanda", "Gabriela", "Helena", "Isabel", "Júlia",
             "Kátia", "Larissa", "Mariana", "Natália", "Olívia", "Patrícia", "Renata", "Sílvia", "Tatiane", "Vera"]
MASCULINOS = ["Antônio", "Bruno", "Carlos", "Daniel", "Eduardo", "Fábio", "Gustavo", "Hugo", "Igor", "Jorge",
              "Leonardo", "Marcos", "Nelson", "Paulo", "Rafael", "Sérgio", "Tiago", "Vinícius", "Wagner", "Rodrigo"]
SOBRENOMES = ["Almeida", "Barbosa", "Cardoso", "Dias", "Ferreira", "Gomes", "Moreira", "Nunes", "Rocha", "Teixeira",
              "Araújo", "Batista", "Campos", "Duarte", "Farias", "Guedes", "Lopes", "Machado", "Pinto", "Ribeiro",
              "Santos", "Silva", "Tavares", "Vieira", "Xavier"]
CIDADES = ["São Paulo", "Campinas", "Santos", "Guarulhos", "Osasco", "Sorocaba", "São Bernardo do Campo", "Barueri"]
CONVENIOS = ["SUS", "Unimed", "Bradesco Saúde", "Amil", "SulAmérica", "Particular"]
SETORES = ["Pronto-socorro", "Observação", "Enfermaria", "UTI"]
SANGUE = ["O+", "O+", "A+", "A+", "B+", "AB+", "O-", "A-", "B-"]
ALERGIAS = [[], [], [], [], ["dipirona"], ["penicilina"], ["sulfa"], ["AAS"], ["látex"]]
MEDICOS = ["Dra. Helena Prado", "Dr. Marcos Vilela", "Dra. Cíntia Borges", "Dr. Renato Salles", "Dra. Paula Menezes"]

COLUNAS = ["id", "nome", "idade", "sexo", "peso_kg", "altura_cm", "tipo_sanguineo", "cidade", "convenio", "setor",
           "data_admissao", "medico", "motivo", "comorbidades", "alergias", "medicamentos",
           "pas", "pad", "fc", "fr", "spo2", "temp"]
SINAIS = ["pas", "pad", "fc", "fr", "spo2", "temp"]
LISTAS = ["comorbidades", "alergias", "medicamentos"]


def _completar(p, r):
    """Acrescenta os dados genéricos de cadastro e internação."""
    p.setdefault("peso_kg", _n(r, 50, 105) if p["sexo"] == "F" else _n(r, 62, 120))
    p.setdefault("altura_cm", _n(r, 150, 175) if p["sexo"] == "F" else _n(r, 162, 190))
    p.setdefault("tipo_sanguineo", r.choice(SANGUE))
    p.setdefault("cidade", r.choice(CIDADES))
    p.setdefault("convenio", r.choice(CONVENIOS))
    p.setdefault("setor", r.choice(SETORES))
    p.setdefault("data_admissao", f"2026-{r.randint(1, 10):02d}-{r.randint(1, 28):02d}")
    p.setdefault("medico", r.choice(MEDICOS))
    return p


def gerar_pacientes(total=300, seed=7):
    r = random.Random(seed)
    pacientes = [_completar(dict(p), r) for p in CURADOS]
    usados = {p["nome"] for p in pacientes}
    while len(pacientes) < total:
        c = CENARIOS[len(pacientes) % len(CENARIOS)]
        sexo = r.choice("FM")
        nome = f"{r.choice(FEMININOS if sexo == 'F' else MASCULINOS)} {r.choice(SOBRENOMES)} {r.choice(SOBRENOMES)}"
        if nome in usados:
            continue
        usados.add(nome)
        comorb = r.sample(c["comorb"], r.randint(c.get("minimo", 0), min(2, len(c["comorb"]))))
        remedios = sorted({m for cm in comorb for m in MEDICAMENTOS.get(cm, [])})
        pacientes.append(_completar({
            "id": len(pacientes) + 1, "nome": nome, "idade": r.randint(*c["idade"]), "sexo": sexo,
            "motivo": r.choice(c["motivos"]), "alergias": list(r.choice(ALERGIAS)), "comorbidades": comorb,
            "medicamentos": remedios, "sinais": c["sinais"](r), "exames": c["exames"](r), "protocolo": c["protocolo"]}, r))
    return pacientes


def salvar_csv(pacientes):
    PASTA.mkdir(parents=True, exist_ok=True)
    with open(PASTA / "pacientes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUNAS)
        for p in pacientes:
            linha = {**p, **p["sinais"], **{c: ";".join(p[c]) for c in LISTAS}}
            w.writerow([linha[c] for c in COLUNAS])
    with open(PASTA / "exames.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paciente_id", "nome", "valor", "unidade", "status"])
        for p in pacientes:
            for e in p["exames"]:
                w.writerow([p["id"], e["nome"], "" if e["valor"] is None else e["valor"], e["unidade"], e["status"]])


# ---------------------------------------------------------------------------------------------------------
# Banco SQLite de prontuários (carregado dos CSVs) e regras clínicas
# ---------------------------------------------------------------------------------------------------------
TIPOS = {"id": "INTEGER PRIMARY KEY", "idade": "INTEGER", "peso_kg": "INTEGER", "altura_cm": "INTEGER", "pas": "INTEGER",
         "pad": "INTEGER", "fc": "INTEGER", "fr": "INTEGER", "spo2": "INTEGER", "temp": "REAL"}


def criar_banco():
    banco = PASTA / "hospital.db"
    banco.unlink(missing_ok=True)
    conn = sqlite3.connect(banco)
    colunas = ", ".join(c + " " + TIPOS.get(c, "TEXT") for c in COLUNAS)
    conn.execute(f"CREATE TABLE pacientes ({colunas})")
    conn.execute("CREATE TABLE exames (paciente_id INTEGER, nome TEXT, valor REAL, unidade TEXT, status TEXT)")
    with open(PASTA / "pacientes.csv", encoding="utf-8", newline="") as f:
        conn.executemany(f"INSERT INTO pacientes VALUES ({','.join('?' * len(COLUNAS))})",
                         [[linha[c] for c in COLUNAS] for linha in csv.DictReader(f)])
    with open(PASTA / "exames.csv", encoding="utf-8", newline="") as f:
        conn.executemany("INSERT INTO exames VALUES (?,?,?,?,?)",
                         [(l["paciente_id"], l["nome"], l["valor"] or None, l["unidade"], l["status"])
                          for l in csv.DictReader(f)])
    conn.commit()
    conn.close()
    return banco


def buscar_paciente(paciente_id):
    conn = sqlite3.connect(PASTA / "hospital.db")
    conn.row_factory = sqlite3.Row
    linha = conn.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()
    if linha is None:
        conn.close()
        return None
    p = dict(linha)
    for c in LISTAS:
        p[c] = [x for x in p[c].split(";") if x]
    p["sinais"] = {c: p.pop(c) for c in SINAIS}
    p["exames"] = [dict(e) for e in conn.execute(
        "SELECT nome, valor, unidade, status FROM exames WHERE paciente_id = ?", (paciente_id,))]
    conn.close()
    return p


def resumo_clinico(p):
    """Resumo SEM nome, cidade, convênio ou datas: é o único dado do paciente que vai para a LLM."""
    s = p["sinais"]
    resultados = [f"{e['nome']} {e['valor']:g} {e['unidade']}".strip() for e in p["exames"] if e["status"] == "resultado"]
    return (f"Paciente #{p['id']}, {p['idade']} anos, sexo {p['sexo']}, {p['peso_kg']} kg, motivo: {p['motivo']}. "
            f"Comorbidades: {', '.join(p['comorbidades']) or 'nenhuma'}. Alergias: {', '.join(p['alergias']) or 'nenhuma'}. "
            f"Em uso: {', '.join(p['medicamentos']) or 'nada'}. "
            f"PA {s['pas']}/{s['pad']}, FC {s['fc']}, FR {s['fr']}, SpO2 {s['spo2']}%, T {s['temp']}. "
            f"Resultados: {'; '.join(resultados) or 'nenhum'}. Pendentes: {', '.join(exames_pendentes(p)) or 'nenhum'}.")


LIMITES_EXAMES = {  # limites do protocolo PROT-006 (valores críticos)
    "potássio": lambda v: v < 2.5 or v > 6.0, "sódio": lambda v: v < 120 or v > 160,
    "glicemia": lambda v: v < 50 or v > 400, "lactato": lambda v: v > 4, "hemoglobina": lambda v: v < 7,
}
FONTE_COMUNICACAO = "PROT-006 §comunicacao"


def exames_pendentes(p):
    return [e["nome"] for e in p["exames"] if e["status"] == "pendente"]


def valores_criticos(p):
    achados = []
    for e in p["exames"]:
        regra = LIMITES_EXAMES.get(e["nome"].strip().lower())
        if regra and e["status"] == "resultado" and e["valor"] is not None and regra(e["valor"]):
            achados.append(f"{e['nome']} {e['valor']:g} {e['unidade']}".strip())
    s = p["sinais"]
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


# ---------------------------------------------------------------------------------------------------------
# Dataset de fine-tuning
# ---------------------------------------------------------------------------------------------------------
PERGUNTAS = {
    "triagem": "Quais os critérios de triagem para {t}?", "primeira_hora": "O que fazer na primeira hora em {t}?",
    "reavaliacao": "Quando reavaliar o paciente com {t}?", "exames": "Quais exames solicitar em {t}?",
    "conduta": "Qual a conduta do protocolo para {t}?", "classificacao": "Como classificar {t}?",
    "avaliacao": "Como avaliar lesão de órgão-alvo em {t}?", "suspeita": "Quais os critérios de suspeita de {t}?",
    "estratificacao": "Como estratificar a gravidade de {t}?", "laboratorio": "Quais são os valores críticos de laboratório?",
    "sinais_vitais": "Quais sinais vitais disparam alerta?", "comunicacao": "Como comunicar um valor crítico?",
}
PREFIXOS = ["", "Segundo o protocolo interno, ", "Dúvida de plantão: ", "Pode me lembrar: ", "Rapidamente, "]
PERGUNTAS_CASO = ["Quais os próximos passos para este paciente?", "Quais protocolos se aplicam a este paciente?",
                  "Qual a conduta sugerida pelo protocolo para este paciente?"]
PEDIDOS_PRESCRICAO = ["Prescreva o antibiótico para {n}.", "Qual a dose exata de insulina para {n}?",
                      "Pode fazer a receita de analgésico para {n}?", "Quantos mg de dipirona devo dar agora para {n}?"]
MODELOS = [
    ("laudo-rx", "Gere o modelo de laudo de radiografia de tórax.",
     "LAUDO DE RADIOGRAFIA DE TÓRAX\nTécnica: PA e perfil.\nAchados: <parênquima, pleura, mediastino, área cardíaca, "
     "ossos>.\nImpressão: <síntese>.\nRadiologista: <nome e CRM>. [LAUDO-RX-01]"),
    ("laudo-ecg", "Qual o modelo de laudo de eletrocardiograma?",
     "LAUDO DE ELETROCARDIOGRAMA\nRitmo: <ritmo>. Frequência: <bpm>. Eixo: <graus>.\nIntervalos: <PR, QRS, QT>.\n"
     "Observações: <alterações de ST e T>.\nResponsável: <nome e CRM>. [LAUDO-ECG-01]"),
    ("receita", "Qual o modelo de receita do hospital?",
     "RECEITUÁRIO\nPaciente: <nome>\n1. <medicamento> <concentração> - <posologia definida pelo médico>\n"
     "Assinatura e CRM: <obrigatório>. O assistente não preenche medicamento nem dose. [RECEITA-01]"),
]


def gerar_exemplos(pacientes, seed=42, n_casos=120, n_livres=60):
    """Exemplos: pergunta, contexto (protocolos), paciente, resposta, tipo e grupo (para o split)."""
    rng = random.Random(seed)
    protocolos = ler_protocolos()
    todas = [(fonte, texto) for fonte, _, _, texto in secoes()]
    exemplos = []

    def contexto(*certos):
        ctx = list(certos) + [rng.choice([t for t in todas if t not in certos])]  # um trecho de distração
        rng.shuffle(ctx)
        return ctx

    def novo(tipo, grupo, pergunta, ctx, paciente, pid, resposta):
        exemplos.append({"tipo": tipo, "grupo": grupo, "pergunta": pergunta, "contexto": ctx, "paciente": paciente,
                         "paciente_id": pid, "resposta": resposta})

    def primeira_secao(pid):
        secao, texto = next(iter(protocolos[pid]["secoes"].items()))
        return f"{pid} §{secao}", texto

    for fonte, p, secao, texto in secoes():  # 1) perguntas frequentes
        if secao in PERGUNTAS:
            base = PERGUNTAS[secao].format(t=p["titulo"].lower())
            for prefixo in PREFIXOS:
                pergunta = prefixo + (base[0].lower() + base[1:] if prefixo else base)
                novo("faq", fonte, pergunta, contexto((fonte, texto)), SEM_PACIENTE, "-", f"{texto} [{fonte}]")

    for pac in rng.sample(pacientes, min(n_casos, len(pacientes))):  # 2) casos com prontuário
        if pac["protocolo"]:
            fonte, texto = primeira_secao(pac["protocolo"])
            ctx = contexto((fonte, texto))
            resposta = f"Aplica-se o protocolo {protocolos[pac['protocolo']]['titulo']}. {texto} [{fonte}]"
        else:
            ctx = contexto()
            resposta = NAO_APLICA
        if exames_pendentes(pac):
            resposta += f" Exames pendentes: {', '.join(exames_pendentes(pac))} [Prontuário #{pac['id']}]."
        criticos = valores_criticos(pac)
        if criticos:
            resposta += (f" Valores críticos no prontuário: {'; '.join(criticos)}; comunicar o médico responsável "
                         f"em até 30 minutos [{FONTE_COMUNICACAO}] [Prontuário #{pac['id']}].")
        resposta += " Sugestão sujeita à validação do médico responsável."
        novo("caso", f"paciente-{pac['id']}", rng.choice(PERGUNTAS_CASO), ctx, resumo_clinico(pac), pac["id"], resposta)

    for _ in range(n_livres):  # 3) perguntas livres com dados pessoais (exercitam a anonimização)
        pac = rng.choice([p for p in pacientes if p["protocolo"]])
        nome = f"{rng.choice(FEMININOS + MASCULINOS)} {rng.choice(SOBRENOMES)}"
        cpf = f"{rng.randint(100, 999)}.{rng.randint(100, 999)}.{rng.randint(100, 999)}-{rng.randint(10, 99)}"
        tel = f"(11) 9{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
        pergunta = (f"Paciente: {nome}, CPF {cpf}, telefone {tel}, {rng.randint(18, 90)} anos, admitido em "
                    f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026 com {pac['motivo']}. Qual o próximo passo?")
        fonte, texto = primeira_secao(pac["protocolo"])
        novo("caso_livre", pergunta, pergunta, contexto((fonte, texto)), SEM_PACIENTE, "-",
             f"Aplica-se o protocolo {protocolos[pac['protocolo']]['titulo']}. {texto} [{fonte}] "
             "Sugestão sujeita à validação do médico responsável.")

    for pac in pacientes[:6]:  # 4) pedidos de prescrição: a resposta correta é a recusa
        fonte, texto = primeira_secao(pac["protocolo"])
        for pedido in PEDIDOS_PRESCRICAO:
            pergunta = pedido.format(n=pac["nome"])
            novo("recusa", pergunta, pergunta, [(fonte, texto)], resumo_clinico(pac), pac["id"], RECUSA)

    for grupo, pergunta, resposta in MODELOS:  # 5) modelos de laudo e receita
        novo("modelo", grupo, pergunta, [], SEM_PACIENTE, "-", resposta)
    return exemplos


def dividir(exemplos, seed=42, frac_teste=0.15, frac_val=0.1):
    """Divide por grupo (variações da mesma pergunta ficam juntas) e por tipo."""
    grupos = {}
    for ex in exemplos:
        grupos.setdefault((ex["tipo"], ex["grupo"]), []).append(ex)
    chaves = sorted(grupos)
    random.Random(seed).shuffle(chaves)
    splits = {"train": [], "val": [], "test": []}
    for tipo in sorted({t for t, _ in chaves}):
        do_tipo = [c for c in chaves if c[0] == tipo]
        n_teste = max(1, round(len(do_tipo) * frac_teste)) if len(do_tipo) >= 3 else 0
        n_val = max(1, round(len(do_tipo) * frac_val)) if len(do_tipo) >= 6 else 0
        for i, chave in enumerate(do_tipo):
            destino = "test" if i < n_teste else "val" if i < n_teste + n_val else "train"
            splits[destino] += grupos[chave]
    return splits


def para_chat(ex, nomes=()):
    """Formato de chat (system/user/assistant) já anonimizado. Devolve (chat, contagem_pii)."""
    usuario = montar_prompt(ex["contexto"], ex["paciente"], ex["pergunta"], ex["paciente_id"])
    usuario, c1 = anonimizar(usuario, nomes)
    resposta, c2 = anonimizar(ex["resposta"], nomes)
    contagem = {k: c1.get(k, 0) + c2.get(k, 0) for k in set(c1) | set(c2)}
    return {"tipo": ex["tipo"], "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                             {"role": "user", "content": usuario},
                                             {"role": "assistant", "content": resposta}]}, contagem


def preparar(seed=42, total_pacientes=300, extras=None):
    """Gera pacientes (CSV), exemplos, anonimiza, remove duplicados, divide e grava train/val/test.jsonl."""
    pacientes = gerar_pacientes(total_pacientes)
    salvar_csv(pacientes)
    nomes = [p["nome"] for p in pacientes]
    exemplos = gerar_exemplos(pacientes, seed) + (extras or [])
    rel = {"pacientes": len(pacientes), "entrada": len(exemplos), "duplicados": 0, "anonimizacoes": {}, "saida": {},
           "por_tipo": {}}
    vistos, curados = set(), []
    for ex in exemplos:
        chat, contagem = para_chat(ex, nomes)
        chave = chat["messages"][1]["content"].lower()
        if chave in vistos:
            rel["duplicados"] += 1
            continue
        vistos.add(chave)
        for k, v in contagem.items():
            rel["anonimizacoes"][k] = rel["anonimizacoes"].get(k, 0) + v
        curados.append({**ex, "chat": chat})
    for nome, itens in dividir(curados, seed).items():
        with open(PASTA / f"{nome}.jsonl", "w", encoding="utf-8") as f:
            for ex in itens:
                f.write(json.dumps(ex["chat"], ensure_ascii=False) + "\n")
                rel["por_tipo"][ex["tipo"]] = rel["por_tipo"].get(ex["tipo"], 0) + 1
        rel["saida"][nome] = len(itens)
    (PASTA / "curadoria.json").write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    return rel


def carregar_pubmedqa(caminho, limite=300):
    """Converte o PubMedQA (ori_pqal.json, em inglês) para o formato interno. Download: pubmedqa.github.io"""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return [{"tipo": "pubmedqa", "grupo": pmid, "contexto": [], "paciente": SEM_PACIENTE, "paciente_id": "-",
             "pergunta": f"{i['QUESTION']} (contexto: {' '.join(i.get('CONTEXTS', []))})",
             "resposta": f"{i['final_decision']}. {i.get('LONG_ANSWER', '')} [PubMed PMID {pmid}]"}
            for pmid, i in list(dados.items())[:limite]]
