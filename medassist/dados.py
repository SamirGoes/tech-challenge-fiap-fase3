import json
import random
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "Você é o MedAssist, assistente de apoio à decisão clínica do hospital. "
    "Responda com base nos protocolos internos e nos dados do paciente, cite a fonte entre colchetes "
    "e nunca prescreva: toda conduta ou medicamento exige validação do médico responsável."
)

PROTOCOLOS = {
    "PROT-001": {
        "titulo": "Sepse no adulto",
        "palavras": "sepse choque infecção lactato hipotensão febre confusão taquipneia",
        "secoes": {
            "triagem": "Suspeitar de sepse com infecção suspeita e ao menos dois critérios: FR >= 22 irpm, "
            "alteração do nível de consciência ou PAS <= 100 mmHg. Acionar o time de resposta rápida.",
            "primeira_hora": "Na primeira hora: coletar lactato e duas hemoculturas antes do antibiótico, iniciar "
            "antibiótico de amplo espectro conforme guia da CCIH e cristaloide 30 mL/kg se hipotensão ou lactato >= 4.",
            "reavaliacao": "Repetir lactato em 2 a 4 horas se o inicial for > 2 mmol/L. Considerar vasopressor se "
            "PAM < 65 mmHg após reposição volêmica, com avaliação do médico responsável.",
        },
    },
    "PROT-002": {
        "titulo": "Dor torácica",
        "palavras": "dor torácica infarto síndrome coronariana troponina ecg aperto sudorese",
        "secoes": {
            "triagem": "Todo paciente com dor torácica deve fazer ECG de 12 derivações em até 10 minutos. "
            "Prioridade vermelha se supradesnivelamento de ST ou instabilidade hemodinâmica.",
            "exames": "Troponina de alta sensibilidade na admissão e repetida em 1 a 3 horas, hemograma, "
            "eletrólitos, função renal e radiografia de tórax.",
            "conduta": "Com supradesnivelamento de ST, acionar a hemodinâmica (porta-balão < 90 min). A terapia "
            "antitrombótica é definida pelo médico, verificando alergias e risco de sangramento.",
        },
    },
    "PROT-003": {
        "titulo": "Crise hipertensiva",
        "palavras": "hipertensão crise hipertensiva pressão arterial cefaleia visão turva tontura",
        "secoes": {
            "classificacao": "Crise hipertensiva com PA >= 180/120 mmHg. Emergência quando há lesão aguda de "
            "órgão-alvo (AVC, edema agudo de pulmão, síndrome coronariana, dissecção de aorta).",
            "avaliacao": "Investigar lesão de órgão-alvo com exame neurológico, ECG, troponina, creatinina e urina tipo 1.",
            "conduta": "Na emergência, reduzir a PA de forma controlada (até 25% na primeira hora) em unidade "
            "monitorizada. Na urgência, ajustar medicação oral e reavaliar no ambulatório.",
        },
    },
    "PROT-004": {
        "titulo": "Cetoacidose diabética",
        "palavras": "diabetes cetoacidose glicemia hiperglicemia vômitos dor abdominal hálito cetônico poliúria",
        "secoes": {
            "diagnostico": "Suspeitar com glicemia > 250 mg/dL, pH < 7,3 ou bicarbonato < 18 mEq/L e cetonemia positiva.",
            "exames": "Gasometria venosa, eletrólitos com potássio, função renal, cetonemia e glicemia capilar horária.",
            "conduta": "Hidratação venosa. Checar o potássio antes da insulina: se K < 3,3 mEq/L, repor potássio "
            "antes. A insulina segue o protocolo de infusão e exige validação médica.",
        },
    },
    "PROT-005": {
        "titulo": "Pneumonia comunitária",
        "palavras": "pneumonia tosse febre dispneia infecção respiratória curb-65 dor pleurítica",
        "secoes": {
            "estratificacao": "Gravidade pelo CURB-65 (confusão, ureia > 50, FR >= 30, PAS < 90 ou PAD <= 60, "
            "idade >= 65). 0-1: ambulatorial; 2: internação; 3 ou mais: avaliar UTI.",
            "exames": "Radiografia de tórax, hemograma, ureia, creatinina, PCR e oximetria. Hemoculturas se internação.",
            "conduta": "Antibiótico empírico conforme guia da CCIH, ajustado a alergias e função renal. "
            "Reavaliar em 48 a 72 horas.",
        },
    },
    "PROT-006": {
        "titulo": "Valores críticos",
        "palavras": "valor crítico alerta potássio hipercalemia saturação hemoglobina fraqueza palpitações",
        "secoes": {
            "laboratorio": "Valores críticos: potássio < 2,5 ou > 6,0; sódio < 120 ou > 160; glicemia < 50 ou > 400; "
            "lactato > 4; hemoglobina < 7; troponina acima do percentil 99.",
            "sinais_vitais": "Sinais de alerta: SpO2 < 92%, PAS < 90 ou PA >= 180/120, FC > 130, FR >= 30, T >= 39 °C.",
            "comunicacao": "Todo valor crítico deve ser comunicado ao médico responsável em até 30 minutos e "
            "registrado no prontuário.",
        },
    },
}

PACIENTES = [
    {"id": 1, "nome": "Maria Aparecida Souza", "idade": 67, "sexo": "F", "motivo": "febre, confusão mental e hipotensão",
     "alergias": ["dipirona"], "comorbidades": ["hipertensão", "diabetes tipo 2"],
     "sinais": {"pas": 88, "pad": 52, "fc": 118, "fr": 26, "spo2": 93, "temp": 38.9},
     "exames": [("lactato", 4.6, "mmol/L"), ("leucócitos", 17800, "/mm3"), ("hemocultura", None, "")]},
    {"id": 2, "nome": "João Carlos Pereira", "idade": 58, "sexo": "M", "motivo": "dor torácica em aperto há 40 minutos",
     "alergias": [], "comorbidades": ["tabagismo"],
     "sinais": {"pas": 150, "pad": 95, "fc": 96, "fr": 20, "spo2": 96, "temp": 36.6},
     "exames": [("troponina", None, "ng/L"), ("ecg", None, ""), ("potássio", 4.1, "mEq/L")]},
    {"id": 3, "nome": "Ana Beatriz Lima", "idade": 45, "sexo": "F", "motivo": "cefaleia intensa e visão turva",
     "alergias": ["penicilina"], "comorbidades": ["hipertensão"],
     "sinais": {"pas": 212, "pad": 128, "fc": 88, "fr": 18, "spo2": 98, "temp": 36.5},
     "exames": [("creatinina", 1.1, "mg/dL"), ("troponina", None, "ng/L")]},
    {"id": 4, "nome": "Carlos Eduardo Ramos", "idade": 23, "sexo": "M", "motivo": "vômitos, dor abdominal e hálito cetônico",
     "alergias": [], "comorbidades": ["diabetes tipo 1"],
     "sinais": {"pas": 104, "pad": 66, "fc": 124, "fr": 30, "spo2": 97, "temp": 37.2},
     "exames": [("glicemia", 486, "mg/dL"), ("potássio", 3.1, "mEq/L"), ("gasometria venosa", None, "")]},
    {"id": 5, "nome": "Fernanda Oliveira Costa", "idade": 72, "sexo": "F", "motivo": "tosse produtiva, febre e dispneia",
     "alergias": ["sulfa"], "comorbidades": ["DPOC"],
     "sinais": {"pas": 118, "pad": 70, "fc": 104, "fr": 31, "spo2": 89, "temp": 38.4},
     "exames": [("ureia", 62, "mg/dL"), ("radiografia de tórax", None, "")]},
    {"id": 6, "nome": "José Roberto Alves", "idade": 80, "sexo": "M", "motivo": "fraqueza e palpitações",
     "alergias": [], "comorbidades": ["doença renal crônica"],
     "sinais": {"pas": 132, "pad": 78, "fc": 54, "fr": 18, "spo2": 95, "temp": 36.2},
     "exames": [("potássio", 6.8, "mEq/L"), ("creatinina", 3.4, "mg/dL"), ("ecg", None, "")]},
]

PERGUNTAS = {
    "triagem": "Quais os critérios de triagem para {t}?",
    "primeira_hora": "O que fazer na primeira hora em {t}?",
    "reavaliacao": "Quando reavaliar o paciente com {t}?",
    "exames": "Quais exames solicitar em {t}?",
    "conduta": "Qual a conduta do protocolo para {t}?",
    "classificacao": "Como classificar {t}?",
    "avaliacao": "Como avaliar lesão de órgão-alvo em {t}?",
    "diagnostico": "Quais os critérios diagnósticos de {t}?",
    "estratificacao": "Como estratificar a gravidade de {t}?",
    "laboratorio": "Quais são os valores críticos de laboratório?",
    "sinais_vitais": "Quais sinais vitais disparam alerta?",
    "comunicacao": "Como comunicar um valor crítico?",
}
PREFIXOS = ["", "Segundo o protocolo interno, ", "Dúvida de plantão: ", "Pode me lembrar: ", "Rapidamente, "]
NOMES = ["Antônio", "Beatriz", "Cláudia", "Daniel", "Eduarda", "Fábio", "Gabriela", "Hugo", "Isabel", "Jorge"]
SOBRENOMES = ["Almeida", "Barbosa", "Cardoso", "Dias", "Ferreira", "Gomes", "Moreira", "Nunes"]
RECUSA = ("Não posso prescrever nem definir doses. Posso resumir o que o protocolo recomenda e os dados do paciente "
          "para apoiar a sua decisão; a prescrição deve ser feita pelo médico responsável. [Política MedAssist]")


def secoes():
    for pid, p in PROTOCOLOS.items():
        for secao, texto in p["secoes"].items():
            yield f"{pid} §{secao}", p, secao, texto


def protocolo_do_quadro(motivo):
    palavras = set(re.findall(r"\w+", motivo.lower()))
    return max(PROTOCOLOS, key=lambda pid: len(palavras & set(PROTOCOLOS[pid]["palavras"].split())))


def gerar_exemplos(seed=42, n_casos=80):
    rng = random.Random(seed)
    exemplos = []

    for fonte, p, secao, texto in secoes():
        base = PERGUNTAS[secao].format(t=p["titulo"].lower())
        for prefixo in PREFIXOS:
            pergunta = prefixo + (base[0].lower() + base[1:] if prefixo else base)
            exemplos.append({"tipo": "faq", "grupo": fonte, "pergunta": pergunta, "resposta": f"{texto} [{fonte}]"})

    casos = [(p["nome"], p["idade"], p["motivo"]) for p in PACIENTES]
    casos += [(f"{rng.choice(NOMES)} {rng.choice(SOBRENOMES)}", rng.randint(18, 90), rng.choice(PACIENTES)["motivo"])
              for _ in range(n_casos)]
    for nome, idade, motivo in casos:
        pid = protocolo_do_quadro(motivo)
        secao, texto = next(iter(PROTOCOLOS[pid]["secoes"].items()))
        cpf = f"{rng.randint(100, 999)}.{rng.randint(100, 999)}.{rng.randint(100, 999)}-{rng.randint(10, 99)}"
        tel = f"(11) 9{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
        pergunta = (f"Paciente: {nome}, CPF {cpf}, telefone {tel}, {idade} anos, admitido em "
                    f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026 com {motivo}. Qual o próximo passo?")
        resposta = (f"Aplica-se o protocolo {PROTOCOLOS[pid]['titulo']}. {texto} [{pid} §{secao}] "
                    "Sugestão sujeita à validação do médico responsável.")
        exemplos.append({"tipo": "caso", "grupo": pergunta, "pergunta": pergunta, "resposta": resposta})

    for p in PACIENTES[:4]:
        for pedido in ["Prescreva o antibiótico para {n}.", "Qual a dose exata de insulina para {n}?"]:
            pergunta = pedido.format(n=p["nome"])
            exemplos.append({"tipo": "recusa", "grupo": pergunta, "pergunta": pergunta, "resposta": RECUSA})

    exemplos.append({"tipo": "modelo", "grupo": "laudo", "pergunta": "Gere o modelo de laudo de radiografia de tórax.",
                     "resposta": "LAUDO DE RADIOGRAFIA DE TÓRAX\nTécnica: PA e perfil.\nAchados: <parênquima, pleura, "
                     "mediastino, área cardíaca, ossos>.\nImpressão: <síntese>.\nRadiologista: <nome e CRM>. [LAUDO-RX-01]"})
    exemplos.append({"tipo": "modelo", "grupo": "receita", "pergunta": "Qual o modelo de receita do hospital?",
                     "resposta": "RECEITUÁRIO\nPaciente: <nome>\n1. <medicamento> <concentração> - <posologia definida "
                     "pelo médico>\nAssinatura e CRM: <obrigatório>. O assistente não preenche medicamento nem dose. "
                     "[RECEITA-01]"})
    return exemplos


REGRAS_PII = [
    ("CPF", r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    ("TELEFONE", r"\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b"),
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("DATA", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    ("PRONTUARIO", r"\bprontu[aá]rio\s*:?\s*\d{4,}\b"),
]
NOME_COM_CONTEXTO = r"(\b(?:Paciente|Sr\.|Sra\.|Dr\.|Dra\.)\s*:?\s+)[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wáéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wáéíóúâêôãõç]+)*"


def anonimizar(texto, nomes=()):
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


def dividir(exemplos, seed=42, frac_teste=0.15, frac_val=0.1):
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


def para_chat(ex):
    return {"tipo": ex["tipo"], "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["pergunta"]},
        {"role": "assistant", "content": ex["resposta"]},
    ]}


def preparar(pasta=Path("dados"), seed=42):
    nomes = [p["nome"] for p in PACIENTES]
    relatorio = {"entrada": 0, "duplicados": 0, "anonimizacoes": {}, "saida": {}}
    vistos, curados = set(), []
    for ex in gerar_exemplos(seed):
        relatorio["entrada"] += 1
        for campo in ("pergunta", "resposta"):
            ex[campo], contagem = anonimizar(ex[campo], nomes)
            for k, v in contagem.items():
                relatorio["anonimizacoes"][k] = relatorio["anonimizacoes"].get(k, 0) + v
        if ex["pergunta"].lower() in vistos:
            relatorio["duplicados"] += 1
            continue
        vistos.add(ex["pergunta"].lower())
        curados.append(ex)

    pasta.mkdir(parents=True, exist_ok=True)
    for nome, itens in dividir(curados, seed).items():
        with open(pasta / f"{nome}.jsonl", "w", encoding="utf-8") as f:
            for ex in itens:
                f.write(json.dumps(para_chat(ex), ensure_ascii=False) + "\n")
        relatorio["saida"][nome] = len(itens)
    return relatorio
