"""Dados do projeto: pacientes sintéticos (CSV), exemplos de fine-tuning, anonimização, curadoria e split.

Tudo é fictício e reprodutível (seed). O dataset de treino usa o mesmo prompt do assistente em produção.
"""

import csv
import json
import random
import re
from pathlib import Path

from medassist.anonimizacao import anonimizar
from medassist.config import config
from medassist.prompt import (RECUSA, SEM_PACIENTE, SEM_PROTOCOLO, SYSTEM_PROMPT, formatar_protocolos,
                              montar_prompt)
from medassist.prontuario import resumo_clinico
from medassist.protocolos import carregar_protocolos, secoes
from medassist.clinica import FONTE_COMUNICACAO, exames_pendentes, valores_criticos


def _ex(nome, valor=None, unidade=""):
    return {"nome": nome, "valor": valor, "unidade": unidade, "status": "pendente" if valor is None else "resultado"}


def _sv(pas, pad, fc, fr, spo2, temp):
    return {"pas": pas, "pad": pad, "fc": fc, "fr": fr, "spo2": spo2, "temp": temp}


# Seis pacientes curados à mão (um por quadro clínico), com valores críticos de propósito.
CURADOS = [
    {"id": 1, "nome": "Maria Aparecida Souza", "idade": 67, "sexo": "F", "motivo": "febre, confusão mental e hipotensão",
     "alergias": ["dipirona"], "comorbidades": ["hipertensão", "diabetes tipo 2"],
     "sinais": _sv(88, 52, 118, 26, 93, 38.9),
     "exames": [_ex("lactato", 4.6, "mmol/L"), _ex("leucócitos", 17800, "/mm3"), _ex("hemocultura")]},
    {"id": 2, "nome": "João Carlos Pereira", "idade": 58, "sexo": "M", "motivo": "dor torácica em aperto há 40 minutos",
     "alergias": [], "comorbidades": ["tabagismo"], "sinais": _sv(150, 95, 96, 20, 96, 36.6),
     "exames": [_ex("troponina", None, "ng/L"), _ex("ecg"), _ex("potássio", 4.1, "mEq/L")]},
    {"id": 3, "nome": "Ana Beatriz Lima", "idade": 45, "sexo": "F", "motivo": "cefaleia intensa e visão turva",
     "alergias": ["penicilina"], "comorbidades": ["hipertensão"], "sinais": _sv(212, 128, 88, 18, 98, 36.5),
     "exames": [_ex("creatinina", 1.1, "mg/dL"), _ex("troponina", None, "ng/L")]},
    {"id": 4, "nome": "Carlos Eduardo Ramos", "idade": 23, "sexo": "M",
     "motivo": "vômitos, dor abdominal e hálito cetônico", "alergias": [], "comorbidades": ["diabetes tipo 1"],
     "sinais": _sv(104, 66, 124, 30, 97, 37.2),
     "exames": [_ex("glicemia", 486, "mg/dL"), _ex("potássio", 3.1, "mEq/L"), _ex("gasometria venosa")]},
    {"id": 5, "nome": "Fernanda Oliveira Costa", "idade": 72, "sexo": "F", "motivo": "tosse produtiva, febre e dispneia",
     "alergias": ["sulfa"], "comorbidades": ["DPOC"], "sinais": _sv(118, 70, 104, 31, 89, 38.4),
     "exames": [_ex("ureia", 62, "mg/dL"), _ex("radiografia de tórax")]},
    {"id": 6, "nome": "José Roberto Alves", "idade": 80, "sexo": "M", "motivo": "fraqueza e palpitações",
     "alergias": [], "comorbidades": ["doença renal crônica"], "sinais": _sv(132, 78, 54, 18, 95, 36.2),
     "exames": [_ex("potássio", 6.8, "mEq/L"), _ex("creatinina", 3.4, "mg/dL"), _ex("ecg")]},
]

PRIMEIROS = ["Antônio", "Beatriz", "Cláudia", "Daniel", "Eduarda", "Fábio", "Gabriela", "Hugo", "Isabel", "Jorge",
             "Kátia", "Leonardo", "Mariana", "Nelson", "Olívia", "Rafael", "Sílvia", "Tiago"]
SOBRENOMES = ["Almeida", "Barbosa", "Cardoso", "Dias", "Ferreira", "Gomes", "Moreira", "Nunes", "Rocha", "Teixeira"]
ALERGIAS = [[], [], [], ["dipirona"], ["penicilina"], ["sulfa"], ["AAS"]]


def _n(r, a, b, casas=0):
    v = r.uniform(a, b)
    return round(v, casas) if casas else int(round(v))


def _talvez(r, nome, valor, unidade="", p_pendente=0.3):
    return _ex(nome, None if r.random() < p_pendente else valor, unidade)


CENARIOS = {
    "sepse": dict(
        motivos=["febre, confusão mental e hipotensão", "febre, confusão e taquipneia após infecção urinária",
                 "febre alta, calafrios e hipotensão"],
        comorb=["hipertensão", "diabetes tipo 2", "doença renal crônica", "câncer em quimioterapia"], idade=(40, 90),
        sinais=lambda r: _sv(_n(r, 76, 98), _n(r, 40, 60), _n(r, 105, 132), _n(r, 22, 31), _n(r, 89, 95),
                             _n(r, 38.2, 39.6, 1)),
        exames=lambda r: [_talvez(r, "lactato", _n(r, 1.8, 6.0, 1), "mmol/L", 0.1),
                          _ex("leucócitos", _n(r, 12000, 21000), "/mm3"),
                          _ex("creatinina", _n(r, 0.9, 2.6, 1), "mg/dL"), _ex("hemocultura")]),
    "dor_toracica": dict(
        motivos=["dor torácica em aperto há 40 minutos", "dor torácica com sudorese irradiada para o braço esquerdo",
                 "dor torácica e aperto no peito após esforço"],
        comorb=["tabagismo", "dislipidemia", "hipertensão", "diabetes tipo 2"], idade=(40, 85),
        sinais=lambda r: _sv(_n(r, 120, 165), _n(r, 70, 100), _n(r, 70, 110), _n(r, 16, 22), _n(r, 94, 99),
                             _n(r, 36.0, 37.0, 1)),
        exames=lambda r: [_talvez(r, "troponina", _n(r, 5, 80), "ng/L", 0.6), _ex("ecg"),
                          _ex("potássio", _n(r, 3.6, 4.8, 1), "mEq/L")]),
    "crise_hipertensiva": dict(
        motivos=["cefaleia intensa e visão turva", "cefaleia e tontura com pressão muito elevada"],
        comorb=["hipertensão", "doença renal crônica", "obesidade"], idade=(35, 85),
        sinais=lambda r: _sv(_n(r, 182, 230), _n(r, 110, 135), _n(r, 70, 100), _n(r, 16, 20), _n(r, 96, 99),
                             _n(r, 36.0, 37.0, 1)),
        exames=lambda r: [_ex("creatinina", _n(r, 0.9, 2.2, 1), "mg/dL"), _talvez(r, "troponina", _n(r, 4, 30), "ng/L", 0.6),
                          _ex("urina tipo 1")]),
    "cetoacidose": dict(
        motivos=["vômitos, dor abdominal e hálito cetônico", "poliúria, desidratação e hálito cetônico"],
        comorb=["diabetes tipo 1", "diabetes tipo 2"], idade=(18, 45),
        sinais=lambda r: _sv(_n(r, 95, 115), _n(r, 55, 72), _n(r, 110, 135), _n(r, 24, 32), _n(r, 95, 99),
                             _n(r, 36.5, 37.8, 1)),
        exames=lambda r: [_ex("glicemia", _n(r, 280, 560), "mg/dL"), _ex("potássio", _n(r, 2.8, 5.4, 1), "mEq/L"),
                          _ex("gasometria venosa"), _ex("cetonemia")]),
    "pneumonia": dict(
        motivos=["tosse produtiva, febre e dispneia", "tosse, febre e dor pleurítica"],
        comorb=["DPOC", "tabagismo", "insuficiência cardíaca", "diabetes tipo 2"], idade=(50, 90),
        sinais=lambda r: _sv(_n(r, 100, 135), _n(r, 60, 80), _n(r, 95, 115), _n(r, 24, 33), _n(r, 86, 94),
                             _n(r, 37.8, 39.2, 1)),
        exames=lambda r: [_ex("ureia", _n(r, 30, 80), "mg/dL"), _ex("PCR", _n(r, 60, 220), "mg/L"),
                          _ex("radiografia de tórax"), _ex("hemocultura")]),
    "valores_criticos": dict(
        motivos=["fraqueza e palpitações", "fraqueza muscular e palpitações"],
        comorb=["doença renal crônica", "insuficiência cardíaca", "diabetes tipo 2"], idade=(55, 90),
        sinais=lambda r: _sv(_n(r, 110, 145), _n(r, 65, 85), _n(r, 45, 70), _n(r, 16, 20), _n(r, 93, 97),
                             _n(r, 36.0, 37.0, 1)),
        exames=lambda r: [_ex("potássio", _n(r, 5.2, 7.4, 1), "mEq/L"), _ex("creatinina", _n(r, 1.8, 4.0, 1), "mg/dL"),
                          _ex("ecg")]),
}


def gerar_pacientes(total: int = 50, seed: int = 7) -> list[dict]:
    """Pacientes sintéticos: os 6 curados e o resto gerado por cenário clínico."""
    r = random.Random(seed)
    pacientes = [dict(p) for p in CURADOS]
    usados = {p["nome"] for p in pacientes}
    nomes = list(CENARIOS)
    while len(pacientes) < total:
        c = CENARIOS[nomes[len(pacientes) % len(nomes)]]
        nome = f"{r.choice(PRIMEIROS)} {r.choice(SOBRENOMES)}"
        if nome in usados:
            continue
        usados.add(nome)
        pacientes.append({
            "id": len(pacientes) + 1, "nome": nome, "idade": r.randint(*c["idade"]), "sexo": r.choice("FM"),
            "motivo": r.choice(c["motivos"]), "alergias": list(r.choice(ALERGIAS)),
            "comorbidades": r.sample(c["comorb"], r.randint(1, 2)), "sinais": c["sinais"](r), "exames": c["exames"](r)})
    return pacientes


def salvar_csv(pacientes: list[dict], pasta=None) -> list[Path]:
    pasta = Path(pasta or config().pasta_dados)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_p, caminho_e = pasta / "pacientes.csv", pasta / "exames.csv"
    with open(caminho_p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "nome", "idade", "sexo", "motivo", "alergias", "comorbidades",
                    "pas", "pad", "fc", "fr", "spo2", "temp"])
        for p in pacientes:
            s = p["sinais"]
            w.writerow([p["id"], p["nome"], p["idade"], p["sexo"], p["motivo"], ";".join(p["alergias"]),
                        ";".join(p["comorbidades"]), s["pas"], s["pad"], s["fc"], s["fr"], s["spo2"], s["temp"]])
    with open(caminho_e, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paciente_id", "nome", "valor", "unidade", "status"])
        for p in pacientes:
            for e in p["exames"]:
                w.writerow([p["id"], e["nome"], "" if e["valor"] is None else e["valor"], e["unidade"], e["status"]])
    return [caminho_p, caminho_e]


PERGUNTAS = {
    "triagem": "Quais os critérios de triagem para {t}?",
    "primeira_hora": "O que fazer na primeira hora em {t}?",
    "reavaliacao": "Quando reavaliar o paciente com {t}?",
    "exames": "Quais exames solicitar em {t}?",
    "conduta": "Qual a conduta do protocolo para {t}?",
    "classificacao": "Como classificar {t}?",
    "avaliacao": "Como avaliar lesão de órgão-alvo em {t}?",
    "suspeita": "Quais os critérios de suspeita de {t}?",
    "estratificacao": "Como estratificar a gravidade de {t}?",
    "laboratorio": "Quais são os valores críticos de laboratório?",
    "sinais_vitais": "Quais sinais vitais disparam alerta?",
    "comunicacao": "Como comunicar um valor crítico?",
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


def protocolo_do_quadro(motivo: str, protocolos: dict) -> str:
    palavras = set(re.findall(r"\w+", motivo.lower()))
    return max(protocolos, key=lambda pid: len(palavras & set(protocolos[pid]["palavras"].lower().split())))


def gerar_exemplos(pacientes: list[dict], seed: int = 42, n_livres: int = 40) -> list[dict]:
    """Exemplos no formato interno: pergunta, contexto (protocolos), paciente, resposta, tipo e grupo."""
    rng = random.Random(seed)
    protocolos = carregar_protocolos()
    todas = [(fonte, texto) for fonte, _, _, texto in secoes()]
    exemplos = []

    def contexto_com(fonte, texto):
        ctx = [(fonte, texto), rng.choice([t for t in todas if t[0] != fonte])]
        rng.shuffle(ctx)
        return ctx

    def base(tipo, grupo, pergunta, contexto, paciente, paciente_id, resposta):
        return {"tipo": tipo, "grupo": grupo, "pergunta": pergunta, "contexto": contexto, "paciente": paciente,
                "paciente_id": paciente_id, "resposta": resposta}

    # 1) Perguntas frequentes de médicos sobre cada seção dos protocolos
    for fonte, p, secao, texto in secoes():
        if secao not in PERGUNTAS:
            continue
        pergunta_base = PERGUNTAS[secao].format(t=p["titulo"].lower())
        for prefixo in PREFIXOS:
            pergunta = prefixo + (pergunta_base[0].lower() + pergunta_base[1:] if prefixo else pergunta_base)
            exemplos.append(base("faq", fonte, pergunta, contexto_com(fonte, texto), SEM_PACIENTE, "-",
                                 f"{texto} [{fonte}]"))

    # 2) Casos com prontuário: a resposta usa protocolo, exames pendentes e valores críticos do paciente
    for pac in pacientes:
        pid = protocolo_do_quadro(pac["motivo"], protocolos)
        secao, texto = next(iter(protocolos[pid]["secoes"].items()))
        fonte = f"{pid} §{secao}"
        resposta = f"Aplica-se o protocolo {protocolos[pid]['titulo']}. {texto} [{fonte}]"
        pendentes = exames_pendentes(pac)
        if pendentes:
            resposta += f" Exames pendentes: {', '.join(pendentes)} [Prontuário #{pac['id']}]."
        criticos = valores_criticos(pac)
        if criticos:
            resposta += (f" Valores críticos no prontuário: {'; '.join(criticos)}; comunicar o médico responsável "
                         f"em até 30 minutos [{FONTE_COMUNICACAO}] [Prontuário #{pac['id']}].")
        resposta += " Sugestão sujeita à validação do médico responsável."
        exemplos.append(base("caso", f"paciente-{pac['id']}", rng.choice(PERGUNTAS_CASO), contexto_com(fonte, texto),
                             resumo_clinico(pac), pac["id"], resposta))

    # 3) Perguntas livres com dados pessoais (exercitam a anonimização)
    for _ in range(n_livres):
        pac = rng.choice(pacientes)
        nome = f"{rng.choice(PRIMEIROS)} {rng.choice(SOBRENOMES)}"
        cpf = f"{rng.randint(100, 999)}.{rng.randint(100, 999)}.{rng.randint(100, 999)}-{rng.randint(10, 99)}"
        tel = f"(11) 9{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
        pergunta = (f"Paciente: {nome}, CPF {cpf}, telefone {tel}, {rng.randint(18, 90)} anos, admitido em "
                    f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026 com {pac['motivo']}. Qual o próximo passo?")
        pid = protocolo_do_quadro(pac["motivo"], protocolos)
        secao, texto = next(iter(protocolos[pid]["secoes"].items()))
        fonte = f"{pid} §{secao}"
        exemplos.append(base("caso_livre", pergunta, pergunta, contexto_com(fonte, texto), SEM_PACIENTE, "-",
                             f"Aplica-se o protocolo {protocolos[pid]['titulo']}. {texto} [{fonte}] "
                             "Sugestão sujeita à validação do médico responsável."))

    # 4) Pedidos de prescrição: a resposta correta é sempre a recusa
    for pac in pacientes[:6]:
        pid = protocolo_do_quadro(pac["motivo"], protocolos)
        secao, texto = next(iter(protocolos[pid]["secoes"].items()))
        for pedido in PEDIDOS_PRESCRICAO:
            pergunta = pedido.format(n=pac["nome"])
            exemplos.append(base("recusa", pergunta, pergunta, [(f"{pid} §{secao}", texto)], resumo_clinico(pac),
                                 pac["id"], RECUSA))

    # 5) Modelos de laudo e de receita
    for grupo, pergunta, resposta in MODELOS:
        exemplos.append(base("modelo", grupo, pergunta, [], SEM_PACIENTE, "-", resposta))
    return exemplos


def dividir(exemplos, seed=42, frac_teste=0.15, frac_val=0.1):
    """Divisão por grupo e estratificada por tipo: variações da mesma pergunta não vazam para o teste."""
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


def para_chat(ex: dict, nomes=()) -> tuple[dict, dict]:
    """Converte para o formato de chat (system/user/assistant), já anonimizado. Devolve (chat, contagem_pii)."""
    usuario = montar_prompt(formatar_protocolos(ex["contexto"]), ex["paciente"], ex["pergunta"], ex["paciente_id"])
    usuario, c1 = anonimizar(usuario, nomes)
    resposta, c2 = anonimizar(ex["resposta"], nomes)
    contagem = {k: c1.get(k, 0) + c2.get(k, 0) for k in set(c1) | set(c2)}
    chat = {"tipo": ex["tipo"], "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                             {"role": "user", "content": usuario},
                                             {"role": "assistant", "content": resposta}]}
    return chat, contagem


def preparar(pasta=None, seed=42, total_pacientes=50, extras=None) -> dict:
    """Gera pacientes (CSV), exemplos, anonimiza, remove duplicados, divide e grava train/val/test.jsonl."""
    pasta = Path(pasta or config().pasta_dados)
    pacientes = gerar_pacientes(total_pacientes)
    salvar_csv(pacientes, pasta)
    nomes = [p["nome"] for p in pacientes]
    exemplos = gerar_exemplos(pacientes, seed) + (extras or [])
    relatorio = {"pacientes": len(pacientes), "entrada": len(exemplos), "duplicados": 0, "anonimizacoes": {}, "saida": {},
                 "por_tipo": {}}
    vistos, curados = set(), []
    for ex in exemplos:
        chat, contagem = para_chat(ex, nomes)
        chave = chat["messages"][1]["content"].lower()
        if chave in vistos:
            relatorio["duplicados"] += 1
            continue
        vistos.add(chave)
        for k, v in contagem.items():
            relatorio["anonimizacoes"][k] = relatorio["anonimizacoes"].get(k, 0) + v
        curados.append({**ex, "chat": chat})
    for nome, itens in dividir(curados, seed).items():
        with open(pasta / f"{nome}.jsonl", "w", encoding="utf-8") as f:
            for ex in itens:
                f.write(json.dumps(ex["chat"], ensure_ascii=False) + "\n")
                relatorio["por_tipo"][ex["tipo"]] = relatorio["por_tipo"].get(ex["tipo"], 0) + 1
        relatorio["saida"][nome] = len(itens)
    (pasta / "curadoria.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    return relatorio


def carregar_pubmedqa(caminho, limite=300) -> list[dict]:
    """Converte o PubMedQA (ori_pqal.json, em inglês) para o formato interno. Download: pubmedqa.github.io"""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    exemplos = []
    for pmid, item in list(dados.items())[:limite]:
        contexto = " ".join(item.get("CONTEXTS", []))
        exemplos.append({"tipo": "pubmedqa", "grupo": pmid, "contexto": [], "paciente": SEM_PACIENTE, "paciente_id": "-",
                         "pergunta": f"{item['QUESTION']} (contexto: {contexto})",
                         "resposta": f"{item['final_decision']}. {item.get('LONG_ANSWER', '')} [PubMed PMID {pmid}]"})
    return exemplos
