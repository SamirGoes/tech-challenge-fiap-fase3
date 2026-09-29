"""Protocolos internos FICTÍCIOS do hospital, criados para fins acadêmicos.

Os textos resumem condutas gerais amplamente difundidas e não substituem
diretrizes oficiais. Cada seção tem um identificador estável usado como
fonte citável nas respostas do assistente (explainability).
"""

from __future__ import annotations

PROTOCOLOS: list[dict] = [
    {
        "id": "PROT-001",
        "titulo": "Sepse no adulto",
        "versao": "2.1",
        "palavras_chave": ["sepse", "choque séptico", "lactato", "infecção", "hipotensão", "febre", "confusão mental", "taquipneia"],
        "secoes": {
            "triagem": "Suspeitar de sepse em paciente com infecção suspeita e ao menos dois critérios: "
            "frequência respiratória >= 22 irpm, alteração do nível de consciência ou pressão sistólica <= 100 mmHg. "
            "Acionar o time de resposta rápida e abrir o pacote de sepse.",
            "primeira_hora": "Na primeira hora: coletar lactato e duas amostras de hemocultura antes do antibiótico, "
            "iniciar antibiótico de amplo espectro conforme guia da CCIH, e iniciar cristaloide 30 mL/kg "
            "se hipotensão ou lactato >= 4 mmol/L.",
            "reavaliacao": "Repetir lactato em até 2-4 horas se o inicial estiver elevado (> 2 mmol/L). "
            "Considerar vasopressor se PAM < 65 mmHg após reposição volêmica, com avaliação do médico responsável.",
        },
    },
    {
        "id": "PROT-002",
        "titulo": "Dor torácica na emergência",
        "versao": "1.4",
        "palavras_chave": ["dor torácica", "infarto", "síndrome coronariana", "troponina", "ecg", "aperto", "sudorese"],
        "secoes": {
            "triagem": "Todo paciente com dor torácica deve realizar ECG de 12 derivações em até 10 minutos da chegada. "
            "Classificar como prioridade vermelha se supradesnivelamento de ST ou instabilidade hemodinâmica.",
            "exames": "Solicitar troponina de alta sensibilidade na admissão e repetir em 1-3 horas conforme o algoritmo "
            "institucional. Solicitar hemograma, eletrólitos, função renal e radiografia de tórax.",
            "conduta": "Com supradesnivelamento de ST, acionar a hemodinâmica imediatamente (meta porta-balão < 90 min). "
            "A terapia antitrombótica deve ser definida pelo médico assistente, verificando alergias e risco de sangramento.",
        },
    },
    {
        "id": "PROT-003",
        "titulo": "Crise hipertensiva",
        "versao": "1.2",
        "palavras_chave": ["hipertensão", "crise hipertensiva", "pressão arterial", "emergência hipertensiva", "cefaleia", "visão turva", "tontura"],
        "secoes": {
            "classificacao": "Considerar crise hipertensiva com PA >= 180/120 mmHg. Diferenciar urgência (sem lesão aguda "
            "de órgão-alvo) de emergência (com lesão aguda: AVC, edema agudo de pulmão, SCA, dissecção de aorta).",
            "avaliacao": "Investigar lesão de órgão-alvo com exame neurológico, ECG, troponina, creatinina, urina tipo 1 "
            "e fundo de olho quando disponível.",
            "conduta": "Na emergência hipertensiva, reduzir a PA de forma controlada (não mais que 25% na primeira hora) "
            "em unidade monitorizada. Na urgência, preferir ajuste de medicação oral e reavaliação ambulatorial.",
        },
    },
    {
        "id": "PROT-004",
        "titulo": "Cetoacidose diabética",
        "versao": "1.0",
        "palavras_chave": ["diabetes", "cetoacidose", "glicemia", "hiperglicemia", "cetonemia", "vômitos", "dor abdominal", "hálito cetônico", "poliúria"],
        "secoes": {
            "diagnostico": "Suspeitar com glicemia > 250 mg/dL, pH < 7,3 ou bicarbonato < 18 mEq/L e cetonemia positiva.",
            "exames": "Solicitar gasometria venosa, eletrólitos com potássio, função renal, cetonemia e glicemia capilar horária.",
            "conduta": "Iniciar hidratação venosa. Checar o potássio antes da insulina: se K < 3,3 mEq/L, repor potássio "
            "antes de iniciar insulina. A prescrição de insulina segue o protocolo de infusão e exige validação médica.",
        },
    },
    {
        "id": "PROT-005",
        "titulo": "Pneumonia adquirida na comunidade",
        "versao": "2.0",
        "palavras_chave": ["pneumonia", "tosse", "febre", "infecção respiratória", "curb-65", "dispneia", "dor pleurítica", "tosse produtiva"],
        "secoes": {
            "estratificacao": "Estratificar gravidade com CURB-65 (confusão, ureia > 50 mg/dL, FR >= 30, PAS < 90 ou "
            "PAD <= 60, idade >= 65). Escore 0-1: considerar tratamento ambulatorial; 2: internação; >= 3: avaliar UTI.",
            "exames": "Radiografia de tórax, hemograma, ureia, creatinina, PCR e oximetria. Hemoculturas se internação.",
            "conduta": "Antibioticoterapia empírica conforme guia da CCIH, ajustada a alergias e função renal. "
            "Reavaliar em 48-72 horas.",
        },
    },
    {
        "id": "PROT-006",
        "titulo": "Valores críticos de laboratório e sinais vitais",
        "versao": "3.0",
        "palavras_chave": ["valor crítico", "alerta", "potássio", "saturação", "hemoglobina", "hipercalemia", "fraqueza", "palpitações", "bradicardia"],
        "secoes": {
            "laboratorio": "São valores críticos: potássio < 2,5 ou > 6,0 mEq/L; sódio < 120 ou > 160 mEq/L; "
            "glicemia < 50 ou > 400 mg/dL; lactato > 4 mmol/L; hemoglobina < 7 g/dL; troponina acima do percentil 99.",
            "sinais_vitais": "São sinais vitais de alerta: SpO2 < 92%, PAS < 90 mmHg ou PA >= 180/120 mmHg, "
            "FC > 130 bpm, FR >= 30 irpm, temperatura >= 39 °C.",
            "comunicacao": "Todo valor crítico deve ser comunicado ao médico responsável em até 30 minutos e registrado "
            "no prontuário com horário e nome de quem recebeu a informação.",
        },
    },
]


def iter_secoes():
    """Gera (fonte, protocolo, secao, texto) para cada seção de cada protocolo."""
    for p in PROTOCOLOS:
        for secao, texto in p["secoes"].items():
            yield f"{p['id']} v{p['versao']} §{secao}", p, secao, texto
