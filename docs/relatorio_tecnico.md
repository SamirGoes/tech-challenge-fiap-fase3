# Relatório técnico: MedAssist

> A versão principal usa apenas RAG (sem LLM). Os trechos **[PREENCHER]** só se aplicam se a fase 2 (fine-tuning) for
> executada no Colab.

## 1. Contexto e objetivo
Assistente virtual médico com dados fictícios do hospital. Responde dúvidas de médicos com base nos protocolos internos
e nos dados atualizados do paciente, executa um fluxo automatizado (exames pendentes, valores críticos, alertas) e
nunca prescreve: toda conduta exige validação do médico responsável.

## 2. Dados
### 2.1 Fontes
- **Pacientes:** 300 pacientes sintéticos em CSV (`pacientes.csv`, `exames.csv`), carregados em SQLite. Seis são
  curados à mão e os demais são gerados por 14 cenários clínicos (10 com protocolo, 4 de rotina) com seed fixa. Campos:
  idade, sexo, peso, altura, tipo sanguíneo, cidade, convênio, setor, data de admissão, médico, motivo, comorbidades,
  alergias, medicamentos e sinais vitais.
- **Protocolos:** 10 protocolos fictícios em Markdown (sepse, dor torácica, crise hipertensiva, cetoacidose,
  pneumonia, valores críticos, AVC isquêmico, asma aguda, infecção urinária, insuficiência cardíaca), 30 seções
  versionadas e citáveis. É a base do RAG.
- **Dataset de treino (fase 2):** 357 exemplos em formato de chat, no mesmo prompt usado em produção:

| Tipo | Quantidade | Descrição |
|---|---|---|
| faq | 150 | perguntas frequentes de médicos sobre cada seção |
| caso | 120 | pergunta com resumo do prontuário; resposta usa protocolo, exames pendentes e valores críticos |
| caso_livre | 60 | pergunta livre com nome, CPF, telefone e data (exercita a anonimização) |
| recusa | 24 | pedidos de prescrição; resposta é sempre a recusa |
| modelo | 3 | modelos de laudo e de receita |

- Opcional: PubMedQA (`preparar --pubmedqa`), em inglês.

### 2.2 Pré-processamento, anonimização e curadoria
| Etapa | O que faz |
|---|---|
| Anonimização | nome, CPF, telefone, e-mail, data, nº de prontuário; aplicada ao prompt e à resposta |
| Deduplicação | pelo prompt completo normalizado (0 duplicados na execução) |
| Split | por grupo e estratificado por tipo (train 270, val 35, test 52) |
| Formato | system / user / assistant, com protocolos recuperados e resumo do paciente no `user` |

Substituições na execução com `seed=42`: 84 nomes, 60 CPFs, 60 telefones, 60 datas.

## 3. Assistente médico (LangChain, sem LLM)
Pipeline LCEL em `assistente.py`, um Runnable nomeado por etapa:
1. **entrada:** anonimiza a pergunta, bloqueia pedidos fora do escopo e marca pedidos de prescrição.
2. **prontuario:** consulta SQL e resumo clínico sem identificadores.
3. **rag:** busca por similaridade na base vetorial Chroma (sem queries), com score mínimo 0,15.
4. **resposta:** monta o texto com os dois trechos mais relevantes e os exames pendentes, citando a fonte
   (`[PROT-001 §secao]`, `[Prontuário #N]`). Em pedidos de prescrição devolve a recusa; sem trecho relevante, informa
   que não há protocolo aplicável.
5. **guardrail_saida:** bloqueia prescrição direta, troca linguagem conclusiva, remove dados pessoais, exige fonte e
   anexa o aviso.
6. **saida:** resposta, fontes, violações e necessidade de revisão.

O prompt (`PROMPT` em `dados.py`) é montado e registrado no log mesmo sem LLM, para que a fase 2 use exatamente o mesmo
formato.

## 4. Fluxo automatizado
`exames pendentes ‖ valores críticos` (limites do PROT-006) → sugestão → revisão humana, quando há valor crítico ou
risco → alertas à equipe médica (críticos) e à enfermagem (pendências). Implementado com LangChain:
`RunnablePassthrough.assign` para a execução em paralelo e `RunnableBranch` para a revisão. Os alertas vão por SMTP
(se configurado) ou para `logs/alertas.txt`.

## 5. Segurança, auditoria e explainability
Eventos do log, todos com o mesmo `trace_id`, ordem e tempo: `entrada`, `prontuario`, `rag`, `resposta`,
`guardrail_saida`, `saida` e, no fluxo, `fluxo_iniciado`, `exames_pendentes`, `valores_criticos`, `revisao_humana`,
`alerta_enviado` e `alertas_emitidos`. O log também aparece no console e o LangSmith é opcional (variáveis de
ambiente). Testes em `tests/test_medassist.py`.

## 6. Avaliação
Conjunto de teste de 52 exemplos. Métricas: F1 de tokens, fonte correta, prescrição direta (deve ser 0), linguagem
conclusiva (deve ser 0) e recusa correta nos pedidos de prescrição.

| Modelo | F1 | Fonte correta | Prescrição direta | Recusa correta |
|---|---|---|---|---|
| RAG sem LLM (versão principal) | 0,62 | 98% | 0% | 100% |
| Qwen2.5-3B base (fase 2) | **[PREENCHER]** | | | |
| Qwen2.5-3B + LoRA (fase 2) | **[PREENCHER]** | | | |

Por tipo (RAG): faq 0,65; caso 0,54; caso_livre 0,60; recusa 1,00; modelo 0,00 (n=1, sem protocolo correspondente).

## 7. Fase 2 (opcional): fine-tuning
Modelo base `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (3B, abaixo do limite de 8B), QLoRA com Unsloth (LoRA r=16),
loss só nos tokens da resposta, 3 épocas, lr 2e-4. Executado em `notebooks/treino_colab.ipynb`. **[PREENCHER]** tempo de
treino, curva de loss e comparação de métricas.

## 8. Limitações e próximos passos
- Protocolos e pacientes sintéticos; validar com dados reais anonimizados e aprovação do comitê de ética.
- Métricas lexicais não medem correção clínica; incluir avaliação por especialista.
- Sem LLM, a resposta é extrativa (trechos dos protocolos): não sintetiza nem raciocina sobre o caso.
- Embeddings padrão são lexicais (hash); usar embeddings multilíngues (`MEDASSIST_EMBEDDINGS=hf`).
- Guardrails baseados em regras; evoluir para um classificador de segurança.
- Orquestração com LangGraph e LLM com fine-tuning como evolução natural.
