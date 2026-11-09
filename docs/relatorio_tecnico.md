# Relatório técnico: MedAssist

> Os trechos marcados com **[PREENCHER]** dependem do treino no Colab.

## 1. Contexto e objetivo
Assistente virtual médico treinado com dados fictícios do hospital. Responde dúvidas de médicos com base nos
protocolos internos e nos dados atualizados do paciente, executa um fluxo automatizado (exames pendentes, valores
críticos, alertas) e nunca prescreve: toda conduta exige validação do médico responsável.

## 2. Dados
### 2.1 Fontes
- **Pacientes:** 50 pacientes sintéticos em CSV (`pacientes.csv`, `exames.csv`), carregados em SQLite. Seis são curados
  à mão (um por quadro clínico, com valores críticos de propósito) e 44 são gerados por cenário clínico com seed fixa.
- **Protocolos:** 6 protocolos fictícios em Markdown (sepse, dor torácica, crise hipertensiva, cetoacidose, pneumonia,
  valores críticos), 18 seções versionadas e citáveis.
- **Dataset de fine-tuning:** 207 exemplos em formato de chat, no mesmo prompt usado em produção:

| Tipo | Quantidade | Descrição |
|---|---|---|
| faq | 90 | perguntas frequentes de médicos sobre cada seção (5 formas de perguntar) |
| caso | 50 | pergunta com resumo do prontuário; resposta usa protocolo, exames pendentes e valores críticos |
| caso_livre | 40 | pergunta livre com nome, CPF, telefone e data (exercita a anonimização) |
| recusa | 24 | pedidos de prescrição; resposta é sempre a recusa |
| modelo | 3 | modelos de laudo (radiografia, ECG) e de receita |

- Opcional: PubMedQA (`preparar --pubmedqa`), em inglês.

### 2.2 Pré-processamento, anonimização e curadoria
| Etapa | O que faz |
|---|---|
| Anonimização | nome (lista conhecida e por contexto), CPF, telefone, e-mail, data, nº de prontuário; aplicada ao prompt e à resposta |
| Deduplicação | pelo prompt completo normalizado |
| Split | por grupo e estratificado por tipo (train 152, val 21, test 34); variações da mesma pergunta ficam no mesmo split |
| Formato | system / user / assistant, com protocolos recuperados e resumo do paciente no `user` |

Substituições na execução com `seed=42`: 64 nomes, 40 CPFs, 40 telefones, 40 datas. Não há duplicados.

## 3. Fine-tuning
- **Modelo base:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (3B, abaixo do limite de 8B do desafio, licença aberta, bom
  em português, cabe numa T4 em 4 bits).
- **Técnica:** QLoRA com Unsloth: base congelada em 4 bits, LoRA r=16, alpha=32 em todas as projeções de atenção e MLP.
- **Loss:** só nos tokens da resposta (`train_on_responses_only`).
- **Hiperparâmetros:** 3 épocas, lr 2e-4 cosseno, warmup 5%, batch efetivo 8, `adamw_8bit`, max_len 2048.
- **[PREENCHER]** tempo de treino, parâmetros treináveis, curva de loss (`saida/loss.png`).

## 4. Assistente médico (LangChain)
Pipeline LCEL em `agente.py`, com um Runnable nomeado por etapa:
1. **entrada:** anonimiza a pergunta, bloqueia pedidos fora do escopo e marca pedidos de prescrição.
2. **prontuario:** consulta SQL e resumo clínico sem identificadores.
3. **rag_protocolos:** busca por similaridade na base vetorial Chroma (sem queries), com score mínimo.
4. **llm:** prompt com protocolos numerados por fonte, resumo do paciente e pergunta; LLM customizada (LoRA).
5. **guardrail_saida:** bloqueia prescrição direta, troca linguagem conclusiva, remove dados pessoais, exige fonte,
   garante a recusa em pedidos de prescrição e anexa o aviso.
6. **saida:** resposta, fontes, violações e necessidade de revisão.

## 5. Fluxo automatizado
`exames pendentes ‖ valores críticos` (limites do PROT-006) → sugestão do agente → revisão humana, quando há valor
crítico ou risco → alertas por e-mail à equipe médica (críticos) e à enfermagem (pendências). Implementado com
LangChain: `RunnablePassthrough.assign` para a execução em paralelo e `RunnableBranch` para a revisão.

## 6. Segurança, auditoria e explainability
Eventos do log, todos com o mesmo `trace_id`, ordem, status e duração: `entrada`, `prontuario`, `rag`, `llm`,
`guardrail_saida`, `saida` e, no fluxo, `fluxo_iniciado`, `exames_pendentes`, `valores_criticos`, `revisao_humana`,
`alerta_enviado` e `alertas_emitidos`. **[PREENCHER]** exemplo real de `medassist auditoria <trace_id>` com o modelo
treinado e, se usado, captura do LangSmith. Testes em `tests/test_seguranca.py` e `tests/test_agente.py`.

## 7. Avaliação
Conjunto de teste de 34 exemplos, com o mesmo prompt do assistente. Métricas: F1 de tokens, fonte correta,
prescrição direta (deve ser 0), linguagem conclusiva (deve ser 0) e recusa correta nos pedidos de prescrição.

| Modelo | F1 | Fonte correta | Prescrição direta | Recusa correta |
|---|---|---|---|---|
| Demo extrativo (baseline sem GPU) | 0,63 | 97% | 0% | 100% |
| Qwen2.5-3B base | **[PREENCHER]** | | | |
| Qwen2.5-3B + LoRA | **[PREENCHER]** | | | |

**[PREENCHER]** análise por tipo (faq, caso, caso_livre, recusa, modelo) e exemplos de erros.

## 8. Limitações e próximos passos
- Protocolos e pacientes sintéticos; validar com dados reais anonimizados e aprovação do comitê de ética.
- Métricas lexicais não medem correção clínica; incluir avaliação por especialista ou LLM-as-judge com rubrica.
- Embeddings padrão são lexicais (hash); usar embeddings multilíngues (`MEDASSIST_EMBEDDINGS=hf`) com mais protocolos.
- Guardrails baseados em regras; evoluir para um classificador de segurança.
- Orquestração com LangGraph (grafo com nós separados e pausa persistente da revisão humana) como evolução natural.
- Integração com o prontuário eletrônico real (HL7 FHIR) e canais de alerta (pager, Teams).
