# Relatório técnico: MedAssist

> Esqueleto do relatório. Os trechos marcados com **[PREENCHER]** dependem do treino no Colab.

## 1. Contexto e objetivo
Assistente de apoio à decisão clínica, treinado com protocolos internos do hospital, que responde dúvidas de
médicos com base nos protocolos e nos dados atualizados do paciente, e organiza fluxos automatizados
(exames pendentes, valores críticos, alertas) sem nunca prescrever diretamente.

## 2. Dados
### 2.1 Fontes
- Protocolos internos fictícios (6 protocolos, 18 seções versionadas): sepse, dor torácica, crise hipertensiva,
  cetoacidose, pneumonia e valores críticos (`src/medassist/data/protocolos.py`).
- Perguntas frequentes de médicos geradas por templates e paráfrases; casos clínicos sintéticos; pedidos de
  prescrição (para ensinar recusa); modelos de laudo e de receita.
- Opcional: PubMedQA (`ori_pqal.json`) e MedQuAD.

### 2.2 Pré-processamento, anonimização e curadoria
| Etapa | O que faz |
|---|---|
| Normalização | Unicode NFC, espaços, quebras de linha |
| Anonimização | nome (lista conhecida + contexto), CPF, CNS, RG, telefone, e-mail, datas, CEP, nº de prontuário |
| Filtros | tamanho mínimo; resposta sem fonte citada é descartada |
| Deduplicação | hash da pergunta normalizada |
| Split | estratificado por tipo; paráfrases da mesma pergunta ficam no mesmo split (sem vazamento) |

Resultado da execução com `seed=42`: 229 exemplos de entrada, 5 duplicados removidos, 173/20/31 em
treino/validação/teste; substituições: 89 nomes, 87 CPFs, 87 telefones, 86 datas.

## 3. Fine-tuning
- Modelo base: `Qwen/Qwen2.5-1.5B-Instruct`. O enunciado cita LLaMA e Falcon como exemplos e aceita outro modelo; o Qwen2.5
  foi escolhido por ter licença aberta (Apache 2.0, sem acesso restrito no Hugging Face), bom desempenho em português e
  caber numa T4 com 4 bits.
- Técnica: QLoRA (NF4, double quant), LoRA r=16, alpha=32, dropout 0,05 em todas as projeções de atenção e MLP.
- Loss calculada apenas sobre os tokens da resposta (prompt mascarado com -100).
- Hiperparâmetros: 3 épocas, lr 2e-4 cosseno, warmup 5%, batch efetivo 16, max_len 1024.
- **[PREENCHER]** tempo de treino, parâmetros treináveis, curva de loss (`docs/loss.png`).

## 4. Assistente médico (LangChain)
1. Consulta ao prontuário (SQLite) e geração de resumo clínico sem identificadores.
2. Recuperação BM25 das seções de protocolo, usando pergunta + motivo da internação.
3. Prompt com os trechos numerados por fonte e os dados do paciente.
4. LLM customizada (LoRA) como `LLM` do LangChain.
5. Guardrails de saída: bloqueio de prescrição direta, exigência de fonte, aviso de validação humana.

## 5. Fluxo automatizado (LangGraph)
Ver diagrama no README. Nós: `carregar_prontuario` → (`verificar_exames_pendentes` ‖ `avaliar_valores_criticos`)
→ `gerar_sugestao` → [`revisao_humana` se valor crítico ou pedido de prescrição] → `emitir_alertas`.
A revisão humana usa `interrupt` com checkpoint, então o fluxo fica pausado até a decisão do médico.

## 6. Segurança, auditoria e explainability
Descrever os eventos do log (exemplo real de `medassist auditoria <trace_id>`) e os testes em `tests/test_seguranca.py`.

## 7. Avaliação
Métricas no conjunto de teste (31 exemplos): token F1, ROUGE-L, citação da fonte correta e taxa de prescrição direta.
O modelo recebe só a pergunta (sem contexto recuperado), para medir o que o fine-tuning ensinou.

| Modelo | Token F1 | ROUGE-L | Fonte correta | Prescrição direta |
|---|---|---|---|---|
| Baseline extrativo (BM25, backend `demo`) | 0,51 | 0,45 | 100% | 0% |
| Qwen2.5-1.5B base | **[PREENCHER]** | | | |
| Qwen2.5-1.5B + LoRA | **[PREENCHER]** | | | |

**[PREENCHER]** análise por tipo (faq, caso, recusa) e exemplos de erros.

## 8. Limitações e próximos passos
- Protocolos e pacientes sintéticos; validar com dados reais anonimizados e aprovação do comitê de ética.
- Métricas lexicais não medem correção clínica; incluir avaliação por especialista ou LLM-as-judge com rubrica.
- Integração com o PEP real (HL7 FHIR) e com canais de alerta (pager, Teams).
