# Plano do projeto

Branch de trabalho: `feat/rafael-fase3`. Escopo: LangChain apenas (LangGraph fica como evolução).

## Pronto (roda em CPU, 27 testes)
- [x] Dados sintéticos: 50 pacientes (CSV -> SQLite), 6 protocolos em Markdown, dataset de treino anonimizado
- [x] RAG vetorial mutável (Chroma): indexar, adicionar, remover e buscar sem queries
- [x] Agente LangChain: entrada -> prontuário -> RAG -> LLM -> guardrail de saída -> saída
- [x] Guardrails: sem prescrição, sem linguagem conclusiva, anonimização, fonte obrigatória, aviso
- [x] Log de ponta a ponta com trace_id, ordem e duração; LangSmith opcional
- [x] Fluxo automatizado: pendências, valores críticos, revisão humana e alertas por e-mail (ou caixa de saída)
- [x] Notebook do Colab com Unsloth, avaliação base x treinado, README e esqueleto do relatório

## Falta (depende de GPU ou de você)
- [ ] Rodar o notebook no Colab: baseline, treino e avaliação (código de treino não foi testado em GPU)
- [ ] Empurrar a branch para o GitHub (o notebook clona `feat/rafael-fase3`)
- [ ] Preencher o relatório técnico (campos **[PREENCHER]**) com métricas, curva de loss e exemplos de log
- [ ] Rodar o assistente e o fluxo com `MEDASSIST_LLM=treinado` e ajustar prompt e guardrails
- [ ] Gravar o vídeo (até 15 min) e entregar na plataforma

## Roteiro do vídeo (15 min)
1. Problema e arquitetura (2 min)
2. Dados: pacientes CSV, protocolos, anonimização (2 min)
3. Fine-tuning no Colab e métricas base x treinado (3 min)
4. Pergunta clínica contextualizada com fontes e RAG mutável (2 min)
5. Fluxo automatizado: valor crítico, revisão humana, alerta por e-mail (3 min)
6. Pedido de prescrição recusado e linha do tempo da auditoria (2 min)
7. Limitações (1 min)
