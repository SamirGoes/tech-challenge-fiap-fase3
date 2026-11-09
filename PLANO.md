# Plano do projeto

Branch de trabalho: `feat/rafael-fase3`. Escopo: LangChain apenas, **sem LLM no início (só RAG)**. Fine-tuning e
LangGraph ficam como evolução.

## Pronto (roda em CPU, 17 testes)
- [x] 5 arquivos Python: `dados`, `assistente`, `fluxo`, `treino`, `main`
- [x] Base com 300 pacientes sintéticos (CSV -> SQLite) e 10 protocolos em Markdown
- [x] RAG vetorial mutável (Chroma): indexar, adicionar, remover, listar e buscar sem queries
- [x] Assistente LangChain sem LLM: entrada -> prontuário -> RAG -> resposta -> guardrail -> saída
- [x] Chat no console (`python -m medassist chat`) com `/paciente`, `/log` e `/sair`
- [x] Guardrails: sem prescrição, sem linguagem conclusiva, anonimização, fonte obrigatória, aviso
- [x] Log de ponta a ponta com trace_id, ordem e tempo; LangSmith por variáveis de ambiente
- [x] Fluxo automatizado: pendências, valores críticos, revisão humana e alertas (SMTP ou `logs/alertas.txt`)
- [x] Avaliação do RAG no conjunto de teste (F1 0,62; fonte correta 98%; recusa 100%; prescrição direta 0%)

## Falta (depende de você)
- [ ] Empurrar a branch para o GitHub
- [ ] Preencher o relatório técnico (campos **[PREENCHER]**, se quiser incluir a fase 2)
- [ ] Gravar o vídeo (até 15 min) e entregar na plataforma

## Fase 2 (opcional, precisa de GPU)
- [ ] Rodar `notebooks/treino_colab.ipynb` no Colab: baseline, treino QLoRA e avaliação
- [ ] Usar `--modelo treinado` no chat e no fluxo; comparar métricas com o RAG

## Roteiro do vídeo (15 min)
1. Problema e arquitetura (2 min)
2. Dados: pacientes CSV, protocolos, anonimização (2 min)
3. RAG mutável: adicionar um protocolo novo e perguntar sobre ele (2 min)
4. Chat no console com paciente, fontes e pedido de prescrição recusado (3 min)
5. Fluxo automatizado: valor crítico, revisão humana, alerta (3 min)
6. Auditoria (`auditoria <trace_id>`) e testes (2 min)
7. Limitações e fase 2 (1 min)
