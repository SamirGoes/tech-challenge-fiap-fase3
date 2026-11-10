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

## Fase 2 (feita, local em GTX 1050 de 3 GB)
- [x] Fine-tuning QLoRA do Qwen2.5-0.5B-Instruct (transformers + peft + bitsandbytes), 3 épocas, cerca de 1 h
- [x] Avaliação no teste (52 exemplos): base F1 0,26; treinado F1 0,93; RAG puro F1 0,62
- [x] `--modelo treinado` testado no assistente; curva de loss e métricas em `docs/fase2/`
- [x] Relatório técnico atualizado com os resultados

## Falta (depende de você)
- [ ] Empurrar a branch para um GitHub onde você tenha permissão (o `origin` do SamirGoes deu 403 para seu usuário)
- [ ] Gravar o vídeo (até 15 min) e entregar na plataforma
- [ ] Opcional: repetir a fase 2 no Colab com modelo maior (3B) via `MEDASSIST_MODELO`

## Roteiro do vídeo (15 min)
1. Problema e arquitetura (2 min)
2. Dados: pacientes CSV, protocolos, anonimização (2 min)
3. RAG mutável: adicionar um protocolo novo e perguntar sobre ele (2 min)
4. Chat no console com paciente, fontes e pedido de prescrição recusado (3 min)
5. Fluxo automatizado: valor crítico, revisão humana, alerta (3 min)
6. Auditoria (`auditoria <trace_id>`) e testes (2 min)
7. Limitações e fase 2 (1 min)
