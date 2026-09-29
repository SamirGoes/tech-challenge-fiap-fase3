# Plano de entrega: Tech Challenge fase 3 (prazo 03/11/2026)

## Semana 1 (29/09 a 05/10): base do projeto ✅ em grande parte
- [x] Estrutura Python modular, CLI e testes
- [x] Protocolos fictícios, dados sintéticos, anonimização e curadoria
- [x] Banco de prontuários (SQLite), retriever de protocolos, assistente LangChain
- [x] Fluxo LangGraph com valores críticos, exames pendentes, alertas e revisão humana
- [x] Guardrails e log de auditoria
- [ ] Criar o repositório no GitHub e subir o código
- [ ] Definir o grupo e dividir as frentes

## Semana 2 (06/10 a 12/10): fine-tuning
- [ ] Rodar o notebook no Colab com o modelo base (baseline)
- [ ] Incluir um recorte do PubMedQA e/ou MedQuAD no dataset
- [ ] Treinar QLoRA (Qwen2.5-1.5B-Instruct)
- [ ] Salvar o adaptador (Hugging Face Hub ou Drive) e a curva de loss

## Semana 3 (13/10 a 19/10): integração com o modelo treinado
- [ ] Rodar assistente e fluxo com `MEDASSIST_LLM=hf`
- [ ] Ajustar prompt e guardrails com base nas respostas reais do modelo
- [ ] Avaliar base vs fine-tuned (F1, ROUGE-L, citação de fonte, taxa de prescrição direta)

## Semana 4 (20/10 a 26/10): relatório
- [ ] Completar `docs/relatorio_tecnico.md` com métricas, gráficos e análise de erros
- [ ] Diagramas finais (arquitetura e fluxo)
- [ ] Revisar README

## Semana 5 (27/10 a 02/11): vídeo e entrega
- [ ] Roteiro do vídeo (até 15 min): treino, fluxo automatizado, perguntas contextualizadas, logs
- [ ] Gravar e editar
- [ ] Entrega na plataforma até 03/11 (folga de 1 dia)

## Roteiro sugerido do vídeo (15 min)
1. Problema e arquitetura (2 min)
2. Dados: geração, anonimização e curadoria (2 min)
3. Fine-tuning no Colab e métricas base vs treinado (3 min)
4. Pergunta clínica contextualizada com fontes (2 min)
5. Fluxo LangGraph: paciente com valor crítico, pausa para revisão, alertas (3 min)
6. Segurança: pedido de prescrição recusado, logs de auditoria (2 min)
7. Limitações e próximos passos (1 min)
