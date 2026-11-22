# Preparar o ambiente da apresentação (Google Colab)

Link do notebook (branch `feat/video-colab`):
https://colab.research.google.com/github/SamirGoes/tech-challenge-fiap-fase3/blob/feat%2Fvideo-colab/notebooks/treino_colab.ipynb

## 1. Antes de gravar (uns 25 minutos, sem gravar)
1. Abra o link e entre na conta Google.
2. **Ambiente de execução → Alterar tipo de ambiente de execução → GPU T4** e clique em **Conectar**.
3. Rode a primeira célula de código. No aviso "Este notebook não foi criado pelo Google", clique em **Executar assim mesmo**.
4. Rode em ordem as seções **1, 2 e 3**: ambiente, dados, fine-tuning, gráfico, avaliação e tabela.
   - O treino leva cerca de 8 minutos na T4.
   - As três avaliações (RAG, base e treinado) levam mais alguns minutos. Veja `minutagem.md`.
   - Essas saídas ficam na tela e aparecem no vídeo **já executadas**.
5. Rode a célula que **carrega o modelo** (início da seção 4). Assim ele já está aquecido na GPU.
6. Aumente o zoom do navegador (Cmd/Ctrl +), feche as abas que não for usar e desligue as notificações.

> Atalho: se o adaptador já estiver no Google Drive (seção 8), use a célula "Atalho" em vez do treino.

## 2. Durante a gravação
Siga a `minutagem.md`. Ela diz em que momento rodar cada célula e o que aparece na tela.

Daqui em diante, cada célula roda em poucos segundos:
- as perguntas ao modelo treinado: cerca de 5 a 15 s cada;
- o fluxo: cerca de 15 s;
- o pytest: cerca de 40 s.

Grave a tela em um take contínuo e encaixe o áudio do ElevenLabs depois, na edição. Onde a célula demorar mais que a fala, acelere ou corte o trecho.

## 3. Se precisar refazer
- Para recomeçar só as demos (seções 4 a 7), rode a célula `!rm -rf logs` para limpar a auditoria. Também dá para reiniciar a sessão e repetir o passo 1, mas aí tudo roda de novo.
- Se a sessão do Colab cair, o adaptador treinado se perde. Por isso vale salvar no Drive (seção 8) depois do primeiro treino.
