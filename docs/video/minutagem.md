# Minutagem do vídeo MedAssist (~7 min 30 s)

Narração: **uma voz** (Samir, gerada no ElevenLabs) a partir de `roteiro_elevenlabs.txt`.
Tela: notebook `notebooks/treino_colab.ipynb` no Colab, com GPU T4, gravado em um take e editado por cima do áudio.

As durações das células vêm da execução de teste na T4 em 22/11/2026, que rodou do início ao fim sem erros.
A duração de cada fala é uma estimativa (cerca de 2,5 palavras por segundo, mais as pausas `<break>`). Depois de gerar o áudio, ajuste os inícios pela duração real de cada bloco.

## Como gerar o áudio
1. No ElevenLabs, use a sua voz clonada com o modelo multilíngue (Multilingual v2 ou v3), em português.
2. **Gere um clipe por bloco** (B01 a B14). Copie só o texto, sem a linha `### Bxx`. Isso facilita encaixar cada fala no momento certo do vídeo, e se uma frase sair ruim você só refaz aquele bloco.
3. Os `<break time="…" />` viram pausas. Se o seu modelo não aceitar a tag, troque por reticências ou ponto final.
4. Pronúncias para conferir no primeiro teste:
   - "Qwen": se sair errado, escreva "Kuen".
   - "LangChain", "Chroma", "trace ID", "LoRA": o roteiro já usa "Q-LoRA".
   - "PROT zero zero um": já está por extenso.
   - Os números já estão escritos por extenso no roteiro.

## Antes de gravar (fora do vídeo, cerca de 27 min)
Siga `preparar-ambiente-colab.md`. Deixe **já executadas** as seções 1, 2 e 3 (ambiente, dados, treino, gráfico, avaliações e tabela) e as células `!rm -rf logs` e "carregar modelo" do começo da seção 4.

| Célula | Tempo medido na T4 |
|---|---|
| `nvidia-smi` | 1 s (+ aviso "não criado pelo Google": clicar em *Executar assim mesmo*) |
| Clone + `pip install` | 26 s |
| `preparar` | 12 s |
| Tabela de pacientes / PROT-001 | 1 s cada |
| **Treino QLoRA (3 épocas, 102 passos)** | **7 min 58 s** (418 s de treino + download do modelo) |
| Gráfico da loss | 1 s |
| **Avaliações rag + base + treinado** | **16 min 58 s** (a do modelo base, sozinha, leva cerca de 10 min) |
| Tabela de métricas | 2 s |
| **Carregar modelo treinado** | **47 s** |

## Durante a gravação

"Rodar" quer dizer apertar Shift+Enter na célula. A coluna "tempo" é quanto a célula levou no teste.

| Bloco | Início–fim (fala) | Tela | Ação do operador | Tempo da célula |
|---|---|---|---|---|
| B01 Abertura | 0:00–0:19 | Topo do notebook (título e tabela) | — | — |
| B02 Problema e regras | 0:20–0:58 | Mesma tela. Em "nunca prescreve", destacar a última linha do markdown | Rolar devagar | — |
| B03 Arquitetura | 0:59–1:33 | **README no GitHub**, diagrama da seção Arquitetura | Trocar de aba para o README | — |
| B04 Ambiente e dados | 1:34–2:19 | Seções 1 e 2 já executadas: clone, saída do `preparar` (contagens), tabela de pacientes, PROT-001 | Rolar acompanhando a fala: "300 pacientes" → tabela; "357 exemplos" → JSON; "anonimização" → bloco `anonimizacoes`; "protocolos" → PROT-001 | já executada |
| B05 Fine-tuning | 2:20–2:54 | Saída do treino: linha `trainable params: 8,798,208 … 1.7497`, depois as linhas de loss e "Adaptador salvo" | Rolar dentro da saída | já executada |
| B06 Loss e métricas | 2:55–3:43 | Gráfico da loss, depois a **tabela de métricas** (RAG 0,615/98%/100%; base 0,220/23%/0%; treinado 0,930/94%/100%) | Rolar até o gráfico; em "Depois avaliamos", ir para a tabela | já executada |
| B07 Modelo e RAG | 3:44–4:01 | Célula "carregar modelo", com a saída `Modelo treinado carregado: qwen-lora` | — | já executada |
| B08 Pergunta 1 | 4:02–4:25 | Célula "Pergunta 1" | **Rodar no início do bloco.** A resposta aparece cerca de 9 s depois, durante a pausa de 1,5 s e a frase sobre o log | 8,6 s |
| B09 Pergunta 2 | 4:26–4:38 | Célula "Pergunta 2" | **Rodar no início do bloco** | 6,6 s |
| B10 Pergunta 3 + log | 4:39–5:04 | Célula "Pergunta 3", depois a célula `/log` | **Rodar a Pergunta 3 no início**; em "E aqui está o log", rodar a célula `/log` (linha `4. resposta 8116 ms`) | 9,7 s + 1,5 s |
| B11 RAG mutável | 5:05–5:38 | Seção 5 | Em "Se buscamos": rodar `rag buscar` (4,5 s). Em "Agora adicionamos": rodar adicionar + buscar (9,8 s). Em "Perguntando": rodar `perguntar` (4,8 s). Em "E podemos removê-lo": rodar remover + listar (8,6 s) | 4,5 / 9,8 / 4,8 / 8,6 s |
| B12 Fluxo | 5:39–6:23 | Seção 6 | **Rodar a célula do fluxo no início do bloco** (cerca de 19 s; a fala cobre esse tempo). Em "Eles ficam neste arquivo": rodar `cat logs/alertas.txt` | 18,8 s + 2,7 s |
| B13 Auditoria e testes | 6:24–6:46 | Seção 7 | Em "Cada requisição": rodar `auditoria` (5,7 s). Em "Esta é a linha do tempo": rodar a linha do tempo (1,8 s). Em "Por fim, os dezessete testes": **rodar o pytest** | 5,7 / 1,8 s; pytest 45 s |
| B14 Limitações e conclusão | 6:47–7:26 | Célula do pytest rodando, até aparecer `17 passed` | Esperar. O `17 passed` aparece por volta de 7:10, antes de "E os dezessete testes passaram" | (pytest termina) |

**Duração total estimada: cerca de 7 min 30 s**, contando 1 s entre blocos. Com a margem de edição, fica entre 8 e 9 min, bem abaixo dos 15 min de limite.

## Dicas de edição
- Se uma célula terminar antes da fala, mantenha a tela parada. Se terminar depois (por exemplo, o pytest), acelere o trecho de espera em 2x ou corte.
- Os únicos momentos em que a tela espera o modelo são B08 a B10 (8 a 10 s cada) e B12 (19 s). A fala já foi escrita para cobrir essas esperas.
- Para a abertura (B01 e B02), dá para usar a câmera ou um slide com o nome dos três integrantes em vez do notebook.
- Os requisitos do PDF e onde aparecem: treinamento e funcionamento da LLM (B05 a B07), fluxo automatizado (B12), perguntas clínicas (B08 a B10), logs e validação (B08, B10, B13).
