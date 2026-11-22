# Minutagem do vídeo MedAssist (~8 min)

Narração: **uma voz** (Samir, gerada no ElevenLabs) a partir de `roteiro_elevenlabs.txt`.
Tela: notebook `notebooks/treino_colab.ipynb` no Colab, com GPU T4, gravado em um take e editado por cima do áudio.

As durações das células vêm da execução de teste na T4 em 22/11/2026, que rodou do início ao fim sem erros.
A duração de cada fala é uma estimativa (cerca de 2,5 palavras por segundo, mais as pausas `<break>`). Depois de gerar o áudio, ajuste os inícios pela duração real de cada bloco.

## Blocos do roteiro
`roteiro_elevenlabs.txt` contém **só o texto falado**, sem títulos, para o ElevenLabs não ler "B09". Cada parágrafo é um bloco, na ordem abaixo.

| Bloco | Assunto | Começa com |
|---|---|---|
| B01 | Abertura | “Oi, pessoal! Eu sou o Samir e,…” |
| B02 | Problema e regras | “Então, qual é o problema? Um hospital…” |
| B03 | Arquitetura | “Bom, esse aqui é o caminho de…” |
| B04 | Ambiente e dados | “Vamos lá. No Colab, a primeira célula…” |
| B05 | Fine-tuning | “Agora, o fine-tuning. A gente partiu do…” |
| B06 | Loss e métricas | “No gráfico dá pra ver a loss…” |
| B07 | Carregar o modelo e RAG | “Agora sim, o assistente funcionando, com o…” |
| B08 | Pergunta 1 (sepse, paciente 1) | “Primeira pergunta, pra paciente um: o que…” |
| B09 | Pergunta 2 (prescrição) | “Agora, vamos pedir uma prescrição: prescreva antibiótico…” |
| B10 | Pergunta 3 (AVC, sem paciente) e log | “Agora sem paciente: qual a janela de…” |
| B11 | RAG mutável | “Outra coisa: a base de conhecimento é…” |
| B12 | Fluxo automatizado | “Além das perguntas, tem o fluxo automático…” |
| B13 | Auditoria e testes | “Cada requisição tem um identificador único, o…” |
| B14 | Limitações e conclusão | “Enquanto os testes rodam... as limitações. Os…” |

## Como gerar o áudio
1. No ElevenLabs, use a sua voz clonada com o modelo multilíngue (Multilingual v2 ou v3), em português.
2. **Gere um clipe por bloco** (B01 a B14): cada parágrafo do arquivo é um bloco. Isso facilita encaixar cada fala no momento certo do vídeo, e se uma frase sair ruim você só refaz aquele bloco.
3. Os `<break time="…" />` viram pausas. Se o seu modelo não aceitar a tag, troque por reticências ou ponto final.
4. Pronúncias para conferir no primeiro teste:
   - "Qwen": se sair errado, escreva "Kuen".
   - "LangChain", "Chroma", "trace ID", "LoRA": o roteiro já usa "Q-LoRA".
   - "PROT zero zero um": já está por extenso.
   - Os números já estão escritos por extenso no roteiro.

## Ensaio completo (22/11/2026, runtime novo na T4)
Preparação do zero em cerca de 26 min: instalação 25 s, dados 11 s, treino 7 min 20 s, avaliações 17 min 23 s e carga do modelo 46 s.
Depois, o `roteiro_tela.js` rodou do B01 ao B14, iniciado pelo atalho Option+Shift+S. Resultado: 31 ações executadas, nenhuma célula não encontrada, nenhuma célula rodando por cima de outra, e todas terminaram antes da ação seguinte.
Tempos ao vivo: perguntas 5,8 a 8 s, fluxo 17 s, pytest 44 s (17 passed).
Diferença encontrada: o F1 do modelo treinado deu 0,906, contra 0,930 no primeiro treino. A fala virou "mais ou menos zero vírgula nove".

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

## Durante a gravação: automático com `roteiro_tela.js`

O script [roteiro_tela.js](roteiro_tela.js) faz no navegador **tudo o que está na coluna "Ação do operador"**: rola até cada célula, roda as células e acompanha a saída enquanto ela cresce. Ele foi testado no Colab de B07 a B14, e as 20 ações saíram na ordem e no tempo certos.

1. Grave numa janela **normal** do Chrome, sem o Claude controlando a aba. Assim não aparece o aviso "o Claude está controlando este navegador". Feche o banner azul do Colab no X.
2. Abra o Console (**Cmd+Option+J**), cole o conteúdo inteiro de `roteiro_tela.js` e aperte Enter. Na primeira vez, o Chrome pede para você digitar `allow pasting`.
3. Feche o Console, clique no notebook, comece a gravar a tela e aperte **Option+Shift+S**. O roteiro começa no B01.
   - **Option+Shift+X** para o roteiro.
   - Para regravar um trecho, digite no Console, por exemplo, `medassistDemo.iniciar('B08')`.
4. Depois de gerar o áudio, troque os valores de `DUR` no topo do script pela duração real de cada clipe, em segundos. Assim a tela acompanha a narração exatamente.

A tabela abaixo é o que o script faz em cada momento. Ela serve também para conduzir à mão (Shift+Enter roda a célula). A coluna "tempo" é quanto a célula levou no teste.

| Bloco | Início–fim (fala) | Tela | Ação do operador | Tempo da célula |
|---|---|---|---|---|
| B01 Abertura | 0:00–0:21 | Topo do notebook (título e tabela) | — | — |
| B02 Problema e regras | 0:22–1:03 | Mesma tela. Em "nunca prescreve", destacar a última linha do markdown | Rolar devagar | — |
| B03 Arquitetura | 1:04–1:46 | Célula **Arquitetura** do notebook (diagrama) | Rolar até o diagrama | — |
| B04 Ambiente e dados | 1:47–2:33 | Seções 1 e 2 já executadas: clone, saída do `preparar` (contagens), tabela de pacientes, PROT-001 | Rolar acompanhando a fala: "300 pacientes" → tabela; "357 exemplos" → JSON; "anonimização" → bloco `anonimizacoes`; "protocolos" → PROT-001 | já executada |
| B05 Fine-tuning | 2:34–3:12 | Saída do treino: linha `trainable params: 8,798,208 … 1.7497`, depois as linhas de loss e "Adaptador salvo" | Rolar dentro da saída | já executada |
| B06 Loss e métricas | 3:13–4:05 | Gráfico da loss, depois a **tabela de métricas** (RAG 0,615/98%/100%; base 0,220/23%/0%; treinado ~0,91–0,93/94%/100%; o F1 varia um pouco a cada treino, por isso a fala diz "mais ou menos zero vírgula nove") | Rolar até o gráfico; em "Depois avaliamos", ir para a tabela | já executada |
| B07 Modelo e RAG | 4:06–4:24 | Célula "carregar modelo", com a saída `Modelo treinado carregado: qwen-lora` | — | já executada |
| B08 Pergunta 1 | 4:25–4:49 | Célula "Pergunta 1" | **Rodar no início do bloco.** A resposta aparece cerca de 9 s depois, durante a pausa de 1,5 s e a frase sobre o log | 8,6 s |
| B09 Pergunta 2 | 4:50–5:03 | Célula "Pergunta 2" | **Rodar no início do bloco** | 6,6 s |
| B10 Pergunta 3 + log | 5:04–5:30 | Célula "Pergunta 3", depois a célula `/log` | **Rodar a Pergunta 3 no início**; em "E aqui está o log", rodar a célula `/log` (linha `4. resposta 8116 ms`) | 9,7 s + 1,5 s |
| B11 RAG mutável | 5:31–6:08 | Seção 5 | Em "Se eu busco": rodar `rag buscar` (4,5 s). Em "Aí eu adiciono": rodar adicionar + buscar (9,8 s). Em "Se eu pergunto": rodar `perguntar` (4,8 s). Em "E dá pra remover": rodar remover + listar (8,6 s) | 4,5 / 9,8 / 4,8 / 8,6 s |
| B12 Fluxo | 6:09–6:55 | Seção 6 | **Rodar a célula do fluxo no início do bloco** (cerca de 19 s; a fala cobre esse tempo). Em "Eles ficam nesse arquivo": rodar `cat logs/alertas.txt` | 18,8 s + 2,7 s |
| B13 Auditoria e testes | 6:56–7:20 | Seção 7 | Em "Cada requisição": rodar `auditoria` (5,7 s). Em "Essa aqui é a linha do tempo": rodar a linha do tempo (1,8 s). Em "Então dá pra auditar" (3 s antes de "E, por fim, os dezessete testes"): **rodar o pytest** | 5,7 / 1,8 s; pytest 45 s |
| B14 Limitações e conclusão | 7:21–8:04 | Célula do pytest rodando, até aparecer `17 passed` | Esperar. O `17 passed` aparece por volta de 7:54, antes de "E os dezessete testes passaram" | (pytest termina) |

**Duração total estimada: cerca de 8 min**, contando 1 s entre blocos. Com a margem de edição, fica entre 8 min 30 s e 9 min 30 s, bem abaixo dos 15 min de limite.

## Dicas de edição
- Se uma célula terminar antes da fala, mantenha a tela parada. Se terminar depois (por exemplo, o pytest), acelere o trecho de espera em 2x ou corte.
- Os únicos momentos em que a tela espera o modelo são B08 a B10 (8 a 10 s cada) e B12 (19 s). A fala já foi escrita para cobrir essas esperas.
- Para a abertura (B01 e B02), dá para usar a câmera ou um slide com o nome dos três integrantes em vez do notebook.
- Os requisitos do PDF e onde aparecem: treinamento e funcionamento da LLM (B05 a B07), fluxo automatizado (B12), perguntas clínicas (B08 a B10), logs e validação (B08, B10, B13).
