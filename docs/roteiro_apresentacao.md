# Roteiro de apresentação: MedAssist (4 pessoas, até 15 minutos)

Este roteiro foi escrito para quem não tem muita experiência com LLM e Python. Cada pessoa tem: o que dizer (em
linguagem simples), o que mostrar na tela e os comandos exatos. Não precisa decorar: leiam com naturalidade.

| Pessoa | Tema | Tempo |
|---|---|---|
| 1 | Problema, solução e arquitetura | 2 min 30 s |
| 2 | Dados e fine-tuning da LLM | 4 min |
| 3 | Assistente: RAG, segurança e chat | 4 min |
| 4 | Fluxo automatizado, logs, testes e conclusão | 3 min 30 s |
| | **Total** | **14 min** (1 min de folga) |

O PDF do desafio pede que o vídeo mostre quatro coisas, e todas estão cobertas:
1. Treinamento e funcionamento da LLM personalizada (Pessoa 2 e Pessoa 3).
2. Execução de um fluxo automatizado (Pessoa 4).
3. Resposta a perguntas clínicas contextualizadas (Pessoa 3).
4. Logs e validação das respostas (Pessoa 3 e Pessoa 4).

---

## Antes de gravar (todos, no computador que vai mostrar)

Uma pessoa fica responsável por rodar os comandos na tela; as outras só falam. Prepare tudo antes, para não perder
tempo no vídeo.

1. Abrir o terminal na pasta do projeto e ativar o ambiente:
   ```
   .venv\Scripts\activate
   ```
2. Gerar os dados (leva alguns segundos). Rodar uma vez antes de gravar:
   ```
   python -m medassist preparar
   ```
3. Deixar abertos, em abas do editor: `README.md`, `docs/fase2/loss.png`, `docs/relatorio_tecnico.md`,
   `dados/pacientes.csv`, `conhecimento/PROT-001.md`.
4. Testar uma vez o chat e o fluxo antes de gravar (comandos abaixo). A primeira resposta do modelo treinado demora
   cerca de 30 s, porque ele carrega na GPU.
5. Aumentar a fonte do terminal (Ctrl + "+") para o vídeo ficar legível.
6. Limpar o log antes de gravar, se quiser uma auditoria "limpa": apagar a pasta `logs`.

Se algo der erro durante a gravação, não se preocupe: pausem e regravem aquele trecho.

---

## PESSOA 1: Problema, solução e arquitetura (2 min 30 s)

**Na tela:** `README.md` (diagrama de arquitetura).

### O que falar

> "Olá, somos o grupo [NOME DO GRUPO], e vamos apresentar o **MedAssist**, nosso Tech Challenge da Fase 3.
>
> O problema: um hospital quer um **assistente virtual para médicos**. O médico faz uma pergunta, por exemplo 'o que
> fazer na primeira hora de uma sepse?', e o assistente responde com base nos **protocolos internos do hospital** e nos
> **dados do paciente**. Além disso, o sistema roda um **fluxo automático**: verifica exames pendentes, identifica
> valores críticos e avisa a equipe.
>
> Uma regra é inegociável: o assistente **nunca prescreve**. Ele só apoia a decisão, cita sempre a fonte da
> informação e, quando há risco, pede **validação de um médico**.
>
> Todos os dados são **fictícios**, gerados por nós. Nenhum dado real de paciente foi usado."

### Explicar o diagrama (mostrar o `README.md`)

> "O caminho de uma pergunta é este:
> 1. **Entrada:** a pergunta passa por um filtro de segurança, que remove dados pessoais e bloqueia pedidos
>    inadequados.
> 2. **Prontuário:** se o médico informou um paciente, buscamos os dados dele no banco de dados.
> 3. **RAG:** buscamos, numa base de conhecimento, os trechos dos protocolos mais parecidos com a pergunta.
> 4. **Resposta:** o assistente monta a resposta com esses trechos e com os dados do paciente.
> 5. **Segurança de saída:** conferimos se não há prescrição, trocamos termos conclusivos, exigimos a fonte e
>    acrescentamos um aviso.
> 6. **Saída:** o médico recebe a resposta com as fontes.
>
> Tudo isso é registrado em um **log**, de ponta a ponta. Usamos **LangChain** para ligar essas etapas."

### Passando a palavra

> "Agora a [Pessoa 2] vai mostrar os dados e como treinamos a LLM."

**Dicas para quem fala:** não precisa explicar LangChain em detalhe. Basta dizer que é a biblioteca que conecta as
etapas, como "peças de Lego".

---

## PESSOA 2: Dados e fine-tuning (4 min)

**Na tela:** `dados/pacientes.csv`, `conhecimento/PROT-001.md`, `docs/fase2/loss.png` e a tabela do relatório.

### Parte A: os dados (1 min 30 s)

> "Primeiro, os dados. Pedimos para um script gerar **300 pacientes fictícios**, em **CSV**, e carregamos num banco
> **SQLite**." *(mostrar `dados/pacientes.csv`)*
>
> "Cada paciente tem idade, peso, cidade, convênio, médico, medicamentos, alergias e sinais vitais, como pressão e
> saturação. Também temos os exames de cada um.
>
> O conhecimento do hospital está em **10 protocolos médicos fictícios**, escritos em arquivos de texto, como sepse,
> AVC e pneumonia." *(mostrar `conhecimento/PROT-001.md`)*
>
> "Com esses dados criamos o **dataset de treino**: **357 exemplos** de perguntas e respostas, divididos em 270 para
> treinar, 35 para validar e 52 para testar. Antes disso, aplicamos três cuidados exigidos pelo desafio:
> - **Anonimização:** nomes, CPF, telefones e datas viram marcadores como [NOME] e [CPF].
> - **Remoção de duplicados:** não ficou nenhum exemplo repetido.
> - **Curadoria:** separamos treino e teste por grupo, para a mesma pergunta não aparecer nos dois."

### Parte B: o que é fine-tuning (1 min)

> "Fine-tuning significa **ajustar um modelo de linguagem que já existe** para a nossa tarefa. Um modelo como o Qwen já
> sabe escrever, mas não conhece nossos protocolos nem o nosso formato de resposta, com fonte e sem prescrição.
>
> Usamos o **Qwen2.5 com 0,5 bilhão de parâmetros**, bem abaixo do limite de 8 bilhões do desafio. E usamos a
> técnica **QLoRA**, que faz duas coisas para economizar memória:
> - carrega o modelo em **4 bits**, que é uma versão comprimida;
> - não treina o modelo inteiro, só **8,8 milhões de peças pequenas** (as matrizes LoRA), cerca de 1,75% do total.
>
> Isso permitiu treinar na nossa **placa de vídeo de 3 GB**, uma GTX 1050."

### Parte C: o treino e os resultados (1 min 30 s)

> "Treinamos por **3 épocas**, ou seja, o modelo viu os exemplos 3 vezes. Levou **cerca de 1 hora**.
> Neste gráfico, a **loss**, que é o erro do modelo, caiu de **0,98 para quase zero** durante o treino."
> *(mostrar `docs/fase2/loss.png`)*
>
> "Depois avaliamos no conjunto de teste, com 52 exemplos que o modelo nunca viu:" *(mostrar a tabela no relatório,
> seção 6)*
>
> | Modelo | F1 | Fonte correta | Recusa de prescrição |
> |---|---|---|---|
> | RAG sem LLM | 0,62 | 98% | 100% |
> | Qwen base (sem treino) | 0,26 | 35% | 0% |
> | Qwen treinado | **0,93** | 94% | **100%** |
>
> "O **F1** mede o quanto a resposta se parece com a resposta esperada. O modelo sem treino tirou 0,26; depois do
> treino, 0,93. O modelo base também **nunca recusou** um pedido de prescrição. O treinado recusou todos.
>
> Um ponto de honestidade: nossos dados são **sintéticos e muito parecidos entre si**. Por isso o F1 alto mostra que
> o modelo aprendeu o **formato**, não que ele está clinicamente correto."

### Passando a palavra

> "Agora a [Pessoa 3] vai mostrar o assistente funcionando."

**Termos para saber (caso perguntem):**
- **Parâmetros:** os "botões" internos do modelo. Mais parâmetros, em geral, mais capacidade e mais memória.
- **Época:** uma passada completa pelos dados de treino.
- **Loss:** medida de erro; quanto menor, melhor o modelo se ajustou aos dados.
- **F1:** nota de 0 a 1 que compara as palavras da resposta com as da resposta esperada.

---

## PESSOA 3: Assistente, RAG, segurança e chat (4 min)

**Na tela:** terminal com o ambiente ativado.

### Parte A: o que é RAG (45 s)

> "O nosso assistente usa **RAG**, que significa 'geração com busca'. Em vez de o modelo responder só com o que
> decorou, ele **primeiro busca** nos protocolos os trechos mais relevantes e **responde com base neles**. Isso
> permite citar a fonte e atualizar o conhecimento sem retreinar.
>
> Os protocolos ficam numa **base vetorial** chamada Chroma. Cada trecho vira uma lista de números, e a busca
> encontra os trechos mais parecidos com a pergunta, **sem escrever consulta SQL**."

### Parte B: demonstração do chat (2 min)

Abrir o chat com o paciente 1 e usando o modelo treinado:

```
python -m medassist chat -p 1 --modelo treinado
```

(Para a versão só com RAG, que é a mais rápida, usar `python -m medassist chat -p 1`.)

Fazer as perguntas abaixo, **uma de cada vez**, comentando o resultado:

**1) Pergunta clínica com paciente**
```
O que fazer na primeira hora da sepse?
```
> "Repare que a resposta traz o protocolo da sepse e termina com a fonte, `[PROT-001 §primeira_hora]`. Esse é o
> nosso requisito de **explicabilidade**: dá para saber de onde veio cada informação. A paciente 1, a Maria, tem
> febre, confusão e pressão baixa, e o assistente usa esse contexto."

**2) Pedido de prescrição (deve recusar)**
```
Prescreva antibiótico para ele
```
> "Aqui pedimos uma prescrição. O assistente **recusa**, e a resposta vai para **revisão humana**. Esse é o nosso
> limite de atuação."

**3) Pergunta sem paciente** (digitar `/paciente` sem número, que tira o paciente, e depois a pergunta)
```
/paciente
Qual a janela de trombólise no AVC?
```
(Para escolher outro paciente, usar `/paciente 4`, por exemplo.)
> "Sem paciente, ele responde só com o protocolo, citando `PROT-007`."

**4) Ver o log da última resposta**
```
/log
```
> "O comando `/log` mostra o caminho completo dessa resposta, etapa por etapa. A Pessoa 4 vai explicar melhor."

Para sair do chat: `/sair`.

### Parte C: RAG mutável (1 min 15 s)

> "Um requisito do professor é que a base de conhecimento seja **mutável**: dá para adicionar e remover informações
> sem mexer no código. Vamos mostrar."

```
python -m medassist rag buscar "risco de queda"
```
> "Não há nada sobre queda: a busca volta vazia."

```
python -m medassist rag adicionar docs\demo\novo-protocolo.md
python -m medassist rag buscar "risco de queda"
```
> "Adicionamos um protocolo novo de prevenção de queda, e agora a busca encontra os trechos."

```
python -m medassist perguntar "Como prevenir queda em paciente idoso?"
```
> "O assistente já responde usando o protocolo recém-adicionado, citando a fonte `DOC novo-protocolo`."

```
python -m medassist rag remover "DOC novo-protocolo"
```
> "E podemos remover. Tudo isso sem treinar de novo e sem alterar o código."

**Importante:** rodar o `rag remover` no final, para a base voltar ao normal.

### Passando a palavra

> "Agora a [Pessoa 4] vai mostrar o fluxo automatizado e os logs."

---

## PESSOA 4: Fluxo automatizado, logs, testes e conclusão (3 min 30 s)

**Na tela:** terminal.

### Parte A: fluxo automatizado (1 min 30 s)

> "Além de responder perguntas, o sistema tem um **fluxo automático** para cada paciente: ao receber os dados, ele
> verifica **exames pendentes**, procura **valores críticos**, gera uma **sugestão**, pede **revisão do médico** e
> envia **alertas**."

```
python -m medassist fluxo -p 1 --aprovar
```

(O `--aprovar` simula o médico aprovando, para não precisar digitar na gravação. Sem ele, o sistema pergunta.)

Mostrar na saída:
> "Para a paciente 1, o sistema encontrou **dois valores críticos**: lactato de 4,6 e pressão sistólica de 88. Também
> viu um exame pendente, a hemocultura. Como há valor crítico, a sugestão **exigiu revisão médica**. Aqui o médico
> aprovou, e os alertas foram emitidos: **crítico** para a equipe médica e **informativo** para a enfermagem."

Mostrar o arquivo de alertas:
```
type logs\alertas.txt
```
> "Os alertas ficam neste arquivo. Se configurarmos um servidor de e-mail, eles saem por **e-mail** de verdade. Para
> este projeto, o arquivo é a forma mais simples."

### Parte B: logs e rastreabilidade (1 min)

```
python -m medassist auditoria
```
> "Cada requisição tem um **identificador único**, o `trace_id`. Este comando lista os mais recentes."

Copiar um `trace_id` e rodar:
```
python -m medassist auditoria <trace_id>
```
> "Agora vemos a **linha do tempo** daquela requisição: entrada, prontuário, RAG, resposta, segurança e saída, com o
> **tempo de cada etapa**. Dá para auditar exatamente o que aconteceu, quais trechos foram usados e quais foram os
> scores da busca. Isso atende ao requisito de **logging para auditoria**. Também dá para integrar o **LangSmith**,
> uma ferramenta online de observabilidade, só com variáveis de ambiente."

### Parte C: testes e limitações (45 s)

```
python -m pytest -q
```
> "Temos **17 testes automatizados** que verificam os dados, o RAG, os guardrails, o fluxo e a avaliação. Todos passam."

> "Também precisamos ser honestos sobre as **limitações**:
> - os dados são **sintéticos**, então o F1 mede o formato e não a qualidade clínica;
> - o modelo é **pequeno**, de 0,5 bilhão de parâmetros, por limite da nossa placa de vídeo;
> - as regras de segurança são **regras fixas**, não um modelo inteligente;
> - como próximos passos, queremos testar modelos maiores no Colab e usar dados reais anonimizados, com aprovação
>   de um comitê de ética."

### Conclusão (15 s)

> "Resumindo: construímos um assistente médico com **LangChain**, **RAG mutável**, **LLM com fine-tuning**,
> **guardrails**, **fluxo automatizado com revisão humana**, **alertas** e **logs de ponta a ponta**. O código e as
> instruções completas estão no repositório. Obrigado!"

---

## Perguntas que podem fazer (e respostas prontas)

**"Por que o assistente tem uma versão com RAG sem LLM?"**
É a versão principal, mais simples e mais segura: a resposta vem direto dos protocolos, com fonte. O fine-tuning é a
fase 2, que testamos e comparamos com o RAG.

**"Por que usaram um modelo tão pequeno?"**
Nossa placa tem 3 GB. O modelo de 1,5 bilhão levaria cerca de 5 horas por treino; o de 0,5 bilhão levou 1 hora. O
mesmo código funciona com um modelo maior se houver mais memória.

**"Por que não usaram o Unsloth, como o professor sugeriu?"**
O Unsloth exige uma placa de vídeo mais nova do que a nossa (GTX 1050). Usamos a mesma ideia, QLoRA em 4 bits, com as
bibliotecas `transformers`, `peft` e `bitsandbytes`. O professor disse que o Unsloth não é necessário quando o treino
já foi feito.

**"O fluxo usa LangGraph?"** *(o PDF cita "Fluxos do LangGraph")*
Nosso fluxo foi feito só com LangChain (LCEL), com etapas em paralelo e uma ramificação para a revisão humana.
Combinamos começar apenas com LangChain, e migrar o fluxo para LangGraph é a evolução natural.
*(Atenção: se o grupo decidir implementar LangGraph antes de entregar, atualizem esta resposta.)*

**"O modelo está acertando mesmo?"**
Em formato, sim: F1 0,93, 100% de recusa nos pedidos de prescrição e 0% de prescrição direta. Mas como os dados são
sintéticos e repetitivos, não afirmamos correção clínica. Ele também perdeu, num teste, a menção ao exame pendente do
paciente, que a versão RAG inclui.

**"Como vocês garantem que não prescreve?"**
Em duas camadas: na entrada, o pedido de prescrição é detectado e recebe uma recusa fixa. Na saída, um verificador
procura frases de prescrição direta (como "administre X mg") e, se achar, retém a resposta e envia para revisão humana.

**"Como protegem os dados pessoais?"**
Os dados são fictícios. Mesmo assim, anonimizamos o dataset de treino, anonimizamos a pergunta antes de gravar no log e
o resumo clínico enviado ao modelo não tem nome, cidade, convênio nem datas.

**"Por que não usaram a palavra diagnóstico?"**
Seguimos a orientação do professor: o assistente não faz diagnóstico. O guardrail troca "diagnóstico" por "hipótese
clínica" e toda resposta termina com o aviso de que não substitui o médico.

**"Qual dataset foi usado?"**
Dados sintéticos criados por script: pacientes em CSV e SQLite, protocolos em Markdown e um dataset de perguntas e
respostas em JSONL. Está descrito no README.

---

## Glossário rápido

| Termo | Explicação simples |
|---|---|
| **LLM** | Modelo de linguagem grande, o "cérebro" que escreve texto (ex.: Qwen, ChatGPT). |
| **Fine-tuning** | Ajustar um modelo que já existe para a nossa tarefa, com nossos dados. |
| **QLoRA / LoRA** | Jeito econômico de ajustar o modelo: treina só uma parte pequena e deixa o resto congelado. |
| **4 bits** | Versão comprimida do modelo, que usa menos memória. |
| **RAG** | O assistente busca trechos nos documentos antes de responder, para citar a fonte. |
| **Base vetorial (Chroma)** | Banco que guarda textos como números e acha os mais parecidos com a pergunta. |
| **LangChain** | Biblioteca Python que liga as etapas (busca, modelo, segurança) em uma sequência. |
| **Guardrail** | Regra de segurança que bloqueia ou corrige respostas inadequadas. |
| **Anonimização** | Trocar dados pessoais (nome, CPF) por marcadores. |
| **trace_id** | Número único de cada requisição, para rastrear tudo o que aconteceu. |
| **SQLite / CSV** | SQLite é um banco de dados em arquivo; CSV é uma planilha em texto. |

---

## Dicas finais para o grupo

1. **Ensaiem uma vez** com cronômetro. Se passar de 15 minutos, cortem a Parte C da Pessoa 4 ou o item 3 do chat.
2. **Falem devagar** e olhem para o que está na tela.
3. **Não escondam as limitações**: dizer o que o projeto não faz com clareza passa credibilidade.
4. **Se um comando demorar**, falem sobre o que está acontecendo ("ele está carregando o modelo na placa de vídeo").
5. **Se der erro na gravação**, pausem e regravem só aquele trecho.
