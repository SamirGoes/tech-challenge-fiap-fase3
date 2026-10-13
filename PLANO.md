# Plano do projeto

Entrega: **03/11/2026**.

## O que será feito

| Semana | O que fazer | Resultado |
|---|---|---|
| 13/10 a 19/10 | Rodar o notebook no Colab: avaliar o modelo base, treinar e avaliar o modelo treinado | adaptador LoRA, métricas e gráfico de loss |
| 20/10 a 26/10 | Rodar o assistente e o fluxo com o modelo treinado e ajustar o que falhar; escrever o relatório técnico | relatório com métricas, diagramas e análise |
| 27/10 a 02/11 | Gravar o vídeo de até 15 minutos e revisar README e relatório | vídeo e entrega |
| 03/11 | Entregar na plataforma | |

Já pronto: código das 8 etapas abaixo, testes automatizados e notebook do Colab.

## Cada etapa explicada

O projeto tem 8 etapas. As etapas 1 a 3 rodam uma vez, para preparar e treinar o modelo. As etapas 4 a 8 rodam a
cada pergunta ou paciente.

### 1. Gerar os dados (`dados.py`)
O enunciado pede treino com protocolos, perguntas de médicos, laudos e receitas do hospital. Como não temos dados
reais, criamos dados fictícios:
- **6 protocolos** (sepse, dor torácica, crise hipertensiva, cetoacidose, pneumonia, valores críticos), cada um
  dividido em seções com um código, por exemplo `PROT-001 §primeira_hora`. Esse código é a "fonte" citada nas respostas.
- **6 pacientes** com sinais vitais e exames. Alguns têm valores críticos de propósito, para o fluxo ter o que alertar.
- **Exemplos de treino** no formato pergunta → resposta:
  - perguntas frequentes sobre cada seção, com 5 jeitos diferentes de perguntar;
  - casos clínicos com nome, CPF, telefone e data falsos;
  - pedidos de prescrição, cuja resposta é sempre uma recusa;
  - um modelo de laudo e um de receita.

### 2. Anonimizar e preparar o dataset (`dados.py`)
- `anonimizar` troca nome, CPF, telefone, e-mail, data e número de prontuário por marcadores como `[NOME]` e `[CPF]`.
- Perguntas repetidas são removidas.
- `dividir` separa os exemplos em treino (75%), validação (10%) e teste (15%). As variações da mesma pergunta ficam
  sempre no mesmo grupo, senão o teste teria perguntas que o modelo já viu.
- Cada exemplo vira uma conversa `system / user / assistant`, que é o formato que o modelo espera.

Comando: `python -m medassist.main preparar`. Gera `dados/train.jsonl`, `val.jsonl`, `test.jsonl` e o banco `dados/hospital.db`.

### 3. Treinar o modelo (`treino.py`)
- Modelo base: **Qwen2.5-1.5B-Instruct**, aberto e bom em português.
- Técnica: **QLoRA**. O modelo é carregado compactado (4 bits) e fica congelado. Só pequenas camadas extras (LoRA) são
  treinadas, cerca de 1% dos parâmetros. Por isso cabe na GPU gratuita do Colab.
- O modelo só é corrigido pela parte da resposta, não pela pergunta.
- Saída: pasta `saida/lora-medassist` com o adaptador treinado e o histórico de loss.

### 4. Consultar o prontuário (`prontuario.py`)
- O banco SQLite tem duas tabelas: `pacientes` e `exames`. Cada exame tem resultado ou está pendente.
- `resumo_clinico` monta um texto com idade, motivo, alergias, sinais vitais e exames, **sem nome nem documentos**.
  É isso que vai para o modelo.

### 5. Buscar o protocolo certo (`assistente.py`)
- `buscar_protocolos` compara as palavras da pergunta e do motivo da internação com as palavras de cada seção de
  protocolo e devolve as duas seções com mais palavras em comum.

### 6. Gerar a resposta com LangChain (`assistente.py`)
A cadeia LangChain faz três passos:
1. busca as seções de protocolo;
2. monta o prompt com protocolos (cada um com sua fonte), dados do paciente e pergunta;
3. chama o modelo.

Há dois modelos:
- `demo`: responde copiando o trecho do protocolo, para rodar sem GPU e servir de comparação;
- `treinado`: o Qwen com o adaptador da etapa 3.

### 7. Aplicar as regras de segurança e registrar o log (`seguranca.py`)
- **Antes do modelo:** bloqueia pedidos fora do escopo ("ignore as instruções", "atestado falso") e marca pedidos de prescrição.
- **Depois do modelo:**
  - se a resposta tiver cara de prescrição ("administre X 1 g a cada 12 h"), ela é retida;
  - se não citar fonte, é marcada;
  - toda resposta ganha o aviso de que precisa de validação médica.
- **Log:** cada etapa grava uma linha em `logs/auditoria.jsonl` com o mesmo `trace_id`. As linhas registram a pergunta,
  o prontuário consultado, os protocolos usados, a resposta do modelo e a validação.

Ver o log: `python -m medassist.main auditoria <trace_id>`.

### 8. Fluxo automatizado com LangGraph (`fluxo.py`)
Ao receber um paciente:
1. `carregar_paciente` busca o prontuário;
2. `verificar_pendentes` e `verificar_criticos` rodam juntos: um lista exames sem resultado e o outro compara
   exames e sinais vitais com os limites do protocolo de valores críticos;
3. `gerar_sugestao` chama o assistente da etapa 6;
4. se houver valor crítico ou pedido de prescrição, o fluxo **pausa** em `revisao_medica` até um médico aprovar ou rejeitar;
5. `emitir_alertas` envia os alertas: críticos para a equipe médica, pendências para a enfermagem.

Comando: `python -m medassist.main fluxo -p 1`.

### Avaliação (`avaliacao.py`)
O conjunto de teste é passado por cada modelo, que recebe só a pergunta. Três medidas são calculadas:
- **f1:** quanto a resposta se parece com a esperada;
- **fonte_certa:** se o modelo citou o protocolo certo;
- **prescreveu:** se o modelo prescreveu algo (precisa dar zero).

Para o relatório, compare `--modelo base` com `--modelo treinado`.

## Roteiro do vídeo (15 min)
1. Problema e arquitetura (2 min)
2. Dados e anonimização (2 min)
3. Treino no Colab e métricas antes e depois (3 min)
4. Pergunta clínica com fonte (2 min)
5. Fluxo com paciente crítico, pausa para revisão e alertas (3 min)
6. Pedido de prescrição recusado e log de auditoria (2 min)
7. Limitações (1 min)
