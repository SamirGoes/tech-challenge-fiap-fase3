// MedAssist: roteiro de tela para gravar o Colab sem o Claude controlando o navegador.
// Roda as células e faz o scroll nos momentos da minutagem (docs/video/minutagem.md).
//
// Como usar: abra o Console do Chrome (Cmd+Option+J), cole este arquivo inteiro e aperte Enter.
// Na primeira vez, o Chrome pede para digitar "allow pasting". Depois feche o Console,
// clique em qualquer lugar do notebook e comece a gravar a tela.
//   Option+Shift+S (Alt+Shift+S)  -> começa do bloco B01
//   Option+Shift+X (Alt+Shift+X)  -> para
//   medassistDemo.iniciar('B08')  -> (no Console) começa de um bloco, para regravar um trecho
//
// Depois de gerar o áudio no ElevenLabs, troque os valores de DUR pela duração real de cada clipe, em segundos.

(() => {
  if (window.medassistDemo) window.medassistDemo.remover();

  const DUR = { B01: 21, B02: 41, B03: 42, B04: 46, B05: 38, B06: 52, B07: 18,
                B08: 24, B09: 13, B10: 26, B11: 37, B12: 46, B13: 24, B14: 43 };
  const PAUSA_ENTRE_BLOCOS = 1;   // segundos de silêncio entre um clipe e o próximo
  const VELOCIDADE = 1;           // 0.5 = tudo duas vezes mais rápido (só para ensaiar)

  // [segundos dentro do bloco, ação, texto que identifica a célula]
  // ações: topo | ir (rola até o início da célula) | fim (rola até o fim da célula) | rodar (roda e acompanha a saída)
  const ROTEIRO = {
    B01: [[0, 'topo']],
    B02: [],
    B03: [[0, 'ir', 'Arquitetura']],
    B04: [[0, 'ir', '1. Ambiente'], [7, 'ir', 'medassist preparar'], [16, 'fim', 'medassist preparar'],
          [27, 'ir', 'pacientes.head'], [36, 'ir', 'PROT-001.md']],
    B05: [[0, 'ir', 'medassist.treino'], [14, 'fim', 'medassist.treino']],
    B06: [[0, 'ir', 'plt.plot'], [10, 'ir', 'DataFrame(linhas)']],
    B07: [[0, 'ir', '4. Assistente'], [5, 'ir', 'criar_assistente(']],
    B08: [[0, 'rodar', '# Pergunta 1']],
    B09: [[0, 'rodar', '# Pergunta 2']],
    B10: [[0, 'rodar', '# Pergunta 3'], [16, 'rodar', '# /log']],
    B11: [[0, 'ir', '5. RAG'], [3, 'rodar', 'rag buscar "risco'], [8, 'rodar', 'rag adicionar'],
          [17, 'rodar', 'perguntar "Como'], [27, 'rodar', 'rag remover']],
    B12: [[0, 'ir', '6. Fluxo'], [2, 'rodar', 'executar_fluxo'], [39, 'rodar', 'cat logs/alertas']],
    B13: [[0, 'ir', '7. Logs'], [1, 'rodar', 'medassist auditoria'], [5, 'rodar', 'linha do tempo da req'],
          [14, 'rodar', 'pytest -q']],
    B14: [],
  };

  const blocos = Object.keys(DUR);
  let timers = [], seguindo = null, cursor = 1;

  const celulas = () => [...document.querySelectorAll('.cell')];
  const texto = c => ((c.querySelector('.view-lines, .text-top-div, .markdown') || c).innerText || '').replace(/ /g, ' ');
  const achar = chave => {
    const cs = celulas();
    for (let i = cursor; i < cs.length; i++) if (texto(cs[i]).includes(chave)) { cursor = i; return cs[i]; }
    const i = cs.findIndex((c, j) => j > 0 && texto(c).includes(chave));  // fallback: procura no notebook todo
    if (i >= 0) { cursor = i; return cs[i]; }
    console.warn('[medassist] célula não encontrada:', chave);
    return null;
  };
  const rodando = c => /pending|running/.test(c.className);
  const pararDeSeguir = () => { if (seguindo) clearInterval(seguindo); seguindo = null; };

  const acoes = {
    topo() { pararDeSeguir(); document.querySelector('colab-scroller')?.scrollTo({ top: 0, behavior: 'smooth' }); },
    ir(c) { pararDeSeguir(); c.scrollIntoView({ behavior: 'smooth', block: 'start' }); },
    fim(c) { pararDeSeguir(); c.scrollIntoView({ behavior: 'smooth', block: 'end' }); },
    rodar(c) {
      pararDeSeguir();
      c.scrollIntoView({ behavior: 'smooth', block: 'start' });
      c.querySelector('colab-run-button')?.click();
      let ultimaAltura = 0, paradoDesde = Date.now();
      seguindo = setInterval(() => {            // acompanha a saída enquanto ela cresce
        const r = c.getBoundingClientRect();
        if (r.height !== ultimaAltura) { ultimaAltura = r.height; paradoDesde = Date.now(); }
        if (r.bottom > window.innerHeight - 20) c.scrollIntoView({ behavior: 'smooth', block: 'end' });
        if (!rodando(c) && Date.now() - paradoDesde > 1500) pararDeSeguir();
      }, 400);
    },
  };

  function parar() { timers.forEach(clearTimeout); timers = []; pararDeSeguir(); console.log('[medassist] parado'); }

  function iniciar(desde = 'B01') {
    parar();
    cursor = 1;
    const t0 = performance.now();
    let inicio = 0;
    for (const b of blocos.slice(blocos.indexOf(desde))) {
      for (const [t, acao, chave] of ROTEIRO[b]) {
        timers.push(setTimeout(() => {
          const c = acao === 'topo' ? null : achar(chave);
          if (acao !== 'topo' && !c) return;
          console.log(`[medassist] ${((performance.now() - t0) / 1000).toFixed(1)}s ${b} ${acao} ${chave || ''}`);
          acoes[acao](c);
        }, (inicio + t) * 1000 * VELOCIDADE));
      }
      inicio += DUR[b] + PAUSA_ENTRE_BLOCOS;
    }
    timers.push(setTimeout(() => console.log('[medassist] fim do roteiro'), inicio * 1000 * VELOCIDADE));
    console.log(`[medassist] roteiro iniciado em ${desde}; duração ~${Math.round(inicio * VELOCIDADE)} s`);
  }

  const tecla = e => {
    if (!(e.altKey && e.shiftKey)) return;
    if (e.code === 'KeyS') { e.preventDefault(); e.stopPropagation(); iniciar('B01'); }
    if (e.code === 'KeyX') { e.preventDefault(); e.stopPropagation(); parar(); }
  };
  document.addEventListener('keydown', tecla, true);

  window.medassistDemo = { iniciar, parar, DUR, ROTEIRO,
    remover() { parar(); document.removeEventListener('keydown', tecla, true); delete window.medassistDemo; } };
  console.log('[medassist] pronto: Option+Shift+S começa, Option+Shift+X para, medassistDemo.iniciar("B08") começa de um bloco');
})();
