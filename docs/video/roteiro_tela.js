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
// DUR = duração real de cada clipe (audios/01.mp3 … 14.mp3). A trilha audios/narracao_completa.m4a tem os 14 clipes
// com 1 s entre eles: na edição, alinhe o início dela com o momento em que você apertou Option+Shift+S.

(() => {
  if (window.medassistDemo) window.medassistDemo.remover();

  const DUR = { B01: 16.0, B02: 35.4, B03: 32.9, B04: 40.0, B05: 30.7, B06: 40.1, B07: 16.0,
                B08: 22.0, B09: 12.1, B10: 20.2, B11: 28.5, B12: 38.1, B13: 22.6, B14: 40.3 };
  const PAUSA_ENTRE_BLOCOS = 1;   // segundos de silêncio entre um clipe e o próximo
  const VELOCIDADE = 1;           // 0.5 = tudo duas vezes mais rápido (só para ensaiar)

  // [segundos dentro do bloco, ação, texto que identifica a célula]
  // ações: topo | ir (rola até o início) | fim (rola até o fim) | rodar (roda e acompanha a saída)
  //        disparar (roda sem mexer na tela) | acompanhar (rola até a célula e acompanha a saída, sem rodar de novo)
  const ROTEIRO = {
    B01: [[0, 'topo']],
    B02: [],
    B03: [[0, 'ir', 'Arquitetura']],
    B04: [[0, 'ir', '1. Ambiente'], [6, 'ir', 'medassist preparar'], [14, 'fim', 'medassist preparar'],
          [23, 'ir', 'pacientes.head'], [31, 'ir', 'PROT-001.md']],
    B05: [[0, 'ir', 'medassist.treino'], [11, 'fim', 'medassist.treino']],
    B06: [[0, 'ir', 'plt.plot'], [8, 'ir', 'DataFrame(linhas)']],
    B07: [[0, 'ir', '4. Assistente'], [4, 'ir', 'criar_assistente(']],
    B08: [[0, 'rodar', '# Pergunta 1']],
    B09: [[0, 'rodar', '# Pergunta 2']],
    B10: [[0, 'rodar', '# Pergunta 3'], [12, 'rodar', '# /log']],
    B11: [[0, 'ir', '5. RAG'], [2, 'rodar', 'rag buscar "risco'], [7, 'rodar', 'rag adicionar'],
          [13, 'rodar', 'perguntar "Como'], [21, 'rodar', 'rag remover']],
    B12: [[0, 'ir', '6. Fluxo'], [2, 'rodar', 'executar_fluxo'], [32, 'rodar', 'cat logs/alertas']],
    B13: [[0, 'ir', '7. Logs'], [1, 'rodar', 'medassist auditoria'], [5, 'rodar', 'linha do tempo da req'],
          [6, 'disparar', 'pytest -q'], [13, 'acompanhar', 'pytest -q']],
    B14: [],
  };

  const blocos = Object.keys(DUR);
  let timers = [], seguindo = null, cursor = 1;

  const celulas = () => [...document.querySelectorAll('.cell')];
  const texto = c => ((c.querySelector('.view-lines, .text-top-div, .markdown') || c).innerText || '').replace(/\u00a0/g, ' ');
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
    disparar(c) { c.querySelector('colab-run-button')?.click(); },
    rodar(c) { acoes.disparar(c); acoes.acompanhar(c); },
    acompanhar(c) {
      pararDeSeguir();
      c.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
