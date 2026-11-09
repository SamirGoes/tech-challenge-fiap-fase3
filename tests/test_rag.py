from medassist.rag import indice_em_memoria


def test_busca_vetorial_encontra_o_protocolo_certo():
    indice = indice_em_memoria()
    assert indice.buscar("primeira hora sepse lactato", k=1)[0]["fonte"] == "PROT-001 §primeira_hora"
    assert indice.buscar("valores críticos de potássio", k=1)[0]["fonte"].startswith("PROT-006")
    assert indice.buscar("dor torácica e ecg", k=1)[0]["fonte"].startswith("PROT-002")


def test_retriever_descarta_trechos_irrelevantes():
    from medassist.rag import RetrieverProtocolos

    docs = RetrieverProtocolos(indice=indice_em_memoria()).invoke("futebol campeonato brasileiro")
    assert docs == []


def test_arquivo_com_bom_e_lido_por_secoes(tmp_path):
    arquivo = tmp_path / "doc.md"
    arquivo.write_text("## triagem\nTexto da triagem de teste.\n", encoding="utf-8-sig")
    from medassist.rag import documentos_de_arquivo

    assert documentos_de_arquivo(arquivo)[0].metadata["fonte"] == "DOC doc §triagem"


def test_base_e_mutavel_adicionar_atualizar_e_remover(tmp_path):
    indice = indice_em_memoria()
    antes = indice.total()
    arquivo = tmp_path / "protocolo-queda.md"
    arquivo.write_text("## prevencao\nPrevenção de queda em idosos: avaliar risco com a escala de Morse na admissão.\n",
                       encoding="utf-8")
    assert indice.adicionar_arquivo(arquivo) == 1 and indice.total() == antes + 1
    achado = indice.buscar("escala de Morse risco de queda", k=1)[0]
    assert achado["fonte"] == "DOC protocolo-queda §prevencao"

    arquivo.write_text("## prevencao\nPrevenção de queda: reavaliar o risco com a escala de Morse a cada 24 horas.\n",
                       encoding="utf-8")
    indice.adicionar_arquivo(arquivo)  # mesma fonte: atualiza sem duplicar
    assert indice.total() == antes + 1
    assert "24 horas" in indice.buscar("escala de Morse", k=1)[0]["texto"]

    assert indice.remover("DOC protocolo-queda") == 1 and indice.total() == antes
    assert "DOC protocolo-queda" not in indice.listar()
