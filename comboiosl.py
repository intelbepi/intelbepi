import streamlit as st
import re
from collections import defaultdict

# Processamento do texto
def process_text(text):
    linhas = [linha.strip() for linha in text.split("\n") if linha.strip()]
    if not linhas:
        st.error("Nenhum texto válido encontrado!")
        return None

    placas = defaultdict(list)
    placa_atual = None

    padrao_placa = re.compile(r'[A-Za-z]{3}\d[A-Za-z]\d{2}|[A-Za-z]{3}\d{4}')
    padrao_data_hora = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})')

    for linha in linhas:
        if padrao_placa.fullmatch(linha.strip()):
            placa_atual = linha.strip().upper()
        elif placa_atual and (match := padrao_data_hora.search(linha)):
            data = match.group(1)
            hora = match.group(2)
            placas[placa_atual].append(f"{data} {hora}")

    return placas

# Exibição dos resultados
def display_results(placas):
    resultado = "## RELATÓRIO DE PLACAS:\n\n"
    resultado += f"**Total de placas únicas encontradas:** {len(placas)}\n\n"

    placas_unicas = {p: d for p, d in placas.items() if len(d) == 1}
    placas_repetidas = {p: d for p, d in placas.items() if len(d) > 1}

    if placas_repetidas:
        resultado += "### 🚨 PLACAS REPETIDAS:\n"
        for placa, registros in sorted(placas_repetidas.items()):
            resultado += f"\n**Placa:** {placa}\n"
            resultado += f"**Quantidade:** {len(registros)} ocorrências\n"
            resultado += "**Registros:**\n"
            for registro in sorted(registros):
                resultado += f"- {registro}\n"

    if placas_unicas:
        resultado += "\n\n### 🔹 PLACAS ÚNICAS (apareceram apenas uma vez):\n"
        for placa, registros in sorted(placas_unicas.items()):
            resultado += f"\n**Placa:** {placa}\n"
            resultado += f"**Registro:** {registros[0]}\n"

    return resultado

# Página principal
def main():
    st.set_page_config(page_title="Analisador de Veículos em Comboio", layout="wide")
    st.title("Analisador de Veículos em Comboio")

    # Inicializa a lista de textos na sessão
    if 'text_blocks' not in st.session_state:
        st.session_state.text_blocks = [""]

    # Botão para adicionar novo bloco
    if st.button("➕ Adicionar Passagem"):
        st.session_state.text_blocks.append("")

    # Renderiza os campos de texto
    for i in range(len(st.session_state.text_blocks)):
        st.session_state.text_blocks[i] = st.text_area(
            f"Bloco {i + 1}",
            value=st.session_state.text_blocks[i],
            height=200,
            key=f"text_block_{i}"
        )

    # Processamento dos textos combinados
    if st.button("Processar"):
        full_text = "\n".join(st.session_state.text_blocks)
        if full_text.strip():
            placas = process_text(full_text)
            if placas is not None:
                results = display_results(placas)
                st.markdown(results)
                st.download_button(
                    label="⬇️ Baixar Relatório",
                    data=results,
                    file_name="relatorio_placas.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Por favor, insira texto nos blocos para processar.")

if __name__ == '__main__':
    main()
