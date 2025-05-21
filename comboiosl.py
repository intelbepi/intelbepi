import streamlit as st
import re
from collections import defaultdict

# Processamento do texto
def process_text(text):
    # Remove linhas vazias
    linhas = [linha.strip() for linha in text.split("\n") if linha.strip()]
    
    if not linhas:
        st.error("Nenhum texto válido encontrado!")
        return None
    
    # Dicionário para armazenar datas e horas por placa
    placas = defaultdict(list)
    placa_atual = None
    
    # Padrão para detectar placa (3 letras + 1 número + 1 letra + 2 números ou formato antigo)
    padrao_placa = re.compile(r'[A-Za-z]{3}\d[A-Za-z]\d{2}|[A-Za-z]{3}\d{4}')
    
    # Padrão para detectar data (dd/mm/aaaa) e hora (HH:MM)
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

# Página principal da aplicação
def main():
    st.set_page_config(page_title="Analisador de Veículos em Comboio", layout="wide")
    
    st.title("Analisador de Veículos em Comboio")
    
    st.subheader("Cole os registros abaixo e clique em 'Processar':")
    text_input = st.text_area("", height=300, placeholder="Cole aqui os dados das placas e horários...", key="text_input")

    if st.button("Processar", type="primary"):
        if text_input.strip():
            placas = process_text(text_input)
            if placas is not None:
                results = display_results(placas)
                st.markdown(results)
                st.download_button(
                    label="Baixar Relatório",
                    data=results,
                    file_name="relatorio_placas.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Por favor, insira algum texto para processar.")

if __name__ == '__main__':
    main()
