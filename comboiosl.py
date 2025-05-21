import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import re, os
from io import BytesIO

sns.set(style="darkgrid")

dias_semana = {
    "Monday": "Segunda-feira", "Tuesday": "Terça-feira", "Wednesday": "Quarta-feira",
    "Thursday": "Quinta-feira", "Friday": "Sexta-feira", "Saturday": "Sábado", "Sunday": "Domingo"
}

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Navegação")
    pagina = st.sidebar.radio("Escolha uma opção:", ["🚗🚗 - Veículos em Comboio", "🕵️ - Análise de Passagens"])

    if pagina == "🚗🚗 - Veículos em Comboio":
        veiculos_em_comboio()
    elif pagina == "🕵️ - Análise de Passagens":
        analise_de_passagens()


# -------------------------------------------
# Página 1 – Veículos em Comboio
# -------------------------------------------
def veiculos_em_comboio():
    import re
    from collections import defaultdict

    def process_text(text):
        linhas = [linha.strip() for linha in text.split("\n") if linha.strip()]
        placas = defaultdict(list)
        padrao_placa = re.compile(r'[A-Za-z]{3}\d[A-Za-z]\d{2}|[A-Za-z]{3}\d{4}')
        padrao_data_hora = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})')
        placa_atual = None

        for linha in linhas:
            if padrao_placa.fullmatch(linha.strip()):
                placa_atual = linha.strip().upper()
            elif placa_atual and (match := padrao_data_hora.search(linha)):
                data, hora = match.groups()
                placas[placa_atual].append(f"{data} {hora}")
        return placas

    def display_results(placas):
        resultado = "## RELATÓRIO DE PLACAS:\n\n"
        resultado += f"**Total de placas únicas encontradas:** {len(placas)}\n\n"
        placas_unicas = {p: d for p, d in placas.items() if len(d) == 1}
        placas_repetidas = {p: d for p, d in placas.items() if len(d) > 1}

        if placas_repetidas:
            resultado += "### 🚨 PLACAS REPETIDAS:\n"
            for placa, registros in sorted(placas_repetidas.items()):
                resultado += f"\n**Placa:** {placa}\n**Quantidade:** {len(registros)} ocorrências\n**Registros:**\n"
                for registro in sorted(registros):
                    resultado += f"- {registro}\n"
        if placas_unicas:
            resultado += "\n\n### 🔹 PLACAS ÚNICAS (apareceram apenas uma vez):\n"
            for placa, registros in sorted(placas_unicas.items()):
                resultado += f"\n**Placa:** {placa}\n**Registro:** {registros[0]}\n"
        return resultado

    st.title("Veículos em Comboio")
    if 'text_blocks' not in st.session_state:
        st.session_state.text_blocks = [""]

    if st.button("➕ Adicionar Passagem"):
        st.session_state.text_blocks.append("")

    for i in range(len(st.session_state.text_blocks)):
        st.session_state.text_blocks[i] = st.text_area(
            f"Passagem {i + 1}",
            value=st.session_state.text_blocks[i],
            height=200,
            key=f"text_block_{i}"
        )

    if st.button("Processar"):
        full_text = "\n".join(st.session_state.text_blocks)
        if full_text.strip():
            placas = process_text(full_text)
            results = display_results(placas)
            st.markdown(results)
            st.download_button("⬇️ Baixar Relatório", data=results, file_name="relatorio_placas.txt", mime="text/plain")


# -------------------------------------------
# Página 2 – Análise de Passagens
# -------------------------------------------
def analise_de_passagens():
    st.title("Análise de Passagens")

    texto = st.text_area("Cole os dados (linha 1: placa, linha 2: título, restante: registros):", height=400)
    texto += '/n0/0'
    if st.button("Processar Registros"):
        if texto:
            processar_analise(texto)
        else:
            st.warning("Por favor, insira o texto antes de processar.")


def processar_analise(texto):
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    if len(linhas) < 2:
        st.error("Texto deve conter pelo menos duas linhas: placa e título.")
        return

    placa = linhas[0].strip()
    titulo = linhas[1].strip()
    texto_limpo = "\n".join(linhas)

    padrao = re.compile(r'(\d{1,3})?\n?(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})\n([^\n]+)\n([^\n]+)', re.VERBOSE)
    registros = []

    for i, match in enumerate(padrao.finditer(texto_limpo)):
        velocidade, data, horario, local, _ = match.groups()
        cidade = local.split(',')[-1].split('-')[0].strip() if ',' in local and '-' in local else local
        data_formatada = datetime.strptime(data, "%d/%m/%Y")
        hora = int(horario.split(":")[0])
        turno = "Manhã" if 0 <= hora < 12 else "Tarde" if hora < 18 else "Noite"
        dia_semana = dias_semana[data_formatada.strftime("%A")]
        intervalo = f"{hora}:00 - {hora+1}:00"
        registros.append([i + 1, velocidade or "", data, horario, turno, dia_semana, intervalo, local, cidade])

    if not registros:
        st.error("Nenhum registro válido encontrado.")
        return

    df = pd.DataFrame(registros, columns=["Ord", "Velocidade", "Data", "Horário", "Turno", "Dia da Semana", "Intervalo", "Local", "Cidade"])
    st.success(f"{len(df)} registros processados.")
    st.dataframe(df)

    # Geração Excel
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    st.download_button("⬇️ Baixar Excel", data=excel_buffer.getvalue(), file_name=f"{placa}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Geração da análise textual e PDF
    analise, imagens = gerar_analise(df, placa)
    st.markdown(analise)

    pdf_buffer = gerar_pdf_streamlit(df, placa, titulo, analise, imagens)
    st.download_button("⬇️ Baixar PDF", data=pdf_buffer.getvalue(), file_name=f"analise_{placa}.pdf", mime="application/pdf")


def gerar_analise(df, placa):
    analise = ""
    imagens = []
    contagem_turno = Counter(df["Turno"])
    contagem_horario = Counter(df["Intervalo"])
    contagem_dia = Counter(df["Dia da Semana"])
    contagem_local = Counter(df["Local"])

    dia_freq = contagem_dia.most_common(1)[0][0]
    horario_freq_dia = Counter(df[df["Dia da Semana"] == dia_freq]["Intervalo"]).most_common(1)[0]
    local_freq_dia = Counter(df[df["Dia da Semana"] == dia_freq]["Local"]).most_common(1)[0]
    horario_local_dia = Counter(df[(df["Dia da Semana"] == dia_freq) & (df["Local"] == local_freq_dia[0])]["Intervalo"]).most_common(1)[0]
    local_geral = contagem_local.most_common(1)[0]
    horario_local_geral = Counter(df[df["Local"] == local_geral[0]]["Intervalo"]).most_common(1)[0]

    analise += f"📌 **Movimentação do veículo {placa}**\n\n"
    analise += f"- Turno mais frequente: **{contagem_turno.most_common(1)[0][0]}**\n"
    analise += f"- Horário mais frequente: **{contagem_horario.most_common(1)[0][0]}**\n\n"
    analise += f"- Dia mais frequente: **{dia_freq}**\n"
    analise += f"   - Melhor horário nesse dia: **{horario_freq_dia[0]}**\n"
    analise += f"   - Local mais frequente nesse dia: **{local_freq_dia[0]}**\n"
    analise += f"   - Melhor horário nesse local: **{horario_local_dia[0]}**\n\n"
    analise += f"- Local mais frequente geral: **{local_geral[0]}**\n"
    analise += f"   - Melhor horário nesse local: **{horario_local_geral[0]}**\n\n"

    for i, (dados, titulo) in enumerate([(contagem_turno, "Turnos"), (contagem_horario, "Horários"), (contagem_dia, "Dias"), (contagem_local, "Locais")]):
        dados = dict(sorted(dados.items(), key=lambda x: x[1], reverse=True))
        if titulo == "Locais":
            dados = dict(list(dados.items())[:10])
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=list(dados.keys()), y=list(dados.values()), ax=ax, palette="dark:blue_r")
        ax.set_title(titulo)
        plt.xticks(rotation=45)
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png")
        imagens.append(buf)
        plt.close()
    return analise, imagens


def gerar_pdf_streamlit(df, placa, titulo, analise, imagens):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from io import BytesIO

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 750, titulo)

    y = 720
    for buf in imagens:
        buf.seek(0)
        c.drawImage(ImageReader(buf), 50, y - 100, width=400, height=100)
        y -= 120
        if y < 100:
            c.showPage()
            y = 750

    c.showPage()
    c.setFont("Helvetica", 10)
    y = 750

    
    for linha in analise.split("\n"):
        c.drawString(50, y, linha)
        y -= 15
        if y < 50:
            c.showPage()
            y = 750

    c.save()
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    main()
