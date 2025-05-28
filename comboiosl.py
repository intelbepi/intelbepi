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
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set(style="darkgrid")
st.set_page_config(layout="wide")

dias_semana = {
    "Monday": "Segunda-feira", "Tuesday": "Terça-feira", "Wednesday": "Quarta-feira",
    "Thursday": "Quinta-feira", "Friday": "Sexta-feira", "Saturday": "Sábado", "Sunday": "Domingo"
}

# Inicializar o modelo InsightFace
@st.cache_resource
def load_face_model():
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

app = load_face_model()

def main():
    
    st.sidebar.title("Navegação")
    pagina = st.sidebar.radio("Escolha uma opção:", 
                             ["🚗🚗 - Veículos em Comboio", 
                              "🕵️ - Análise de Passagens",
                              "👥 - Comparação Facial"])

    if pagina == "🚗🚗 - Veículos em Comboio":
        veiculos_em_comboio()
    elif pagina == "🕵️ - Análise de Passagens":
        analise_de_passagens()
    elif pagina == "👥 - Comparação Facial":
        comparacao_facial()

# -------------------------------------------
# Página 3 – Comparação Facial
# -------------------------------------------
def comparacao_facial():
    st.title("👥 Comparação Facial")
    st.write("""
    🔍 Faça upload de duas imagens para comparar os rostos detectados. 
    O sistema mostrará:
    - Imagens com bounding boxes nos rostos detectados
    - Similaridade entre os rostos mais próximos
    - Visualização dos embeddings em 2D e 3D
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem 1")
        img1 = st.file_uploader("Selecione a primeira imagem", type=["jpg", "jpeg", "png"], key="img1")
        
    with col2:
        st.subheader("Imagem 2")
        img2 = st.file_uploader("Selecione a segunda imagem", type=["jpg", "jpeg", "png"], key="img2")

    if img1 and img2:
        processar_comparacao_facial(img1, img2)

# -------------------------------------------
# Processamento
# -------------------------------------------
def processar_comparacao_facial(img1, img2):
    def extract_face_details(image_file):
        img_bytes = image_file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Erro ao ler a imagem.")
            return None
        
        faces = app.get(img)  # InsightFace detecta e extrai embeddings dos rostos
        if not faces:
            st.error("Nenhum rosto detectado na imagem.")
            return None
        
        img_with_boxes = img.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        return {
            'image': img,
            'image_with_boxes': img_with_boxes,
            'faces': faces
        }

    def calculate_similarity(embedding1, embedding2):
        # Calcula similaridade usando distância de cosseno
        cos_distance = cosine(embedding1, embedding2)
        similarity = (1 - cos_distance) * 100  # Converte para escala percentual
        similarity = max(min(similarity, 100), 0)
        return similarity, cos_distance


    # Processar imagens
    result1 = extract_face_details(img1)
    result2 = extract_face_details(img2)
    
    if not result1 or not result2:
        return
    
    # Mostrar imagens com bounding boxes
    st.subheader("🖼️ Rostos Detectados")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(result1['image_with_boxes'], cv2.COLOR_BGR2RGB), 
                caption="Imagem 1 com rostos detectados", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(result2['image_with_boxes'], cv2.COLOR_BGR2RGB), 
                caption="Imagem 2 com rostos detectados", use_container_width=True)
    
    # Comparar todos os rostos entre as duas imagens
    best_similarity = 0
    best_pair = (None, None)
    best_distance = 1  # maior possível
    
    for face1 in result1['faces']:
        for face2 in result2['faces']:
            sim, dist = calculate_similarity(face1.embedding, face2.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_distance = dist
                best_pair = (face1, face2)

    face1, face2 = best_pair
    
    # Resultados da comparação
    st.subheader("📊 Resultados da Comparação")
    st.metric("Maior Similaridade Encontrada", f"{best_similarity:.2f}%")
    
    # Barra de similaridade
    st.progress(int(best_similarity))
    
    # Interpretação baseada em testes práticos e literatura
    if best_similarity >= 70:
        st.success("✅ Alta chance de ser a MESMA pessoa (similaridade ≥ 70%)")
    elif 60 <= best_similarity < 70:
        st.info("🤔 Provavelmente a mesma pessoa (similaridade entre 60% e 70%)")
    elif 50 <= best_similarity < 60:
        st.warning("⚠️ Possivelmente a mesma pessoa, mas com incerteza (50-60%)")
    else:
        st.error("❌ Provavelmente NÃO são a mesma pessoa (similaridade < 50%)")
    
    # Explicação do limiar
    with st.expander("ℹ️ Por que o limiar varia tanto?"):
        st.write("""
        ✅ A similaridade entre rostos não é fixa e pode variar muito devido a:
        - Diferenças de **iluminação**, **ângulo do rosto**, **expressões faciais**, **resolução da imagem** e **obstruções** como óculos ou bonés.
        - O modelo faz um redimensionamento interno dos rostos para **112x112 pixels**, mas rostos muito distantes ou fotos de baixa qualidade impactam os resultados.
        - Na prática, é comum que fotos da mesma pessoa em condições diferentes tenham similaridade na faixa de **55% a 70%**.
        
        👉 Este app utiliza um limiar mais flexível, pensado para uso investigativo, não restrito como sistemas biométricos de segurança.
        """)
    
    # Distância de cosseno
    st.write(f"**Distância de Cosseno:** {best_distance:.4f}")
    
    # Visualização dos embeddings
    st.subheader("📈 Visualização dos Embeddings")
    
    embeddings = [face1.embedding, face2.embedding]
    labels = ["Imagem 1", "Imagem 2"]
    
    plot_embeddings(embeddings, labels)
    
    # Detalhes técnicos
    with st.expander("🔍 Detalhes Técnicos"):
        st.write("**Embedding da Imagem 1:**", face1.embedding)
        st.write("**Embedding da Imagem 2:**", face2.embedding)
        st.write("**Dimensão dos embeddings:**", len(face1.embedding))





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
    if st.button("Processar Registros"):
        if texto:
            processar_analise(texto)
        else:
            st.warning("Por favor, insira o texto antes de processar.")


def processar_analise(texto):
        # Adiciona "0/0" no final do texto se ainda não estiver presente
    if not texto.strip().endswith("0/0"):
        texto = texto + "\n0/0"
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
    width, height = letter

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, titulo)

    # Inserir imagens (gráficos)
    y = height - 80
    for buf in imagens:
        buf.seek(0)
        c.drawImage(ImageReader(buf), 50, y - 120, width=500, height=120, preserveAspectRatio=True)
        y -= 140
        if y < 150:
            c.showPage()
            y = height - 50

    c.showPage()  # Nova página para análise textual

    # Texto da análise
    c.setFont("Helvetica", 12)
    y = height - 50
    margem_esquerda = 50
    linhas = analise.split('\n')
    for linha in linhas:
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50
        c.drawString(margem_esquerda, y, linha)
        y -= 18

    # Opcional: incluir tabela resumo ou dados do dataframe aqui, se quiser

    c.save()
    buffer.seek(0)
    return buffer



if __name__ == "__main__":
    main()
