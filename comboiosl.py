# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Inicializar o modelo InsightFace
@st.cache_resource
def load_face_model():
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

app = load_face_model()

# Funções auxiliares
def calculate_similarity(embedding1, embedding2):
    cos_distance = cosine(embedding1, embedding2)
    similarity = (1 - cos_distance) * 100
    similarity = max(min(similarity, 100), 0)
    return similarity

def extract_faces(image):
    """Extrai rostos com bounding boxes e embeddings"""
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    faces = app.get(img)
    if not faces:
        return None
    
    # Desenhar bounding boxes na imagem
    img_with_boxes = img.copy()
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    return {
        'image': img,
        'image_with_boxes': img_with_boxes,
        'faces': faces
    }

def plot_embeddings(embeddings, labels):
    """Visualiza embeddings em 2D e 3D usando PCA e t-SNE"""
    
    # Converter para numpy array
    embeddings = np.array(embeddings)
    
    # PCA 2D
    pca_2d = PCA(n_components=2)
    embeddings_pca_2d = pca_2d.fit_transform(embeddings)
    
    # t-SNE 2D
    tsne_2d = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1))
    embeddings_tsne_2d = tsne_2d.fit_transform(embeddings)
    
    # Criar figuras
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot PCA 2D
    for i, (x, y) in enumerate(embeddings_pca_2d):
        ax1.scatter(x, y, label=labels[i])
        ax1.text(x, y, f"{i+1}", fontsize=12)
    ax1.set_title('PCA 2D')
    ax1.legend()
    
    # Plot t-SNE 2D
    for i, (x, y) in enumerate(embeddings_tsne_2d):
        ax2.scatter(x, y, label=labels[i])
        ax2.text(x, y, f"{i+1}", fontsize=12)
    ax2.set_title('t-SNE 2D')
    ax2.legend()
    
    st.pyplot(fig)
    
    # PCA 3D se tivermos pelo menos 3 pontos
    if len(embeddings) >= 3:
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        pca_3d = PCA(n_components=3)
        embeddings_pca_3d = pca_3d.fit_transform(embeddings)
        
        for i, (x, y, z) in enumerate(embeddings_pca_3d):
            ax_3d.scatter(x, y, z, label=labels[i])
            ax_3d.text(x, y, z, f"{i+1}", fontsize=12)
        
        ax_3d.set_title('PCA 3D')
        ax_3d.legend()
        st.pyplot(fig_3d)

# Interface Streamlit
st.title("🔍 Comparador Avançado de Rostos")
st.write("""
Faça upload de duas imagens para comparar os rostos detectados. 
A aplicação mostrará:
- Imagens com bounding boxes nos rostos detectados
- Similaridade entre os rostos
- Visualização dos embeddings em 2D e 3D
""")

# Upload das imagens
col1, col2 = st.columns(2)
with col1:
    st.subheader("Imagem 1")
    img1 = st.file_uploader("Selecione a primeira imagem", type=["jpg", "jpeg", "png"], key="img1")

with col2:
    st.subheader("Imagem 2")
    img2 = st.file_uploader("Selecione a segunda imagem", type=["jpg", "jpeg", "png"], key="img2")

if img1 and img2:
    # Processar imagens
    result1 = extract_faces(img1)
    result2 = extract_faces(img2)
    
    if not result1 or not result2:
        st.error("Não foi possível detectar rostos em uma ou ambas as imagens.")
    else:
        # Mostrar imagens com bounding boxes
        st.subheader("Rostos Detectados")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(result1['image_with_boxes'], cv2.COLOR_BGR2RGB), 
                    caption="Imagem 1 com rostos detectados", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(result2['image_with_boxes'], cv2.COLOR_BGR2RGB), 
                    caption="Imagem 2 com rostos detectados", use_column_width=True)
        
        # Comparar os primeiros rostos de cada imagem
        face1 = result1['faces'][0]
        face2 = result2['faces'][0]
        
        similarity = calculate_similarity(face1.embedding, face2.embedding)
        
        # Resultados da comparação
        st.subheader("📊 Resultados da Comparação")
        st.metric("Similaridade entre os rostos", f"{similarity:.2f}%")
        
        # Barra de similaridade
        st.progress(int(similarity))
        
        # Interpretação
        if similarity < 60.0:
            st.warning("👤 As imagens NÃO são da mesma pessoa (similaridade abaixo de 60.0%)")
        elif 60.0 <= similarity < 75.0:
            st.info("🤔 As imagens são PROVAVELMENTE da mesma pessoa (similaridade entre 60.0% e 75.0%)")
        else:
            st.success("✅ As imagens SÃO da mesma pessoa (similaridade acima de 75.0%)")
        
        # Métricas adicionais
        st.write(f"**Distância de cosseno:** {cosine(face1.embedding, face2.embedding):.4f}")
        
        # Visualização dos embeddings
        st.subheader("📈 Visualização dos Embeddings")
        
        embeddings = [face1.embedding, face2.embedding]
        labels = ["Imagem 1", "Imagem 2"]
        
        plot_embeddings(embeddings, labels)
        
        # Detalhes técnicos
        with st.expander("🔍 Detalhes técnicos"):
            st.write("**Embedding da Imagem 1:**", face1.embedding)
            st.write("**Embedding da Imagem 2:**", face2.embedding)
            st.write("**Dimensão dos embeddings:**", len(face1.embedding))
            
            # Matriz de distância
            dist_matrix = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    dist_matrix[i, j] = cosine(embeddings[i], embeddings[j])
            
            st.write("**Matriz de distância de cosseno:**")
            st.dataframe(dist_matrix, index=labels, columns=labels)

# Rodapé
st.markdown("---")
st.caption("Aplicação desenvolvida usando InsightFace, Streamlit e técnicas de redução de dimensionalidade")
Funcionalidades desta versão:
Upload de duas imagens:

Interface simples para selecionar duas imagens do computador

Processamento em tempo real

Detecção de rostos com bounding boxes:

Mostra as imagens originais com retângulos verdes ao redor dos rostos detectados

Usa o detector do InsightFace para localização precisa

Análise de similaridade:

Calcula a similaridade entre os rostos em porcentagem

Mostra a distância de cosseno entre os embeddings

Classificação em 3 categorias com feedback visual

Visualização dos embeddings:

Gráficos em segunda dimensão usando PCA e t-SNE

Gráfico 3D usando PCA (quando aplicável)

Rótulos claros para cada ponto

Informações técnicas:

Expansível com detalhes dos embeddings e matriz de distância

Como executar:
Salve o código em um arquivo face_comparison.py

Instale as dependências:

pip install streamlit opencv-python numpy insightface scikit-learn matplotlib scipy
Execute com:

streamlit run face_comparison.py
Melhorias adicionais possíveis:
Adicionar opção para comparar múltiplos rostos na mesma imagem

Permitir ajuste dos limiares de similaridade

Adicionar animações durante o processamento

Opção para salvar os resultados da comparação

Visualização mais interativa dos gráficos 3D (usando Plotly)

Esta aplicação fornece uma análise completa da similaridade facial com visualizações intuitivas dos embeddings, mantendo a simplicidade de uso com apenas duas imagens de upload.

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

Adicione a esse código da aplicação.
Vou adicionar a funcionalidade de comparação facial ao seu código existente, criando uma nova página no menu de navegação. Aqui está a versão integrada:

python
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
    st.set_page_config(layout="wide")
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
    Faça upload de duas imagens para comparar os rostos detectados. 
    A aplicação mostrará:
    - Imagens com bounding boxes nos rostos detectados
    - Similaridade entre os rostos
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

def processar_comparacao_facial(img1, img2):
    def extract_face_details(image_file):
        img_bytes = image_file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Erro ao ler a imagem.")
            return None
        
        faces = app.get(img)
        if not faces:
            st.error("Nenhum rosto detectado na imagem.")
            return None
        
        # Desenhar bounding boxes
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
        cos_distance = cosine(embedding1, embedding2)
        similarity = (1 - cos_distance) * 100
        similarity = max(min(similarity, 100), 0)
        return similarity

    def plot_embeddings(embeddings, labels):
        embeddings = np.array(embeddings)
        
        # PCA 2D
        pca_2d = PCA(n_components=2)
        embeddings_pca_2d = pca_2d.fit_transform(embeddings)
        
        # t-SNE 2D
        tsne_2d = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1))
        embeddings_tsne_2d = tsne_2d.fit_transform(embeddings)
        
        # Criar figuras
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot PCA 2D
        for i, (x, y) in enumerate(embeddings_pca_2d):
            ax1.scatter(x, y, label=labels[i])
            ax1.text(x, y, f"{i+1}", fontsize=12)
        ax1.set_title('PCA 2D')
        ax1.legend()
        
        # Plot t-SNE 2D
        for i, (x, y) in enumerate(embeddings_tsne_2d):
            ax2.scatter(x, y, label=labels[i])
            ax2.text(x, y, f"{i+1}", fontsize=12)
        ax2.set_title('t-SNE 2D')
        ax2.legend()
        
        st.pyplot(fig)
        
        # PCA 3D se tivermos pelo menos 3 pontos
        if len(embeddings) >= 3:
            fig_3d = plt.figure(figsize=(8, 6))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            
            pca_3d = PCA(n_components=3)
            embeddings_pca_3d = pca_3d.fit_transform(embeddings)
            
            for i, (x, y, z) in enumerate(embeddings_pca_3d):
                ax_3d.scatter(x, y, z, label=labels[i])
                ax_3d.text(x, y, z, f"{i+1}", fontsize=12)
            
            ax_3d.set_title('PCA 3D')
            ax_3d.legend()
            st.pyplot(fig_3d)

    # Processar imagens
    result1 = extract_face_details(img1)
    result2 = extract_face_details(img2)
    
    if not result1 or not result2:
        return
    
    # Mostrar imagens com bounding boxes
    st.subheader("Rostos Detectados")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(result1['image_with_boxes'], cv2.COLOR_BGR2RGB), 
                caption="Imagem 1 com rostos detectados", use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(result2['image_with_boxes'], cv2.COLOR_BGR2RGB), 
                caption="Imagem 2 com rostos detectados", use_column_width=True)
    
    # Comparar os primeiros rostos de cada imagem
    face1 = result1['faces'][0]
    face2 = result2['faces'][0]
    
    similarity = calculate_similarity(face1.embedding, face2.embedding)
    
    # Resultados da comparação
    st.subheader("📊 Resultados da Comparação")
    st.metric("Similaridade entre os rostos", f"{similarity:.2f}%")
    
    # Barra de similaridade
    st.progress(int(similarity))
    
    # Interpretação
    if similarity < 60.0:
        st.warning("👤 As imagens NÃO são da mesma pessoa (similaridade abaixo de 60.0%)")
    elif 60.0 <= similarity < 75.0:
        st.info("🤔 As imagens são PROVAVELMENTE da mesma pessoa (similaridade entre 60.0% e 75.0%)")
    else:
        st.success("✅ As imagens SÃO da mesma pessoa (similaridade acima de 75.0%)")
    
    # Métricas adicionais
    st.write(f"**Distância de cosseno:** {cosine(face1.embedding, face2.embedding):.4f}")
    
    # Visualização dos embeddings
    st.subheader("📈 Visualização dos Embeddings")
    
    embeddings = [face1.embedding, face2.embedding]
    labels = ["Imagem 1", "Imagem 2"]
    
    plot_embeddings(embeddings, labels)
    
    # Detalhes técnicos
    with st.expander("🔍 Detalhes técnicos"):
        st.write("**Embedding da Imagem 1:**", face1.embedding)
        st.write("**Embedding da Imagem 2:**", face2.embedding)
        st.write("**Dimensão dos embeddings:**", len(face1.embedding))
        
        # Matriz de distância
        dist_matrix = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                dist_matrix[i, j] = cosine(embeddings[i], embeddings[j])
        
        st.write("**Matriz de distância de cosseno:**")
        st.dataframe(dist_matrix, index=labels, columns=labels)

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
