import streamlit as st
from streamlit_authenticator import Authenticate
import yaml
from yaml.loader import SafeLoader
import re
from collections import defaultdict

# Configuração inicial do autenticador
def setup_authenticator():
    try:
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        # Cria um arquivo de configuração padrão se não existir
        config = {
            'credentials': {
                'usernames': {
                    'admin': {
                        'email': 'admin@example.com',
                        'name': 'Administrador',
                        'password': 'admin123'  # Será hashado automaticamente
                    }
                }
            },
            'cookie': {
                'name': 'placas_cookie',
                'key': 'uma_chave_secreta_aleatoria_muito_longa',  # Substitua por uma chave real
                'expiry_days': 30
            },
            'preauthorized': {
                'emails': []
            }
        }
        with open('config.yaml', 'w') as file:
            yaml.dump(config, file)
    
    return Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

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
        # Verifica se é uma placa
        if padrao_placa.fullmatch(linha.strip()):
            placa_atual = linha.strip().upper()
        
        # Verifica se é uma data/hora e se já temos uma placa identificada
        elif placa_atual and (match := padrao_data_hora.search(linha)):
            data = match.group(1)
            hora = match.group(2)
            placas[placa_atual].append(f"{data} {hora}")
    
    return placas

# Exibição dos resultados
def display_results(placas):
    # Prepara o resultado
    resultado = "## RELATÓRIO DE PLACAS:\n\n"
    resultado += f"**Total de placas únicas encontradas:** {len(placas)}\n\n"
    
    # Separa em placas únicas e repetidas
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
def main_page(authenticator):
    st.set_page_config(page_title="Analisador de Placas Veiculares", layout="wide")
    
    # Barra lateral com informações do usuário e logout
    with st.sidebar:
        st.write(f"Bem-vindo, *{st.session_state['name']}*")
        authenticator.logout('Sair', 'sidebar')
    
    st.title("Analisador de Placas Veiculares")
    
    # Área de texto para entrada
    st.subheader("Cole os registros abaixo e clique em 'Processar':")
    text_input = st.text_area("", height=300, placeholder="Cole aqui os dados das placas e horários...", key="text_input")

    # Botão para processar
    if st.button("Processar", type="primary"):
        if text_input.strip():
            placas = process_text(text_input)
            if placas is not None:
                results = display_results(placas)
                st.markdown(results)
                
                # Opção para download do relatório
                st.download_button(
                    label="Baixar Relatório",
                    data=results,
                    file_name="relatorio_placas.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Por favor, insira algum texto para processar.")

# Página de gerenciamento de usuários (apenas para admin)
def user_management(authenticator):
    st.title("Gerenciamento de Usuários")
    
    try:
        if authenticator.register_user('Registrar novo usuário', preauthorization=False):
            st.success('Usuário registrado com sucesso')
            # Atualiza o arquivo de configuração
            with open('config.yaml', 'w') as file:
                yaml.dump(authenticator.config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)
    
    try:
        if authenticator.reset_password(st.session_state['username'], 'Redefinir senha'):
            st.success('Senha modificada com sucesso')
            with open('config.yaml', 'w') as file:
                yaml.dump(authenticator.config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)
    
    try:
        if authenticator.update_user_details(st.session_state['username'], 'Atualizar detalhes do usuário'):
            st.success('Detalhes atualizados com sucesso')
            with open('config.yaml', 'w') as file:
                yaml.dump(authenticator.config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)

# Página de administração
def admin_page(authenticator):
    st.set_page_config(page_title="Administração", layout="wide")
    
    with st.sidebar:
        st.write(f"Bem-vindo, *{st.session_state['name']}* (Admin)")
        authenticator.logout('Sair', 'sidebar')
    
    tab1, tab2 = st.tabs(["Analisador de Placas", "Gerenciamento de Usuários"])
    
    with tab1:
        main_page(authenticator)
    
    with tab2:
        user_management(authenticator)

# Inicialização do aplicativo
def run_app():
    authenticator = setup_authenticator()
    
    # Tenta autenticar o usuário
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    if authentication_status:
        st.session_state['authentication_status'] = authentication_status
        st.session_state['name'] = name
        st.session_state['username'] = username
        
        # Verifica se é admin
        if username == 'admin':
            admin_page(authenticator)
        else:
            main_page(authenticator)
    
    elif authentication_status is False:
        st.error('Usuário ou senha incorretos')
    elif authentication_status is None:
        st.warning('Por favor, insira seu usuário e senha')

if __name__ == '__main__':
    run_app()