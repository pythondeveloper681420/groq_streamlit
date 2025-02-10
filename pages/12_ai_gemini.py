import streamlit as st
from google.generativeai import configure, GenerativeModel
import google.generativeai as genai
import time
from PIL import Image
import pdf2image
import io
import fitz  # PyMuPDF

# Page configuration
st.set_page_config(
    page_title="Chatbot com Streamlit e Google AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuração da API do Google
configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Inicialização dos modelos - usando Gemini 1.5
chat_model = GenerativeModel('gemini-1.5-pro')
vision_model = GenerativeModel('gemini-1.5-flash')  # Modelo mais rápido para processamento de imagens

# Inicializa o histórico de mensagens na sessão se não existir
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat" not in st.session_state:
    st.session_state.chat = chat_model.start_chat(history=[])

def processar_documento(arquivo):
    if arquivo.type == "application/pdf":
        # Processar PDF
        pdf_document = fitz.open(stream=arquivo.read(), filetype="pdf")
        texto = ""
        for pagina in pdf_document:
            texto += pagina.get_text()
        return texto
    else:
        # Para outros tipos de documento, retornar mensagem de erro
        return "Formato de documento não suportado. Por favor, envie um arquivo PDF."

def gerar_resposta(prompt, imagem=None, documento=None):
    try:
        with st.spinner('Gerando resposta...'):
            if imagem:
                # Prepara a imagem para o modelo vision
                if isinstance(imagem, Image.Image):
                    # Converte a imagem para bytes
                    img_byte_arr = io.BytesIO()
                    imagem.save(img_byte_arr, format=imagem.format or 'PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                # Gera o conteúdo usando o modelo vision
                response = vision_model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_byte_arr}
                ])
                return response.text
            elif documento:
                # Combina o texto do documento com o prompt do usuário
                prompt_completo = f"Documento: {documento}\n\nPergunta: {prompt}"
                response = st.session_state.chat.send_message(prompt_completo)
                return response.text
            else:
                # Resposta normal do chat
                response = st.session_state.chat.send_message(prompt)
                time.sleep(0.5)
                return response.text
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

st.markdown('## **🤖 :rainbow[Chatbot com Google AI]**')

# Adiciona upload de arquivo na sidebar
with st.sidebar:
    st.header("Upload de Arquivos")
    arquivo_imagem = st.file_uploader("Upload de Imagem", type=["jpg", "jpeg", "png"])
    arquivo_documento = st.file_uploader("Upload de Documento", type=["pdf"])

# Container principal do chat
chat_container = st.container()

# Exibe o histórico da conversa
with chat_container:
    for autor, mensagem in st.session_state.chat_history:
        if autor == "Você":
            st.write(f"👤 **Você:** {mensagem}")
        else:
            st.write(f"🤖 **Assistente:** {mensagem}")

# Formulário para input do usuário
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6,1])
    with col1:
        entrada_usuario = st.text_input("Digite sua mensagem:", key="user_input")
    with col2:
        enviado = st.form_submit_button("Enviar")

    if enviado and entrada_usuario:
        # Armazena a mensagem do usuário
        st.session_state.chat_history.append(("Você", entrada_usuario))
        
        # Processa a entrada com base no contexto (imagem ou documento)
        if arquivo_imagem:
            imagem = Image.open(arquivo_imagem)
            resposta_chat = gerar_resposta(entrada_usuario, imagem=imagem)
        elif arquivo_documento:
            texto_documento = processar_documento(arquivo_documento)
            resposta_chat = gerar_resposta(entrada_usuario, documento=texto_documento)
        else:
            resposta_chat = gerar_resposta(entrada_usuario)
        
        # Armazena a resposta
        st.session_state.chat_history.append(("Assistente", resposta_chat))
        
        # Recarrega a página
        st.rerun()

# Botão para limpar o histórico
if st.button("Limpar Conversa"):
    st.session_state.chat_history = []
    st.session_state.chat = chat_model.start_chat(history=[])
    st.rerun()