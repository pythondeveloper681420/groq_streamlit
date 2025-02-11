import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import time
from PIL import Image
import fitz  # PyMuPDF
import io
import os
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Chatbot Gemini com Processamento de Documentos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração do Gemini (usando chave gratuita)
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]  # Chave API gratuita do Google
genai.configure(api_key=GOOGLE_API_KEY)

# Inicialização dos modelos
model = genai.GenerativeModel('gemini-1.5-pro')
vision_model = genai.GenerativeModel('gemini-1.5-flash')

# Configuração do modelo de embeddings gratuito
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Inicialização do embedding model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

# Função para processar documentos
def processar_documento(arquivo, embeddings):
    try:
        if arquivo.type == "application/pdf":
            # Processar PDF
            pdf_document = fitz.open(stream=arquivo.read(), filetype="pdf")
            texto = ""
            for pagina in pdf_document:
                texto += pagina.get_text()
            
            # Dividir texto em chunks menores
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(texto)
            
            # Criar base de conhecimento vetorial
            vectorstore = FAISS.from_texts(chunks, embeddings)
            
            return {
                "status": "success",
                "texto": texto,
                "vectorstore": vectorstore,
                "n_paginas": len(pdf_document),
                "n_chunks": len(chunks)
            }
        else:
            return {
                "status": "error",
                "message": "Formato não suportado. Por favor, envie um arquivo PDF."
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro ao processar documento: {str(e)}"
        }

# Função para gerar resposta contextualizada
def gerar_resposta(prompt, chat_context=None, vectorstore=None):
    try:
        with st.spinner('Processando sua pergunta...'):
            if vectorstore:
                # Buscar contexto relevante
                docs = vectorstore.similarity_search(prompt, k=3)
                contexto = "\n".join([doc.page_content for doc in docs])
                
                # Criar prompt enriquecido
                prompt_enriquecido = f"""
                Contexto do documento:
                {contexto}
                
                Pergunta do usuário:
                {prompt}
                
                Por favor, responda à pergunta usando apenas as informações fornecidas no contexto acima.
                """
                
                response = model.generate_content(prompt_enriquecido)
            else:
                response = model.generate_content(prompt)
            
            return response.text
            
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

# Inicialização do estado da sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documento_atual" not in st.session_state:
    st.session_state.documento_atual = None
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Carregar embeddings
embeddings = get_embeddings()

# Interface principal
st.markdown('## **🤖 :rainbow[Chatbot Gemini com Processamento de Documentos]**')

# Sidebar para upload e informações
with st.sidebar:
    st.header("📁 Upload e Informações")
    arquivo_documento = st.file_uploader("Upload de Documento", type=["pdf"])
    
    if arquivo_documento:
        resultado = processar_documento(arquivo_documento, embeddings)
        if resultado["status"] == "success":
            st.session_state.vectorstore = resultado["vectorstore"]
            st.session_state.documento_atual = arquivo_documento.name
            
            st.success("✅ Documento processado com sucesso!")
            st.info(f"""
            📊 Informações do documento:
            - Nome: {arquivo_documento.name}
            - Páginas: {resultado['n_paginas']}
            - Chunks processados: {resultado['n_chunks']}
            """)
        else:
            st.error(resultado["message"])
    
    if st.session_state.documento_atual:
        st.markdown("---")
        st.markdown(f"📄 **Documento atual:** {st.session_state.documento_atual}")

# Container principal do chat
chat_container = st.container()

# Exibir histórico
with chat_container:
    for autor, mensagem in st.session_state.chat_history:
        if autor == "Você":
            st.write(f"👤 **Você:** {mensagem}")
        else:
            st.write(f"🤖 **Assistente:** {mensagem}")

# Formulário de entrada
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6,1])
    with col1:
        entrada_usuario = st.text_input("Digite sua mensagem:", key="user_input")
    with col2:
        enviado = st.form_submit_button("Enviar")

    if enviado and entrada_usuario:
        # Armazenar mensagem do usuário
        st.session_state.chat_history.append(("Você", entrada_usuario))
        
        # Gerar resposta
        resposta_chat = gerar_resposta(
            entrada_usuario,
            st.session_state.chat,
            st.session_state.vectorstore
        )
        
        # Armazenar resposta
        st.session_state.chat_history.append(("Assistente", resposta_chat))
        
        # Recarregar página
        st.rerun()

# Botões de controle
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.chat_history = []
        st.session_state.chat = model.start_chat(history=[])
        st.rerun()
with col2:
    if st.button("📄 Limpar Documento"):
        st.session_state.vectorstore = None
        st.session_state.documento_atual = None
        st.rerun()