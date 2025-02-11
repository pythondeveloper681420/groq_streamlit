import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pypdf
import pandas as pd
from io import BytesIO
from PIL import Image
import base64
import re
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuração da página
st.set_page_config(
    page_title="Assistente Inteligente de Documentos",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

def configure_gemini():
    """Configura a API do Gemini."""
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.warning("API key não encontrada. Configure sua API key no sidebar.")
            return False
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Erro ao configurar a API Gemini: {e}")
        return False

def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF."""
    try:
        if not pdf_file:
            raise ValueError("Nenhum arquivo PDF fornecido")
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text.append(f"\n=== Página {page_num} ===\n{page_text}")
            except Exception as e:
                st.warning(f"Erro ao extrair texto da página {page_num}: {e}")
                continue
        return "\n".join(text) if text else None
    except Exception as e:
        st.error(f"Erro ao processar PDF: {e}")
        return None

def extract_text_from_excel(excel_file):
    """Extrai texto de um arquivo Excel."""
    try:
        if not excel_file:
            raise ValueError("Nenhum arquivo Excel fornecido")
        excel_data = pd.ExcelFile(excel_file)
        all_sheets_text = []
        for sheet_name in excel_data.sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if not df.empty:
                    all_sheets_text.append(f"\n=== Planilha: {sheet_name} ===\n{df.to_string()}")
            except Exception as e:
                st.warning(f"Erro ao processar planilha '{sheet_name}': {e}")
                continue
        return "\n".join(all_sheets_text) if all_sheets_text else None
    except Exception as e:
        st.error(f"Erro ao processar Excel: {e}")
        return None

def process_image(image_file):
    """Processa e codifica a imagem para a API."""
    try:
        if not image_file:
            raise ValueError("Nenhuma imagem fornecida")
        image = Image.open(image_file)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        max_size = 1600
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        st.error(f"Erro ao processar imagem: {e}")
        return None

def handle_rate_limit(retry_state):
    """Gerencia erros de rate limit."""
    wait_time = min(getattr(retry_state.outcome.exception(), 'retry_after', 10), 60)
    st.warning(f"Limite de requisições atingido. Aguardando {wait_time} segundos...")
    sleep(wait_time)
    return True

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=handle_rate_limit
)
def get_gemini_response(content, question, file_type=None, context=None):
    """Obtém resposta do Gemini com integração de conhecimento base."""
    try:
        if not content or not question:
            raise ValueError("Conteúdo ou pergunta não fornecidos")

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]

        # Prompts aprimorados para diferentes tipos de arquivo
        if file_type == "image":
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""Por favor, analise esta imagem e responda à seguinte pergunta:

            Pergunta: {question}

            Instruções específicas:
            1. Use seu conhecimento geral para fornecer contexto adicional relevante
            2. Faça conexões com conceitos relacionados mesmo que não estejam explícitos na imagem
            3. Se identificar elementos na imagem, explique sua relevância e relações com outros conceitos
            4. Forneça explicações detalhadas baseadas tanto na imagem quanto em seu conhecimento geral
            
            Contexto adicional do usuário: {context if context else 'Nenhum contexto adicional fornecido'}"""

            response = model.generate_content(
                contents=[prompt, {"mime_type": "image/jpeg", "data": content}],
                safety_settings=safety_settings
            )
        else:
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""Analise o seguinte texto e responda à pergunta utilizando tanto o conteúdo fornecido quanto seu conhecimento base:

            Texto para análise:
            {content}

            Pergunta: {question}

            Instruções específicas:
            1. Primeiro, analise as informações explícitas no documento
            2. Em seguida, utilize seu conhecimento base para:
               - Fazer conexões com conceitos relacionados
               - Explicar relações implícitas
               - Fornecer contexto adicional relevante
               - Classificar elementos dentro de categorias apropriadas
            3. Combine as informações do documento com seu conhecimento para uma resposta completa
            4. Se uma informação não estiver no documento mas for relevante, indique que está usando conhecimento adicional
            
            Contexto adicional do usuário: {context if context else 'Nenhum contexto adicional fornecido'}"""

            response = model.generate_content(
                prompt,
                safety_settings=safety_settings
            )

        if not response.text or response.text.strip() in ["True", "False"]:
            raise ValueError("Resposta inválida recebida da API")
        return response.text

    except Exception as e:
        error_message = str(e)
        if "429" in error_message:
            st.warning("Limite de requisições atingido. Tentando novamente...")
            raise
        elif "403" in error_message:
            st.error("Erro de autenticação. Verifique sua API key.")
        else:
            st.error(f"Erro ao obter resposta: {e}")
        return "Não foi possível gerar uma resposta adequada. Por favor, tente novamente."

def init_session_state():
    """Inicializa variáveis da sessão."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "context" not in st.session_state:
        st.session_state.context = ""

def show_chat_history():
    """Exibe o histórico do chat."""
    if not st.session_state.chat_history:
        st.info("Faça uma pergunta para começar a conversa!")
        return

    for question, answer in st.session_state.chat_history:
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(
                    f"""<div style='background-color:#f0f2f6; 
                    padding: 10px; border-radius: 10px; 
                    margin-bottom: 10px;'>
                    <strong>Pergunta:</strong><br>{question}</div>""",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""<div style='background-color:#d4edda; 
                    padding: 10px; border-radius: 10px; 
                    margin-bottom: 10px;'>
                    <strong>Resposta:</strong><br>{answer}</div>""",
                    unsafe_allow_html=True
                )

def main():
    st.title("🤖 Assistente Inteligente de Documentos")
    st.markdown("""
    Este assistente usa IA avançada para analisar seus documentos e responder perguntas,
    combinando o conteúdo do documento com conhecimento geral para respostas mais completas.
    """)

    with st.sidebar:
        st.header("⚙️ Configuração")
        api_key = st.text_input(
            "API Key do Google",
            type="password",
            help="Insira sua API key se não estiver configurada no .env"
        )
        if api_key:
            genai.configure(api_key=api_key)

        st.header("🎯 Contexto Adicional")
        st.session_state.context = st.text_area(
            "Adicione contexto para melhorar as respostas",
            help="Informações adicionais que podem ajudar a IA a entender melhor suas perguntas"
        )

    if not configure_gemini():
        st.stop()

    init_session_state()

    uploaded_file = st.file_uploader(
        "📤 Carregue seu arquivo",
        type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg", "bmp", "tiff"],
        help="Arquivos suportados: PDF, Excel, PNG, JPG, JPEG, BMP, TIFF"
    )

    if uploaded_file:
        file_type = None
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".pdf":
            file_type = "pdf"
        elif file_extension in [".xlsx", ".xls"]:
            file_type = "excel"
        elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            file_type = "image"

        content = None

        if file_type == "pdf":
            with st.spinner("📄 Processando PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    content = text
                    st.success("✅ PDF processado com sucesso!")
                    with st.expander("👀 Visualizar conteúdo do PDF"):
                        st.text_area("Conteúdo do PDF", value=content, height=200, label_visibility="collapsed")

        elif file_type == "excel":
            with st.spinner("📊 Processando Excel..."):
                text = extract_text_from_excel(uploaded_file)
                if text:
                    content = text
                    st.success("✅ Excel processado com sucesso!")
                    with st.expander("👀 Visualizar conteúdo do Excel"):
                        st.text_area("Conteúdo do Excel", value=content, height=200, label_visibility="collapsed")

        elif file_type == "image":
            with st.spinner("🖼️ Processando imagem..."):
                encoded_image = process_image(uploaded_file)
                if encoded_image:
                    content = encoded_image
                    st.success("✅ Imagem processada com sucesso!")
                    st.image(uploaded_file, caption="Imagem carregada", use_container_width=True)

        if content:
            st.markdown("### 💭 Faça sua pergunta")
            question = st.text_input(
                "Digite sua pergunta sobre o documento",
                key="question_input",
                help="Seja específico em sua pergunta. O assistente combinará o conteúdo do documento com conhecimento geral."
            )

            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                ask_button = st.button(
                    "🚀 Perguntar",
                    use_container_width=True,
                    help="Clique para enviar sua pergunta"
                )

            with col2:
                clear_button = st.button(
                    "🗑️ Limpar Histórico",
                    use_container_width=True,
                    help="Limpa todo o histórico de perguntas e respostas"
                )

            if clear_button:
                st.session_state.chat_history = []
                st.rerun()

            if ask_button or question:
                if question.strip():
                    with st.spinner("🤔 Analisando e gerando resposta..."):
                        answer = get_gemini_response(
                            content, 
                            question, 
                            file_type, 
                            st.session_state.context
                        )
                        if answer:
                            st.session_state.chat_history.append((question, answer))
                else:
                    st.warning("⚠️ Por favor, digite uma pergunta.")

            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("### 📝 Histórico da Conversa")
                show_chat_history()

if __name__ == "__main__":
    main()