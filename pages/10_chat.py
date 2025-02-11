import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pypdf
import pandas as pd
from io import BytesIO
from PIL import Image
import base64
import asyncio
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Configurações e Constantes
MAX_CHUNK_SIZE = 30000
MAX_CONCURRENT_REQUESTS = 3
REQUEST_INTERVAL = 1.0

# Configuração da página Streamlit
st.set_page_config(
    page_title="Assistente Inteligente de Documentos",
    page_icon="🤖",
    layout="wide"
)

# Carrega variáveis de ambiente
load_dotenv()

class RateLimitError(Exception):
    """Exceção customizada para rate limiting."""
    def __init__(self, retry_after=None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds")

class RequestManager:
    """Gerencia requisições com rate limiting."""
    def __init__(self):
        self.last_request_time = 0
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.lock = asyncio.Lock()
    
    async def wait_for_rate_limit(self):
        """Espera o tempo necessário entre requisições."""
        async with self.lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < REQUEST_INTERVAL:
                await asyncio.sleep(REQUEST_INTERVAL - time_since_last)
            self.last_request_time = asyncio.get_event_loop().time()

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

def chunk_text(text, max_size=MAX_CHUNK_SIZE):
    """Divide texto em chunks menores de forma inteligente."""
    if not text or len(text) <= max_size:
        return [text] if text else []
    
    chunks = []
    current_chunk = ""
    
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_size:
            current_chunk += paragraph + '\n\n'
        else:
            if len(paragraph) > max_size:
                sentences = paragraph.replace('\n', ' ').split('. ')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > max_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                    current_chunk += sentence + '. '
            else:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF."""
    try:
        if not pdf_file:
            raise ValueError("Nenhum arquivo PDF fornecido")
        
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = []
        total_pages = len(pdf_reader.pages)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                status_text.text(f"Processando página {page_num} de {total_pages}")
                page_text = page.extract_text()
                if page_text:
                    text.append(f"\n=== Página {page_num} ===\n{page_text}")
                progress_bar.progress(page_num / total_pages)
            except Exception as e:
                st.warning(f"Erro ao extrair texto da página {page_num}: {e}")
                continue
                
        progress_bar.empty()
        status_text.empty()
        
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
        total_sheets = len(excel_data.sheet_names)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, sheet_name in enumerate(excel_data.sheet_names, 1):
            try:
                status_text.text(f"Processando planilha {idx} de {total_sheets}: {sheet_name}")
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if not df.empty:
                    all_sheets_text.append(f"\n=== Planilha: {sheet_name} ===\n{df.to_string()}")
                progress_bar.progress(idx / total_sheets)
            except Exception as e:
                st.warning(f"Erro ao processar planilha '{sheet_name}': {e}")
                continue
                
        progress_bar.empty()
        status_text.empty()
        
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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError)  # Aqui estava o erro: use "retry=" ao invés de apenas passando o argumento direto
)

async def process_chunk(chunk, question, model, request_manager, safety_settings):
    """Processa um chunk individual com rate limiting."""
    async with request_manager.semaphore:
        await request_manager.wait_for_rate_limit()
        try:
            prompt = f"""Analise este trecho do documento e responda à pergunta considerando o contexto geral:

            Texto: {chunk}

            Pergunta: {question}

            Instruções:
            1. Foque nas informações relevantes deste trecho
            2. Mantenha consistência com o contexto geral
            3. Indique se a informação parece incompleta ou precisa de mais contexto"""

            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                safety_settings=safety_settings
            )
            
            if response.text.strip() in ["True", "False"]:
                raise ValueError("Resposta inválida recebida da API")
            
            return response.text
        except Exception as e:
            if "429" in str(e):
                raise RateLimitError(retry_after=60)
            raise

async def process_large_document(content, question, model, safety_settings):
    """Processa documento grande em chunks paralelos."""
    chunks = chunk_text(content)
    request_manager = RequestManager()
    
    total_chunks = len(chunks)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_chunks = 0
    tasks = []
    
    # Criação das tasks com todos os argumentos nomeados
    for chunk in chunks:
        task = process_chunk(
            chunk=chunk,
            question=question,
            model=model,
            request_manager=request_manager,
            safety_settings=safety_settings
        )
        tasks.append(task)
    
    responses = []
    async for task in asyncio.as_completed(tasks):
        try:
            response = await task
            responses.append(response)
            processed_chunks += 1
            progress_bar.progress(processed_chunks / total_chunks)
            status_text.text(f"Processando chunk {processed_chunks} de {total_chunks}")
        except Exception as e:
            st.warning(f"Erro ao processar chunk: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    if not responses:
        raise ValueError("Não foi possível obter respostas válidas")
    
    combined_response = "\n\n".join(responses)
    
    summary_prompt = f"""Sumarize as seguintes respostas em uma única resposta coerente:

    Respostas: {combined_response}

    Pergunta original: {question}

    Instruções:
    1. Combine informações complementares
    2. Resolva possíveis contradições
    3. Mantenha apenas informações relevantes
    4. Organize em uma resposta clara e concisa"""
    
    # Uso de argumentos nomeados na chamada de generate_content
    final_response = await asyncio.to_thread(
        model.generate_content,
        contents=summary_prompt,
        safety_settings=safety_settings
    )
    
    return final_response.text

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

        if file_type == "image":
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""Por favor, analise esta imagem e responda à seguinte pergunta:

            Pergunta: {question}

            Instruções específicas:
            1. Use seu conhecimento geral para fornecer contexto adicional relevante
            2. Faça conexões com conceitos relacionados mesmo que não estejam explícitos na imagem
            3. Se identificar elementos na imagem, explique sua relevância
            4. Forneça explicações detalhadas baseadas tanto na imagem quanto em seu conhecimento geral
            
            Contexto adicional: {context if context else 'Nenhum contexto adicional fornecido'}"""

            response = model.generate_content(
                contents=[prompt, {"mime_type": "image/jpeg", "data": content}],
                safety_settings=safety_settings
            )
            return response.text
        else:
            model = genai.GenerativeModel('gemini-pro')
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    process_large_document(content, question, model, safety_settings)
                )
                return response
            finally:
                loop.close()

    except Exception as e:
        error_message = str(e)
        if isinstance(e, RateLimitError):
            st.warning(f"Limite de requisições atingido. Aguardando {e.retry_after} segundos...")
            sleep(e.retry_after)
            return get_gemini_response(content, question, file_type, context)
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

        st.header("⚡ Configurações Avançadas")
        max_chunk_size = st.slider(
            "Tamanho máximo do chunk (caracteres)",
            min_value=1000,
            max_value=50000,
            value=MAX_CHUNK_SIZE,
            step=1000,
            help="Ajuste o tamanho dos chunks para processamento. Valores menores podem ser mais precisos mas mais lentos."
        )

        max_concurrent = st.slider(
            "Máximo de requisições simultâneas",
            min_value=1,
            max_value=5,
            value=MAX_CONCURRENT_REQUESTS,
            help="Ajuste o número máximo de requisições simultâneas. Valores maiores podem ser mais rápidos mas podem atingir limites da API."
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
            with st.spinner("📄 Processando PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    content = text
                    st.success("✅ PDF processado com sucesso!")
                    with st.expander("👀 Visualizar conteúdo do PDF"):
                        st.text_area("Conteúdo do PDF", value=content, height=200, label_visibility="collapsed")

        elif file_extension in [".xlsx", ".xls"]:
            file_type = "excel"
            with st.spinner("📊 Processando Excel..."):
                text = extract_text_from_excel(uploaded_file)
                if text:
                    content = text
                    st.success("✅ Excel processado com sucesso!")
                    with st.expander("👀 Visualizar conteúdo do Excel"):
                        st.text_area("Conteúdo do Excel", value=content, height=200, label_visibility="collapsed")

        elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            file_type = "image"
            with st.spinner("🖼️ Processando imagem..."):
                encoded_image = process_image(uploaded_file)
                if encoded_image:
                    content = encoded_image
                    st.success("✅ Imagem processada com sucesso!")
                    st.image(uploaded_file, caption="Imagem carregada", use_container_width=True)

        if 'content' in locals():  # Verifica se content foi definido
            st.markdown("### 💭 Faça sua pergunta")
            
            # Adiciona sugestões de perguntas baseadas no tipo de arquivo
            suggestions = {
                "pdf": [
                    "Qual é o tema principal deste documento?",
                    "Pode fazer um resumo dos pontos principais?",
                    "Quais são as conclusões mais importantes?"
                ],
                "excel": [
                    "Quais são os padrões observados nos dados?",
                    "Pode identificar tendências importantes?",
                    "Quais são os valores mais significativos?"
                ],
                "image": [
                    "O que você pode me dizer sobre esta imagem?",
                    "Quais são os elementos principais desta imagem?",
                    "Que tipo de ambiente/contexto esta imagem mostra?"
                ]
            }

            if file_type in suggestions:
                with st.expander("💡 Sugestões de perguntas"):
                    for suggestion in suggestions[file_type]:
                        if st.button(suggestion, key=suggestion):
                            st.session_state.question = suggestion

            question = st.text_input(
                "Digite sua pergunta sobre o documento",
                key="question",
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

            # Adiciona opção de exportar histórico
            if st.session_state.chat_history:
                if st.button("📥 Exportar Histórico"):
                    export_text = "Histórico de Perguntas e Respostas\n\n"
                    for q, a in st.session_state.chat_history:
                        export_text += f"Pergunta: {q}\n\nResposta: {a}\n\n---\n\n"
                    
                    st.download_button(
                        label="💾 Baixar Histórico",
                        data=export_text.encode('utf-8'),
                        file_name="historico_chat.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()