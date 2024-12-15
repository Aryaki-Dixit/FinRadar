import streamlit as st
import os
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize performance metrics in session state
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {
        'response_times': [],
        'embedding_times': [],
        'accuracy_scores': []
    }

def reset_metrics():
    st.session_state.performance_metrics = {
        'response_times': [],
        'embedding_times': [],
        'accuracy_scores': []
    }

def calculate_answer_relevancy(answer, source_docs, embeddings):
    """
    Calculate relevancy score between the answer and source documents
    using cosine similarity of embeddings.
    """
    try:
        # Get embeddings for the answer and source documents
        answer_embedding = embeddings.embed_query(answer)
        
        # Combine all source documents into one text
        source_text = " ".join([doc.page_content for doc in source_docs])
        source_embedding = embeddings.embed_query(source_text)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            np.array(answer_embedding).reshape(1, -1),
            np.array(source_embedding).reshape(1, -1)
        )[0][0]
        
        # Convert similarity to percentage
        relevancy_score = (similarity + 1) / 2 * 100
        return round(relevancy_score, 2)
    except Exception as e:
        st.error(f"Error calculating relevancy: {e}")
        return None

def display_performance_metrics():
    metrics = st.session_state.performance_metrics
    
    # Calculate averages
    avg_response_time = sum(metrics['response_times']) / len(metrics['response_times']) if metrics['response_times'] else 0
    avg_embedding_time = sum(metrics['embedding_times']) / len(metrics['embedding_times']) if metrics['embedding_times'] else 0
    avg_accuracy = sum(metrics['accuracy_scores']) / len(metrics['accuracy_scores']) if metrics['accuracy_scores'] else 0
    
    # Create metrics dashboard
    st.sidebar.markdown("### 📊 Performance Metrics")
    
    st.sidebar.markdown(f"""
        ⏱ Avg Response Time: {avg_response_time:.2f}s<br>
        🔄 Avg Embedding Time: {avg_embedding_time:.2f}s<br>
        🎯 Avg Answer Relevancy: {avg_accuracy:.2f}%
    """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(page_title="FinRadar", page_icon="📈", layout="wide")

# CSS styles
st.markdown("""
    <style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #3FCCA8;
        color: #1a1a1a;
        border-radius: 25px;
        padding: 0.75rem 2.5rem;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2AAA8A;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(42,170,138,0.3);
    }
    .header-text {
        text-align: center;
        color: #ffffff;
        padding: 2rem 0;
    }
    .big-title {
        font-size: 4rem;
        font-weight: bold;
        background: linear-gradient(120deg, #2AAA8A, #3FCCA8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .quote-text {
        font-size: 1.5rem;
        font-style: italic;
        color: #000000;
        margin: 2rem 0;
    }
    .about-section {
        background-color: #2a2a2a;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    .tool-section {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session states
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'landing'
if 'urls_processed' not in st.session_state:
    st.session_state.urls_processed = False

# Landing Page
if st.session_state.current_page == 'landing':
    st.markdown('<h1 class="big-title">FinRadar</h1>', unsafe_allow_html=True)
    st.markdown('<p class="quote-text">"Transforming Financial News into Actionable Intelligence"</p>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown("### About Us")
    st.write("""
    FinRadar is your AI-powered companion for navigating the complex world of financial news. 
    Our cutting-edge platform harnesses the power of artificial intelligence to analyze multiple news sources simultaneously, 
    providing you with deep insights and answers to your most pressing financial questions.
    
    Whether you're a seasoned investor, financial analyst, or just getting started in the world of finance, 
    FinRadar helps you stay ahead of the curve by:
    - 📊 Processing multiple news sources in seconds
    - 🤖 Utilizing advanced AI for comprehensive analysis
    - 💡 Providing intelligent answers to your specific queries
    - 🎯 Delivering source-backed insights
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Let's Get Started! 🚀"):
        st.session_state.current_page = 'tool'
        st.rerun()

# Tool Page
elif st.session_state.current_page == 'tool':
    st.markdown('<h1 style="color: #2AAA8A;">Your Financial News Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Unlock the Power of AI-Driven Financial Analysis")

    # Reset metrics button in sidebar
    if st.sidebar.button("Reset Metrics"):
        reset_metrics()
        st.rerun()

    url_col, spacer, result_col = st.columns([2, 0.5, 2])

    with url_col:
        st.subheader("📰 Add Your News Sources")
        urls = []
        for i in range(3):
            url = st.text_input(f"URL {i+1}")
            if url:
                urls.append(url)

        button_text = "Processed URLs! ✅" if st.session_state.urls_processed else "Process URLs"
        process_clicked = st.button(button_text, disabled=st.session_state.urls_processed)
        
        if process_clicked and urls:
            try:
                start_time = time.time()
                with st.spinner("Processing articles... "):
                    # Measure embedding time
                    embedding_start = time.time()
                    loader = UnstructuredURLLoader(urls=urls)
                    data = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=['\n\n', '\n', '.', ','],
                        chunk_size=1000
                    )
                    docs = text_splitter.split_documents(data)
                    
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    embedding_time = time.time() - embedding_start
                    
                    st.session_state.performance_metrics['embedding_times'].append(embedding_time)
                    
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_urls = urls
                    st.session_state.urls_processed = True
                    
                    total_time = time.time() - start_time
                    st.session_state.performance_metrics['response_times'].append(total_time)
                    
                    st.success("Articles processed successfully! 🎉")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing URLs: {e}")

    with result_col:
        st.subheader("🔍 Ask Your Questions")
        
        if 'vectorstore' in st.session_state:
            query = st.text_input("What insights would you like to discover?")
            
            if query:
                try:
                    start_time = time.time()
                    with st.spinner("Analyzing articles... 🤔"):
                        llm = ChatOpenAI(
                            model="gpt-3.5-turbo", 
                            temperature=0.7,
                            api_key=OPENAI_API_KEY
                        )
                        chain = RetrievalQAWithSourcesChain.from_llm(
                            llm=llm,
                            retriever=st.session_state.vectorstore.as_retriever(
                                search_kwargs={"k": 3}  # Fetch top 3 most relevant documents
                            )
                        )
                        
                        # Get source documents for relevancy calculation
                        source_docs = st.session_state.vectorstore.similarity_search(query, k=3)
                        
                        result = chain.invoke({"question": query})
                        
                        if "answer" in result:
                            st.markdown("#### 💡 Insights")
                            st.write(result["answer"])
                            
                            # Calculate and display relevancy score
                            embeddings = OpenAIEmbeddings()
                            relevancy_score = calculate_answer_relevancy(
                                result["answer"],
                                source_docs,
                                embeddings
                            )
                            
                            if relevancy_score is not None:
                                st.session_state.performance_metrics['accuracy_scores'].append(relevancy_score)
                                
                                # Display confidence meter
                                st.markdown(f"""
                                    <div style='margin: 1rem 0;'>
                                        <h4>🎯 Answer Relevancy Score: {relevancy_score}%</h4>
                                        <div style='
                                            background-color: #ddd;
                                            border-radius: 10px;
                                            height: 20px;
                                            width: 100%;
                                        '>
                                            <div style='
                                                background-color: {"#2AAA8A" if relevancy_score >= 70 else "#FFA500" if relevancy_score >= 50 else "#FF4444"};
                                                width: {relevancy_score}%;
                                                height: 100%;
                                                border-radius: 10px;
                                                transition: width 0.5s ease-in-out;
                                            '></div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Add interpretation of the score
                                if relevancy_score >= 70:
                                    st.markdown("✅ High confidence in answer relevancy")
                                elif relevancy_score >= 50:
                                    st.markdown("⚠ Moderate confidence in answer relevancy")
                                else:
                                    st.markdown("❗ Low confidence in answer relevancy")
                        
                        if "sources" in result and result["sources"]:
                            st.markdown("#### 📚 Sources")
                            sources_list = result["sources"].split("\n")
                            for source in sources_list:
                                if source.strip():
                                    st.write(f"- {source}")
                        
                        total_time = time.time() - start_time
                        st.session_state.performance_metrics['response_times'].append(total_time)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.info("Start by processing some articles using the form on the left.")

    # Display performance metrics
    display_performance_metrics()

    # Home button in the sidebar
    if st.sidebar.button("🏠 Back to Home"):
        st.session_state.current_page = 'landing'
        st.session_state.urls_processed = False
        st.rerun()

    # Display processed URLs
    if 'processed_urls' in st.session_state:
        st.sidebar.markdown("### 📑 Processed Articles")
        for idx, url in enumerate(st.session_state.processed_urls, 1):
            st.sidebar.markdown(f"{idx}. {url}")