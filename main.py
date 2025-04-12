import streamlit as st
import sys
import importlib

# Configure page settings
st.set_page_config(
    page_title="Financial Document Analysis RAG System",
    page_icon="💰",
    layout="wide"  # Wider layout for better use of space
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        text-align: center;
    }
    .stMarkdown p {
        text-align: center;
        font-size: 18px;
    }
    .stMarkdown img {
        display: block;
        margin: 0 auto;
    }
    .stMarkdown .highlight {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 8px;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a session state to track the active page
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to navigate to different pages
def navigate_to(page):
    st.session_state.page = page
    # Clear any cached results when switching pages
    if page == 'home':
        for key in list(st.session_state.keys()):
            if key != 'page':
                del st.session_state[key]

# Home Page
def home_page():
    st.title("Welcome to the Financial Document Analysis RAG System 💰")
    
    st.markdown("""
    ### About This Application
    This **financial analysis tool** provides two powerful functionalities to help you analyze and query financial documents effectively:
    
    1. **RAG System** - Query our pre-processed financial knowledge base using advanced Retrieval Augmented Generation (RAG) techniques.
    2. **Document Q&A** - Upload your own financial PDF documents (e.g., 10-K reports, earnings calls, financial statements) and ask questions about them using the same powerful RAG system.
    
    Select an option below to get started:
    """)
    
    # Add a financial-themed image or icon
    col1, col2, col3 = st.columns([1, 2, 1])  # Create 3 columns with the middle column wider
    with col2:
        st.image("FA.jpg", width=800)  # Adjust the width as needed
    
    # Create a 2-column layout for the buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("RAG System", use_container_width=True, key="rag_btn"):
            navigate_to('rag')
            
    with col2:
        if st.button("Document Q&A", use_container_width=True, key="doc_btn"):
            navigate_to('doc_qa')
    
    st.markdown("---")
    st.info("""
    ### Why Use This Tool?
    - **Efficient Financial Document Analysis**: Quickly extract insights from large financial documents like 10-K reports, earnings calls, and financial statements.
    - **Advanced AI-Powered Search**: Combines semantic and keyword search for accurate results tailored to financial data.
    - **User-Friendly Interface**: Simple and intuitive design for financial professionals and analysts.
    - **Customizable Queries**: Ask specific financial questions and get precise answers.
    """)

# Main app logic - determine which page to show
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'rag':
    # Add back button
    if st.button("← Back to Home"):
        navigate_to('home')
    st.markdown("<h1 style='text-align: center;'>RAG System</h1>", unsafe_allow_html=True)
    
    # Dynamically import and run App1.py
    app1 = importlib.import_module("App1")
    # If App1.py has a main function, call it
    if hasattr(app1, 'main'):
        app1.main()
    # Otherwise, it will just execute the file
    
elif st.session_state.page == 'doc_qa':
    # Add back button
    if st.button("← Back to Home"):
        navigate_to('home')
    st.markdown("<h1 style='text-align: center;'>Financial Document Q&A</h1>", unsafe_allow_html=True)
    
    # Dynamically import and run App2.py
    app2 = importlib.import_module("App2")
    # If App2.py has a main function, call it
    if hasattr(app2, 'main'):
        app2.main()