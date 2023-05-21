import streamlit as st
import pdfplumber
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback 
import streamlit.components.v1 as components
import my_key
langchain.verbose = False
# Additional install: pip install faiss-cpu  

st.title('(1) Knowledge-Base Q&A Bot🤖')
st.header('👜Enterprise AIGC Practice Series')
st.write('❤️🐕Knowledge Sharing by CodingLucas: Langchain Text Vector Parsing + Facebook FAISS Dense Retrieval + OpenAI LLM Q&A Parsing')
st.write('✴️ Supports querying of Word, TXT, PDF, and other text content')
st.write('✴️ Supports content query, information Q&A')
st.write('✴️ Employee training, customer service Q&A, product features, data application integration')

def read_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    with st.spinner('Reading PDF...'):
        text = read_pdf(uploaded_file)
    st.success('Finished reading PDF.')

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 400,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(text)

    # Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=my_key.get_key())
    knowledge_base = FAISS.from_texts(chunks,embeddings)
    
    # Question web component
    my_question = st.text_input('Enter your question:')
    if my_question:        
        
        with st.spinner('Processing...'):
            docs = knowledge_base.similarity_search(my_question)
            print(str(docs))
            llm = OpenAI(openai_api_key=my_key.get_key(),temperature=0.3)
            chain = load_qa_chain(llm,chain_type="stuff")
            response = chain.run(input_documents = docs, question = my_question)
        
        st.success('Completed vector computation')
        st.write('Answer:', response)
