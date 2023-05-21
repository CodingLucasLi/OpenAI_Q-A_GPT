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
# 额外安装 pip install faiss-cpu  

st.title('(一)知识库文答机器人🤖')
st.header('👜企业AIGC实践系列')
st.write('❤️🐕CodingLucas的知识分享:Langchan文本向量解析 + Facebook FAISS稠密检索 + OpenAI LLM 问答解析')
st.write('✴️ 支持Word、TXT、PDF等任意文本内容查询')
st.write('✴️ 支持内容查询，信息问答')
st.write('✴️ 员工培训，客服问答，产品要素，数据应用集成')

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

    # 生成 embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=my_key.get_key())
    knowledge_base = FAISS.from_texts(chunks,embeddings)
    
    # 问题web组件
    my_question = st.text_input('输入问题:')
    if my_question:        
        
        with st.spinner('运行中...'):
            docs = knowledge_base.similarity_search(my_question)
            print(str(docs))
            llm = OpenAI(openai_api_key=my_key.get_key(),temperature=0.3)
            chain = load_qa_chain(llm,chain_type="stuff")
            response = chain.run(input_documents = docs, question = my_question)
        
        st.success('完成向量运算')
        st.write('回答:', response)
