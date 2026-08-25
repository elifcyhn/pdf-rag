import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
def get_pdf_text(docs):
    text= ""
    for pdf in docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter= CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks= text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDF's")

    st.header("Chat with multiple PDF's")
    st.text_input("Ask a question about your PDF's")

    with st.sidebar:
        st.subheader("Your documents")
        docs=st.file_uploader("Upload your PDF's and click the button below", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(docs)

                text_chunks= get_text_chunks(raw_text)

                vectorstore= get_vectorstore(text_chunks)


if __name__ == "__main__":
    main()