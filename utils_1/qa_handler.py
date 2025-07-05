from langchain_together import Together
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st

def ask_question(db, question):
     
    llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    
    api_key = st.secrets["together_api_key"]
    )

    # Search for relevant chunks from vector store
    relevant_docs = db.similarity_search(question, k=3)

    # Custom prompt 
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use ONLY the following context to answer the question below.
If the answer isn't in the context, say "Not found in document."
Always justify your answer with a snippet from the context.

Context:
{context}

Question: {question}
Answer:
"""
    )

    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
    return chain.run(input_documents=relevant_docs, question=question)
