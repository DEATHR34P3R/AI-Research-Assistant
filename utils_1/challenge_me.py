from langchain.prompts import PromptTemplate
import streamlit as st

def generate_questions(document_text,num_questions):
    

    from langchain_together import Together

    llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    api_key = st.secrets["together_api_key"]
    

    )


    prompt = PromptTemplate(
        input_variables=["text", "num_questions"],
        template="""
    You are an AI that generates reasoning-based comprehension questions from academic text.

    Your task is to generate exactly {num_questions} numbered questions. These must:
    - Be based on the provided document
    - Require reasoning, inference, or conceptual understanding
    - Avoid simple fact recall or yes/no answers



    TEXT:
    {text}

    QUESTIONS:
    """
    )


    return llm.invoke(prompt.format(text=document_text,num_questions=num_questions))

def evaluate_answer(db, question, user_answer):

    from langchain_together import Together

    llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    api_key = st.secrets["together_api_key"]
    
    )
    context_docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
You are a strict but fair evaluator.

Evaluate the user's answer to the question below using ONLY the provided context.

Context:
{context}

Question:
{question}

User's Answer:
{user_answer}

Respond with:
- Whether the answer is correct or not
- One-line explanation
- A snippet from the context that supports or contradicts the answer
"""

    return llm.invoke(prompt)
