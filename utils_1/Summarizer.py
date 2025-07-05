from langchain.prompts import PromptTemplate
from langchain_together import Together
import streamlit as st

# Secure API key
api_key = st.secrets["together_api_key"]

def chunk_text(text, chunk_size=25000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize_chunk(chunk):
    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        together_api_key=api_key,
        max_tokens=300
    )

    prompt_template = PromptTemplate(
        input_variables=["content"],
        template='''
You are a document summarizer.
Summarize the following content in no more than 100 words.
Use full sentences. Do not add any information not present in the content.

CONTENT:
{content}

SUMMARY:
'''
    )

    prompt = prompt_template.format(content=chunk)
    return llm.invoke(prompt)

def summarize(text):
    chunks = chunk_text(text)

    if len(chunks) == 1:
        return summarize_chunk(chunks[0])
    
    summaries = []
    for i, chunk in enumerate(chunks):
        #st.info(f"Summarizing chunk {i+1} of {len(chunks)}...")
        summary = summarize_chunk(chunk)
        summaries.append(summary)

    combined_summary = " ".join(summaries)

    final_llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        together_api_key=api_key,
        max_tokens=300
    )

    final_prompt = PromptTemplate(
        input_variables=["content"],
        template='''
You are a document summarizer.
The following are summaries of different sections of a document.
Create a final comprehensive summary in no more than 150 words.
Use full sentences. Do not add any information not present in the content.

SECTION SUMMARIES:
{content}

FINAL SUMMARY:
'''
    )

    final_prompt_text = final_prompt.format(content=combined_summary)
    return final_llm.invoke(final_prompt_text)
