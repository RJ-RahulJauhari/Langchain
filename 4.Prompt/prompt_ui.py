# Dynamic Prompt Generation UI using Streamlit

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

model = ChatOllama(model="qwen3-coder:30b", temperature=1.5)

st.header("Reasearch Tool using Dynamic Prompts")

paper_input = st.selectbox(
    "Select a research paper",
    ("AI in Healthcare", "Machine Learning Basics", "Natural Language Processing", "Computer Vision")
)

style_input = st.selectbox(
    "Select a writing style",
    ("Formal", "Informal", "Technical", "Conversational")
)


length_input = st.slider("Select the length of the summary", 50, 500, 150, step=25)

if st.button("Generate Summary"):
    prompt = PromptTemplate(
        input_variables=["paper", "style", "length"],
        template="Summarize the research paper on '{paper}' in a '{style}' style within {length} words."
    )
    
    chain = prompt | model
    # You provide the inputs for the chain here, as the first step is a prompt template
    # you will need to provide the inputs for that prompt template
    # The model will then be invoked with the formatted prompt
    response = chain.invoke({"paper": paper_input, "style": style_input, "length": length_input})
    
    st.subheader("Summary:")
    st.write(response.content)