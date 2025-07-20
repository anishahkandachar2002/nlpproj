# streamlit_app.py

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# --- Admission Contexts ---
btech_context = """For Bachelor of Technology (B.Tech), admissions can be through:
- KCET entrance exam and counselling,
- JEE Main or KCET scores via PES University counselling,
- Higher Fee Category (Management Quota) by contacting the admissions office.
Eligibility requires:
- Minimum 60% aggregate in Physics, Chemistry, and Mathematics (PCM) in 10+2 or equivalent,
- A valid JEE Main or KCET score.
JEE Main Quota Fee Structure (per annum):
- Computer Science and Engineering (CSE): INR 5,00,000
- CSE (Artificial Intelligence & Machine Learning – AIML): INR 5,00,000
- Electronics and Communication Engineering (ECE): INR 4,50,000
- Mechanical Engineering (ME): INR 4,00,000
- Electrical and Electronics Engineering (EEE): INR 4,00,000
- Biotechnology (BT): INR 4,00,000
For Higher Fee Category / Management / NRI / Foreign / Sponsored / OCI / PIO seats:
- Please contact the Office of Admissions or write to admissions@pes.edu
Other Charges:
- One-time non-refundable administrative fee: INR 10,000
"""

btech_lateral_context = """For B.Tech Diploma Lateral Entry, admission is possible through DCET entrance
and counselling or PES University Entrance Test. Eligibility requires 10+3 Diploma with minimum 60%
from a DTE-affiliated institution."""

bca_context = """For Bachelor of Computer Applications (BCA), admission is through registration, PES Entrance
Test (online interview) with minimum 50% aggregate in 10+2 and Mathematics as a compulsory subject.
Registration fee is typically INR 1,200. Counseling, verification, and fee payment complete the admission process."""

mtech_context = """Master of Technology (M.Tech) Admissions
Admission Pathways:
- Direct Application with PES Entrance Test (online interview)
- PGCET Quota
- Valid GATE Score
Eligibility: Minimum 50% in relevant engineering degree from a UGC-recognized university."""

# --- Section Mapping ---
section_map = {
    "1. B.Tech Regular": btech_context,
    "2. B.Tech Lateral": btech_lateral_context,
    "3. BCA": bca_context,
    "4. M.Tech": mtech_context
}

docs = [
    Document(page_content=content, metadata={"section": section})
    for section, content in section_map.items()
]

# --- Embeddings & QA Model ---
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_name_qa = "deepset/roberta-base-squad2"
tokenizer_qa = AutoTokenizer.from_pretrained(model_name_qa)
model_qa = AutoModelForQuestionAnswering.from_pretrained(model_name_qa)

device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline("question-answering", model=model_qa, tokenizer=tokenizer_qa, device=device)

# --- Streamlit UI ---
st.set_page_config(page_title="PES Admission Q&A Bot", layout="centered")
st.title("📘 PES University Admission Q&A")
st.markdown("Ask admission-related questions based on course-specific details.")

section_choice = st.selectbox("Select a Course Section", list(section_map.keys()))
question = st.text_input("Your Question", placeholder="e.g., What is the fee for CSE?")

if st.button("Get Answer"):
    if not section_choice or not question:
        st.warning("Please select a section and enter your question.")
    else:
        relevant_docs = [doc for doc in docs if doc.metadata['section'] == section_choice]
        try:
            vectorstore = FAISS.from_documents(relevant_docs, embedder)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
            retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                st.error("No relevant information found.")
            else:
                context = retrieved_docs[0].page_content
                result = qa_pipeline(question=question, context=context)
                answer = result.get('answer', 'No answer found.')
                if not answer.strip() or answer.lower() in ["[cls]", "[sep]", "i don't know."]:
                    st.warning("Could not extract a specific answer. Try rephrasing.")
                else:
                    st.success("✅ Answer:")
                    st.write(answer.strip())
        except Exception as e:
            st.error(f"An error occurred: {e}")
