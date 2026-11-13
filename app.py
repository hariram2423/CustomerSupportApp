
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


# =====================================================
# 1. CONFIGURATION
# =====================================================

load_dotenv()

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "NovaCare_Customer_Support_Knowledge_Base.pdf"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")


# =====================================================
# 2. STREAMLIT PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="NovaCare Customer Support AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 NovaCare Multi-Agent Customer Support System")

st.write(
    "A customer support application powered by CrewAI, "
    "RAG, FAISS and web research."
)


# =====================================================
# 3. CHECK API KEYS AND PDF
# =====================================================

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing in your .env file.")
    st.stop()

if not SERPER_API_KEY:
    st.error("SERPER_API_KEY is missing in your .env file.")
    st.stop()

if not PDF_PATH.exists():
    st.error(f"Knowledge base PDF not found: {PDF_PATH.name}")
    st.stop()


# =====================================================
# 4. LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


# =====================================================
# 5. READ PDF AND CREATE TEXT CHUNKS
# =====================================================

def create_chunks(text, chunk_size=1000, overlap=150):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =====================================================
# 6. BUILD FAISS VECTOR INDEX
# =====================================================

@st.cache_resource
def build_pdf_index(pdf_path, modified_time):

    reader = PdfReader(pdf_path)

    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if not page_text:
            continue

        chunks = create_chunks(page_text)

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "page": page_number
            })

    if not all_chunks:
        raise ValueError(
            "No readable text found in the PDF."
        )

    embedding_model = load_embedding_model()

    texts = [
        item["text"] for item in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, all_chunks, embedding_model


# =====================================================
# 7. RETRIEVE RELEVANT PDF INFORMATION
# =====================================================

def retrieve_pdf_context(
    question,
    index,
    chunks,
    embedding_model,
    top_k=5
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    number_of_results = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        number_of_results
    )

    retrieved_context = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        chunk = chunks[idx]

        retrieved_context.append(
            f"""
            PDF Page: {chunk['page']}
            Similarity Score: {score:.3f}

            Information:
            {chunk['text']}
            """
        )

    return "\n\n".join(retrieved_context)


# =====================================================
# 8. INITIALIZE LLM AND WEB SEARCH TOOL
# =====================================================

llm = LLM(
    model=f"openai/{MODEL_NAME}",
    api_key=OPENAI_API_KEY,
    temperature=0.2
)

web_search_tool = SerperDevTool()


# =====================================================
# 9. CREATE AGENT 1 - KNOWLEDGE SUPPORT AGENT
# =====================================================

knowledge_agent = Agent(

    role="Customer Support Knowledge Specialist",

    goal=(
        "Answer customer questions accurately using "
        "the company knowledge base and general knowledge."
    ),

    backstory="""
    You are an experienced customer support specialist.

    You receive relevant information retrieved from the
    company's PDF knowledge base.

    Your responsibilities:

    1. Prioritize company PDF information for company policies.
    2. Never invent company policies.
    3. If the PDF does not contain the answer, clearly say so.
    4. You may provide general technical guidance separately.
    5. Distinguish company policy from general guidance.
    6. Give clear and practical answers.
    """,

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# =====================================================
# 10. CREATE AGENT 2 - WEB RESEARCH AGENT
# =====================================================

web_agent = Agent(

    role="Web Research Specialist",

    goal=(
        "Research customer questions using reliable "
        "online sources and provide evidence-based answers."
    ),

    backstory="""
    You are a professional web research specialist.

    Your responsibilities:

    1. Search the internet for relevant information.
    2. Prefer official documentation and trusted sources.
    3. Verify information before presenting it.
    4. Include source URLs in your answer.
    5. Clearly explain when information cannot be verified.
    6. Never invent URLs or research findings.
    """,

    tools=[web_search_tool],

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# =====================================================
# 11. CREATE AGENT 3 - REPORT ANALYST
# =====================================================

report_agent = Agent(

    role="Customer Support Report Analyst",

    goal=(
        "Compare the knowledge base answer and web research "
        "answer and prepare a useful customer support report."
    ),

    backstory="""
    You are an experienced customer support quality analyst.

    You receive two answers:

    1. Internal company knowledge base answer.
    2. External web research answer.

    Your responsibilities:

    1. Compare both answers.
    2. Identify agreements and differences.
    3. Identify missing information.
    4. Highlight any conflict between company policy
       and external information.
    5. Recommend a safe response for the customer.
    6. Never treat external web information as company policy.
    """,

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# =====================================================
# 12. STREAMLIT CUSTOMER QUERY INPUT
# =====================================================

st.divider()

st.subheader("Customer Query")

customer_query = st.text_area(
    "Enter the customer's problem:",
    placeholder=(
        "Example: I requested a refund 3 days ago, "
        "but I have not received my money. What should I do?"
    ),
    height=120
)

run_button = st.button(
    "🚀 Run Multi-Agent Support System",
    type="primary"
)


# =====================================================
# 13. RUN THE COMPLETE MULTI-AGENT WORKFLOW
# =====================================================

if run_button:

    if not customer_query.strip():

        st.warning("Please enter a customer query.")

    else:

        try:

            with st.spinner(
                "Loading PDF and building the RAG knowledge index..."
            ):

                index, chunks, embedding_model = build_pdf_index(
                    str(PDF_PATH),
                    PDF_PATH.stat().st_mtime
                )

                pdf_context = retrieve_pdf_context(
                    customer_query,
                    index,
                    chunks,
                    embedding_model
                )

            st.success("PDF knowledge retrieval completed.")

            with st.expander("View retrieved PDF information"):

                st.write(pdf_context)

            # -----------------------------------------
            # TASK 1 - KNOWLEDGE BASE ANSWER
            # -----------------------------------------

            knowledge_task = Task(

               description="""
Customer Question:
{customer_query}

Retrieved PDF Knowledge:
{pdf_context}

Answer the customer's question using the retrieved PDF information.

Instructions:
1. Prioritize company policy from the PDF.
2. Do not invent company policies.
3. If information is missing, mention it.
4. Use simple English.
5. Answer in 3-5 bullet points.
6. Keep the answer within 80 words.
""",

expected_output="""
A short customer support answer in simple English,
with 3-5 bullet points and a maximum of 80 words.
""",

                agent=knowledge_agent
            )

            # -----------------------------------------
            # TASK 2 - WEB RESEARCH
            # -----------------------------------------

            web_task = Task(

                description="""
Research the following customer question using web search.

Customer Question:
{customer_query}

Instructions:
1. Search reliable and relevant websites.
2. Prefer official sources.
3. Give only the 3 most important findings.
4. Use simple English.
5. Keep the answer within 80 words, excluding URLs.
6. Include a maximum of 3 relevant source URLs.
7. Never invent information or URLs.
""",

expected_output="""
A short web research answer with up to 3 findings
and relevant source URLs.
""",

                agent=web_agent
            )

            # -----------------------------------------
            # TASK 3 - COMPARE AND GENERATE REPORT
            # -----------------------------------------

            report_task = Task(

               description="""
Create a short customer support record.

Customer Query:
{customer_query}

Use the outputs from Agent 1 and Agent 2.

Include:
1. Customer Query
2. Agent 1 Answer
3. Agent 2 Answer

Instructions:
- Keep both original answers unchanged.
- Do not add unnecessary explanations.
- Do not repeat the answers.
- Keep the format clean and readable.
""",

expected_output="""
A short text record containing the customer query,
Agent 1 answer and Agent 2 answer.
""",

                agent=report_agent,

                context=[
                    knowledge_task,
                    web_task
                ]
            )

            # -----------------------------------------
            # CREATE CREW
            # -----------------------------------------

            support_crew = Crew(

                agents=[
                    knowledge_agent,
                    web_agent,
                    report_agent
                ],

                tasks=[
                    knowledge_task,
                    web_task,
                    report_task
                ],

                process=Process.sequential,

                verbose=True
            )

            # -----------------------------------------
            # EXECUTE ALL THREE AGENTS
            # -----------------------------------------

            with st.spinner(
                "Agents are working. Please wait..."
            ):

                crew_result = support_crew.kickoff(

                    inputs={
                        "customer_query": customer_query,
                        "pdf_context": pdf_context
                    }

                )

            # -----------------------------------------
            # GET INDIVIDUAL AGENT OUTPUTS
            # -----------------------------------------

            knowledge_answer = knowledge_task.output.raw

            web_answer = web_task.output.raw

            final_report = report_task.output.raw

            # -----------------------------------------
            # SAVE REPORT
            # -----------------------------------------

            report_content = f"""
NOVA CARE CUSTOMER SUPPORT REPORT

Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

CUSTOMER QUERY:
{customer_query}


AGENT 1 - KNOWLEDGE BASE ANSWER:
{knowledge_answer}


AGENT 2 - WEB RESEARCH:
{web_answer}


AGENT 3 - FINAL SUPPORT REPORT:
{final_report}
"""

            report_path = (
                OUTPUT_DIR / "support_report.txt"
            )

            report_path.write_text(
                report_content,
                encoding="utf-8"
            )

            # -----------------------------------------
            # DISPLAY RESULTS
            # -----------------------------------------

            st.success(
                "All three agents completed successfully!"
            )

            st.divider()

            st.subheader(
                "Agent 1 - Knowledge Base Answer"
            )

            st.markdown(knowledge_answer)

            st.divider()

            st.subheader(
                "Agent 2 - Web Research"
            )

            st.markdown(web_answer)

            st.divider()

            st.subheader(
                "Agent 3 - Final Support Report"
            )

            st.markdown(final_report)

            # -----------------------------------------
            # DOWNLOAD REPORT
            # -----------------------------------------

            st.divider()

            st.download_button(

                label="📥 Download Support Report",

                data=report_content,

                file_name="support_report.txt",

                mime="text/plain"

            )

            st.caption(
                f"Report saved locally at: {report_path}"
            )

        except Exception as error:

            st.error(
                f"An error occurred: {str(error)}"
            )

            st.exception(error)
