
import requests
import xml.etree.ElementTree as ET
from collections import Counter

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama


# ==========================================
# LLM CONFIGURATION
# ==========================================

llm = ChatOllama(
    model="llama3",
    temperature=0
)


# ==========================================
# SEARCH RESEARCH PAPERS USING ARXIV
# ==========================================

def search_papers(topic, max_results=10):

    if not topic or not topic.strip():
        raise ValueError("Please enter a research topic.")

    url = "https://export.arxiv.org/api/query"

    params = {
        "search_query": f"all:{topic.strip()}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    root = ET.fromstring(response.content)

    namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    papers = []

    for entry in root.findall("atom:entry", namespace):

        title = entry.findtext(
            "atom:title",
            default="No title",
            namespaces=namespace
        ).strip().replace("\n", " ")

        summary = entry.findtext(
            "atom:summary",
            default="No abstract available",
            namespaces=namespace
        ).strip().replace("\n", " ")

        published = entry.findtext(
            "atom:published",
            default="",
            namespaces=namespace
        )

        year = int(published[:4]) if published else None

        authors = []

        for author in entry.findall(
            "atom:author",
            namespace
        ):

            name = author.findtext(
                "atom:name",
                default="Unknown",
                namespaces=namespace
            )

            authors.append(name)

        paper_url = entry.findtext(
            "atom:id",
            default="",
            namespaces=namespace
        )

        papers.append({
            "title": title,
            "authors": ", ".join(authors),
            "year": year,
            "summary": summary,
            "url": paper_url
        })

    return papers


# ==========================================
# CREATE DOCUMENTS
# ==========================================

def create_documents(papers):

    documents = []

    for paper in papers:

        content = f"""
Title: {paper['title']}

Authors: {paper['authors']}

Year: {paper['year']}

Summary: {paper['summary']}

URL: {paper['url']}
"""

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"] or 0,
                    "url": paper["url"]
                }
            )
        )

    return documents


# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

def create_vector_db(documents):

    if not documents:
        raise ValueError(
            "No documents available to create the vector database."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return db


# ==========================================
# RETRIEVE RELEVANT CONTEXT
# ==========================================

def retrieve_context(db, topic):

    retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(topic)

    if not docs:
        return "No relevant research papers were retrieved."

    return "\n\n".join(
        doc.page_content for doc in docs
    )


# ==========================================
# COMMON LLM HELPER
# ==========================================

def generate_ai_response(prompt):

    response = llm.invoke(prompt)

    return response.content


# ==========================================
# 1. RESEARCH ANALYSIS
# ==========================================

def analyze_research(topic, context):

    prompt = f"""
You are an academic research assistant.

Research Topic:
{topic}

Research Paper Context:
{context}

Analyze the research papers.

Include:

1. Major Findings
2. Methodologies
3. Strengths
4. Weaknesses
5. Applications

Use clear academic language.

Only use information supported by the supplied context.
Mention when information is not available.
Do not invent research results or citations.
"""

    return generate_ai_response(prompt)


# ==========================================
# 2. CITATION EXTRACTION
# ==========================================

def extract_citations(context):

    prompt = f"""
Extract the research paper citations from the context.

For every paper, provide:

1. Paper Title
2. Authors
3. Publication Year
4. Paper URL

Use only the information provided in the context.

Do not invent authors, publication years, or URLs.

Context:
{context}
"""

    return generate_ai_response(prompt)


# ==========================================
# 3. RESEARCH GAP DETECTION
# ==========================================

def detect_gaps(context):

    prompt = f"""
Analyze the supplied research papers.

Identify:

1. Research Gaps
2. Limitations
3. Unsolved Problems
4. Future Research Opportunities

Explain each point clearly.

Distinguish limitations explicitly mentioned in the papers
from possible gaps inferred from the available papers.

Do not claim that a gap is universal based on this
limited collection.

Context:
{context}
"""

    return generate_ai_response(prompt)


# ==========================================
# 4. LITERATURE REVIEW
# ==========================================

def generate_review(topic, context):

    prompt = f"""
Generate an academic literature review.

Research Topic:
{topic}

Research Context:
{context}

Use the following structure:

1. Introduction
2. Related Work
3. Major Findings
4. Research Challenges
5. Future Scope
6. Conclusion

Write in a clear academic style.

Use only the supplied research context.
Do not invent references or findings.
"""

    return generate_ai_response(prompt)


# ==========================================
# 5. TREND ANALYSIS
# ==========================================

def analyze_trends(papers):

    years = []

    for paper in papers:

        year = paper.get("year")

        if year:
            years.append(year)

    return dict(sorted(Counter(years).items()))


# ==========================================
# 6. PAPER SUMMARIZATION
# ==========================================

def summarize_papers(context):

    prompt = f"""
Summarize each research paper separately.

For each paper provide:

1. Paper Title
2. Objective
3. Methodology
4. Key Findings
5. Limitations, if available

Keep each summary separate.

Only use the information in the supplied context.
Do not invent missing details.

Context:
{context}
"""

    return generate_ai_response(prompt)


# ==========================================
# 7. RESEARCH PAPER COMPARISON
# ==========================================

def compare_papers(context):

    prompt = f"""
Compare the research papers provided below.

Include:

1. Research Objectives
2. Methodologies
3. Advantages
4. Limitations
5. Key Findings
6. Similarities
7. Differences

Present a clear comparison.

If information is missing, mention that it is not
available rather than guessing.

Context:
{context}
"""

    return generate_ai_response(prompt)


# ==========================================
# 8. KNOWLEDGE DISCOVERY
# ==========================================

def knowledge_discovery(context):

    prompt = f"""
You are an academic knowledge discovery assistant.

Analyze the research papers.

Discover:

1. Emerging Trends
2. Common Research Patterns
3. Less Explored Areas
4. Future Opportunities
5. Potential Novel Research Ideas

Explain why each insight may be useful.

Clearly distinguish evidence from your suggestions.

Use only the supplied research context as evidence.

Context:
{context}
"""

    return generate_ai_response(prompt)
