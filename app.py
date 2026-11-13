
import pandas as pd
import streamlit as st

from AI_research_assistant import (
    search_papers,
    create_documents,
    create_vector_db,
    retrieve_context,
    analyze_research,
    extract_citations,
    detect_gaps,
    generate_review,
    analyze_trends,
    summarize_papers,
    compare_papers,
    knowledge_discovery
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #eef5ff,
            #f8faff,
            #edf9f7
        );
    }

    .main-title {
        color: #173b67;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
    }

    .sub-title {
        color: #52677f;
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }

    .info-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dce6f2;
        margin-bottom: 15px;
    }

    div.stButton > button {
        background-color: #1769aa;
        color: white;
        border-radius: 9px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #104d80;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# INITIALIZE SESSION STATE
# ==========================================

if "report" not in st.session_state:
    st.session_state.report = None

if "papers" not in st.session_state:
    st.session_state.papers = []


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">📚 AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Academic Research Analysis using RAG,
    LangChain, Hugging Face and Ollama
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Project Information")

    st.write("**AI Model:** Llama 3")

    st.write("**Framework:** LangChain")

    st.write("**Embeddings:** Hugging Face")

    st.write("**Vector Database:** ChromaDB")

    st.write("**Paper Source:** arXiv")

    st.divider()

    st.info(
        "Enter a research topic to search papers "
        "and generate an AI-powered research report."
    )

    if st.button("Clear Report"):

        st.session_state.report = None
        st.session_state.papers = []

        st.rerun()


# ==========================================
# TOPIC INPUT
# ==========================================

st.subheader("🔍 Research Topic")

topic = st.text_input(
    "Enter your research topic",
    placeholder="Example: Artificial Intelligence in Healthcare"
)


max_results = st.slider(
    "Number of research papers",
    min_value=1,
    max_value=20,
    value=10
)


# ==========================================
# GENERATE REPORT
# ==========================================

if st.button("🚀 Generate Research Report"):

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        try:

            with st.spinner(
                "Searching research papers from arXiv..."
            ):

                papers = search_papers(
                    topic,
                    max_results=max_results
                )

                if not papers:

                    st.warning(
                        "No research papers found. "
                        "Try another research topic."
                    )

                    st.stop()

                docs = create_documents(papers)

            with st.spinner(
                "Creating vector database and retrieving context..."
            ):

                db = create_vector_db(docs)

                context = retrieve_context(
                    db,
                    topic
                )

            with st.spinner(
                "Generating AI research report..."
            ):

                analysis = analyze_research(
                    topic,
                    context
                )

                review = generate_review(
                    topic,
                    context
                )

                citations = extract_citations(
                    context
                )

                gaps = detect_gaps(
                    context
                )

                trends = analyze_trends(
                    papers
                )

                paper_summaries = summarize_papers(
                    context
                )

                research_comparison = compare_papers(
                    context
                )

                discovery = knowledge_discovery(
                    context
                )

            # Save report in session state

            st.session_state.papers = papers

            st.session_state.report = {
                "topic": topic,
                "analysis": analysis,
                "review": review,
                "citations": citations,
                "gaps": gaps,
                "trends": trends,
                "paper_summaries": paper_summaries,
                "research_comparison": research_comparison,
                "discovery": discovery
            }

            st.success(
                "Research report generated successfully!"
            )

        except Exception as e:

            st.error(
                f"An error occurred: {str(e)}"
            )

            st.info(
                "Check that Ollama is running, the Llama 3 "
                "model is installed, and your internet connection "
                "is working."
            )


# ==========================================
# DISPLAY REPORT
# ==========================================

if st.session_state.report is not None:

    report = st.session_state.report

    st.divider()

    st.header("📊 Research Report")

    st.write(
        f"**Research Topic:** {report['topic']}"
    )

    st.write(
        f"**Papers Found:** {len(st.session_state.papers)}"
    )

    # ======================================
    # DISPLAY PAPER DETAILS
    # ======================================

    with st.expander("📄 View Retrieved Research Papers"):

        for index, paper in enumerate(
            st.session_state.papers,
            start=1
        ):

            st.markdown(
                f"### {index}. {paper['title']}"
            )

            st.write(
                f"**Authors:** {paper['authors']}"
            )

            st.write(
                f"**Year:** {paper['year']}"
            )

            st.write(
                f"**Abstract:** {paper['summary']}"
            )

            st.markdown(
                f"[Read Research Paper]({paper['url']})"
            )

            st.divider()

    # ======================================
    # CREATE EIGHT TABS
    # ======================================

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "📊 Analysis",
            "📖 Literature Review",
            "📚 Citations",
            "🔍 Research Gaps",
            "📈 Trends",
            "📝 Paper Summaries",
            "⚖️ Comparison",
            "💡 Knowledge Discovery"
        ]
    )

    # ======================================
    # TAB 1: RESEARCH ANALYSIS
    # ======================================

    with tab1:

        st.subheader("📊 Research Analysis")

        st.markdown(report["analysis"])

    # ======================================
    # TAB 2: LITERATURE REVIEW
    # ======================================

    with tab2:

        st.subheader("📖 Literature Review")

        st.markdown(report["review"])

    # ======================================
    # TAB 3: CITATIONS
    # ======================================

    with tab3:

        st.subheader("📚 Research Paper Citations")

        st.markdown(report["citations"])

    # ======================================
    # TAB 4: RESEARCH GAPS
    # ======================================

    with tab4:

        st.subheader("🔍 Research Gap Detection")

        st.markdown(report["gaps"])

    # ======================================
    # TAB 5: TREND ANALYSIS
    # ======================================

    with tab5:

        st.subheader("📈 Research Publication Trends")

        trends = report["trends"]

        if trends:

            trend_df = pd.DataFrame(
                list(trends.items()),
                columns=["Year", "Paper Count"]
            )

            trend_df = trend_df.sort_values("Year")

            st.line_chart(
                trend_df.set_index("Year")
            )

            st.dataframe(
                trend_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Publication year information is unavailable."
            )

    # ======================================
    # TAB 6: PAPER SUMMARIZATION
    # ======================================

    with tab6:

        st.subheader("📝 Individual Paper Summaries")

        st.markdown(
            report["paper_summaries"]
        )

    # ======================================
    # TAB 7: RESEARCH COMPARISON
    # ======================================

    with tab7:

        st.subheader("⚖️ Research Paper Comparison")

        st.markdown(
            report["research_comparison"]
        )

    # ======================================
    # TAB 8: KNOWLEDGE DISCOVERY
    # ======================================

    with tab8:

        st.subheader("💡 Knowledge Discovery")

        st.markdown(
            report["discovery"]
        )

    # ======================================
    # DOWNLOAD REPORT
    # ======================================

    st.divider()

    st.subheader("📥 Download Research Report")

    report_text = f"""
AI RESEARCH ASSISTANT
=====================

Research Topic:
{report['topic']}

RESEARCH ANALYSIS
-----------------
{report['analysis']}

LITERATURE REVIEW
-----------------
{report['review']}

CITATIONS
---------
{report['citations']}

RESEARCH GAPS
-------------
{report['gaps']}

TREND ANALYSIS
--------------
{report['trends']}

PAPER SUMMARIES
---------------
{report['paper_summaries']}

RESEARCH COMPARISON
-------------------
{report['research_comparison']}

KNOWLEDGE DISCOVERY
-------------------
{report['discovery']}
"""

    st.download_button(
        label="⬇️ Download Complete Report",
        data=report_text,
        file_name="AI_Research_Report.txt",
        mime="text/plain"
    )
