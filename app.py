"""
app.py
Intelligent Resume-to-Job Matching & Explainable Recommendation System.

Core Design Philosophy:
- Direct, focused user journey: Upload -> Profile -> Ranked Matches -> Skill Gap -> AI Assistant
- Deterministic, mathematically grounded recommendations
- Fully explainable: every match has exact feature attributions and matched/missing skills
- High-performance caching for low-latency interactive evaluation
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple

# Ensure repository root and src directory are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_CURRENT_DIR, "src")
for _p in [_SRC_DIR, _CURRENT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analyzer import (
    extract_text_from_pdf,
    load_jobs,
    get_job_recommendations,
    classify_resume,
    get_skill_gap,
    evaluate_classifier,
    chat_job_search
)
from nlp.resume_parser import parse_resume, ResumeProfile
from explainability.recommendation_explainer import generate_match_breakdown, explain_recommendation
from explainability.llm_assistant import get_groq_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── PAGE CONFIGURATION ───────────────────────────────────────────

st.set_page_config(
    page_title="Smart Resume Analyzer & Job Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STREAMLINED PREMIUM CSS ───────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }

    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #334155 100%);
        border-radius: 14px;
        padding: 32px 28px;
        color: #ffffff;
        margin-bottom: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .hero-title {
        color: #ffffff !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        margin: 0 0 8px 0 !important;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin: 0;
        line-height: 1.5;
    }
    .hero-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #c7d2fe;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
        border: 1px solid rgba(165, 180, 252, 0.3);
    }

    /* Content Cards */
    .dashboard-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    /* Stat Cards */
    .stat-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .stat-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 20px;
        font-weight: 800;
        color: #0f172a;
    }

    /* Skill Badges */
    .skill-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
        margin: 3px 4px 3px 0;
    }
    .pill-green {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .pill-amber {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .pill-neutral {
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }

    /* Match Score Badges */
    .score-badge-high {
        background: #dcfce7;
        color: #166534;
        font-weight: 800;
        font-size: 14px;
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid #86efac;
    }
    .score-badge-med {
        background: #e0e7ff;
        color: #3730a3;
        font-weight: 800;
        font-size: 14px;
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid #a5b4fc;
    }
    .score-badge-low {
        background: #fef3c7;
        color: #92400e;
        font-weight: 800;
        font-size: 14px;
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid #fcd34d;
    }

    /* Job Card */
    .job-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .job-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .job-title {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .job-meta {
        font-size: 12px;
        color: #64748b;
        margin-top: 2px;
        margin-bottom: 12px;
    }

    /* Attribution Bar */
    .attr-container {
        display: flex;
        height: 6px;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0 12px 0;
        background: #e2e8f0;
    }
    .attr-segment {
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ─── CACHED RESOURCE LOADERS ──────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_cached_job_data(sample_size: int = 500) -> pd.DataFrame:
    """Loads and caches job postings dataframe."""
    return load_jobs(sample_size=sample_size)


# ─── PRESET INTERVIEW DEMO RESUMES ────────────────────────────────

PRESET_PROFILES = {
    "None (Use Uploaded / Pasted Text)": "",
    "Machine Learning Engineer (Senior)": """
Alex Rivera
Senior Machine Learning Engineer | San Francisco, CA | alex.rivera@example.com
Summary:
Machine Learning Engineer with 6+ years of experience designing and deploying end-to-end AI systems.
Deep expertise in Python, PyTorch, TensorFlow, Scikit-learn, SQL, Docker, and Kubernetes on AWS.
Built scalable real-time recommendation systems, NLP transformers, and computer vision models.
Experience:
Senior ML Engineer at TechCorp (2020 - Present):
- Architected candidate recommendation ranking engine using Sentence-Transformers and FAISS vector index.
- Productionized ML pipelines with Docker, Kubernetes, and CI/CD GitHub Actions.
- Improved retrieval NDCG@10 by 24% using hybrid lexical and semantic embedding search.
Education:
Master of Science in Computer Science, Stanford University
Skills:
Python, PyTorch, TensorFlow, Scikit-learn, Pandas, NumPy, SQL, AWS, Docker, Kubernetes, NLP, Git, CI/CD
Certifications:
AWS Certified Machine Learning - Specialty, Deep Learning Specialization
Projects:
- Neural Search Engine: Hybrid dense-sparse information retrieval pipeline using MiniLM and BM25.
- Real-time Fraud Detection: Gradient boosting model handling 10k QPS with sub-15ms latency.
""",
    "Financial Analyst": """
Morgan Chase
Senior Financial Analyst | New York, NY | morgan.chase@example.com
Summary:
Quantitative Financial Analyst with 5 years of experience in financial modeling, valuation, and forecasting.
Proficient in Financial Analysis, Accounting, Budgeting, Excel, Power BI, Tableau, and Risk Management.
Experience:
Financial Analyst at Global Investment Partners (2021 - Present):
- Constructed multi-scenario discounted cash flow (DCF) models and quarterly variance forecasts.
- Designed executive dashboard suites in Power BI and Tableau automating compliance reporting.
Education:
Bachelor of Science in Finance and Accounting, NYU Stern
Skills:
Financial Analysis, Budgeting, Forecasting, Accounting, Excel, Power BI, Tableau, Risk Management, Compliance
Certifications:
CFA Level II Candidate, Financial Modeling and Valuation Analyst (FMVA)
Projects:
- Enterprise Capital Allocation Model: Dynamic forecasting engine for $500M portfolio planning.
""",
    "HR Business Partner & Talent Acquisition": """
Samantha Brooks
Human Resources Generalist & Talent Acquisition Lead | Chicago, IL
Summary:
HR Professional with 6 years of expertise in talent acquisition, recruitment, and employee relations.
Skilled in Performance Management, Compensation and Benefits, Onboarding, HRIS, Payroll, and Labor Laws.
Experience:
Lead Recruiter & HRBP at Horizon Health (2020 - Present):
- Managed full-lifecycle recruiting across technical and clinical functional departments.
- Overhauled onboarding workflow reducing 90-day voluntary turnover by 18%.
Education:
Bachelor's Degree in Human Resources Management, University of Illinois
Skills:
Talent Acquisition, Recruitment, Employee Relations, Performance Management, Onboarding, HRIS, Payroll, Labor Laws, Workforce Planning
Certifications:
SHRM-CP (Society for Human Resource Management Certified Professional)
Projects:
- DEI Recruitment Initiative: Implemented structured competency interview rubrics across 12 teams.
"""
}


# ─── SIDEBAR CONFIGURATION ────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # Fast interview demo profile selector
    selected_preset = st.selectbox(
        "⚡ 1-Click Demo Profiles:",
        options=list(PRESET_PROFILES.keys()),
        index=0,
        help="Instantly load pre-built resumes for quick testing without uploading files."
    )

    st.markdown("---")
    st.markdown("#### 🎯 Retrieval & Ranking Weights")
    
    alpha_val = st.slider(
        "Retrieval Balance (α):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="α = 1.0 is Pure Lexical (TF-IDF); α = 0.0 is Pure Semantic (Sentence-Transformers). 0.5 is Balanced Hybrid."
    )
    st.caption(f"**Current:** {int(alpha_val * 100)}% TF-IDF / {int((1 - alpha_val) * 100)}% Embeddings")

    top_k_val = st.slider("Top Recommendations:", min_value=3, max_value=20, value=5, step=1)
    
    st.markdown("---")
    st.markdown("#### 📂 Corpus Size")
    sample_size_choice = st.selectbox(
        "Working Jobs Corpus:",
        options=[500, 1000, 5000, 23425],
        index=0,
        help="500 jobs provides sub-50ms CPU execution. Full dataset has 23,425 postings."
    )

    # Groq API Key Check
    groq_key = get_groq_api_key()
    st.markdown("---")
    if groq_key and groq_key != "your_groq_api_key_here":
        st.success("🟢 Groq LLaMA Connected")
    else:
        st.info("⚪ LLaMA Assistant: Offline Mode\n(ML ranking is 100% operational)")


# ─── HERO HEADER ──────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <div class="hero-tag">Explainable Two-Stage RecSys • NLP Pipeline</div>
    <h1 class="hero-title">Smart Resume Analyzer & Job Recommendation System</h1>
    <p class="hero-subtitle">
        Transparent two-stage matching: Lexical TF-IDF + Dense Sentence-Transformers (MiniLM-L6) 
        retrieval combined with multi-factor ranking and grounded skill-gap explainability.
    </p>
</div>
""", unsafe_allow_html=True)


# ─── STEP 1: RESUME INGESTION ─────────────────────────────────────

st.markdown("### 📄 1. Candidate Resume Ingestion")

resume_text = ""
uploaded_file = st.file_uploader(
    "Upload Candidate Resume (PDF format):",
    type=["pdf"],
    help="Upload any standard resume PDF. Text is extracted with pdfplumber."
)

if uploaded_file is not None:
    try:
        # Check size (limit to 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("Uploaded file exceeds 10MB limit. Please upload a standard resume.")
        else:
            with st.spinner("Extracting text from PDF..."):
                extracted = extract_text_from_pdf(uploaded_file)
                if not extracted or len(extracted.strip()) < 40:
                    st.warning(
                        "⚠️ **No selectable text detected in this PDF.**\n\n"
                        "This file appears to be a scanned image or photo. Please provide a text-based PDF "
                        "or paste your resume text directly below."
                    )
                else:
                    resume_text = extracted
    except Exception as e:
        st.error(f"Error reading PDF: {e}. You can paste your resume text below instead.")

# If demo profile selected, use that
if PRESET_PROFILES[selected_preset]:
    resume_text = PRESET_PROFILES[selected_preset].strip()

# Direct text paste fallback
with st.expander("✏️ Or Paste / Edit Resume Text Manually", expanded=not bool(resume_text)):
    resume_text_input = st.text_area(
        "Resume Content:",
        value=resume_text,
        height=180,
        placeholder="Paste plain text resume here (e.g. skills, experience, education)..."
    )
    if resume_text_input.strip():
        resume_text = resume_text_input.strip()

if not resume_text:
    st.info("👆 Upload a resume PDF or select a demo profile from the left sidebar to start analysis.")
    st.stop()


# ─── STEP 2: PROFILE PARSING & EXTRACTION ─────────────────────────

t_start = time.time()
with st.spinner("Parsing resume profile and identifying skills..."):
    profile: ResumeProfile = parse_resume(resume_text)
    jobs_df = get_cached_job_data(sample_size=sample_size_choice)
    
    # Role Classification
    predicted_category, category_probs = classify_resume(resume_text, jobs_df)
t_parse_ms = int((time.time() - t_start) * 1000)

st.markdown("### 🔍 2. Candidate Profile Summary")

# Metric KPI Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    exp_disp = f"{profile.years_of_experience:.1f} Yrs" if profile.years_of_experience is not None else "Unspecified"
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Estimated Experience</div>
        <div class="stat-value">{exp_disp}</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    edu_disp = profile.education[0] if profile.education else "Degree Unstated"
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Education Detected</div>
        <div class="stat-value">{edu_disp}</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    n_skills = len(profile.skills_normalized)
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Skills Extracted</div>
        <div class="stat-value">{n_skills}</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    top_prob = category_probs.get(predicted_category, 0.0)
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Predicted Domain</div>
        <div class="stat-value" style="color:#4f46e5;">{predicted_category} <span style="font-size:12px; font-weight:600;">({top_prob}%)</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Skills Categorized View
with st.expander("🏷️ Extracted Skills by Technical Domain", expanded=True):
    if profile.skills_by_category:
        for cat, s_list in profile.skills_by_category.items():
            pills = "".join([f'<span class="skill-pill pill-neutral">{s}</span>' for s in s_list])
            st.markdown(f"**{cat}:** {pills}", unsafe_allow_html=True)
    else:
        st.warning("No explicit dictionary skills detected. Matching will rely on dense semantic embeddings.")

    if profile.certifications:
        st.markdown(f"**📜 Certifications:** {', '.join(profile.certifications)}")
    if profile.projects:
        st.markdown(f"**🚀 Projects Identified:** {', '.join(profile.projects[:3])}")


# ─── STEP 3: HYBRID RECOMMENDATIONS & MULTI-FACTOR RANKING ────────

st.markdown("### 🎯 3. Top Job Recommendations & Explanations")

t_rec_start = time.time()
with st.spinner("Computing hybrid retrieval and candidate re-ranking..."):
    ranked_jobs_df, _, _ = get_job_recommendations(
        resume_text=resume_text,
        jobs_df=jobs_df,
        top_k=top_k_val,
        alpha=alpha_val,
        use_hybrid=True,
        resume_profile=profile
    )
t_rec_ms = int((time.time() - t_rec_start) * 1000)

st.caption(f"⚡ *Pipeline latency: Profile parsing {t_parse_ms}ms • Retrieval & Ranking {t_rec_ms}ms over {len(jobs_df)} postings*")

if ranked_jobs_df.empty:
    st.warning("No candidate jobs found matching the search criteria.")
    st.stop()

# Display each recommended job card
for rank_idx, (_, row) in enumerate(ranked_jobs_df.iterrows(), start=1):
    job_title = str(row.get('job_title', 'Role Title Unavailable'))
    job_cat = str(row.get('category', 'General'))
    score_pct = float(row.get('ranking_score_pct', 0.0))
    desc = str(row.get('job_description', ''))
    
    breakdown = row.get('match_breakdown', {})
    if not isinstance(breakdown, dict):
        breakdown = generate_match_breakdown(row)
        
    matched = breakdown.get('matched_skills', [])
    missing = breakdown.get('missing_skills', [])
    strong_factors = breakdown.get('strong_factors', [])
    gap_factors = breakdown.get('gap_factors', [])
    explanation_text = str(row.get('explanation', ''))

    # Score Badge Color
    if score_pct >= 70.0:
        badge_html = f'<span class="score-badge-high">{score_pct}% Match</span>'
    elif score_pct >= 50.0:
        badge_html = f'<span class="score-badge-med">{score_pct}% Match</span>'
    else:
        badge_html = f'<span class="score-badge-low">{score_pct}% Match</span>'

    with st.container():
        st.markdown(f"""
        <div class="job-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div>
                    <h3 class="job-title">#{rank_idx}. {job_title}</h3>
                    <div class="job-meta">Domain: <strong>{job_cat}</strong> • Job ID: {row.get('job_id', rank_idx)}</div>
                </div>
                <div>{badge_html}</div>
            </div>
        """, unsafe_allow_html=True)

        # Matched and Missing Pills
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**✅ Matched Skills:**")
            if matched:
                matched_pills = "".join([f'<span class="skill-pill pill-green">✓ {s}</span>' for s in matched])
                st.markdown(matched_pills, unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size:12px; color:#64748b;'>No exact skill overlap</span>", unsafe_allow_html=True)

        with col_m2:
            st.markdown("**⚠️ Missing Job Skills:**")
            if missing:
                missing_pills = "".join([f'<span class="skill-pill pill-amber">! {s}</span>' for s in missing])
                st.markdown(missing_pills, unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size:12px; color:#15803d;'>No missing technical requirements detected</span>", unsafe_allow_html=True)

        # Explainability Synthesis
        st.markdown(f"""
        <div style="background:#f8fafc; border-left:3px solid #6366f1; padding:8px 12px; margin:12px 0 6px 0; border-radius:0 6px 6px 0;">
            <span style="font-size:13px; color:#1e293b;"><strong>Why this match:</strong> {explanation_text}</span>
        </div>
        """, unsafe_allow_html=True)

        # Expandable Job Details & Mathematical Breakdown
        with st.expander(f"📊 Detailed Match Scoring Breakdown & Job Description"):
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Skill Match", f"{breakdown.get('skill_match_pct', 0)}%")
            b2.metric("Semantic Fit", f"{breakdown.get('semantic_similarity_pct', 0)}%")
            b3.metric("Lexical Fit", f"{breakdown.get('lexical_similarity_pct', 0)}%")
            b4.metric("Experience", f"{breakdown.get('experience_match_pct', 0)}%")
            b5.metric("Education", f"{breakdown.get('education_match_pct', 0)}%")

            st.markdown("**Primary Match Drivers:**")
            for sf in strong_factors:
                st.markdown(f"- 🟢 {sf}")
            for gf in gap_factors:
                st.markdown(f"- 🟡 {gf}")

            st.markdown("**Full Job Description:**")
            st.text_area("Description Text", value=desc, height=120, disabled=True, key=f"desc_{rank_idx}")

        st.markdown("</div>", unsafe_allow_html=True)


# ─── STEP 4: INTERACTIVE SKILL-GAP ANALYSIS ───────────────────────

st.markdown("### ⚡ 4. Deep-Dive Skill Gap Analysis")

with st.container():
    job_titles_list = [
        f"#{i} {r.get('job_title', 'Role')} ({r.get('ranking_score_pct', 0)}% Match)"
        for i, (_, r) in enumerate(ranked_jobs_df.iterrows(), start=1)
    ]
    selected_job_idx = st.selectbox(
        "Select target job to analyze skill alignment:",
        options=range(len(job_titles_list)),
        format_func=lambda i: job_titles_list[i]
    )
    
    target_job = ranked_jobs_df.iloc[selected_job_idx]
    target_skill_str = str(target_job.get('job_skill_set', '[]'))
    matched_skills, missing_skills = get_skill_gap(profile.skills_raw, target_skill_str)
    
    col_gap1, col_gap2 = st.columns([1, 1])
    with col_gap1:
        st.markdown("#### ✅ Skills You Have For This Role")
        if matched_skills:
            for s in matched_skills:
                st.markdown(f"- **{s.title()}** (Verified on resume)")
        else:
            st.info("No direct skill overlap found for this posting.")

    with col_gap2:
        st.markdown("#### 🚀 Recommended Skills To Acquire")
        if missing_skills:
            for s in missing_skills:
                st.markdown(f"- 🎯 **{s.title()}** (Required by job posting)")
        else:
            st.success("You possess 100% of the explicitly listed skills for this role!")


# ─── STEP 5: GROUNDED AI CAREER ASSISTANT ──────────────────────────

st.markdown("### 🤖 5. Grounded AI Career Assistant")
st.caption("Answers career questions strictly grounded in the calculated recommendations and candidate profile.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question (e.g. 'How can I bridge the skill gap for Job #1?' or 'Why was Job #2 recommended?')")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing recommendation context..."):
            reply = chat_job_search(
                messages=st.session_state.messages,
                resume_profile=profile,
                top_jobs=ranked_jobs_df
            )
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})


# ─── STEP 6: TECHNICAL BENCHMARKS & EVALUATION (INTERVIEW TAB) ────

with st.expander("🔬 Model Benchmarks & IR Evaluation (Technical Deep-Dive for Interviewers)"):
    st.markdown("""
    #### Information Retrieval (IR) Offline Evaluation
    Evaluating the 4 recommendation paradigms across standard ranking metrics:
    - **Precision@5 (P@5):** Proportion of top-5 recommendations matching target domain and skill criteria.
    - **MRR (Mean Reciprocal Rank):** Reciprocal rank of the first relevant recommendation.
    - **NDCG@10:** Normalized Discounted Cumulative Gain, penalizing relevant items placed lower in the list.
    """)

    benchmark_csv_path = os.path.join(_CURRENT_DIR, "experiments", "results", "benchmark_metrics.csv")
    if os.path.exists(benchmark_csv_path):
        bench_df = pd.read_csv(benchmark_csv_path)
        st.dataframe(bench_df, width="stretch")
    else:
        st.info("Run `python experiments/run_experiments.py` to generate offline benchmark metrics.")

    st.markdown("""
    #### 🛡️ Data Leakage Prevention Guarantee
    - **Classification Evaluation:** Uses `sklearn.pipeline.Pipeline` combining `TfidfVectorizer` and `LogisticRegression`.
    - **Stratified Split:** Split is performed *before* vectorizer fitting. Vocabulary and inverse document frequencies are learned **strictly on training folds**.
    - **Recommendation Evaluation:** Evaluated on held-out candidate profiles using explicit domain proxy criteria without synthetic label contamination.
    """)
