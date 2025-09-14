import streamlit as st
import pandas as pd
import numpy as np
import time

# Try to import Plotly and show a friendly error if missing
try:
    import plotly.express as px
except ImportError:
    st.error(
        "Plotly is required for this dashboard but is not installed.\n\n"
        "Please install it by running:\n"
        "`pip install plotly`\n\n"
        "After installation, restart the app."
    )
    st.stop()

# Optional: if you want to keep your original evaluation pipeline,
def lazy_import_langchain():
    from langchain_openai import OpenAIEmbeddings, OpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.agents import initialize_agent, Tool
    return {
        "OpenAIEmbeddings": OpenAIEmbeddings,
        "OpenAI": OpenAI,
        "FAISS": FAISS,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "RetrievalQA": RetrievalQA,
        "PromptTemplate": PromptTemplate,
        "initialize_agent": initialize_agent,
        "Tool": Tool,
    }

# ------------------------------------------------------------
# Original helper functions (preserved)
# ------------------------------------------------------------
def compute_f1(pred, truth):
    pred_tokens = set(pred.lower().split())
    true_tokens = set(truth.lower().split())
    if not pred_tokens or not true_tokens:
        return 0
    precision = len(pred_tokens & true_tokens) / (len(pred_tokens) + 1e-9)
    recall = len(pred_tokens & true_tokens) / (len(true_tokens) + 1e-9)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def build_langchain_objects(openai_api_key, uploaded_files):
    # Lazy import to avoid requiring these deps for synthetic mode
    L = lazy_import_langchain()
    OpenAIEmbeddings = L["OpenAIEmbeddings"]
    OpenAI = L["OpenAI"]
    FAISS = L["FAISS"]
    RecursiveCharacterTextSplitter = L["RecursiveCharacterTextSplitter"]
    RetrievalQA = L["RetrievalQA"]
    PromptTemplate = L["PromptTemplate"]

    documents = [f.read().decode("utf-8", errors="ignore") for f in uploaded_files]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    base_prompt = PromptTemplate(
        template="Context: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"],
    )
    reflection_prompt = PromptTemplate(
        template="Question: {question}\nInitial Answer: {answer}\nCritique and improved answer:",
        input_variables=["question", "answer"],
    )

    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    return retriever, llm, base_prompt, reflection_prompt, RetrievalQA, L["initialize_agent"], L["Tool"]

def run_rag(query, mode, retriever, llm, base_prompt, reflection_prompt, RetrievalQA, initialize_agent, Tool):
    start = time.perf_counter()
    if mode == "Normal RAG":
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        answer = qa_chain.run(query)
    elif mode == "Self-RAG":
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, chain_type="stuff",
            chain_type_kwargs={"prompt": base_prompt}
        )
        initial_answer = qa_chain.run(query)
        reflection_input = reflection_prompt.format(question=query, answer=initial_answer)
        answer = llm(reflection_input)
    else:  # Agentic RAG
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        tools = [Tool(name="KBQA", func=qa_chain.run, description="Answer from knowledge base")]
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
        answer = agent.run(query)
    latency = (time.perf_counter() - start) * 1000
    return answer, latency

# ------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------
RAG_TYPES = ["Vanilla RAG", "Self-RAG", "Agentic RAG"]

def truncated_normal(mean, std, size, lower=0.0, upper=1.0, seed=None):
    rng = np.random.default_rng(seed)
    vals = rng.normal(mean, std, size=size)
    return np.clip(vals, lower, upper)

def sample_latency(mean_ms, std_ms, size, lower=50, upper=5000, seed=None):
    rng = np.random.default_rng(seed)
    vals = rng.normal(mean_ms, std_ms, size=size)
    return np.clip(vals, lower, upper)

def generate_synthetic_dataset(n_queries=50, seed=42):
    """
    Returns a DataFrame with n_queries * 3 rows (one per RAG type per query).
    Columns:
      - query_id, rag_type
      - factual_accuracy, retrieval_precision, retrieval_recall, hallucination_rate,
        multi_hop_reasoning, latency_ms
      - error flags (6 booleans)
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Distribution assumptions per RAG type (reflecting your findings)
    params = {
        "Vanilla RAG": {
            "f1":       {"mean": 0.75, "std": 0.10},
            "prec":     {"mean": 0.70, "std": 0.10},
            "recall":   {"mean": 0.75, "std": 0.10},
            "hall":     {"mean": 0.15, "std": 0.05},
            "mh":       {"mean": 0.60, "std": 0.15},
            "lat":      {"mean": 500,  "std": 100},
            "errors_p": {
                "misinterp": 0.30,
                "overweight": 0.40,
                "specialized": 0.50,
                "long_tail": 0.50,
                "irrelevant": 0.30,
                "superficial": 0.40,
            }
        },
        "Self-RAG": {
            "f1":       {"mean": 0.85, "std": 0.08},
            "prec":     {"mean": 0.80, "std": 0.10},
            "recall":   {"mean": 0.80, "std": 0.10},
            "hall":     {"mean": 0.05, "std": 0.02},
            "mh":       {"mean": 0.70, "std": 0.10},
            "lat":      {"mean": 700,  "std": 150},
            "errors_p": {
                "misinterp": 0.20,
                "overweight": 0.20,
                "specialized": 0.30,
                "long_tail": 0.40,
                "irrelevant": 0.10,
                "superficial": 0.30,
            }
        },
        "Agentic RAG": {
            "f1":       {"mean": 0.80, "std": 0.12},
            "prec":     {"mean": 0.75, "std": 0.10},
            "recall":   {"mean": 0.85, "std": 0.10},
            "hall":     {"mean": 0.10, "std": 0.03},
            "mh":       {"mean": 0.80, "std": 0.12},
            "lat":      {"mean": 1200, "std": 300},
            "errors_p": {
                "misinterp": 0.10,
                "overweight": 0.10,
                "specialized": 0.20,
                "long_tail": 0.30,
                "irrelevant": 0.20,
                "superficial": 0.20,
            }
        }
    }

    for qid in range(1, n_queries + 1):
        for rag in RAG_TYPES:
            p = params[rag]

            f1 = float(truncated_normal(p["f1"]["mean"], p["f1"]["std"], 1, 0, 1, seed=rng.integers(1e9))[0])
            prec = float(truncated_normal(p["prec"]["mean"], p["prec"]["std"], 1, 0, 1, seed=rng.integers(1e9))[0])
            rec = float(truncated_normal(p["recall"]["mean"], p["recall"]["std"], 1, 0, 1, seed=rng.integers(1e9))[0])
            hall = float(truncated_normal(p["hall"]["mean"], p["hall"]["std"], 1, 0, 1, seed=rng.integers(1e9))[0])
            mh = float(truncated_normal(p["mh"]["mean"], p["mh"]["std"], 1, 0, 1, seed=rng.integers(1e9))[0])
            lat = float(sample_latency(p["lat"]["mean"], p["lat"]["std"], 1, 50, 5000, seed=rng.integers(1e9))[0])

            # Error flags (Bernoulli)
            e = p["errors_p"]
            misinterp = int(rng.random() < e["misinterp"])
            overweight = int(rng.random() < e["overweight"])
            specialized = int(rng.random() < e["specialized"])
            long_tail = int(rng.random() < e["long_tail"])
            irrelevant = int(rng.random() < e["irrelevant"])
            superficial = int(rng.random() < e["superficial"])

            rows.append({
                "query_id": qid,
                "rag_type": rag,
                "factual_accuracy": f1,
                "retrieval_precision": prec,
                "retrieval_recall": rec,
                "hallucination_rate": hall,
                "multi_hop_reasoning": mh,
                "latency_ms": lat,
                "err_misinterpretation": misinterp,
                "err_over_reliance_high_weight": overweight,
                "err_specialized_knowledge": specialized,
                "err_long_tail": long_tail,
                "err_contextual_irrelevance": irrelevant,
                "err_superficial_accuracy": superficial,
            })
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------
st.set_page_config(page_title="RAG Evaluation Lab", layout="wide")
st.title("📊 RAG Evaluation Lab")

tab1, tab2 = st.tabs(["🧪 Synthetic Benchmark (No API needed)", "🧭 Run Real Evaluation (Optional)"])

with tab1:
    st.sidebar.header("Synthetic Benchmark Settings")
    n_queries = st.sidebar.slider("Number of queries", min_value=50, max_value=500, value=50, step=10)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
    st.sidebar.write("_The dataset will have n_queries × 3 rows (one per RAG type)._")

    df_syn = generate_synthetic_dataset(n_queries=n_queries, seed=seed)

    st.subheader("📄 Synthetic Dataset Preview")
    st.dataframe(df_syn.head(30), use_container_width=True)
    st.download_button("Download synthetic CSV", df_syn.to_csv(index=False), "synthetic_rag_benchmark.csv", mime="text/csv")

    # Aggregates
    metrics = ["factual_accuracy", "retrieval_precision", "retrieval_recall",
               "hallucination_rate", "multi_hop_reasoning"]
    pretty = {
        "factual_accuracy": "Factual Accuracy (F1)",
        "retrieval_precision": "Retrieval Precision",
        "retrieval_recall": "Retrieval Recall",
        "hallucination_rate": "Hallucination Rate",
        "multi_hop_reasoning": "Multi-hop Reasoning Score",
        "latency_ms": "End-to-End Latency (ms)",
    }

    agg = df_syn.groupby("rag_type").agg({
        "factual_accuracy": "mean",
        "retrieval_precision": "mean",
        "retrieval_recall": "mean",
        "hallucination_rate": "mean",
        "multi_hop_reasoning": "mean",
        "latency_ms": "mean"
    }).reset_index()

    # Scale some metrics to percentage view for charting clarity
    agg_pct = agg.copy()
    agg_pct["Retrieval Precision (%)"] = agg_pct["retrieval_precision"] * 100
    agg_pct["Retrieval Recall (%)"] = agg_pct["retrieval_recall"] * 100
    agg_pct["Hallucination Rate (%)"] = agg_pct["hallucination_rate"] * 100
    agg_pct["Factual Accuracy (F1)"] = agg_pct["factual_accuracy"]
    agg_pct["Multi-hop Reasoning Score"] = agg_pct["multi_hop_reasoning"]
    agg_pct["End-to-End Latency (ms)"] = agg_pct["latency_ms"]

    st.markdown("## 1) Quantitative Metrics — System Comparison")
    cols_show = ["Factual Accuracy (F1)", "Retrieval Precision (%)", "Retrieval Recall (%)",
                 "Hallucination Rate (%)", "Multi-hop Reasoning Score"]
    for i in range(0, len(cols_show), 2):
        c1, c2 = st.columns(2)
        for c, col in zip((c1, c2), cols_show[i:i+2]):
            fig = px.bar(
                agg_pct,
                x="rag_type",
                y=col,
                color="rag_type",
                title=col,
                text=agg_pct[col].round(2),
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
            c.plotly_chart(fig, use_container_width=True)

    st.markdown("## 2) End-to-End Latency Comparison")
    fig_lat = px.bar(
        agg_pct,
        x="rag_type",
        y="End-to-End Latency (ms)",
        color="rag_type",
        text=agg_pct["End-to-End Latency (ms)"].round(1),
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Average Latency by System"
    )
    fig_lat.update_traces(textposition="outside")
    fig_lat.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig_lat, use_container_width=True)

    st.dataframe(
        agg_pct[["rag_type", "End-to-End Latency (ms)"]]
        .rename(columns={"rag_type": "System"})
        .round(2),
        use_container_width=True
    )

    st.markdown("## 3) Error Analysis — Category Frequencies")
    error_cols = [
        "err_misinterpretation",
        "err_over_reliance_high_weight",
        "err_specialized_knowledge",
        "err_long_tail",
        "err_contextual_irrelevance",
        "err_superficial_accuracy",
    ]
    error_pretty = {
        "err_misinterpretation": "Misinterpretation of Ambiguous Queries",
        "err_over_reliance_high_weight": "Over-reliance on High-Weight Documents",
        "err_specialized_knowledge": "Errors in Specialized Knowledge",
        "err_long_tail": "Long-tail Query Handling",
        "err_contextual_irrelevance": "Contextual Irrelevance",
        "err_superficial_accuracy": "Superficial Accuracy",
    }

    # Compute percentage of queries exhibiting each error per system
    err_agg = (
        df_syn
        .groupby("rag_type")[error_cols]
        .mean()
        .reset_index()
    )
    # Convert to tidy format for grouped bar chart
    err_long = err_agg.melt(id_vars="rag_type", var_name="error", value_name="rate")
    err_long["rate_pct"] = err_long["rate"] * 100
    err_long["error_pretty"] = err_long["error"].map(error_pretty)

    fig_err = px.bar(
        err_long,
        x="error_pretty",
        y="rate_pct",
        color="rag_type",
        barmode="group",
        title="Error Category Frequency by System",
        labels={"rate_pct": "Frequency (%)", "error_pretty": "Error Category", "rag_type": "System"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_err.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig_err, use_container_width=True)

    with st.expander("Show aggregated table"):
        st.dataframe(
            err_agg.rename(columns=error_pretty).set_index("rag_type").applymap(lambda v: round(v*100, 1)),
            use_container_width=True
        )

    st.caption(
        "Notes: Self-RAG shows lower hallucination and higher precision; Agentic RAG excels in multi-hop reasoning at higher latency; "
        "Vanilla RAG remains competitive for single-hop recall with lower latency."
    )

with tab2:
    st.subheader("Run Real Evaluation (Optional)")
    st.write("Use your own knowledge base and dataset to run the original pipelines. This requires an OpenAI-compatible API and LangChain deps.")

    st.sidebar.header("Real Evaluation Settings")
    openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
    rag_mode = st.sidebar.selectbox("Choose RAG Mode", ["Normal RAG", "Self-RAG", "Agentic RAG"])

    uploaded_files = st.file_uploader("Upload knowledge base (TXT)", accept_multiple_files=True, key="kb_upload")
    dataset_file = st.file_uploader("Upload evaluation dataset (CSV: query, ground_truth, relevant_docs)", type=["csv"], key="ds_upload")

    if uploaded_files and openai_api_key:
        try:
            retriever, llm, base_prompt, reflection_prompt, RetrievalQA, initialize_agent, Tool = build_langchain_objects(openai_api_key, uploaded_files)
            st.success("✅ Knowledge base indexed.")
        except Exception as e:
            st.error(f"Failed to initialize KB/LLM: {e}")
            st.stop()
    else:
        st.info("Provide API key and upload KB to enable evaluation.")
        retriever = llm = base_prompt = reflection_prompt = RetrievalQA = initialize_agent = Tool = None

    if dataset_file is not None and retriever is not None:
        df_real = pd.read_csv(dataset_file)

        results = []
        for _, row in df_real.iterrows():
            q = row.get("query", "")
            gt = str(row.get("ground_truth", ""))
            relevant_docs = str(row.get("relevant_docs", "")).split(",")

            try:
                answer, latency = run_rag(q, rag_mode, retriever, llm, base_prompt, reflection_prompt, RetrievalQA, initialize_agent, Tool)
            except Exception as e:
                answer, latency = f"ERROR: {e}", np.nan

            f1 = compute_f1(answer, gt) if isinstance(answer, str) else 0.0

            # Placeholder retrieval metrics (since we don't surface retriever hits here)
            retrieved = relevant_docs[:1]
            retrieved_set, relevant_set = set(retrieved), set(relevant_docs)
            precision = len(retrieved_set & relevant_set) / (len(retrieved_set) + 1e-9) if len(retrieved_set) > 0 else 0.0
            recall = len(retrieved_set & relevant_set) / (len(relevant_set) + 1e-9) if len(relevant_set) > 0 else 0.0

            hallucination = 1 if isinstance(answer, str) and not any(w in gt.lower() for w in answer.lower().split()) else 0
            is_multihop = any(x in q.lower() for x in ["who", "which company", "author of", "how many", "steps"])
            multihop_score = 1 if is_multihop and f1 > 0.7 else 0

            results.append({
                "query": q,
                "answer": answer,
                "ground_truth": gt,
                "F1": round(f1, 3),
                "Retrieval Precision": round(precision * 100, 2),
                "Retrieval Recall": round(recall * 100, 2),
                "Hallucination": hallucination,
                "Multi-hop Score": multihop_score,
                "Latency (ms)": None if pd.isna(latency) else round(latency, 2),
            })

        results_df = pd.DataFrame(results)
        st.subheader("📊 Evaluation Results")
        st.dataframe(results_df, use_container_width=True)
        st.download_button("Download Results as CSV", results_df.to_csv(index=False), "rag_eval_results.csv", mime="text/csv")
    else:
        st.info("Upload both KB and dataset (and provide API key) to run real evaluation.")