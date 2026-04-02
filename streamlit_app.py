import streamlit as st
import json
from pathlib import Path
import time


from app.main import run_rag_pipeline   

LOG_FILE = Path("logs.json")

st.set_page_config(page_title="RAG Agent System", layout="wide")

st.title("🤖 Autonomous Multi-Agent RAG System")
st.write("Ask questions grounded in documents or general AI knowledge.")

query = st.text_input(
    "Enter your question:",
    placeholder="Ask the question:"
)

if st.button("Run RAG"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running AI agents..."):
            try:
                start_time = time.time()

                result = run_rag_pipeline(query)

                end_time = time.time()

                st.success("Done!")

                response_text = result.get("response", "No response generated")

                st.subheader("Final Answer")
                st.write(response_text)

                # Metrics
                latency = end_time - start_time
                response_length = len(response_text)

                metrics = {
                    "latency_seconds": latency,
                    "response_length": response_length
                }

                st.subheader("📊 Evaluation Metrics")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Response Length", response_length)

                with col2:
                    st.metric("Latency (seconds)", round(latency, 3))

                with st.expander("See full metrics"):
                    st.json(metrics)

                # Save logs
                log_entry = {
                    "query": query,
                    "response": response_text,
                    "metrics": metrics
                }

                if LOG_FILE.exists():
                    with open(LOG_FILE, "r") as f:
                        data = json.load(f)
                else:
                    data = []

                data.append(log_entry)

                with open(LOG_FILE, "w") as f:
                    json.dump(data, f, indent=4)

            except Exception as e:
                st.error(f"Error: {str(e)}")


st.markdown("---")
st.subheader("📈 Performance Dashboard")

if LOG_FILE.exists():
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)

        if len(data) > 0:
            latencies = [item["metrics"]["latency_seconds"] for item in data]
            lengths = [item["metrics"]["response_length"] for item in data]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Queries", len(data))

            with col2:
                st.metric("Avg Latency", round(sum(latencies)/len(latencies), 3))

            with col3:
                st.metric("Avg Response Length", int(sum(lengths)/len(lengths)))

            st.subheader("⏱ Latency Trend")
            st.line_chart(latencies)

            st.subheader("📝 Response Length Trend")
            st.line_chart(lengths)

            st.subheader("🗂 Query History")

            for item in reversed(data[-5:]):
                with st.expander(f"Q: {item['query']}"):
                    st.write("**Answer:**", item["response"])

        else:
            st.info("No logs yet")

    except Exception as e:
        st.error(f"Dashboard Error: {str(e)}")

else:
    st.info("No logs file found. Run some queries first.")