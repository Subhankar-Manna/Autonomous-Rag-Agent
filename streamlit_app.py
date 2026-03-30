import streamlit as st
import requests
import json
from pathlib import Path

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/chat"
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
                response = requests.post(
                    API_URL,
                    json={"query": query}
                )

                result = response.json()

                st.success("Done!")

                st.subheader("🧠 Final Answer")

                if "response" in result:
                    st.write(result["response"])
                else:
                    st.error(result.get("error", "Something went wrong"))

               
                if "metrics" in result:
                    st.subheader("📊 Evaluation Metrics")

                    metrics = result["metrics"]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Response Length", metrics.get("response_length", 0))

                    with col2:
                        st.metric("Latency (seconds)", round(metrics.get("latency_seconds", 0), 3))

                    with st.expander("See full metrics"):
                        st.json(metrics)

            except Exception as e:
                st.error(f"Error: {str(e)}")


st.markdown("---")
st.subheader("📈 Performance Dashboard")

if LOG_FILE.exists():
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)

        if len(data) > 0:
            # Extract metrics
            latencies = [item["metrics"]["latency_seconds"] for item in data]
            lengths = [item["metrics"]["response_length"] for item in data]
            queries = [item["query"] for item in data]

            
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