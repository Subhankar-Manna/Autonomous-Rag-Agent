import re
from app.graph.state import AgentState


class ReviewerAgent:

    def clean_text(self, text: str) -> str:
        """
        Clean messy output:
        - Remove extra spaces/newlines
        - Remove repeated sentences
        """
        text = text.replace("\n", " ").strip()

        sentences = list(dict.fromkeys(text.split(". ")))
        return ". ".join(sentences).strip()

    def has_page_reference(self, text: str) -> bool:
        return bool(re.search(r"Page\s+\d+", text))

    def contains_hallucination(self, text: str) -> bool:
        forbidden_phrases = [
            "generally",
            "typically",
            "it appears",
            "it seems",
            "based on knowledge",
            "in general"
        ]
        return any(p in text.lower() for p in forbidden_phrases)

    def run(self, state: AgentState) -> AgentState:
        """
        HYBRID-AWARE REVIEWER (FINAL):
        - Cleans output
        - Supports both LLM + RAG
        - Applies strict checks ONLY for RAG
        """


        if not state.result or state.result.strip() == "":
            state.result = "I don't know"
            return state

        answer = state.result.strip()

        # Clean text
        answer = self.clean_text(answer)

        if len(answer) < 10:
            state.result = "I don't know"
            return state

        source = getattr(state, "source", "")

      
        if source == "llm":
            state.result = answer
            return state

        if source == "rag":
            if not self.has_page_reference(answer):
                answer = f"(Source not explicit) {answer}"

        if self.contains_hallucination(answer):
            state.result = "I don't know"
            return state

        state.result = answer
        return state