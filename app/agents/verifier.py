import re
from app.graph.state import AgentState


class VerifierAgent:

    def is_supported(self, sentence: str, context: str) -> bool:
        """
        Check if sentence is grounded in retrieved research
        """
        sentence_words = set(sentence.lower().split())
        context_words = set(context.lower().split())

        overlap = sentence_words.intersection(context_words)

        return len(overlap) >= 3

    def clean_text(self, text: str) -> str:
        """
        Final cleanup:
        - remove extra spaces
        - remove page numbers
        """
        text = re.sub(r"\(Page\s+\d+.*?\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def contains_hallucination(self, text: str) -> bool:
        forbidden = [
            "in general",
            "typically",
            "it seems",
            "based on knowledge",
            "widely used"
        ]
        return any(p in text.lower() for p in forbidden)

    def run(self, state: AgentState) -> AgentState:
        """
        FINAL HYBRID VERIFIER:
        - LLM → minimal filtering
        - RAG → grounding check
        """

        if not state.result or state.result.strip() == "I don't know":
            return state

        answer = state.result.strip()

        source = getattr(state, "source", "")

       
        if source == "llm":
            final_answer = self.clean_text(answer)

            if len(final_answer) < 20:
                state.result = "I don't know"
            else:
                state.result = final_answer

            return state

        
        if source == "rag":

            research = getattr(state, "research", "")

            if not research:
                return state 

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', answer)

            verified_sentences = []
            for s in sentences:
                if self.is_supported(s, research):
                    verified_sentences.append(s)

        
            if not verified_sentences:
                print("No grounded sentences → keeping original answer")
                final_answer = self.clean_text(answer)
            else:
                final_answer = " ".join(verified_sentences)
                final_answer = self.clean_text(final_answer)

            if len(final_answer) < 10:
                state.result = "I don't know"
            else:
                state.result = final_answer

            return state

        
        final_answer = self.clean_text(answer)

        if len(final_answer) < 10:
            state.result = "I don't know"
        else:
            state.result = final_answer

        return state