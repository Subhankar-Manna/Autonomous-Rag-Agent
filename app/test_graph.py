from app.main import run_rag

query = "Ask the question"

state = run_rag(query)

print("\nPLAN:")
if state.get("plan"):
    for step in state["plan"]:
        print("-", step)
else:
    print("No plan generated")

print("\nRESEARCH NOTES:")
print(state.get("research", "No research notes"))

print("\nFINAL ANSWER:")
print(state.get("result", "No result generated"))







