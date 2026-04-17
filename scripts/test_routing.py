"""Test LLM routing latency for the supervisor."""
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOllama(model="qwen3:8b", temperature=0, num_ctx=4096)

print("Testing routing LLM call latency...")
t0 = time.time()
resp = llm.invoke([
    SystemMessage(content="Respond ONLY with a JSON array. Example: [\"diagnostic\"]"),
    HumanMessage(content='Query: What is HER2 status for IHC 3+?\nClinical data: {"ihc_score": "3+"}'),
])
elapsed = time.time() - t0
print(f"Time: {elapsed:.1f}s")
print(f"Response: {repr(resp.content[:500])}")
