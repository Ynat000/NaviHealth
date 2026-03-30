import os
from llama_index.llms.groq import Groq

# check API key
api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    print("ERROR: GROQ_API_KEY environment variable not set!")
    print("Run: $env:GROQ_API_KEY = 'your-api-key'")
    exit(1)

print("API key found.")

supported_models = [
    "qwen/qwen3-32b",            # Qwen QWQ
    "llama-3.3-70b-versatile",   # Llama as fallback
    "openai/gpt-oss-120b"        # OpenAI as additional fallback
] #note: from https://console.groq.com/docs/models

working_model = None

print("\nTesting model availability...")
for model_name in supported_models:
    try:
        print(f"  Trying {model_name}...", end=" ")
        llm = Groq(model=model_name, api_key=api_key)
        response = llm.complete("Say 'hello' in one word.")
        print(f"OK - Response: {response.text.strip()[:50]}")
        working_model = model_name
        break
    except Exception as e:
        print(f"Failed - {str(e)[:50]}")

if not working_model:
    print("\nERROR: No working model found!")
    print("Please check Groq's documentation for available models.")
    exit(1)

print(f"\nUsing model: {working_model}")
llm = Groq(model=working_model, api_key=api_key)

# multilingual test
print("\n" + "="*60)
print("MULTILINGUAL TEST")
print("="*60)

test_cases = [
    {
        "name": "Chinese question, Chinese answer",
        "prompt": "请用中文简短回答：感冒了应该怎么办？只需要三句话。"
    },
    {
        "name": "French question, French answer", 
        "prompt": "Répondez en français en trois phrases: Que faire quand on a un rhume?"
    },
    {
        "name": "English question with Chinese context",
        "prompt": """Based on this policy text, answer the question in Chinese.

Policy: "Urgent and Primary Care Centres (UPCCs) provide same-day care for conditions such as high fevers, sprains, minor cuts requiring stitches, and mild breathing difficulty."

Question: 我扭伤了脚踝，应该去哪里看？

Answer in Chinese:"""
    },
]

for case in test_cases:
    print(f"\n📝 Test: {case['name']}")
    print(f"   Prompt: {case['prompt'][:60]}...")
    
    try:
        response = llm.complete(case['prompt'])
        answer = response.text.strip()
        # only display up to 200 chars
        display_answer = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"   Response: {display_answer}")
    except Exception as e:
        print(f"   ERROR: {e}")

# # Simulation of RAG
# print("\n" + "="*60)
# print("RAG SIMULATION PRE-TEST")
# print("="*60)

# rag_prompt = """你是BC省医疗分流助手。根据以下政策文档回答用户的问题。
# 用用户提问的语言回答。必须引用政策依据。如果不确定，建议拨打811。

# 政策文档:
# ---
# Urgent and Primary Care Centres (UPCCs) provide access to same-day, urgent, non-emergency health care.
# Patients who require medical attention within 12 to 24 hours can receive care at UPCCs for conditions such as:
# - High fevers
# - Sprains and strains caused by minor accidents and falls
# - Minor bleeding or cuts that require stitches
# - Mild to moderate breathing difficulty or asthma attacks
# ---

# 用户问题: 我走路的时候腿很痛，有点肿，应该去哪里？

# 请回答:"""

# print(f"Testing RAG-style prompt...")
# try:
#     response = llm.complete(rag_prompt)
#     print(f"\nResponse:\n{response.text.strip()}")
# except Exception as e:
#     print(f"ERROR: {e}")

print("\n" + "="*60)
print("LLM TEST COMPLETE")
print("="*60)
