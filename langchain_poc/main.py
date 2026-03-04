from langchain_ollama import ChatOllama
from langchain.tools import tool
# from langchain

@tool
def get_best_applicant(job_id): 
    """ return best applicant
    
    Args: 
        job_id (int) the jobs ID.
    """
    print( "The D&A Recommendation for {job_id} Job is Application-323, Andy Wang")


llm = ChatOllama(model="llama3.2").bind_tools([get_best_applicant])

messages = [("system", "you are an assistant that says hello"), ("human", "hi")]

result = llm.invoke(messages)

print(result.content)


# from langchain.agents import create_agent 


# agent = create_agent(
#     model="",
#     tools=[get_best_app],
#     system_prompt="You are an Hiring Managers Assistant",
# )

# agent.invoke(
#     {"messages": [{
#         "role": "user", 
#         "content": "what is the weather in sf"
#     }]}
# )