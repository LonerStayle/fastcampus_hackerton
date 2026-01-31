from langchain.agents import AgentState 

class TransferState(AgentState):
    temp_recommendation: str = ""
    recommendation: str = ""