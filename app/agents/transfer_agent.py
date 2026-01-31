from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.messages import SystemMessage, HumanMessage
from .transfer_state import TransferState
from langgraph.graph import START, END, StateGraph
from langchain_anthropic import ChatAnthropic
from app.core.prompt import CASE_1_SYSTEM_PROMPT, CASE_2_SYSTEM_PROMPT, ROUTER_PROMPT
from langchain_groq import ChatGroq


common_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens = 80
)

router_model = common_llm
case_model = common_llm

graph_builder = StateGraph(TransferState)

def router(state:TransferState):
    last_user_message = state["messages"][-1].content
    response = router_model.invoke(
        [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=last_user_message)
        ]
    )

    content = response.content
    if "CASE_1" in content:
        return "case_1"
    else:
        return "case_2"


def case_1_transfer(state:TransferState):
    result = case_model.invoke(
        [
            SystemMessage(content=CASE_1_SYSTEM_PROMPT),
            HumanMessage(content=state["messages"][-1].content)
        ]
    )
    return {
        "messages": [result]
    }
    
def case_2_transfer(state:TransferState):
    result = case_model.invoke(
        [
            SystemMessage(content=CASE_2_SYSTEM_PROMPT),
            HumanMessage(content=state["messages"][-1].content)
        ]
    )
    return {
        "messages": [result]
    }


graph_builder.add_node("case_1_transfer", case_1_transfer)
graph_builder.add_node("case_2_transfer", case_2_transfer)

graph_builder.add_conditional_edges(
    START,
    router,
    {
        "case_1": "case_1_transfer",
        "case_2": "case_2_transfer",
    },
)
graph_builder.add_edge("case_1_transfer", END)
graph_builder.add_edge("case_2_transfer", END)
graph = graph_builder.compile()




