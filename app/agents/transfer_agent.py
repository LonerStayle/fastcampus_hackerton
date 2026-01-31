from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.messages import SystemMessage, HumanMessage
from .transfer_state import TransferState
from langgraph.graph import START, END, StateGraph
from langchain_anthropic import ChatAnthropic
from app.core.prompt import CASE_1_SYSTEM_PROMPT, CASE_2_SYSTEM_PROMPT, ROUTER_PROMPT, RAG_PROMPT
from langchain_groq import ChatGroq
from app.core.retriever import get_ensemble_retriever

case_model = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens = 80
)

router_model = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens = 10
)
recommand_model = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens = 200
)

graph_builder = StateGraph(TransferState)


def recommend_answer(state:TransferState):
    retriever = get_ensemble_retriever(
        bm25_k = 3,
        dense_k= 3,
        final_k= 3,
    )
    last_user_message = state["messages"][-1].content
    docs = retriever.invoke(last_user_message)
    combined_content = "\n\n".join([doc.page_content for doc in docs])
    response = recommand_model.invoke(
        [
            SystemMessage(content=RAG_PROMPT.format(context=combined_content, question=last_user_message)),
        ]
    )
    
    return {
        "temp_recommendation": response.content
    }

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
    elif "CASE_2" in content:
        return "case_2"
    else:
        return "case_3"  


def case_1_transfer(state:TransferState):
    result = case_model.invoke(
        [
            SystemMessage(content=CASE_1_SYSTEM_PROMPT),
            HumanMessage(content=state["messages"][-1].content)
        ]
    )
    return {
        "messages": [result],
    }
    
def case_2_transfer(state:TransferState):
    result = case_model.invoke(
        [
            SystemMessage(content=CASE_2_SYSTEM_PROMPT),
            HumanMessage(content=state["messages"][-1].content)
        ]
    )
    return {
        "messages": [result],
        "recommendation": state.get("temp_recommendation", "")
    }
def case_3_transfer(state:TransferState):
    return {
        "recommendation": state.get("temp_recommendation", "")
    }


graph_builder.add_node("recommend_answer", recommend_answer)
graph_builder.add_node("case_1_transfer", case_1_transfer)
graph_builder.add_node("case_2_transfer", case_2_transfer)
graph_builder.add_node("case_3_transfer", case_3_transfer)

graph_builder.add_edge(START, "recommend_answer")
graph_builder.add_conditional_edges(
    "recommend_answer",
    router,
    {
        "case_1": "case_1_transfer",
        "case_2": "case_2_transfer",
        "case_3": "case_3_transfer",
    },
)
graph_builder.add_edge("case_1_transfer", END)
graph_builder.add_edge("case_2_transfer", END)
graph_builder.add_edge("case_3_transfer", END)
graph = graph_builder.compile()




