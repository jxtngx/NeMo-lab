# see: https://langchain-ai.github.io/langgraph/#example
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from nemo.collections import llm

# steps
# 1. initialize the models and tools
# 2. initialize graph with state
# 3. define graph with state
# 4. define entrypoint and graph edges
# 5. compile the graph
# 6. execute the graph


# 1. initialize the models and tools
config = llm.Llama31Config8B()
model = llm.LlamaModel(config=config, tokenizer="meta-llama/Llama-3.1-8B")


@tool
def a_tool(): ...


tools = [a_tool]
tool_node = ToolNode(tools)


# 2. initialize graph with state
def should_continue(): ...
def call_model(): ...


workflow = StateGraph(MessagesState)

# 3. define graph with state
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 4. define entrypoint and graph edges
workflow.add_edge(START, "agent")  # entrypoint
workflow.add_conditional_edges("agent", should_continue)

# 5. compile the graph
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 6. execute the graph
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]}, config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
