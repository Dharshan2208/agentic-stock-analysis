from typing import Annotated,TypedDict,List,Any

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.market import price_ohlcv
from tools.news import new_summariser
from tools.fundamentals import fundamentals
from tools.trend import trend_analysis
from colorama import Fore,Style

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.23
)

tools = [price_ohlcv,fundamentals,new_summariser,trend_analysis]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
                You are a professional financial analyst. For any user input about a ticker or stock/company you MUST:
                1) Use ALL available tools to gather evidence:
                - price_ohlcv: fetch recent market price and OHLCV data (prefer last 10min / intraday / latest candle)
                - fundamentals: fetch company fundamentals and use all the data.
                - new_summariser: fetch the recent news and headlines
                - trend_analysis : fetch the 7 days and 30 days and 200 days trend
                2) Synthesize the tool outputs into a final recommendation with three possible sentiments: 'bullish', 'bearish', or 'neutral'.
                3) Return a concise JSON object ONLY (no extra chatter) with these keys:
                {
                    "symbol": "<INPUT_SYMBOL>",
                    "sentiment": "bullish|bearish|neutral",
                    "confidence": <0-100 integer>,
                    "rationale": "<2-6 sentence summary tying together price/fundamentals/news>",
                    "signals": ["signal 1", "signal 2", ...],  # short bullet signals used to decide
                    "tool_outputs": {
                        "price_ohlcv": <short summary>,
                        "fundamentals": <short summary>,
                        "new_summariser": <short summary>,
                        "trend_analysis": "<short summary>"
                    }
                }
                4) Provide numeric reasoning (e.g., "% change in price", "PE=xx", "latest headline sentiment") inside the rationale where useful.
                5) If tools return errors or no data, explicitly show that in tool_outputs and still try to give a cautious recommendation.
                6) Use a neutral, professional tone. Keep rationale concise and focused on evidence.
                7) Example output for user 'AAPL':
                {
                "symbol":"AAPL",
                "sentiment":"bullish",
                "confidence":78,
                "rationale":"Recent intraday price breakout (+3.2% 1h) combined with improving fundamentals (PE 23 vs sector 28) and positive exec-level news indicate momentum. Volume is above average.",
                "signals":["price_breakout","improving_PE_vs_sector","positive_news_headlines","above_average_volume"],
                "tool_outputs":{
                    "alpaca_price": "...",
                    "fundamentals": "...",
                    "new_summariser":"...",
                    "trend_analysis":"..."
                }
                }
                """


# State
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]


# Chatbot Node
def chatbot(state: State):
    """
    LLM node — ensure system prompt is included and invite tool usage.
    Returns messages expected by the graph (list-like), catching exceptions and returning an assistant error message.
    """
    try:
        messages_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
        response = llm_with_tools.invoke(messages_with_system)
        return {"messages": [response]}
    except Exception as e:
        return {
            "messages": [{
                "role": "assistant",
                "content": {
                    "error": f"Error in LLM node: {str(e)}"
                }
            }]
        }

# Router Node
def router(state: State):
    """
    If last message has tool_calls → go to tools
    else → end the graph
    """
    last_msg = state["messages"][-1]

    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END

# Graph Build
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", router)

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def safe_print(message):
    """Ensures output is always readable regardless of type."""
    if message is None:
        return ""

    if isinstance(message, str):
        return message

    if isinstance(message, list):
        return "\n".join([safe_print(m) for m in message])

    if isinstance(message, dict):
        import json
        return json.dumps(message, indent=2)

    # If message is an object with a 'content' attribute, try to fetch it.
    content = getattr(message, "content", None)
    if content is not None:
        return safe_print(content)

    # fallback
    return str(message)

if __name__ == "__main__":
    print(Fore.GREEN + "Financial Analyst Chatbot (type a ticker or question, e.g. 'AAPL' or 'What about MSFT?')" + Fore.RESET)
    while True:
        try:
            prompt = input("USER : ").strip()
            if not prompt:
                continue

            # Pass user message into graph; configurable thread_id preserved
            result = graph.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": 1234}},
            )

            # get the last assistant message and robustly print content
            final_msg = result["messages"][-1]
            content = getattr(final_msg, "content", final_msg)

            # If content is a dict and has "text", extract only that.
            if isinstance(content, dict) and "text" in content:
                printable = content["text"]

            # If content is a list and element contains "text"
            elif isinstance(content, list) and len(content) > 0:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    printable = first["text"]
                else:
                    printable = safe_print(content)

            # fallback
            else:
                printable = safe_print(content)

            # print(content[0]["text"])
            # printable = safe_print(content)

            print(Fore.LIGHTYELLOW_EX + printable + Fore.RESET)
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(Fore.RED + f"Runtime error: {e}" + Fore.RESET)

