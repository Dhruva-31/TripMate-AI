import asyncio
import os
from dotenv import load_dotenv
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from mcp_client import (
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    mcp_tavily_search,
    weather_mcp_search,
)

load_dotenv()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_groq import ChatGroq
from tools.travily_tool import tavily_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing.")

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")


llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=GROQ_API_KEY)


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str
    llm_calls: int


FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:

        airports = asyncio.run(aviation_mcp_call("list_airports"))

        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )

        flight_data = response.content

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"

    state["flight_results"] = flight_data
    state["messages"] = [AIMessage(content="Flight recommendations generated")]
    state["llm_calls"] = state.get("llm_calls", 0) + 1
    return state


def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = asyncio.run(mcp_tavily_search(query))

    state["hotel_results"] = hotel_results
    state["messages"] = [AIMessage(content="Hotel information fetched.")]
    state["llm_calls"] = state.get("llm_calls", 0) + 1
    return state


def weather_agent(state: TravelState):

    city = extract_destination(state["user_query"])
    weather_data = asyncio.run(weather_mcp_search(city))
    forecast_data = asyncio.run(forecast_mcp_search(city))

    return {
        "weather_results": f"""
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """,
        "messages": [AIMessage(content="Weather information fetched")],
    }


def itinerary_agent(state: TravelState) -> TravelState:
    prompt = f"""
    Create a complete travel itinerary.
    
    User Query:
    {state['user_query']}
    
    Flight Results:
    {state['flight_results']}
    
    Weather Results:
    {state['weather_results']}
    
    Hotel Results:
    {state['hotel_results']}

    
    Make the itinerary practical, budget-aware, and easy to follow.
    """

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner"),
            HumanMessage(content=prompt),
        ]
    )

    state["itinerary"] = response.content
    state["messages"] = [response]
    state["llm_calls"] = state.get("llm_calls", 0) + 1
    return state


def final_agent(state: TravelState) -> TravelState:
    prompt = f"""
        Generate the final travel response for the user.

        User Request:
        {state['user_query']}

        Flights:
        {state['flight_results']}

        Hotels:
        {state['hotel_results']}

        Weather:
        {state['weather_results']}

        Itinerary:
        {state['itinerary']}

        Format the final answer beautifully using these sections:

        1. Trip Summary
        2. Flight Information
        3. Hotel Suggestions
        4. Weather Information
        5. Day-by-Day Itinerary
        6. Estimated Budget
        7. Final Recommendations


        Important:
        - Be clear and practical.
        - Mention that live flight API may not provide ticket prices if pricing is unavailable.
        - Include weather-based travel advice.
        - Keep the response useful for real travel planning.
        """

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=prompt),
        ]
    )

    state["messages"] = [response]
    state["llm_calls"] = state.get("llm_calls", 0) + 1
    return state


graph = StateGraph(TravelState)

graph.add_node("flight", flight_agent)
graph.add_node("hotel", hotel_agent)
graph.add_node("weather", weather_agent)
graph.add_node("itinerary", itinerary_agent)
graph.add_node("final", final_agent)

graph.add_edge(START, "flight")
graph.add_edge("flight", "hotel")
graph.add_edge("hotel", "weather")
graph.add_edge("weather", "itinerary")
graph.add_edge("itinerary", "final")
graph.add_edge("final", END)


DATABASE_URL = get_database_url()

_conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)


def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    state = TravelState(
        user_query=user_input,
        flight_results="",
        hotel_results="",
        weather_results="",
        itinerary="",
        messages=[HumanMessage(content=user_input)],
        llm_calls=0,
    )
    result = app.invoke(state, config=config)

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
