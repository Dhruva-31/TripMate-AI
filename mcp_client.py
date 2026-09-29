import os
from pathlib import Path
import sys

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AVIATION_ENV = os.environ.copy()
WEATHER_ENV = os.environ.copy()
AVIATION_ENV["AVIATION_STACK_API_KEY"] = AVIATION_STACK_API_KEY or ""
WEATHER_ENV["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY or ""

PROJECT_DIR = Path(__file__).resolve().parent
WEATHER_SERVER_PATH = PROJECT_DIR / "custom_weather_mcp_server.py"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=GROQ_API_KEY)

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": ("https://mcp.tavily.com/mcp/" f"?tavilyApiKey={TAVILY_API_KEY}"),
        },
        "aviationstack": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["aviationstack-mcp"],
            "env": AVIATION_ENV,
        },
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(WEATHER_SERVER_PATH)],
            "env": WEATHER_ENV,
        },
    }
)


async def get_search_tool():
    search_tool = None

    tools = await client.get_tools(server_name="tavily")
    for tool in tools:
        if tool.name == "tavily_search":
            search_tool = tool

        print(tool.name)

    print(search_tool.name)
    if search_tool is None:
        raise RuntimeError(
            "Tavily MCP connected, but the 'tavily_search' tool was not found. "
        )
    return search_tool


async def mcp_tavily_search(query: str):
    search_tool = await get_search_tool()
    response = await search_tool.ainvoke({"query": query})
    return response


async def get_aviation_tool():
    aviation_tools = {}

    tools = await client.get_tools(server_name="aviationstack")
    for tool in tools:
        aviation_tools[tool.name] = tool
        print(tool.name)

    if not aviation_tools:
        raise RuntimeError("AviationStack MCP connected but " "returned no tools.")

    return aviation_tools


async def aviation_mcp_call(tool_name: str, tool_args: dict = None):
    aviation_tools = await get_aviation_tool()
    tool = aviation_tools[tool_name]
    if tool is None:
        raise ValueError(f"AviationStack tool '{tool_name}' " "was not found. ")

    response = await tool.ainvoke(tool_args or {})
    return response


async def get_weather_tool():
    weather_tool = None

    tools = await client.get_tools(server_name="weather")
    for tool in tools:
        if tool.name == "get_current_weather":
            weather_tool = tool

        print(tool.name)

    print(weather_tool.name)
    if weather_tool is None:
        raise RuntimeError(
            "Tavily MCP connected, but the 'tavily_search' tool was not found. "
        )
    return weather_tool


async def get_forcast_tool():
    forcast_tool = None

    tools = await client.get_tools(server_name="weather")
    for tool in tools:
        if tool.name == "get_forecast":
            forcast_tool = tool

        print(tool.name)

    print(forcast_tool.name)
    if forcast_tool is None:
        raise RuntimeError(
            "Tavily MCP connected, but the 'tavily_search' tool was not found. "
        )
    return forcast_tool


async def weather_mcp_search(city: str):
    weather_tool = await get_weather_tool()
    result = await weather_tool.ainvoke({"city": city})
    return result


async def forecast_mcp_search(city: str):
    forecast_tool = await get_forcast_tool()
    result = await forecast_tool.ainvoke({"city": city})
    return result


def extract_destination(query: str):
    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """
    response = llm.invoke(prompt)
    return response.content.strip()
