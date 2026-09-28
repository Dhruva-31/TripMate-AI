from backend import run_travel_agent
from tools.travily_tool import tavily_search
from tools.flight_tool import search_flights

# res = tavily_search("Best hotels in India")
# print(res)

# res = search_flights("Plan a 7 days Japan trip from Bangladesh")
# print(res)

response = run_travel_agent(
    "Plan a complete 7 days India trip from Bangladesh including flights, hotels and sightseeing under 2 lakhs",
    thread_id="test_user",
)

print(response["answer"])
