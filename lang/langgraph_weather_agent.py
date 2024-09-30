import requests
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")

# def get_coordinates(location: str):
#     """Get the longitude and latitude of a location"""
#     prompt = f"""Please provide me the exact coordinates (latitude, longitude) of {location}.
#     Your response should only include the longitude and latitude, with nothing additional.
#     Only Provide exact and correct coordinates. Never provide wrong ones. If the location is not clear to you ask for clarification.
#     Your response should be in the following dictionary format:
#     {{"latitude": <latitude_value>, "longitude": <longitude_value>}}
#     """

#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
#     response = llm.invoke(prompt)
#     return response # Converting string response to dictionary

def get_coordinates(city_name: str):
    """
    This function retrieves the geographical coordinates (latitude and longitude) for a given city name 
    using the OpenCage Geocoding API.

    Parameters:
    city_name (str): The name of the city for which to retrieve the coordinates.
    
    Returns:
    dict: A dictionary containing the latitude and longitude of the specified city. The dictionary will 
    have the following structure:
        {
            "latitude": <latitude_value>,
            "longitude": <longitude_value>
        }
    In case of errors or no results, the function returns a dictionary containing an "error" key with a 
    corresponding message:
        {
            "error": <error_message>
        }
    
    Example:
    >>> get_coordinates("Faisalabad")
    {'latitude': 31.4504, 'longitude': 73.1350}

    API Key:
    This function uses the OpenCage Geocoding API, which requires an API key for usage. The key should 
    be included in the `url` as a query parameter.

    Response Handling:
    - If the API request is successful (status code 200), the function extracts the latitude and longitude 
      from the first result in the JSON response.
    - If no results are found or the request fails, an appropriate error message is returned.
    """
    
    url = f"https://api.opencagedata.com/geocode/v1/json?q={city_name}&key=0c22b1de2c9a4eff8bac2f6a4e9c0863"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            latitude = results[0]["geometry"]["lat"]
            longitude = results[0]["geometry"]["lng"]
            return {"latitude": latitude, "longitude": longitude}
        else:
            return {"error": "No results found for the provided city."}
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

import requests

def get_weather(latitude: str, longitude: str):
    """
    Retrieves the current weather data for a specific location using the OpenWeatherMap API
    based on the provided geographical coordinates (latitude and longitude).

    Parameters:
    latitude (str): The latitude of the location.
    longitude (str): The longitude of the location.

    Returns:
    dict: A dictionary containing current weather information if the API request is successful. 
    The returned dictionary typically includes the following keys:

        - 'coord': Coordinates of the location (longitude and latitude).
            - 'lon' (float): Longitude.
            - 'lat' (float): Latitude.

        - 'weather': List containing weather condition details.
            - 'id' (int): Weather condition ID.
            - 'main' (str): Group of weather parameters (e.g., Rain, Snow).
            - 'description' (str): Detailed description of the weather (e.g., light rain).
            - 'icon' (str): Weather icon code.

        - 'base': Internal parameter used by the API.

        - 'main': Weather measurements.
            - 'temp' (float): Current temperature (in Celsius if metric units are used).
            - 'feels_like' (float): Perceived temperature.
            - 'temp_min' (float): Minimum temperature.
            - 'temp_max' (float): Maximum temperature.
            - 'pressure' (int): Atmospheric pressure at sea level (in hPa).
            - 'humidity' (int): Humidity percentage.
            - 'sea_level' (int, optional): Atmospheric pressure at sea level (if available).
            - 'grnd_level' (int, optional): Atmospheric pressure at ground level (if available).

        - 'visibility' (int): Visibility distance in meters.

        - 'wind': Wind data.
            - 'speed' (float): Wind speed in meters/second.
            - 'deg' (int): Wind direction in degrees.

        - 'rain': Rain volume (if applicable).
            - '1h' (float): Rain volume for the last 1 hour (in mm).

        - 'clouds': Cloudiness data.
            - 'all' (int): Cloudiness percentage.

        - 'dt' (int): Time of data calculation (UNIX timestamp).

        - 'sys': Additional system data.
            - 'type' (int): Internal parameter.
            - 'id' (int): Internal ID.
            - 'country' (str): Country code (ISO 3166-1 alpha-2).
            - 'sunrise' (int): Sunrise time (UNIX timestamp).
            - 'sunset' (int): Sunset time (UNIX timestamp).

        - 'timezone' (int): Shift in seconds from UTC.
        
        - 'id': Location ID.

        - 'name' (str): Location name.

        - 'cod' (int): Status code of the API response.

    If the request fails, a dictionary with an error message is returned:
        {
            "error": "Unable to fetch weather data. Error code: <status_code>"
        }

    Example:
    >>> get_weather("51.5072", "-0.1276")
    {
        "coord": {"lon": -0.1276, "lat": 51.5072},
        "weather": [
            {
                "id": 500,
                "main": "Rain",
                "description": "light rain",
                "icon": "10d"
            }
        ],
        "base": "stations",
        "main": {
            "temp": 17.55,
            "feels_like": 17.63,
            "temp_min": 15.99,
            "temp_max": 18.23,
            "pressure": 1003,
            "humidity": 87,
            "sea_level": 1003,
            "grnd_level": 999
        },
        "visibility": 10000,
        "wind": {"speed": 3.6, "deg": 200},
        "rain": {"1h": 0.12},
        "clouds": {"all": 75},
        "dt": 1727084244,
        "sys": {
            "type": 2,
            "id": 2011528,
            "country": "GB",
            "sunrise": 1727070458,
            "sunset": 1727114154
        },
        "timezone": 3600,
        "id": 7302135,
        "name": "Abbey Wood",
        "cod": 200
    }
    """

    # Calling the weather API with the provided latitude and longitude
    response = requests.get(f"https://api.openweathermap.org/data/3.0/onecall?lat={latitude}&lon={longitude}&exclude=current&appid=de21f28aa1a816f1f11e0a17458bb719&units=metric")

    if response.status_code == 200:
        return response.json()  # Return weather data if the call succeeds
    else:
        return {"error": f"Unable to fetch weather data. Error code: {response.status_code}"}

sys_msg = SystemMessage(content="""You are a weather-helpful assistant tasked with helping the user with weather-related queries. You must provide users with detailed responses, including all weather-related parameters like temperature, humidity, wind speed, visibility, and other available data. You have access to two tools to assist in fulfilling this task:

1. **get_coordinates**: This tool helps convert location names (like cities or regions) into geographical coordinates (latitude and longitude). You can use this tool to find the precise coordinates for a given location, which can then be passed to the get_weather tool to get the corresponding weather data. The tool uses services like OpenCageData to retrieve accurate geographical details based on input queries.

2. **get_weather**: This tool allows you to retrieve current and forecasted weather data based on specific geographical coordinates (latitude and longitude). The weather data includes parameters such as:
   - Current temperature
   - Feels-like temperature
   - Minimum and maximum temperatures
   - Atmospheric pressure
   - Humidity percentage
   - Wind speed and direction
   - Cloud coverage
   - Rainfall information (if applicable)
   - Sunrise and sunset times
   - Visibility
   - Location information (city, country)

Use these tools together to efficiently assist users with their weather-related queries.")
""")

# Define the LLM with tools
tools = [get_weather, get_coordinates]
llm_with_tools = llm.bind_tools(tools)

# Node definition
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "123"}}
while True:
  inp = input("Enter:")
  messages = [HumanMessage(content=f"{inp}")]
  messages = react_graph_memory.invoke({"messages": messages}, config)
  print("Message>>>", messages)
  for m in messages['messages']:
      m.pretty_print()