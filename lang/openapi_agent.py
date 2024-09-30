#This is a complete prototype using which we can create an agent which can call apis using openapi specification of our custom api. You have to provide the url of the openapi.json specification and it will pass it to the agent

# NOTE: set allow_dangerous_requests manually for security concern https://python.langchain.com/docs/security
import os
import yaml
from langchain.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi import planner
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
#to download the specs
import subprocess
import os
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
# Define the URL and output file: This code is to download the openapi.yaml file which will be used by the agent
output_file = "openapi.yaml"
url = "https://your-openapi-spec-url/openapi.json"

# Check if the file already exists
if os.path.isfile(output_file):
    print(f"{output_file} already exists. Skipping download.")
else:
    # Run the curl command to download the OpenAPI spec
    try:
        subprocess.run(["curl", "-o", output_file, url], check=True)
        print(f"Successfully downloaded the OpenAPI spec to {output_file}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading: {e}")


# To reduce the openapi specs
with open("openapi.yaml") as f:
    raw_api_spec = yaml.load(f, Loader=yaml.Loader)
openai_api_spec = reduce_openapi_spec(raw_api_spec)
ALLOW_DANGEROUS_REQUEST = True
requests_wrapper = RequestsWrapper()
todo_agent = planner.create_openapi_agent(
    openai_api_spec,
    requests_wrapper,
    llm=llm,
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
)
user_query = (
    "i want to read all the todos I have added."
)
response = todo_agent.invoke(user_query)
print("RESPONSE>>>>>>", response)