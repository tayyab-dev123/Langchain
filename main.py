import argparse
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--code", default="print the list of numbers from 1 to 10.")
parser.add_argument("--language", default="Python")
args = parser.parse_args()

# Initialize the ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192",
)

# Create the prompt template using the parsed arguments
code_prompt = PromptTemplate(
    template="Write a {code} in {language}.",
    input_variables=["code", "language"],
)

# Format the prompt with the provided code and language
formatted_prompt = code_prompt.format(code=args.code, language=args.language)

# Invoke the LLM with the formatted prompt
result = llm.invoke(formatted_prompt)
print(result.content)


# echo "# Langchain" >> README.md
# git init
# git add README.md
# git commit -m "first commit"
# git branch -M main
# git remote add origin https://github.com/tayyab-dev123/Langchain.git
# git push -u origin main
