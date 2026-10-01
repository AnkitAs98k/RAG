from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Components
model = ChatGroq(model="openai/gpt-oss-120b")
parser = StrOutputParser()

# Two different prompts
short_prompt = ChatPromptTemplate.from_template(
    "Write a 2 line passage on {topic}"
)

long_prompt = ChatPromptTemplate.from_template(
    "Write a 6 line summary on {topic}"
)

# Parallel chaining
chain = RunnableParallel({

    "short":
        RunnableLambda(lambda x: x["short"])
        | short_prompt
        | model
        | parser,

    "long":
        RunnableLambda(lambda x: x["long"])
        | long_prompt
        | model
        | parser
})

# Input
result = chain.invoke({
    "short": {"topic": "Machine Learning"},
    "long": {"topic": "Deep Learning"}
})

print(result)