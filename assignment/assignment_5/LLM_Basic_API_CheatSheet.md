# LangChain Basic API Call Cheat Sheet

This cheat sheet summarizes how to initialize and use various LLM providers (OpenAI, Google Gemini, Groq, NVIDIA) using LangChain, based on the `7_1_LLM_Basic_API_Call_LangChain.ipynb` notebook.

## 1. Common Setup (API Keys)
Most integrations require setting an environment variable or passing the key directly.
```python
import getpass
import os

# Example for setting environment variable safely
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key: ")
```

## 2. OpenAI (`langchain-openai`)
**Installation:**
```bash
!pip install -qU langchain-openai
```
**Initialization:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini", # or "gpt-3.5-turbo", etc.
    temperature=0,
    # api_key="..." # Optional if env var is set
)
```

## 3. Google Gemini (`langchain-google-genai`)
**Installation:**
```bash
!pip install -qU langchain-google-genai
```
**Initialization:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "..." # Set API Key

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)
```
*Note: Free tier rate limits may apply (approx. 20 RPD for some models).*

## 4. Groq (`langchain-groq`)
*Free-tier friendly alternative for high-speed inference.*

**Installation:**
```bash
!pip install -qU langchain-groq
```
**Initialization:**
```python
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "..." # Set API Key

llm = ChatGroq(
    model="llama-3.1-8b-instant", # Check Groq console for available models
    temperature=0
)
```

## 5. NVIDIA (`langchain-nvidia-ai-endpoints`)
**Installation:**
```bash
!pip install -qU langchain_nvidia_ai_endpoints
```
**Initialization:**
```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

os.environ["NVIDIA_API_KEY"] = "..." # Set API Key (starts with nvapi-)

# Check available models
# ChatNVIDIA.get_available_models()

llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
```

## 6. Basic Usage (Invoke)
The pattern is consistent across all `Chat` models.

```python
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "Translate 'I love programming' to French."),
]

# Invoke the model
response = llm.invoke(messages)

# Access content
print(response.content)
```
