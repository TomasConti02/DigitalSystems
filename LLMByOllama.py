#step 1 ollama download 
#step 2 install API -> pip install langchain-ollama
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Leggi il contenuto del file di testo
with open("/content/drive/MyDrive/Colab Notebooks/provaTesto.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()

# Definizione del prompt con il contesto estratto dal file
template = """Use the following knowledge to answer the question:

{knowledge}

Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.2:latest")  # Assicurati che il nome del modello sia corretto

chain = prompt | model

response = chain.invoke({"knowledge": knowledge, "question": "who is the protagonist?"})
print(response)
