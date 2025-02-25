# Step 1: ollama download
# Step 2: pip install langchain-ollama langchain-community langchain-text-splitters

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Leggi il contenuto del file di testo (solo la prima volta)
with open("/content/drive/MyDrive/Colab Notebooks/provaTesto.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()

# Dividi il testo in chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(knowledge)

# Crea un database vettoriale con gli embeddings
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = Chroma.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Crea la memoria per mantenere il contesto della conversazione
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Crea il modello LLM
llm = OllamaLLM(model="llama3.2:latest")

# Crea una catena di conversazione che può recuperare informazioni pertinenti
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True
)

# Poi puoi fare domande senza ripetere il testo:
def ask(question):
    response = qa_chain.invoke({"question": question})
    return response["answer"]

# Esempio di utilizzo
print(ask("Who is the protagonist?"))
# Puoi continuare a fare domande senza ricaricare il testo
print(ask("What happens next?"))
