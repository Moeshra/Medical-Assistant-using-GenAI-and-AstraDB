# We will start by importing the list of mudule, Each module plays a specific role in the process of extracting information from text, 
# transforming this text into vector representations, and finding semantic similarities among these vector representations for information 
# retrieval and appropriate response generation.

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import Cassandra
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl


'''
    Cassandra AstraDB part
'''

# Create an AstraDB vectore store, create a keyspace called qa_docs and a table called qa_table, generate an admin tokens and download 
# the secure-connect bundle and add its path to the following code. Also, don't forget to replace the ASTRA_DB_Client_ID and Astra_DB_CLIENT_SECRET
# with your tokens credentienls. Now we will be connecting to AstraDB cluster.  

keyspace_name = "qa_docs"
table_name = "qa_table"

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config= {
  'secure_connect_bundle': 'ASTRA_DB_SECURE_BUNDLE_PATH'
}
auth_provider = PlainTextAuthProvider('ASTRA_DB_CLIENT_ID', 'ASTRA_DB_CLIENT_SECRET')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
#session = cluster.connect()

#Generate an openAI key and add it the following variable
os.environ["OPENAI_API_KEY"] = "Add your OpenAI AI key here"

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100): This line is initializing an object text_splitter using 
#the RecursiveCharacterTextSplitter class. The RecursiveCharacterTextSplitter is a class that is used to split a long text string into smaller chunks of a specified size.
# In this case, you're setting chunk_size=1000, which means the text will be split into chunks of 1000 characters each. You're also setting chunk_overlap=100, which indicates that each chunk will overlap with the next by 100 characters. This overlap can be useful in ensuring that the context is maintained between chunks, especially when these chunks are processed separately.
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# System prompt message template
system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
...
Begin!
----------------
{summaries}"""

# Creating a list of system and human message prompts
messages = [
    SystemMessagePromptTemplate.from_template(system_template),  # System message prompt
    HumanMessagePromptTemplate.from_template("{question}"),  # Human message prompt
]

# Creating a chat prompt template from the above messages
prompt = ChatPromptTemplate.from_messages(messages)

# A dictionary to hold the prompt
chain_type_kwargs = {"prompt": prompt}


@cl.langchain_factory(use_async=True)
async def init():  # This function initializes the language model and sets up necessary variables and components
    files = None

    # A loop that waits for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["text/plain"]
        ).send()

    file = files[0]

    # Message to notify the user that file processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # File loading and decoding
    loader = TextLoader(file.path)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)  # Splitting the documents into chunks

    # Creating metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Creating embeddings for texts using OpenAI's API
    embeddings = OpenAIEmbeddings()

    # Creating a vector store for storing and retrieving embeddings
    session = cluster.connect()
    db = Cassandra.from_documents(
        documents=texts,
        embedding=embeddings,
        session=session,
        keyspace=keyspace_name,
        table_name=table_name,
        metadatas=metadatas)

    # Creating a chat chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        #ChatOpenAI(temperature=0),
        llm=OpenAI(),
        chain_type="stuff",
        retriever = db.as_retriever(),
        return_source_documents=True
    )

    # Saving the metadata and texts in the user session for later use
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Message to notify the user that the system is ready
    await msg.update(content=f"`{file.name}` processed. You can now ask questions!")

    # Returning the chat chain
    return chain


@cl.langchain_postprocess
async def process_response(res):  # This function post-processes the response from the language model
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Retrieving the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Loop to add sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += ""

    # Sending the response message
    await cl.Message(content=answer, elements=source_elements).send()
