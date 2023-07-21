# Medical-Assistant-using-GenAI-and-AstraDB

The code leverages the LangChain framework to extract information from text documents, transform the text into vector representations, and find semantic similarities between these vector representations for information retrieval and appropriate response generation. The code is set up to connect with Cassandra's AstraDB, load documents, split them, store their vector representations in a vector store, and retrieve the information based on user input.

Prerequisites
To run this code, you'll need:
Python 3.x installed on your system.
LangChain and Cassandra modules installed in your Python environment.
Cassandra AstraDB setup with a keyspace called qa_docs and a table called qa_table.
Secure-connect bundle for Cassandra AstraDB downloaded and its path available.
AstraDB admin tokens generated and their credentials available.
OpenAI API key.

Configuration
Firstly, replace the placeholders ASTRA_DB_SECURE_BUNDLE_PATH, ASTRA_DB_CLIENT_ID, and ASTRA_DB_CLIENT_SECRET with your actual secure-connect bundle path, AstraDB client ID, and AstraDB client secret, respectively.

Then, replace Add your OpenAI AI key here with your actual OpenAI API key.

Running the Code
Upon running the code, it will connect to the Cassandra AstraDB cluster, initialize several components (document loaders, text splitters, etc.), and set up a chat environment. You will be asked to upload a text file, which will be processed, and its information stored in the vector store.

After the setup, you can ask questions related to the uploaded text document, and the system will retrieve the most relevant information as an answer.

Advanced Usage
You can customize the code according to your needs. You can change the Cassandra table and keyspace, the OpenAI API key, the text chunk size and overlap in the RecursiveCharacterTextSplitter, and the system prompt message template, among others.
