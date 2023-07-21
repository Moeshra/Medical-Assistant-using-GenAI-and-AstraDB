# Medical-Assistant-using-GenAI-and-AstraDB

Medical assistants can leverage GenAI to create medical chatbots, which serve as conversational agents interacting with patients^[5^]. These AI-enabled agents provide instant access to medical advice, answer patients' questions, and offer round-the-clock support. By doing so, they play a crucial role in enhancing patient engagement and satisfaction with healthcare services

This code leverages the LangChain framework to extract information from text documents, transform the text into vector representations, and find semantic similarities between these vector representations for information retrieval and appropriate response generation. The code is set up to connect with Cassandra's AstraDB, load documents, split them, store their vector representations in a vector store, and retrieve the information based on user input.

#You need to install the list of Prerequisites Python packages in the requirements file 

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
To run this code on your local machine, you need the following command 
chainlit run qa_demo.py 

Upon running the code, it will connect to the Cassandra AstraDB cluster, initialize several components (document loaders, text splitters, etc.), and set up a chat environment. You will be asked to upload a text file, attached is a dataset for some illness and sympthoms which you can upload  which will be processed, and its information stored in the vector store.

Now, you can ask questions related to the uploaded text document, and the system will retrieve the most relevant information as an answer.
Example: 
- What is the sympthoms of Asthma ? 
- What are the medecines for Asmthma ? 
- Give me the list of doctors in London
- Give me a the name, address, booking address of a doctor next to Kind's rd street

Advanced Usage
You can customize the code according to your needs. You can change the astra table and keyspace, the OpenAI API key, the text chunk size and overlap in the RecursiveCharacterTextSplitter, and the system prompt message template, among others.
You can also use your own dataset. 
