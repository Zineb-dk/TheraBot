Second Method
=============

In this page, we will be explaining how we retrieved data from a PDF file using Llama_Parse.



Parsing, Embedding, and Querying
================================

This part provides a comprehensive guide to understanding and working with the ``rag.py`` script. The script uses **LlamaParse**, **LangChain**, and **Chroma** to parse a psychology PDF, create embeddings, and store data in a vector database for similarity-based retrieval.


Overview
--------

The script performs the following tasks:

1. **Parse a PDF file**: Extracts text and saves it into a Markdown file.
2. **Text splitting**: Splits the extracted text into manageable chunks for processing.
3. **Embeddings**: Converts the text chunks into numerical representations for storage and querying.
4. **Vector database setup**: Stores the embeddings in a Chroma vector database for similarity search.
5. **Query the database**: Retrieves relevant information from the database based on similarity.


Environment Configuration
-------------------------

Set up the Llama Cloud API key as an environment variable:

.. code-block:: python

   import os
   os.environ["LLAMA_CLOUD_API_KEY"] = "API_KEY"

Replace ``API_KEY`` with your Llama Cloud API key.

Step-by-Step Explanation
------------------------

1. **Import Libraries**

   Import the required modules for parsing, text splitting, embeddings, and vector database operations:

   .. code-block:: python

      from llama_parse import LlamaParse
      from llama_parse.base import ResultType, Language
      from langchain.text_splitter import RecursiveCharacterTextSplitter
      from langchain.vectorstores import Chroma
      from langchain_community.embeddings.ollama import OllamaEmbeddings
      from langchain_core.documents import Document

2. **Define the Parser**

   Configure the parser to extract data from the PDF file:

   .. code-block:: python

      parser = LlamaParse(result_type=ResultType.MD, language=Language.ENGLISH)

3. **Parse the PDF**

   Load the text from the PDF and save it to a Markdown file:

   .. code-block:: python

      documents = parser.load_data("PsychologyKeyConcepts.pdf")

      # Save to a file
      filename = "psychology_data.md"
      with open(filename, 'w') as f:
          f.write(documents[0].text)

4. **Load and Split Text**

   Load the text from the Markdown file and split it into smaller chunks:

   .. code-block:: python

      with open("psychology_data.md", encoding='utf-8') as f:
          doc = f.read()

      r_splitter = RecursiveCharacterTextSplitter(
          chunk_size=2000,
          chunk_overlap=0,
          separators=["\n\n", "\n", "(?<=\. )", " ", ""]
      )
      docs = r_splitter.split_text(doc)
      docs = [Document(page_content=d) for d in docs]

      print("Text has been split.")

5. **Create Embeddings**

   Use the `OllamaEmbeddings` model to create embeddings for the text chunks:

   .. code-block:: python

      embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
      print("Embeddings created.")

6. **Set Up the Vector Database**

   Define and populate the Chroma vector database:

   .. code-block:: python

      persist_directory = "Psycho_db"

      vecdb = Chroma(
          persist_directory=persist_directory,
          embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
          collection_name="rag-chroma"
      )

      vecdb.add_documents(docs)
      vecdb.persist()

      print("Data has been ingested into the vector database.")

7. **Query the Database**

   Perform a similarity search on the database:

   .. code-block:: python

      question = "What is depression?"
      documents = vecdb.similarity_search(question, k=5)

      print(documents[0].page_content)

Outputs
-------

- **Parsed Data**: The text is saved to a Markdown file named ``psychology_data.md``.
- **Vector Database**: The embeddings are stored in a Chroma database at ``Psycho_db``.
- **Search Results**: Queries retrieve the most relevant document chunks from the database.


Therapy Chatbot: Interactive Mental Health Support
===================================================

This section of the script creates a **therapy chatbot** using **Streamlit**, **Ollama**, and **Chroma**. The chatbot is designed to support mental health conversations by retrieving contextually relevant information from a database and engaging users in meaningful interactions.

Overview
--------

The chatbot is built with the following features:

- **Text and Speech Input**: Accepts user messages via text input or audio recording.
- **Role-Specific Responses**: Provides empathetic and supportive replies tailored to mental health topics.
- **Retrieval-Augmented Generation (RAG)**: Combines user queries with relevant data from the Chroma vector database for informed responses.
- **Session State**: Maintains a history of the conversation for context continuity.



Key Components
--------------
^^^^^^^^^^^^^^^^^^^^^^^^^^
Initialize Chroma Database
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Chroma vector database is loaded with embedded documents to enable similarity-based retrieval:

.. code-block:: python

   persist_directory = "rag/Psycho_db"
   vecdb = Chroma(
       persist_directory=persist_directory,
       embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
       collection_name="rag-chroma"
   )
^^^^^^^^^^^^^^^
Retrieval Logic
^^^^^^^^^^^^^^^

The `retrieve_from_db` function retrieves relevant documents from the Chroma database based on the user's query:

.. code-block:: python

   def retrieve_from_db(question):
       model = OllamaLLM(model="llama3.2")
       retriever = vecdb.as_retriever()
       retrieved_docs = retriever.invoke(question)
       return retrieved_docs[1].page_content

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 Chatbot Response Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `generate_response` function creates a reply to the user's query. It defines the chatbot's role, ensuring responses are empathetic and contextually relevant:

.. code-block:: python

   def generate_response(user_message: str, chat_history: list = [], doc=""):
       system_msg = (
           """You are a Chatbot for mental health support, don't overtalk. When the users are trying to harm themselves, remind them that they're loved by someone.
           When asked about someone (celebrity for example) say "sorry, I don't wanna talk about other people". Stick to the context of mental health. 
           If the situation is serious refer to Moroccan health services. Combine what you know and verify it using the Relevant Documents : {document}
           Question: {question}. Don't say "Based on the provided context". If there is no answer, say "I'm sorry, the context is not enough to answer the question." """
       )
       my_message = [{"role": "system", "content": system_msg, "document": doc}]
       for chat in chat_history:
           my_message.append({"role": chat["name"], "content": chat["msg"]})
       my_message.append({"role": "user", "content": user_message, "document": doc})

       response = ollama.chat(
           model="llama3.2",
           messages=my_message
       )
       return response["message"]["content"]

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 Streamlit UI and Interaction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The chatbot interface is implemented using Streamlit:

1. **Chat History**: Displays the history of user and chatbot interactions.
2. **Input Methods**:
   - Text input: Users can type messages in a text box.
   - Audio input: Users can record their voice, which is transcribed into text.
3. **Real-Time Responses**: The chatbot processes the input and displays a response.

^^^^^^^^^^^^^
Main Function
^^^^^^^^^^^^^

The `main` function initializes the chatbot interface and handles user inputs:

.. code-block:: python

   def main():
       if "chat_log" not in st.session_state:
           st.session_state.chat_log = []

       for chat in st.session_state.chat_log:
           with st.chat_message(chat["name"]):
               st.write(chat["msg"])

       input_container = st.empty()

       with input_container:
           col1, col2 = st.columns([4, 1])

           with col1:
               user_message = st.chat_input("What is up?", key="user_input")
           with col2:
               record_audio = st.button("🎙️")

       if user_message:
           with st.chat_message("user"):
               st.write(user_message)
           doc = retrieve_from_db(user_message)
           response = generate_response(user_message, chat_history=st.session_state.chat_log, doc=doc)

           if response:
               with st.chat_message("assistant"):
                   st.write(response)

               st.session_state.chat_log.append({"name": "user", "msg": user_message})
               st.session_state.chat_log.append({"name": "assistant", "msg": response})

       elif record_audio:
           r = sr.Recognizer()
           with sr.Microphone() as source:
               st.write("Talk...")
               audio_text = r.listen(source)
               try:
                   user_message = r.recognize_google(audio_text)
                   with st.chat_message("user"):
                       st.write(user_message)
                   doc = retrieve_from_db(user_message)
                   response = generate_response(user_message, chat_history=st.session_state.chat_log, doc=doc)

                   if response:
                       with st.chat_message("assistant"):
                           st.write(response)

                       st.session_state.chat_log.append({"name": "user", "msg": user_message})
                       st.session_state.chat_log.append({"name": "assistant", "msg": response})
               except:
                   st.write("Sorry, I did not get that.")

       if __name__ == "__main__":
           main()

Outputs
-------

- **Interactive Chat Interface**: Provides real-time interactions with users.
- **Mental Health Support**: Tailored responses based on user queries.
- **Document-Aided Replies**: Incorporates data from the Chroma database to provide relevant answers.

General Notes
--------------

- Ensure the input PDF is in the same directory as the script.
- Customize the ``chunk_size`` and ``chunk_overlap`` parameters to suit your needs.
- Use the correct model version for embeddings (e.g., ``mxbai-embed-large:latest``).
- Ensure the Chroma database is initialized with the appropriate data.
- Configure the API keys and microphone permissions correctly for full functionality.
