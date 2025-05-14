RAG : Retrieving data using Web Scraping with Firecrawl
=======================================================

To prepare data for retreival, we created a create_db function that returns as an output a retreiver .
This retriever is used to retreive revelant text from the FAISS vector database, it is built by scraping content from multiple URLs using the FireCrawlLoader and then splitting the content into smaller chunks.
These chunks are then embedded using a HuggingFace transformer model to create vector
representations which are stored in the FAISS database. The documents or chunks are
then stored on the database locally and can be retrieved based on similarity with a given
query.

API Environment Configuration
-----------------------------
First, we need to configure the environment by getting the needed API addresses to run the code properly. 

Visit the official LangChain  and Firecrawl websites to get the API endpoints and acquire your API keys.

You will need to register or log in to get these details.

   .. code-block:: python

      os.environ['LANGCHAIN_TRACING_V2']='true'
      os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
      os.environ['LANGCHAIN_API_KEY']="YOUR_API_KEY"
      os.environ['LANGCHAIN_PROJECT']="therabot"
      
      FireCrawl_API = "YOUR_API_KEY"

Database Creation
=================

The database was created using the collected data from the provided URLs, these links contain reliable informations and documents about mental health.

   .. code-block:: python

      urls = [
          "https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/cognitive-behaviour-therapy",
          "https://www.mentalhealth.org.uk/explore-mental-health/publications/how-manage-and-reduce-stress",
          "https://www.who.int/news-room/fact-sheets/detail/anxiety-disorders",
          "https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
          "https://www.who.int/news-room/fact-sheets"
      ]

Scraping and Storing Website Content with FireCrawlLoader
---------------------------------------------------------

The `FireCrawlLoader` tool is used to scrape each URL for content, and then the extracted text from each website will be stored as an object in the **docs** list.

   .. code-block:: python

      docs = [FireCrawlLoader(api_key=FireCrawl_API,url = url,mode="scrape").load() for url in urls]
      docs_list = [item for sublist in docs for item in sublist]

Splitting Text into Chunks
--------------------------

The extracted content is a vast amount of unstructured text data. To manage this large text efficiently, the content is split into smaller chunks using **RecursiveCharacterTextSplitter**. These chunks make it easier to search for and retrieve specific pieces of information, boosting the accuracy of information retrieval tasks.

   .. code-block:: python

      text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=50)
      doc_splits = text_splitter.split_documents(docs_list)

The **overlap** argument is used to avoid the risk of losing context.
If chunks are created without overlap, the model might lose key contextual informations between adjacent segments, reducing its ability to understand the complete context.

Filtering Metadata
------------------

Metadata is cleaned by iterating through a list of documents, checking for valid **Document** objects, and then filtering the metadata to only include values of specific types (str, int, float, bool).

   .. code-block:: python

      cleaned_docs = []

      for doc in doc_splits:
          if isinstance(doc, Document) and hasattr(doc, 'metadata'):
              clean_metadat = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
              cleaned_docs.append(Document(page_content=doc.page_content, metadata=clean_metadat))


Creating Embeddings
-------------------

Embeddings are created using the Hugging Face `sentence-transformers/all-MiniLM-L6-v2` model.

   .. code-block:: python

      embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


Creating the FAISS Vector Database
----------------------------------

The cleaned documents and their embeddings are then stored in a **FAISS vector store**.

   .. code-block:: python

      db = FAISS.from_documents(documents=cleaned_docs, embedding=embeddings)
      db.save_local(DB_FAISS_PATH)
      retriever = db.as_retriever()


Generating Chatbot Output
=========================

The chatbot generates responses using LLaMa3.1 model. 
The `GenerateResponse` class was implemented  to handle the response generation process.

Prompt Template
---------------

A prompt template is created to shape the chatbot's responses, ensuring empathy and relevance, and to define the tone, the style and the constraints for generating responses.

   .. code-block:: python

      self.prompt_template = """
      You are a therapist, and your primary goal is to offer support, understanding, and guidance...
      Relevant Documents : {document}
      Question: {question}
      Answer:
      """


Check if RAG is necessary to generate an accurate response
----------------------------------------------------------

Before generating a response, the chatbot evaluates the user query to decide if external documents are necessary to answer properly, using the check_need_for_rag function.

This function uses the predefined logic in the **rag_check_prompt** , this prompt will be combined with the user query and passed to the model.

   .. code-block:: python

      self.rag_check_prompt = """
      You are a highly intelligent assistant designed to decide whether a query requires additional information from external sources...
      Query: "{query}"
      Needs External Information (True/False):
      """

The model evaluates the query to determine if RAG is necessary, based on whether the query requires additional context, such as scientific information or other detailed data.
If additional information is needed, the model responds with "true", and the function returns True, and False otherwise.
However, if there are issues during this process, the function returns False by default.

   .. code-block:: python

      def check_need_for_rag(self, user_query):
          try:
              rag_check_input = self.rag_check_prompt.format(query=user_query)
              response = self.model.invoke({"question": rag_check_input})
              return response.strip().lower() == "true"
          except Exception as e:
              print(f"Error checking for RAG need: {str(e)}")
              return False
      """

Retrieve Documents
------------------

After checking if RAG is necessary . If it is required , the **retreive()** method returns the retreived document's text from the FAISS vector store.
The FAISS vector store compares the query embedding with the stored document embeddings using a similarity search, and returns the documents with the highest similarity scores.

   .. code-block:: python

      def retreive(self, user_query):
          retriever = create_db()
          retreived_docs = retriever.invoke(user_query)
          retreived_docs_txt = retreived_docs[1].page_content
          return retreived_docs_txt

Generate Response
-----------------

The **generate_answer()** method uses the predefined prompt template, the retrieved documents, and the chat history to generate a response using the **llama3.1** model via **ollama.chat**.

   .. code-block:: python

      def generate_answer(self, user_query, chat_history: list=[]):
          try:
              # Check if external information is needed
              needs_rag = self.check_need_for_rag(user_query)
              if needs_rag:
                  retrieved_docs_txt = self.retreive(user_query)
              else:
                  retrieved_docs_txt = ""

              # Create input for the model
              my_message = [
                  {"role": "system", "content": self.prompt_template, "document": retrieved_docs_txt}
              ]

              # Add previous chat history
              for chat in chat_history:
                  my_message.append({"role": chat["role"], "content": chat["content"]})

              # Append the current user query
              my_message.append({"role": "user", "content": user_query, "document": retrieved_docs_txt})

              # Call the model to generate the response
              generated_answer = ollama.chat(
                  model="llama3.1",
                  messages=my_message
              )

              # Save the conversation to the chat history
              self.log_chat(user_query, generated_answer)
              return generated_answer["message"]["content"]
          except Exception as e:
              error_message = f"An error occurred: {str(e)}"
              return error_message



Chat History
------------

Finally, the conversation between the user and the model is logged to maintain a record of user queries and assistant responses, ensuring that the context is preserved.

   .. code-block:: python

      def log_chat(self, user_query, response):
          chat = {"user": user_query, "assistant": response}
          self.chat_history.append(chat)








