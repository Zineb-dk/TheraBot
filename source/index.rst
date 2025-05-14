.. TheraBot documentation master file, created by
   sphinx-quickstart on Fri Dec  6 19:47:13 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TheraBot
========


This project is made by **Mouad BOULAID** & **Zineb DKIER**.

Supervised by: **Dr. Tawfik MASROUR**

This documentation explains the process of creating a therapeutic chatbot using an LLM with detailed explanations of each step. Links to additional resources and documentation are included for further reference.

Table of contents
-----------------
- `Introduction <index.html#id3>`_
- `Features <index.html#id4>`_
- `Requirements <index.html#id5>`_
- `Pipeline <index.html#id6>`_
- `Select The Text Generation Model <index.html#id7>`_
- `Database Creation <index.html#id8>`_
   - `Database Creation Using Web Scraping with Firecrawl <index.html#id9>`_
   - `Database Creation Using PDF Parsing with Llama Parse <index.html#id10>`_
- `RAG Implementation and Text Generation <index.html#id11>`_
   - `Using Web Scraping <index.html#id12>`_
   - `Using PDFs <index.html#id13>`_
- `Speech to Text <index.html#id14>`_
- `UI Development (TheraBot Interface) <index.html#id15>`_
   - `Using Web Scraping <index.html#id16>`_
   - `Using PDFs <index.html#id17>`_

Introduction
============
AI in mental health is not a new topic for discussion. In the 1960s,the first chatbot, ELIZA,
was created by Joseph Weizenbaum at MIT.It operated using a script called DOCTOR,
which allowed it to respond like a psychotherapist. But despite of it’s simplicity it was
one of the first attempts to use and define chatbots as a resource for mental health.
Since then, the evolution had led to the development of chatbots that are much smarter
and empathetic making them able to hold meaningful conversations.One such chatbot is
Therabot, designed to provide meantal health support to users to help them in their daily
life.
Such mental health issues like anxiety, depression, stress and many others have recently
became more common in our society. However, many people still struggle as they search
for assistance, for instance they may be busy, therapies can be expensive, or they just feel
ashamed. Some of these may be solved by a therapeutic chatbot by providing confidential
and cheap service when compared to other therapeutic solutions.
One of the key benefits of a therapeutic chatbot is that it can be available anytime and
anywhere at the press of a button. This makes it a good choice for individuals who have
no time or cannot afford to attend therapy sessions or those who prefer to do a self-help
program in the comfort of their rooms away from people’s interference... And since the
chatbot is anonymous, users might feel more comfortable opening up about their thoughts
and emotions, and they are more likely to disclose more about their feelings.
Although, therapeutic chatbot is not a replacement for professional therapy, it can
be a helpful tool for those who need immediate support or as an addition to clinical
therapy. And as the technology continues to improve, these chatbots will definitely be
more efficient in identifying and responding to human emotions, a plus to mental health
care.

Features
========
Before we dive into building the chatbot, let’s break down its main components. Think of these as the building blocks that make the chatbot work.

First, there’s answer generation. This is the core of the chatbot, where a machine learning model (in our case, Llama 3.1) generates responses based on user input. The model runs locally on your computer, thanks to a tool called Ollama, which handles the heavy lifting.

Next, we have the knowledge base. This is essentially a database of verified information that the chatbot can reference to ensure its answers are accurate. For example, if a user asks about coping strategies for anxiety, the chatbot can pull information from trusted sources like therapy websites or mental health guides.

Finally, there are additional features that make the chatbot more user-friendly and accessible. For instance, a speech-to-text feature allows users to speak instead of typing, which can be especially helpful for those who find typing difficult or prefer verbal communication. A clean and intuitive interface also makes the chatbot easier to use.

Requirements
============
The implementation of this project requires the following:

**• Programming Language**
- `Python`

**• Libraries and Frameworks**

- `LangChain:`
   - `langchain`
   - `langchain-ollama`
   - `langchain_community`
   - `langchain_core`

- `ollama`
- `speechRecognition`
- `streamlit`
- `chromadb`
- `llama_parse`
- `os`

Pipeline
========
The pipeline for our therapeutic chatbot involves the following stages :

1. **Model Selection**: We chose the LLaMa3.1 model from Ollama's LLMs for response generation.
2. **Database Preparation**: Collection of relevant mental health resources through web scraping and PDF parsing.
3. **RAG Implementation**: Integration of Chroma-based and FAISS-based retrievers for fetching revelant documents from the prepared data.
4. **Voice-to-Text**: The voice input will be processed and converted to text using Google's SpeechRecognition python module.
5. **UI Development**: We deployed the chatbot using Streamlit


Select The Text Generation Model
================================

The first step in building the chatbot is to load the Llama 3.1 model locally on your computer. But why use Llama 3.1 and Ollama?

Llama 3.1 is a powerful large language model that excels at generating human-like responses. It’s perfect for creating conversational chatbots. Ollama, on the other hand, is a tool that lets you run Llama models locally. This means your chatbot won’t rely on external servers or APIs, ensuring privacy and independence.

Hardware Requirements
---------------------

Before proceeding, make sure your computer meets the hardware requirements for running large language models. Refer to the Ollama GitHub page for detailed specifications. : https://github.com/ollama/ollama

Install Ollama
--------------

Visit the Ollama website and download the installation package for your operating system (Windows, macOS, or Linux): 
https://ollama.com

Follow the instructions to install it on your computer.

With Ollama installed, you’ll need to download the Llama 3.1 model. Open your terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run the following command:

   .. code-block:: python
      
      ollama pull llama3.1 


This command downloads the model to your computer. Depending on your internet speed, this may take some time. Once the download is complete, the model is ready to use! You can now generate responses without relying on external APIs or an internet connection.


Database Creation
=================

Why a Knowledge Base is Necessary
---------------------------------

While Llama 3.1 is great at generating conversational responses, it may not always have the specific knowledge needed for therapy-related questions. That’s where the knowledge base comes in.

The knowledge base acts as a source of truth for the chatbot, ensuring it provides accurate and reliable information. 

You can build this knowledge base in two ways: by scraping data from trusted websites or by extracting content from therapy-related PDF documents.

Database Creation Using Web Scraping with Firecrawl
===================================================

Let’s start with scraping data from websites. 

First, let’s understand why we’re using this tool to gather data. Websites like government health portals, mental health organizations, and therapy blogs provide verified and reliable information. Scraping these sources ensures the chatbot’s responses are based on accurate and evidence-based data. Websites are often updated with the latest research, guidelines, and best practices, and web scraping allows us to keep the knowledge base current without manual intervention. Manually collecting data from multiple websites is time-consuming and error-prone, but web scraping automates this process, enabling us to gather large amounts of data quickly. Firecrawl is a specialized tool designed to simplify and enhance the web scraping process, which is why we chose it. With Firecrawl, you don’t need to write complex scraping scripts from scratch.


To prepare data for retreival, we created a create_db function that returns as an output a retreiver . This retriever is used to retreive revelant text from the FAISS vector database, it is built by scraping content from multiple URLs using the FireCrawlLoader and then splitting the content into smaller chunks. These chunks are then embedded using a HuggingFace transformer model to create vector representations which are stored in the FAISS database. The documents or chunks are then stored on the database locally and can be retrieved based on similarity with a given query.

   .. code-block:: python

      
      from langchain_core.prompts import ChatPromptTemplate
      from langchain_ollama import OllamaLLM
      from langchain.vectorstores import FAISS
      from langchain.embeddings import HuggingFaceEmbeddings
      import os
      import ollama
      from langchain.text_splitter import RecursiveCharacterTextSplitter
      from langchain_community.vectorstores import FAISS
      from langchain.embeddings import HuggingFaceEmbeddings
      from langchain_community.document_loaders import FireCrawlLoader
      from langchain_community.vectorstores.utils import filter_complex_metadata
      from langchain.docstore.document import Document
      
      os.environ['LANGCHAIN_TRACING_V2']='true'
      os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
      os.environ['LANGCHAIN_API_KEY']="YOUR_LANGCHAIN_API"
      os.environ['LANGCHAIN_PROJECT']="therabot"
      
      def create_db():
          FireCrawl_API = 'YOUR_FireCrawl_API'
          DB_FAISS_PATH = 'vectorstores/db_faiss'
          urls = [
          "https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/cognitive-behaviour-therapy",
          "https://www.mentalhealth.org.uk/explore-mental-health/publications/how-manage-and-reduce-stress",
          "https://www.who.int/news-room/fact-sheets/detail/anxiety-disorders",
          "https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
          "https://www.who.int/news-room/fact-sheets"
          ]
          docs = [FireCrawlLoader(api_key=FireCrawl_API,url = url,mode="scrape").load() for url in urls]
          docs_list = [item for sublist in docs for item in sublist]
          text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 512,chunk_overlap = 50)
          doc_splits = text_splitter.split_documents(docs_list)
          cleaned_docs = []
          for doc in doc_splits : 
              if isinstance(doc, Document) and hasattr(doc, 'metadata'):
                  clean_metadat = {k: v for k ,v in doc.metadata.items() if isinstance(v, (str,int,float,bool))}
                  cleaned_docs.append(Document(page_content=doc.page_content,metadata = clean_metadat))
          embeddings = HuggingFaceEmbeddings(
          model_name='sentence-transformers/all-MiniLM-L6-v2'
          )
          db = FAISS.from_documents(
          documents= cleaned_docs, embedding= embeddings
          )
          db.save_local(DB_FAISS_PATH)
          retreiver = db.as_retriever()
          return retreiver



This code is used to gather reliable information about therapy and mental health from trusted websites, break it into smaller pieces, and store it in a format that the chatbot can understand and use to answer user questions. This ensures that the chatbot provides accurate and helpful responses.
We’ll explain everything in detail so you can understand why each part is necessary and how it contributes to the overall process.
						

1. Importing Libraries
----------------------

   .. code-block:: python

      from langchain_core.prompts import ChatPromptTemplate
      from langchain_ollama import OllamaLLM
      from langchain.vectorstores import FAISS
      from langchain.embeddings import HuggingFaceEmbeddings
      import os
      import ollama
      from langchain.text_splitter import RecursiveCharacterTextSplitter
      from langchain_community.vectorstores import FAISS
      from langchain.embeddings import HuggingFaceEmbeddings
      from langchain_community.document_loaders import FireCrawlLoader
      from langchain_community.vectorstores.utils import filter_complex_metadata
      from langchain.docstore.document import Document

-**FireCrawlLoader :** Used to scrape content from websites.
-**RecursiveCharacterTextSplitter :** Splits large documents into smaller chunks for processing.
-**HuggingFaceEmbeddings :** Generates vector representations of text.
-**FAISS :** A library for efficient similarity search and storage of vector embeddings.


2. Setting Up Environment Variables
-----------------------------------
First, we need to configure the environment by getting the needed API addresses to run the code properly. 

Visit the official LangChain and Firecrawl websites to get the API endpoints and acquire your API keys.

You will need to register or log in to get these details.

   .. code-block:: python

      os.environ['LANGCHAIN_TRACING_V2']='true'
      os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
      os.environ['LANGCHAIN_API_KEY']="YOUR_LANGCHAIN_API_KEY"
      os.environ['LANGCHAIN_PROJECT']="YOUR_PROJECT_NAME"
      
      FireCrawl_API = "YOUR_FireCrawl_API_KEY"


3. Defining the create_db Function
----------------------------------

   .. code-block:: python

   	def create_db():
   	    FireCrawl_API = 'YOUR_FireCrawl_API'
   	    DB_FAISS_PATH = 'vectorstores/db_faiss'

4. Specifying URLs to Scrape
----------------------------

The database was created using the collected data from the provided URLs, these links contain reliable informations and documents about mental health.

   .. code-block:: python

      urls = [
          "https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/cognitive-behaviour-therapy",
          "https://www.mentalhealth.org.uk/explore-mental-health/publications/how-manage-and-reduce-stress",
          "https://www.who.int/news-room/fact-sheets/detail/anxiety-disorders",
          "https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
          "https://www.who.int/news-room/fact-sheets"
      ]

5. Scraping Content with FireCrawl
----------------------------------

The `FireCrawlLoader` tool is used to scrape each URL for content. It takes a website URL, visits the page, and extracts the text.
To use the FireCrawl service, an api_key is required, which acts as a special access key. The url refers to the website address we want to scrape. By setting the mode to **scrape** , FireCrawl is instructed to extract the text content from the specified page. The scraped data from each URL is stored as an object in the **docs** list, while **docs_list** provides a flattened version of this collection, combining multiple layers of lists into one.

   .. code-block:: python

      docs = [FireCrawlLoader(api_key=FireCrawl_API,url = url,mode="scrape").load() for url in urls]
      docs_list = [item for sublist in docs for item in sublist]

6. Splitting Documents into Smaller Chunks
------------------------------------------

The extracted content is a vast amount of unstructured text data. To manage this large text efficiently, and to make it easier for the chatbot to understand and use this text, the content is split into smaller pieces called chunks using **RecursiveCharacterTextSplitter**. 

These chunks make it easier to search for and retrieve specific pieces of information, boosting the accuracy of information retrieval tasks.

   .. code-block:: python

      text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=50)
      doc_splits = text_splitter.split_documents(docs_list)

The **overlap** argument is used to avoid the risk of losing context. So, each chunk will overlap with the next one by 50 characters. This ensures no important context is lost between chunks.
If chunks are created without overlap, the model might lose key contextual informations between adjacent segments, reducing its ability to understand the complete context.

7. Cleaning Metadata
--------------------
Metadata is additional information about the text, like the title, author, or date. Sometimes, this metadata can be messy or unnecessary, so we clean it up.
It is cleaned by iterating through a list of documents, checking for valid **Document** objects, and then filtering the metadata to only include values of specific types (str, int, float, bool).

   .. code-block:: python

      cleaned_docs = []

      for doc in doc_splits:
          if isinstance(doc, Document) and hasattr(doc, 'metadata'):
              clean_metadat = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
              cleaned_docs.append(Document(page_content=doc.page_content, metadata=clean_metadat))

**cleaned_docs** is a new list where we store cleaned documents. First, we check if an object is a valid document using **isinstance(doc, Document)**. Then, we check if the document has metadata with **hasattr(doc, 'metadata')**. The **clean_metadata** function filters the metadata to keep only simple types like strings, numbers, or booleans. Finally, we add the cleaned document to the new list.

8. Generating Embeddings
------------------------

Embeddings are like numbers that represent text in a way that computers can understand. They help the chatbot figure out how similar or different pieces of text are. We need to convert text chunks into embeddings so they can be stored in the **FAISS vectorstore**. This makes it easy to quickly search and compare chunks when answering questions or retrieving information.
To create these embeddings, we used **HuggingFaceEmbeddings**, which is a tool that turns text into embeddings using a pre-trained model called sentence-transformers/all-MiniLM-L6-v2.

   .. code-block:: python

      embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

9. Creating and Saving the Vector Database
------------------------------------------

A vector database is a system used to store text embeddings, making it easy to search for similar text.

FAISS (Facebook AI Similarity Search) is a library that stores and retrieves embeddings quickly, perfect for fast similarity searches. Using **FAISS.from_documents**, we create a FAISS database from cleaned documents and their embeddings, which can be saved locally with db.save_local.

A retriever works with the FAISS database to find relevant documents based on user queries, helping provide accurate answers efficiently.

   .. code-block:: python

      db = FAISS.from_documents(documents=cleaned_docs, embedding=embeddings)
      db.save_local(DB_FAISS_PATH)
      retriever = db.as_retriever()


Database Creation Using PDF Parsing with Llama Parse
====================================================

For this method, you could use multiple documents as a reference. For our case we'll be trying to use only 1 pdf file containing a pretty good amount of informations that are both basic (somewhat general) and also a bit specific and detailed (specifically for the fields of psychology and psychiatry). The book we're using is Psychology : The Key Concepts, written by Graham Richards. This book provides a comprehensive overview of 200 key concepts essential for a strong understanding of psychology, incorporating the latest guidelines from the British Psychological Society (BPS). The focus is on practical uses of Psychology in settings such as nursing, education and human resources, with topics ranging from Gender to Psychometrics and Perception.

In order to use the book as a source of informations for the LLM. We first need to extract and organize the data from that book.

We will do that in 3 main step. 

1. Parsing the PDF files
------------------------

We first need to parse the PDF file. Meaning we need to analyze the text contained in the PDF file, extract relevant data and then structure that data into a good usable format. For that, we're using LlamaParse.

LlamaParse is a GenAI-native parser built with LLMs and for LLM use cases. It can parse and transform complex documents into LLM-ready formats with unparalleled accuracy (for any downstream LLM use case such as advanced RAG).
What makes this a really good choice is that it exists as a standalone API. The API is self-serve and available to everyone, meaning it's FREE. Also, LlamaParse can be prompted to get your data structured however you want.

In this guide we'll be saving the data into a markdown (MD) file.

2. Text Splitting
-----------------

After successfully parsing the pdf file, the second step is to split the text contained in our MD file. To do that we'll be using Langchain's RecursiveCharacterTextSplitter function.

LangChain is basically a framework designed for developing applications powered by LLMs. It simplifies building advanced applications by integrating language models with external tools. 

The RecursiveCharacterTextSplitter function in LangChain is a utility for breaking down long text into smaller, manageable chunks. It’s particularly useful when processing lengthy documents with large language models, which often have token limits. This splitter is designed to preserve meaning and context as much as possible while dividing the text. We're using it instead of the CharacterTextSplitter function in order to :

	• Preserve the context and meaning:
By breaking text hierarchically using multiple delimiters (e.g., paragraphs → sentences → words → characters). This ensures chunks are split at logical points, preserving meaning and coherence as much as possible.

While CharacterTextSplitter function splits text strictly by a single delimiter or character count, regardless of the content structure.

	• Flexible Delimiters:
RecursiveCharacterTextSplitter tries to split at meaningful delimiters (\n\n, \n, spaces) before resorting to splitting by individual characters. For example, it first tries to split text by paragraphs, then by sentences, and finally by individual words/characters if the chunk size isn't met. While, CharacterTextSplitter in the other hand, uses a single, rigid delimiter or a fixed character count. If the text doesn’t align perfectly with the delimiter, it may split mid-word or mid-sentence.

As a conclusion, for our case the RecursiveCharacterTextSplitter is the smarter choice since the context and natural breaks are critical. It'll ensure that the chunks make more sense.

3. Embedding  and Setting up the Vector Database
------------------------------------------------

You might be wondering, how is it possible to save those chunks so that the LLM can access them and know exactly the context? What format should be used? Where are we going to save them? …etc.

Well, in order to differenciate between the chunks, context-wise, we can't keep the chunks as a text. We need something that would make it possible to understand that there is a difference between chunks. A machine learning model can't understand that "being depressed" is diifferent from "being sad". But it can understand that the vector (1,1) is different from (1,0). So that's exactly what we will be doing.

We need to convert the text chunks into numerical representations (vectors) for storage and quering, and that's exactly the process of Creating Embeddings. Those embeddings will be stored in a vector database in order to do a similarity search and retrieve data from it.

For the Vector database, we used Chroma in this method. Since we're associating metadata like topics, section titles. Chroma allows to attach metadata to each chunk and query it alongside the embeddings, making it easy to filter and sort results based on context. The storage is also very persistent, this ensures the saved chunks are available accross the sessions. Chroma integrates smoothly with Langchain as well. Finally, if  we used more PDF documents, Chroma can handle a large volume of embeddings efficiently.

Implementation
--------------

Now let's get practical.

First thing to do is to import the libraries :

   .. code-block:: python

	import os
	from llama_parse import LlamaParse
	from llama_parse.base import ResultType, Language
	from langchain.text_splitter import RecursiveCharacterTextSplitter
	from langchain.vectorstores import Chroma
	from langchain_community.embeddings.ollama import OllamaEmbeddings
	from langchain_core.documents import Document

You can set up the LlamaCloud API key as an environment variable using :

   .. code-block:: python

	os.environ["LLAMA_CLOUD_API_KEY"]= "API_KEY"

You should replace "API_KEY" with your LlamaCloud API key.


Now we need to define the parser :

   .. code-block:: python

	parser = LlamaParse(result_type=ResultType.MD,language=Language.ENGLISH)

The next thing to do is to load the text from the PDF file, using our parser, and save it to a Markdown file: 

   .. code-block:: python

	#Parsing into the documents variable
	documents = parser.load_data(
	    "PsychologyKeyConcepts.pdf"
	)
	#Writing the result of parsing into the file psychology_data.md
	filename = "psychology_data.md"
	with open(filename, 'w') as f:
	    f.write(documents[0].text)

Now we got our data extracted into a markdown file called "psychology_data". The next step is to split it into chunks!

   .. code-block:: python

	with open("psychology_data.md", encoding='utf-8') as f:
	    doc = f.read()
	
	#defining the splitter
	r_splitter = RecursiveCharacterTextSplitter(
	    chunk_size=2000,
	    chunk_overlap=0,
	    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
	)
	#Splitting the text
	docs = r_splitter.split_text(doc)
	print("data has been split.")

To create embeddings for the text chunks, we will be using OllamaEmbeddings since we want a local, privacy-friendly, and cost-effective solution. The mxbai-embed-large model is a compelling choice for generating embeddings due to its specific design for high-quality, contextually aware vector representations:

   .. code-block:: python

	# Convert the list of strings to a list of Document objects
	docs = [Document(page_content=d) for d in docs]
	embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
	print("embeddings created.")

The last thing to do now, is to define the vector database and populate it.

   .. code-block:: python

	#defining vector database directory
	persist_directory = "Psycho_db"

	# Load the database
	vecdb = Chroma(persist_directory=persist_directory, 
	embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"), 
	collection_name="rag-chroma")

	vecdb.add_documents(docs)
	print("data has been added to the database.")
	vecdb.persist()
	print("Data has been ingested into vector database.")

After doing that, it's practical to test it out, to see if everything is working out well.

   .. code-block:: python

	#Testing
	question = "what is depression?"
	documents = vecdb.similarity_search(question,k=5)
	print(documents[0].page_content)

RAG Implementation and Text Generation
======================================

1- Using Web Scraping
=====================

Now that we’ve built the knowledge base, let’s talk about how the chatbot uses it to generate responses. We’ll do this using a class called GenerateResponse. A class is like a blueprint that defines how the chatbot works. It organizes all the steps and logic into one place, making it easier to manage and reuse.

   .. code-block:: python

   	class GenerateResponse:
   	    def __init__(self, model_name="llama3.1"):
   	        self.model = OllamaLLM(model=model_name)
   	        self.db_faiss_path = 'vectorstores/db_faiss'
   	        self.context = ""
   	        self.chat_history = []  # used to store chat history during the session
   	        self.prompt_template = """
   	        You are a therapist, and your primary goal is to offer support, understanding, and guidance to the user in a                  
   	        compassionate and professional manner.
   	        Always respond empathetically, non-judgmentally, and with respect.
   	        Your role is to help the user feel heard and understood, not to judge.
   	        Respond with empathy and only with evidence-based advice, referencing only to the relevant documents provided.
   	        Provide support using active listening and ask open-ended questions to explore the user's feelings and thoughts.
   	        Only provide information that you are sure about.
   	        Relevant Documents : {document}
   	        Question: {question}
   	        Answer:
   	        """
   	        #prompt to check if RAG is needed
   	        self.rag_check_prompt = """
   	        You are a highly intelligent assistant designed to decide whether a query requires additional information from external sources (like documents) to provide a complete answer.
   	        Respond with "True" if the query involves scientific, medical, or evidence-based information, such as mental health conditions, medical conditions, or psychological coping strategies. In these cases, external references like research articles, therapeutic methods, or clinical guidelines are necessary.
   	        Example: Queries like "How can I deal with amnesia?" or "What are effective ways to manage anxiety?" require scientific and evidence-based details, so respond with "True."
   	        Respond only with " True " or " False "
   	        Query: "{query}"
   	        Needs External Information (True/False):
   	        """
   	    def check_need_for_rag(self,user_query):
   	        #determine if the user's query needs RAG.
   	        try:
   	            #check for RAG requirement
   	            check_prompt = ChatPromptTemplate.from_template(self.rag_check_prompt)
   	            query_grader = check_prompt | self.model
   	            query_grade = query_grader.invoke({"query":user_query})
   	            return query_grade.strip().lower() == "true"
   	        except Exception as e:
   	            print(f"Error checking for RAG need: {str(e)}")
   	            return False  #default to no RAG on failure
   	    def generate_answer(self, user_query,chat_history: list=[]):
   	        try:
   	            needs_rag = self.check_need_for_rag(user_query)
   	            if needs_rag:
   	                retrieved_docs_txt = self.retreive(user_query)
   	            else:
   	                retrieved_docs_txt = ""
   	            #generate response
   	            my_message = [{"role": "system", "content": self.prompt_template,  "document": retrieved_docs_txt }]
   	            #Append history in message 
   	            for chat in chat_history:                      
   	                my_message.append({"role": chat["role"], "content": chat["content"]})
   	            #Append the latest question in message
   	            my_message.append({"role": "user", "content": user_query, "document": retrieved_docs_txt })
   	            generated_answer = ollama.chat(                      
   	            model="llama3.1",
   	            messages=my_message
   	            ) 
   	            #save the chat
   	            self.log_chat(user_query, generated_answer)
   	            return generated_answer["message"]["content"]
   	        except Exception as e:
   	            error_message = f"An error occurred: {str(e)}"
   	            return error_message
   	    def retreive(self,user_query):
   	        #load FAISS vectorstore
   	        retriever = create_db()
   	        retreived_docs = retriever.invoke(user_query)
   	        retreived_docs_txt = retreived_docs[1].page_content
   	        return retreived_docs_txt
   	    def log_chat(self, user_query, response):
   	        #add the user query and the generated answeer to chat history
   	        chat = {"user": user_query, "assistant": response}
   	        self.chat_history.append(chat)

The class has several key components:

"__init__" Method
-----------------

This method initializes the GenerateResponse class with default attributes.  
We start first by loading the language model specified by `model_name` (default is `"llama3.1"`), which is used to generate answers. Additionally, we specify the path where the FAISS vectorstore is stored, used for document retrieval.

   .. code-block:: python

        self.model = OllamaLLM(model=model_name)
        self.db_faiss_path = 'vectorstores/db_faiss'

We will have to keep track of the entire chat session to allow responses to consider the previous conversation. We do this using the `chat_history` list.

   .. code-block:: python

        self.chat_history = []  # Used to store chat history during the session

A prompt template is created to shape the chatbot's responses, ensuring empathy and relevance, and to define the tone, style, and constraints for generating responses.

   .. code-block:: python

        self.prompt_template = """
        You are a therapist, and your primary goal is to offer support, understanding, and guidance...
        Relevant Documents : {document}
        Question: {question}
        Answer:
        """

Then we are going to define a secondary prompt to determine if a query requires external information to provide a complete response.

   .. code-block:: python

        self.rag_check_prompt = """
        You are a highly intelligent assistant designed to decide whether a query requires additional information 
        from external sources (like documents) to provide a complete answer.
        Query: "{query}"
        Needs External Information (True/False):
        """


"check_need_for_rag" Method
---------------------------

Before generating a response, the chatbot evaluates the user query to decide if external documents are necessary to answer properly, using the check_need_for_rag function.

This function uses the predefined logic in the **rag_check_prompt** , this prompt will be combined with the user query and passed to the model.
The model evaluates the query to determine if RAG is necessary, based on whether the query requires additional context, such as scientific information or other detailed data.
If additional information is needed, the model responds with "true", and the function returns True, and False otherwise.
However, if there are issues during this process, the function returns False by default.

   .. code-block:: python

    def check_need_for_rag(self,user_query):
        try:
            check_prompt = ChatPromptTemplate.from_template(self.rag_check_prompt)
            query_grader = check_prompt | self.model
            query_grade = query_grader.invoke({"query":user_query})
            return query_grade.strip().lower() == "true"
        except Exception as e:
            print(f"Error checking for RAG need: {str(e)}")
            return False


"retrieve" Method 
-----------------

After checking if RAG is necessary . If it is required , the **retreive()** method returns the retreived document's text from the FAISS vector store.
The FAISS vector store compares the query embedding with the stored document embeddings using a similarity search, and returns the documents with the highest similarity scores.

   .. code-block:: python

      def retreive(self, user_query):
          retriever = create_db()
          retreived_docs = retriever.invoke(user_query)
          retreived_docs_txt = retreived_docs[1].page_content
          return retreived_docs_txt

"generate_answer" Method
------------------------

The **generate_answer()** method uses the predefined prompt template, the retrieved documents, and the chat history to generate a response using the **llama3.1** model via **ollama.chat**.

   .. code-block:: python

	    def generate_answer(self, user_query,chat_history: list=[]):
	        try:
	            needs_rag = self.check_need_for_rag(user_query)
	            if needs_rag:
	                retrieved_docs_txt = self.retreive(user_query)
	            else:
	                retrieved_docs_txt = ""
	            #generate response
	            my_message = [{"role": "system", "content": self.prompt_template,  "document": retrieved_docs_txt }]
	            #Append history in message 
	            for chat in chat_history:                      
	                my_message.append({"role": chat["role"], "content": chat["content"]})
	            #Append the latest question in message
	            my_message.append({"role": "user", "content": user_query, "document": retrieved_docs_txt })
	            generated_answer = ollama.chat(                      
	            model="llama3.1",
	            messages=my_message
	            ) 
	            #save the chat
	            self.log_chat(user_query, generated_answer)
	            return generated_answer["message"]["content"]
	        except Exception as e:
	            error_message = f"An error occurred: {str(e)}"
	            return error_message

This method creates responses to user queries by first checking if the query requires external information using the check_need_for_rag method.If additional context is needed, it retrieves relevant documents through the retrieve method. And  stores them in **retrieved_docs_txt**.
   .. code-block:: python

     try:
         needs_rag = self.check_need_for_rag(user_query)
         if needs_rag:
             retrieved_docs_txt = self.retreive(user_query)
         else:
             retrieved_docs_txt = ""

It then prepares a list of messages that includes the system’s prompt, the chat history, the user’s query, and any relevant documents. 
   .. code-block:: python

      my_message = [{"role": "system", "content": self.prompt_template,  "document": retrieved_docs_txt }]
      #Append history in message 
      for chat in chat_history:                      
          my_message.append({"role": chat["role"], "content": chat["content"]})
      #Append the latest question in message
      my_message.append({"role": "user", "content": user_query, "document": retrieved_docs_txt })

Using the OllamaLLM language model, it generates a response based on this structured input. 

   .. code-block:: python

      generated_answer = ollama.chat(                      
      model="llama3.1",
      messages=my_message
      ) 
The conversation (user query and assistant response) is logged in the chat_history to maintain context throughout the session. 

   .. code-block:: python

      self.log_chat(user_query, generated_answer)
      return generated_answer["message"]["content"]

If any error occurs during this process, the method returns an error message. This approach ensures that the response is contextually relevant and incorporates external information when necessary.

   .. code-block:: python

      except Exception as e:
         error_message = f"An error occurred: {str(e)}"
         return error_message
	
"log_chat" Method
-----------------

Finally, the conversation between the user and the model is logged to maintain a record of user queries and assistant responses, ensuring that the context is preserved.

   .. code-block:: python

      def log_chat(self, user_query, response):
          chat = {"user": user_query, "assistant": response}
          self.chat_history.append(chat)

2- Using PDFs
=============

In this part, we will be focusing on generating text based on the user's query and our vector database. The user will have to enter an input, that will be then processed by the model Llama3.1 and based on the user's input it should understand whether it's necessary to go back to the vector database and retrieve data, or answer based on what it knows.

The libraries used in this file are : 

   .. code-block:: python

      	import ollama
	from langchain.vectorstores import Chroma
	from langchain_community.embeddings.ollama import OllamaEmbeddings
	from langchain_ollama import OllamaLLM
	from langchain_core.prompts import ChatPromptTemplate

So first, let's load the vector database with embedded documents to enable similarity-based retrieval: 

   .. code-block:: python

      	persist_directory = "rag/Pyscho_db"
	vecdb = Chroma(
	    persist_directory=persist_directory,
	    embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
	    collection_name="rag-chroma"
	)

Now we need to make the retrieval function :

   .. code-block:: python

	def retrieve_from_db(question):
	    # get the model
	    model = OllamaLLM(model="llama3.1")
	    # initialize the vector store
	    retriever = vecdb.as_retriever()
	    retrieved_docs = retriever.invoke(question)
	    retrieved_docs_txt = retrieved_docs[1].page_content
	    return retrieved_docs_txt

We should make the main generation function, which will be the main function that the model uses in order to answer the user. But we need to combine it with the retrieval logic.
There are many ways to do that, we can for example make a verification function that'll enable us to know whether the user's input relates to mental health or not.  But, we will keep everything simple and just add up everything into the model's prompt.

   .. code-block:: python

	def generate_response(user_message: str, chat_history: list=[], doc=""):
		#give role to Chatbot    
		system_msg=("""You are a Chatbot for mental health support, don't overtalk. When the users are trying to harm themselves, remind them that they're loved by someone.
		When asked about someone say "sorry, I don't wanna talk about people". Stick to the context of mental health. If the situation is serious refer to moroccan health services.
		Don't insist on questions, try to be friendly and make the client feel comfortable talking with you.
		don't repeat the same questions in the same message.
		Don't say "Based on the provided context" or "According to the provided document" or any such phrases.
		if there is no answer, please answer with "I m sorry, the context is not enough to answer the question.
		Don't keep on questioning what's happening, your main job is to listen actively and make the client feel comfortable with you.
		Combine what you know and verify it using the Relevant Documents : {document}
		User input: {question}
		
			""")        
		my_message = [{"role": "system", "content": system_msg,  "document": doc}]
		#Append history in message 
		for chat in chat_history:                      
			my_message.append({"role": chat["name"], "content": chat["msg"]})
		#Append the latest question in message
		my_message.append({"role": "user", "content": user_message, "document": doc})
		response = ollama.chat(                      
		model="llama3.1",
		messages=my_message
		) 
		return response["message"]["content"]

So, as you can see. By prompting the model we make it understand its job (being an assistant for mental health support). Also, we tell it to combine what it knows with what's in the "documents". Basically what we're trying to do is :
	- Get the user input
	- Use the retrieval function
	- Save the result in a variable called "document" 
	- Use that "document" variable as a verification tool for what the model knows.


Speech to Text
==============
We integrated the audio input feature using the Streamlit framework and the SpeechRecognition library. Users can speak to the chatbot instead of typing, and their speech is
transcribed into text using Google’s SpeechRecognition Python module.


   .. code-block:: python

       elif recorder:
           recognizer = sr.Recognizer()
           with sr.Microphone() as source:
               st.toast("You can start talking...", icon='🎤')
               recognizer.adjust_for_ambient_noise(source, duration=0.2)  
               audio = recognizer.listen(source)
               try:
                   user_query = recognizer.recognize_google(audio)
                   # Display user message
                   with st.chat_message("user"):
                       st.markdown(user_query)
                   # Generate response, and add it to the chat history
                   response = generator.generate_answer(user_query,chat_history=st.session_state.chat_history)
   
                   # Display the generated response
                   with st.chat_message("assistant"):
                       st.markdown(response)
                   # Update chat history
                   st.session_state.chat_history.append({"role": "user", "content": user_query})
                   st.session_state.chat_history.append({"role": "assistant", "content": response})
               except:
                   st.markdown("Sorry, I did not get that")


First, the user presses the microphone button in the chatbot interface to activate the audio recording feature. 
If the recorder condition is active (microphone button clicked), indicating that the user has triggered the voice recording feature, a Recognizer object from the SpeechRecognition library is created. 
This object will handle audio processing and transcription:

   .. code-block:: python

      recognizer = sr.Recognizer()

Then, the audio is captured using the sr.Microphone() to access the user’s microphone. The with statement ensures that the microphone is properly closed after use:

   .. code-block:: python

      with sr.Microphone() as source:
         st.toast("You can start talking...", icon='🎤')
         recognizer.adjust_for_ambient_noise(source, duration=0.2)  
         audio = recognizer.listen(source)

The user starts speaking when a message **"You can start talking..."** is displayed using the toast method to notify the user that the system is ready to record.
We used **recognizer.adjust_for_ambient noise(source, duration=0.2)** to filter out background noise for better accuracy. This process lasts 0.2 seconds.
Then, **recognizer.listen(source)** records audio from the microphone and stores it in the **audio** variable.

If the transcription fails (for example due to poor audio quality or accents), an error message is displayed to the user :

   .. code-block:: python

      except :
            st.markdown ("Sorry ,I did not get that ")

If the transcription is successful, the **user_query** variable will store the the user's audio input converted into text, using the **recognizer.recognize google(audio)** method:

   .. code-block:: python

      user_query = recognizer.recognize_google(audio)

Finally, the audio input converted into a text format can be handled as a text input and passed to RAG (Retrieval-Augmented Generation) and the llama model for response generation.


UI Development (`TheraBot Interface`)
=====================================

1- Using Web Scraping
---------------------

We will create a Streamlit web application for the chatbot.We will be using streamlit for the interface, GenerateResponse for generating chatbot responses, and speech_recognition for voice input. 

   .. code-block:: python

	import streamlit as st
	from generate import GenerateResponse
	import speech_recognition as sr
	import time
	
	import streamlit as st
	from streamlit_float import *
	
	# initialize float feature/capability
	float_init()
	
	#initialize generator
	generator = GenerateResponse()
	
	st.markdown("<h1 style='text-align : center;'>TheraBot</h1>", unsafe_allow_html=True)
	
	def main():
	    # Initialize chat history and recording state
	    if "chat_history" not in st.session_state:
	        st.session_state.chat_history = []
	
	    # Display chat messages from history on app rerun
	    for message in st.session_state.chat_history:
	        with st.chat_message(message["role"]):
	            st.markdown(message["content"])
	
	    # Create footer container and add content
	    input_placeholder = st.container()
	
	    with input_placeholder.container():
	        col1, col2 = st.columns([11, 1], gap="small")
	        with col1:
	            user_query = st.chat_input("What is up?", key="user_input")
	        with col2:
	            recorder = st.button("🎙️")
	
	    # Float the footer container and provide CSS to target it with
	    input_placeholder.float("bottom: 0; padding : 30px 20px 50px 20px; border-radius: 10px; background:#0E1117;")
	
	    # Accept user input from text or transcription
	    if user_query:
	        # Add user message to chat history
	        st.session_state.chat_history.append({"role": "user", "content": user_query})
	        # Display user message
	        with st.chat_message("user"):
	            st.markdown(user_query)
	
	        # Generate response, and add it to the chat history
	        response = generator.generate_answer(user_query,chat_history=st.session_state.chat_history)
	
	        # Display the generated response
	        with st.chat_message("assistant"):
	            st.markdown(response)
	        # Update chat history
	        st.session_state.chat_history.append({"role": "assistant", "content": response})
	    # Handle recorder button functionality
	    elif recorder:
	        recognizer = sr.Recognizer()
	        with sr.Microphone() as source:
	            st.toast("You can start talking...", icon='🎤')
	            recognizer.adjust_for_ambient_noise(source, duration=0.2)  
	            audio = recognizer.listen(source)
	            try:
	                user_query = recognizer.recognize_google(audio)
	                # Display user message
	                with st.chat_message("user"):
	                    st.markdown(user_query)
	                # Generate response, and add it to the chat history
	                response = generator.generate_answer(user_query,chat_history=st.session_state.chat_history)
	
	                # Display the generated response
	                with st.chat_message("assistant"):
	                    st.markdown(response)
	                # Update chat history
	                st.session_state.chat_history.append({"role": "user", "content": user_query})
	                st.session_state.chat_history.append({"role": "assistant", "content": response})
	            except:
	                st.markdown("Sorry, I did not get that")
	
	if __name__ == "__main__":
	    main()

2- Using PDFs
-------------
For this method, you will use pretty much a similar code to the one mentioned in the first method, except that you would have to use the name of the functions mentioned for this method in the RAG and text generation Section, and you will have to use the retrieval logic mentioned in the rag explanation for this method.

There are some few changes in this code, here is what it looks like:

   .. code-block:: python

	def main():
	    if "chat_log" not in st.session_state:
	        st.session_state.chat_log = []
	
	    # Display chat history
	
	    for chat in st.session_state.chat_log:
	        with st.chat_message(chat["name"]):
	            st.write(chat["msg"])
	
	    # Placeholder for the input section (audio + text)
	    # Create footer container and add content
	
	    input_placeholder = st.container()
	    with input_placeholder.container():
	        col1, col2 = st.columns([11, 1], gap="small")
	        with col1:
	            user_message = st.chat_input("What is up?", key="user_input")
	        with col2:
	            record_audio = st.button("🎙️")
	    
	        # Float the footer container and provide CSS to target it with
	
	    input_placeholder.float("bottom: 0px;padding : 30px 20px 50px 20px; border-radius: 10px; background:#0E1117;")
	
	    # Process user input
	
	    if user_message:
	        with st.chat_message("user"):
	            st.markdown(user_message)
	        doc = retrieve_from_db(user_message)
	
	        # Generate response
	
	        response = generate_response(user_message, chat_history=st.session_state.chat_log, doc= doc)
	        if response:
	            with st.chat_message("assistant"):
	                assistant_response_area = st.empty()
	                assistant_response_area.write(response)
	
	            # Update chat history
	
	            st.session_state.chat_log.append({"name": "user", "msg": user_message})
	            st.session_state.chat_log.append({"name": "assistant", "msg": response})
	    elif record_audio:
	
	        # Handle audio recording
	
	        r = sr.Recognizer()
	        with sr.Microphone() as source:
	            st.markdown("You can start talking...")
	            r.adjust_for_ambient_noise(source, duration=0.2)  
	            audio_text = r.listen(source)
	            try:
	                user_message = r.recognize_google(audio_text)
	                with st.chat_message("user"):
	                    st.markdown(user_message)
	                doc = retrieve_from_db(user_message)
	
	                # Generate response
	
	                response = generate_response(user_message, chat_history=st.session_state.chat_log, doc=doc)
	                if response:
	                    with st.chat_message("assistant"):
	                        assistant_response_area = st.empty()
	                        assistant_response_area.write(response)
	
	                    # Update chat history
	
	                    st.session_state.chat_log.append({"name": "user", "msg": user_message})
	                    st.session_state.chat_log.append({"name": "assistant", "msg": response})
	            except:
	                st.markdown("Sorry, I did not get that")
    
	if __name__ == "__main__":
	    main()


