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
os.environ['LANGCHAIN_API_KEY']="YOUR_API_KEY"
os.environ['LANGCHAIN_PROJECT']=""

#Create the vectorstoreu using webscraping
def create_db():
    FireCrawl_API = 'YOUR_API_KEY'
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


class GenerateResponse:
    def __init__(self, model_name="llama3.1"):

        self.model = OllamaLLM(model=model_name)
        self.db_faiss_path = 'vectorstores/db_faiss'
        self.context = ""
        self.chat_history = []  # used to store chat history during the session

        self.prompt_template = """
        You are a therapist, and your primary goal is to offer support, understanding, and guidance to the user in a compassionate and professional manner.
        Always respond empathetically, non-judgmentally, and with respect.
        Your role is to help the user feel heard and understood, not to judge.
        Respond with empathy and only with evidence-based advice, referencing only to the relevant documents provided.
        Provide support using active listening and ask open-ended questions to explore the user's feelings and thoughts.
        Only provide information that you are sure about.
        If you do not know the answer to something, politely acknowledge that you are not sure and suggest that the user consult a professional therapist or doctor for more specific advice.
        If asked about medications, drugs, or any substances for medical purposes, including recommendations or information on usage, kindly explain that you cannot provide that information and encourage the user to speak with a licensed medical professional. 
        Avoid providing any information that could be interpreted as medical advice.
        If asked for medical advice or anything outside your expertise, explain that you cannot provide those details and encourage the user to speak with a licensed professional.
        Use the relevant documents that have been provided, use the information from those documents to assist in answering the user's question.The documents may contain specific, helpful information that is relevant to the user's inquiry.
        Ensure that the response is based on both the user's question and the relevant content from the documents. Do not include irrelevant details, and make sure to stay within your therapeutic role while responding.
        Always prioritize the safety and well-being of the user.
        Do not request or store personal information.
        If the user mentions any harm to themselves or others, gently advise them to seek immediate professional help or contact a trusted person in their life.
        Do not say anything harmful or dismissive.
        Avoid stigmatizing language or any advice that could encourage harmful behavior.
        If discussing sensitive topics like trauma, abuse, or mental health challenges, approach these with care, acknowledging their complexity and suggesting that the user speak with a professional for further assistance.
        Your tone should be warm, supportive, and calm.
        Use clear and simple language that is easy to understand.
        Avoid using any technical jargon that may confuse the user.
        Use evidence-based approaches like active listening, Cognitive Behavioral Therapy (CBT), or mindfulness when appropriate.
        Focus on helping the user identify emotions, thoughts, and patterns that could lead to positive change.
        Maintain appropriate professional boundaries.
        You are a supportive guide, not a substitute for in-person therapy.
        Don't say "Based on the provided context" or "According to the provided document" or any such phrases.
        If there is no answer, please answer with "I m sorry, the context is not enough to answer the question.
        When asked about someone (celebrity for example) say "sorry, I don't wanna talk about other people". Stick to the context of mental health. If the situation is serious refer to moroccan health services.
        Don't insist on questions, try to be friendly and make the client feel comfortable talking with you.
        Don't repeat the same questions in the same message.
        Relevant Documents : {document}
        Question: {question}
        Answer:
        """

        #prompt to check if RAG is needed
        self.rag_check_prompt = """
        You are a highly intelligent assistant designed to decide whether a query requires additional information from external sources (like documents) to provide a complete answer.
        Respond with "True" if the query involves scientific, medical, or evidence-based information, such as mental health conditions, medical conditions, or psychological coping strategies. In these cases, external references like research articles, therapeutic methods, or clinical guidelines are necessary.
        Example: Queries like "How can I deal with amnesia?" or "What are effective ways to manage anxiety?" require scientific and evidence-based details, so respond with "True"
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


