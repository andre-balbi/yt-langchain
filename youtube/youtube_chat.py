import textwrap

from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"


def create_db_from_youtube_video_url(video_url):
    """
    Creates a document database from a YouTube video URL.

    Args:
        video_url (str): The URL of the YouTube video.

    Returns:
        db: A document database created from the video transcript.
    """
    # Create a loader for the YouTube video transcript
    loader = YoutubeLoader.from_youtube_url(video_url)

    # Load the transcript of the video
    transcript = loader.load()

    # Create a text splitter to divide the transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Split the transcript into smaller documents
    docs = text_splitter.split_documents(transcript)

    # Create a document database using FAISS and embeddings
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    Retrieve a response using the given query from a document database.

    Args:
        db (FAISS): A document database to search for responses.
        query (str): The query to search for in the database.
        k (int): The number of similar documents to retrieve.

    Returns:
        str: The response generated based on the query.
        List[Document]: The list of similar documents from the database.
    """

    # Search for similar documents in the database based on the query
    docs = db.similarity_search(query, k=k)

    # Combine the page content of similar documents into a single string
    docs_page_content = " ".join([d.page_content for d in docs])

    # Create an instance of ChatOpenAI with settings for generating a response
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Define a template for the system message prompt
    template = """
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript: {docs}
        
        Only use factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    # Create a system message prompt template from the defined template
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Define a template for the human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Create a chat prompt template from system and human message prompts
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # Create an LLMChain instance for the conversation with the model
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    # Run the chain to generate a response based on the query and documents
    response = chain.run(question=query, docs=docs_page_content)

    # Replace line breaks in the response
    response = response.replace("\n", "")

    return response, docs


# Example usage:
video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
db = create_db_from_youtube_video_url(video_url)

query = "What are they saying about Microsoft?"
response, docs = get_response_from_query(db, query)
print(textwrap.fill(response, width=50))
