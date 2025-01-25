from haystack.utils import Secret
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
#from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from QASystem.utils import pinecone_config  # Ensure this function correctly initializes Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve Hugging Face API token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"

# Define the prompt template
prompt_template = """Answer the following query based on the provided context. If the context
                   does not include an answer, reply with 'I don't know'.\n
                   Query: {{query}}
                   Documents:
                   {% for doc in documents %}
                      {{ doc.content }}
                   {% endfor %}
                   Answer:
                   """

def get_result(query):
    # Initialize the pipeline
    query_pipeline = Pipeline()

    # Initialize embedding model
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2")

    # Initialize Pinecone retriever
    retriever = PineconeEmbeddingRetriever(document_store=pinecone_config())

    # Build the prompt dynamically
    prompt_builder = PromptBuilder(template=prompt_template)

    # Initialize Hugging Face LLM
    llm = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                    api_params={"model": "tiiuae/falcon-7b-instruct"},
                                    token=Secret.from_token(HF_TOKEN))

    # Add components to the pipeline
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("llm", llm)

    # Connect pipeline components
    query_pipeline.connect("text_embedder", "retriever")
    query_pipeline.connect("retriever", "prompt_builder")
    query_pipeline.connect("prompt_builder", "llm")

    # Run the query pipeline
    results = query_pipeline.run(
        {
            "text_embedder": {"text": query},
            "prompt_builder": {"query": query}
        }
    )
    #answer = result["prompt_node"]["answers"]
    #return answer[0] if answer else "I don't know"
    return results['llm']['replies'][0]  # Extract the answer from the response

if __name__ == "__main__":
    query = "What is rag token?"
    result = get_result(query)
    print("Answer:", result)
