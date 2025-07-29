import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# UniEval imports
from metric.evaluator import get_evaluator
from utils import convert_to_json # Assuming 'utils.py' is in the same directory

# --- Configuration ---
load_dotenv()

# --- Data Loading and Preparation ---
urls = [
    "https://src.pata.org/post-covid-recovery/phuket-action-plan/",
    "https://thinkhazard.org/en/report/240-thailand/TS",
    "https://ait.ac.th/2024/12/20-years-after-the-tsunami-thailand-reflects-on-lessons-for-future-disaster-resilience/"
]
loader = WebBaseLoader(urls)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Print metadata of the loaded documents (optional, but good for understanding data)
avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(texts)
print(f'Average length among {len(documents)} pages loaded is {avg_char_count_pre} characters.')
print(f'After the split you have {len(texts)} chunks.')
print(f'Average length among {len(texts)} chunks is {avg_char_count_post} characters.')

# Initialize vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Initialize OpenAI client
client = OpenAI()

# --- Predefined RAG Questions ---
questions = [
    "What were the five operational focus areas of the Phuket Action Plan developed after the 2004 tsunami?",
    "Why was marketing and communication considered the most important element of the Phuket Action Plan, and how was it implemented?",
    "What was the timeline of Phuket after 2004 tsunami?",
    "What does a 'medium' tsunami hazard classification indicate for Thailand, and what timeframe does it consider?",
    "How should critical infrastructure projects near the Thai coastline account for tsunami risk, especially in relation to network dependencies?",
    "What role did the Royal Thai Army Science Department play during the disaster relief efforts following the 2004 tsunami in Thailand?",
    "How did the expertise developed during the 2004 tsunami disaster relief efforts contribute to other areas beyond disaster management, according to His Serene Highness Prince Chalermsak Yugala?"
]

# --- Generate Response Function ---
def generate_response(prompt: str, model_name: str):
    """
    Generates a response from the specified OpenAI model.
    Includes adjustments for model-specific parameter limitations.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide concise and factual answers based on the context. If the information is not in the context, state that you cannot answer based on the provided information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            # For gpt-4o-2024-08-06, temperature is often fixed at 1.
            # To reduce randomness and blanks, we apply top_p.
            # If a model truly doesn't support a parameter, an error will still occur.
            temperature=1.0, # Explicitly set to 1.0 if the model requires it or defaults to it
            top_p=0.1      # Attempt to make output more deterministic and less prone to blanks
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response for {model_name} with prompt: '{prompt[:50]}...' - {e}")
        return "Error: Could not generate response." # Return a clear error message

# --- UniEval RAG Evaluator ---
# The 'summarization' task in UniEval assesses aspects like coherence, consistency, and fluency.
# 'src_list' is your context, 'output_list' is your generated answer.
# 'ref_list' is typically for human-written reference answers (left empty here).
evaluator = get_evaluator("summarization", device="cpu")

# --- Models to Evaluate ---
models_to_evaluate = [
    "gpt-4o-2024-08-06",
    "o3-2025-04-16", # Assuming 'o3-2025-04-16' is a valid model name you can access
]

# --- Main Evaluation Loop ---
for model_name in models_to_evaluate:
    print(f"\n{'='*50}")
    print(f"--- Evaluating Model: {model_name} ---")
    print(f"{'='*50}\n")

    for i, question in enumerate(questions):
        print(f"Question {i+1}/{len(questions)} for {model_name}: {question}")

        # Step 1: Retrieve top-3 docs
        relevant_docs = vectorstore.similarity_search(question, k=3)
        context_list = [doc.page_content for doc in relevant_docs]
        context = "\n\n".join(context_list)

        # Print a snippet of the context for verification
        print(f"Retrieved Context (first 200 chars):\n{context[:200]}...\n")

        # Step 2: Generate answer
        # The generate_response function now correctly accepts the model_name
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        answer = generate_response(prompt, model_name)

        # Step 3: Evaluate answer using UniEval
        # The 'src_list' is the context, 'output_list' is the generated answer.
        # 'ref_list' is an empty list as we don't have ground truth references.
        data_for_unieval = convert_to_json(output_list=[answer], src_list=[context], ref_list=[''])
        
        # We print results for each question for immediate feedback
        print(f"\nGenerated Answer: {answer}")
        
        # UniEval scores
        scores = evaluator.evaluate(data_for_unieval, print_result=True) # Keep print_result=False for cleaner main loop
        #print(f"UniEval Scores for this response: {scores}")

        print("\n" + "*"*80 + "\n") # Separator between questions