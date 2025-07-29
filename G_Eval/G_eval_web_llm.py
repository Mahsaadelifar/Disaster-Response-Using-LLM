from dotenv import load_dotenv
import os
import promptquality as pq
from promptquality import EvaluateRun
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Set up Galileo Eval 
load_dotenv()
pq.login()
PROJECT_NAME = "evaluate-web-sourced-rag-chatbot"
metrics = [pq.Scorers.correctness, pq.Scorers.instruction_adherence_plus]

# Loading and preparimg data
urls = [
    "https://src.pata.org/post-covid-recovery/phuket-action-plan/",
    "https://thinkhazard.org/en/report/240-thailand/TS",
    "https://ait.ac.th/2024/12/20-years-after-the-tsunami-thailand-reflects-on-lessons-for-future-disaster-resilience/"
]
loader = WebBaseLoader(urls)
documents = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Print metadata of the loaded documents
avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(texts)
print(f'Average length among {len(documents)} pages loaded is {avg_char_count_pre} characters.')
print(f'After the split you have {len(texts)}')
print(f'Average length among {len(texts)} chunks is {avg_char_count_post} characters.')

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create a vector store
vectorstore = FAISS.from_documents(texts, embeddings)

# Create a function to generate a response using Open AI
client = OpenAI()
def generate_response(prompt: str, history: list = [], model_name: str = "gpt-4o-2024-08-06"):
    
    response = client.chat.completions.create(
        model=model_name,
        messages=history + [{"role": "user", "content": prompt}],
        max_completion_tokens=1000,
        temperature=1,
        top_p=1
    )
    
    response_text = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    
    return response_text, input_tokens, output_tokens, total_tokens

USE_PREDEFINED_QUESTIONS = True # If the questions are propted by hand, turn this to False

questions = [
    "What were the five operational focus areas of the Phuket Action Plan developed after the 2004 tsunami?",
    "Why was marketing and communication considered the most important element of the Phuket Action Plan, and how was it implemented?",
    "What was the timeline of Phuket after 2004 tsunami?",
    "What does a 'medium' tsunami hazard classification indicate for Thailand, and what timeframe does it consider?",
    "How should critical infrastructure projects near the Thai coastline account for tsunami risk, especially in relation to network dependencies?",
    "What role did the Royal Thai Army Science Department play during the disaster relief efforts following the 2004 tsunami in Thailand?",
    "How did the expertise developed during the 2004 tsunami disaster relief efforts contribute to other areas beyond disaster management, according to His Serene Highness Prince Chalermsak Yugala?"
]

# Evaluate models
models = [
    "gpt-4o-2024-08-06",
    "o3-2025-04-16",
    ]

max_rounds = 7 # set this to the number of questions you want to ask

for model_name in models[0:]:
    rounds = 0
    evaluate_run = EvaluateRun(run_name=model_name, project_name=PROJECT_NAME, scorers=metrics)
    history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while rounds < max_rounds:
        question = questions[rounds] if USE_PREDEFINED_QUESTIONS else input("You: ")
        if question.lower() == 'exit':
            break
        # Retrieve relevant documents from the vector store
        relevant_docs = vectorstore.similarity_search(question, k=3)
        context_list = [doc.page_content for doc in relevant_docs]
        context = " ".join(context_list)
        print(f"\nCONTEXT = {context[0:50]}")

        # Crafting the prompt
        prompt = f"""Context: {context}

        Question: {question}

        Answer: """

        # Create your workflow to log to Galileo.
        wf = evaluate_run.add_workflow(input={"question": question, "model_name": model_name}, name=model_name, metadata={"env": "demo"})
        wf.add_retriever(
            input=question,
            documents=context_list,
            metadata={"env": "demo"},
            name=f"{model_name}_RAG",
        )
        
        # Generate the response with the updated history
        model_response, input_tokens, output_tokens, total_tokens = generate_response(prompt, history, model_name)
        
        # Add the current question to the history
        history.append({"role": "user", "content": question})
        # Update history with the new interaction
        history.append({"role": "assistant", "content": model_response})

        print("You: ", question)
        print(f"Assistant: {model_response}")
        print("*" * 100)


        # Log your llm call step to Galileo.
        wf.add_llm(
            input=prompt,
            output=model_response,
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            metadata={"env": "demo"},
            name=f"{model_name}_QA"
        )


        # Conclude the workflow.
        wf.conclude(output={"output": model_response, "expected_output": context})
        rounds += 1
    evaluate_run.finish()