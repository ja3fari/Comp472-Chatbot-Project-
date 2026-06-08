import pandas as pd

from sentiment import SentimentAnalyzer
from escalation import get_escalation_message
from semantic_search import load_model, generate_embeddings, find_best_answer

def load_knowledge_base(filename):
    try:
        data = pd.read_csv(filename)

        knowledge_base = []

        for _, row in data.iterrows():
            knowledge_base.append({
                "question": row["question"],
                "answer": row["answer"]
            })

        return knowledge_base

    except FileNotFoundError:
        print("Error: knowledge_base.csv file not found.")
        return []

    except Exception as error:
        print(f"Error loading knowledge base: {error}")
        return []


def conversation_loop(knowledge_base, analyzer, model, kb_questions, kb_answers, kb_embeddings):
    print("\nWelcome to Student Support AI")
    print("Type 'quit' to exit.\n")

    kb_questions = [item["question"] for item in knowledge_base]
    kb_answers   = [item["answer"]   for item in knowledge_base]

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() == "quit":
            print("Goodbye!")
            break

        sentiment = analyzer.analyze(user_input)
        print(f"Sentiment: {sentiment}")

        escalation = get_escalation_message(sentiment)
        if escalation:
            print(f"Recommended escalation: {escalation}")

        answer = find_best_answer(user_input, model, kb_questions, kb_answers, kb_embeddings)
        print(f"Answer: {answer}\n")


def main():
    knowledge_base = load_knowledge_base("knowledge_base.csv")

    if not knowledge_base:
        print("Knowledge base could not be loaded.")
        return

    print("Knowledge base loaded successfully.")
    print(f"Total questions loaded: {len(knowledge_base)}")

    analyzer = SentimentAnalyzer()
    
    model         = load_model()
    kb_questions  = [item["question"] for item in knowledge_base]
    kb_answers    = [item["answer"]   for item in knowledge_base]
    kb_embeddings = generate_embeddings(model, kb_questions)

    conversation_loop(knowledge_base, analyzer, model, kb_questions, kb_answers, kb_embeddings)

if __name__ == "__main__":
    main()