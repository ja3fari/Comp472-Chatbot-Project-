import csv

def load_knowledge_base(filename):
    knowledge_base = []

    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            knowledge_base.append({
                "question": row["question"],
                "answer": row["answer"]
            })

    return knowledge_base


def conversation_loop(knowledge_base):
    print("Chatbot is ready. Type 'quit' to exit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        print("Bot: I received your question.")
        print("Bot: Semantic search will answer this later.")


def main():
    knowledge_base = load_knowledge_base("knowledge_base.csv")
    print("Knowledge base loaded successfully.")
    print(f"Total questions loaded: {len(knowledge_base)}")

    conversation_loop(knowledge_base)


if __name__ == "__main__":
    main()