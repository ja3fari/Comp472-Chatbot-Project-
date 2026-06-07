import pandas as pd


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


def conversation_loop(knowledge_base):
    print("\nWelcome to Student Support AI")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        print("Bot: Processing your question...")
        print("Bot: Semantic search will answer this later.\n")


def main():
    knowledge_base = load_knowledge_base("knowledge_base.csv")

    if not knowledge_base:
        print("Knowledge base could not be loaded.")
        return

    print("Knowledge base loaded successfully.")
    print(f"Total questions loaded: {len(knowledge_base)}")

    conversation_loop(knowledge_base)


if __name__ == "__main__":
    main()