# AI Student Advising Chatbot
This project is a Python-based Concordia student advising chatbot. It loads a knowledge base from a CSV file and runs a conversation loop where users can type questions. Using semantic search and sentiment anaylsis, the program understands user sentiment and matches best answers to user questions to respond accordingly to users' concerns.

## Current Features
- Loads questions and answers from `knowledge_base.csv`
- Starts a chatbot conversation loop
- Allows the user to type questions
- Exits when the user types `quit`
- Converts knowledge_base questions and answers into embeddings using pre-trained model
- Find closest user question and answer matches using cosine similarity and semantic search
- Detect user question sentiment, whether positive neutral, negative
- Recommends special actions based on sentiment criteria

## How to Run
```bash
python3 main.py
```
