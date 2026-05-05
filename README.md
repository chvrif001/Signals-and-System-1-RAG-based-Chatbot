# **Signals and Systems 1 RAG-based chatbot**

This is a Llama-powered chatbot designed to assist students in learning and understanding Signals and Systems 1 concepts through interactive conversations, mathematical problem solving, and retrieval of course-specific knowledge using Retrieval-Augmented Generation (RAG).

This chatbot serves as an academic support tool that extends beyond traditional university resources and lecture hours, providing students with academic help anytime and anywhere.

## **Overview**
This project implements a chatbot that:
- Answers course-related questions
- Solves signals and systems problems
- Provides step-by-step explanations
- Uses **RAG** to retrieve information from course materials
- The chatbot is acccessible via Telegram

## **Technologies Used**
- Python
- Telegram Bot API
- LLM (Open-weights **Llama-3 8B**)
- RAG Pipeline
- Sentence transformers
- Chunking Strategy
- Vector Database
- SymPy, NumPy, Matplotlib
- Implemented on Google Colab
- Deployed on Railway Hosting via Github

## **System Architecture**
```markdown
- User
  ↓
- Telegram
  ↓
- Bot Interface
  ↓
- Query Processing
  ↓
- RAG Pipeline → Vector Database
  ↓
- LLM (Response Generation)
  ↓
- Mathematical Engine
  ↓
- Final Response
