import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import re
from collections import deque


load_dotenv()

app = Flask(__name__)
from flask_cors import CORS

# After app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# Initialize conversation history with a maximum of 5 turns
conversation_history = deque(maxlen=5)

with open("courses.json") as f:
    courses = json.load(f)

from difflib import get_close_matches

def normalize_term(term):
    # Convert to lowercase and replace common separators with spaces
    term = term.lower()
    term = re.sub(r'[_-]', ' ', term)
    return term

def recommend_courses(user_input):
    user_input = normalize_term(user_input)
    user_words = user_input.split()
    matches = []

    for course in courses:
        for tag in course["tags"]:
            normalized_tag = normalize_term(tag)
            # Check for exact matches in normalized terms
            if any(normalized_tag == normalize_term(word) for word in user_words):
                matches.append(course["title"])
                break
            # Check for partial matches (e.g., "data" in "data science")
            elif any(word in normalized_tag or normalized_tag in word for word in user_words):
                matches.append(course["title"])
                break
    return matches

def format_chat_template(messages):
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append(f"<s>[INST] {msg['content']} [/INST]")
        else:
            formatted_messages.append(f"{msg['content']}</s>")
    return "".join(formatted_messages)

def ask_llm(messages):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }
    
    # Format the messages according to Mistral's chat template
    formatted_prompt = format_chat_template(messages)
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        },
    }
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL}",
        headers=headers,
        json=payload
    )
    
    # Clean the response by removing chat template artifacts and system messages
    raw_response = response.json()[0]["generated_text"]
    
    # Remove all system messages and chat template tags
    cleaned_response = re.sub(r'You are a helpful e-learning assistant.*?Be conversational and helpful\.</s>', '', raw_response)
    cleaned_response = re.sub(r'<s>\[INST\].*?\[/INST\]', '', cleaned_response)
    cleaned_response = re.sub(r'</s>', '', cleaned_response)
    
    # Remove any leading/trailing whitespace and empty lines
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Get course recommendations
    recommended = recommend_courses(user_input)
    course_list = ", ".join(recommended) if recommended else "No matches found."
    
    # Create system message with context
    system_message = {
        "role": "system",
        "content": f"You are a helpful e-learning assistant. You can recommend courses from our catalog. Available courses: {course_list}. Be conversational and helpful."
    }
    
    # Prepare messages for the model
    messages = [system_message] + list(conversation_history)
    
    # Get response from the model
    reply = ask_llm(messages)
    
    # Add assistant's response to conversation history
    conversation_history.append({"role": "assistant", "content": reply})
    
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)
