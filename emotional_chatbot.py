import openai
import os
from dotenv import load_dotenv
from textblob import TextBlob

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def analyze_emotion(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def generate_response(prompt, emotion):
    openai.api_key = API_KEY
    system_prompt = f"You are an empathetic assistant. The user's mood is {emotion}. Respond with supportive and caring language."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response["choices"][0]["message"]["content"]

if __name__ == '__main__':
    user_input = input("You: ")
    emotion = analyze_emotion(user_input)
    response = generate_response(user_input, emotion)
    print("AI:", response)
