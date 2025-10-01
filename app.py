import streamlit as st
import requests
import json
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

# Configuration
st.set_page_config(page_title="Personal Voice Bot", page_icon="🎤", layout="wide")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []

# Knowledge Base - Customize this with your personal information
DEFAULT_KNOWLEDGE_BASE = [
    {
        "topic": "life_story",
        "content": """I grew up in Kolkata and pursued my BTech in Computer Science and Engineering at UEM Kolkata. 
       I also completed MTech  in Artificial Intelligence at IISc Bangalore.  Currently I am working as a Data Science intern at Scaler
        Over the years, I’ve developed strong expertise in deep learning, NLP, and generative AI through projects, research, 
        and internships. I enjoy solving technical challenges that combine research rigor with real-world impact.""",
        "keywords": ["life story", "background", "history", "about you", "who are you"]
    },
    {
    "topic": "superpower",
    "content": "My #1 superpower is quickly grasping complex systems and designing end-to-end AI solutions. Whether it’s building a deep learning model from scratch, deploying a scalable recommendation engine, or experimenting with retrieval-augmented generation, I can connect theory with practical implementation seamlessly. Alongside this, I’m a fast learner with the patience to dive deep into challenges, and I have a natural habit of asking 'why' which helps me uncover root causes, challenge assumptions, and design smarter solutions.",
    "keywords": ["superpower", "strength", "best at", "excel at", "good at", "patience", "fast learner", "curiosity", "asking why"]
},
    {
        "topic": "growth_areas",
        "content": """The top 3 areas I'd like to grow in are:
        1. Public speaking and technical communication – to explain AI research in an engaging and clear manner
        2. Leadership and mentorship – to guide teams and junior engineers in building impactful projects
        3. Entrepreneurship and product thinking – to translate AI research into scalable, user-focused products""",
        "keywords": ["grow", "improve", "development", "learning", "weakness", "areas"]
    },
    {
    "topic": "misconceptions",
    "content": """People often think I’m just the nerd who only studies and codes. 
    In reality, I do a lot of other things outside academics and tech. 
    I’m great at table tennis, I hit the gym regularly, I love playing cricket, 
    and I’m a big movie buff. I enjoy balancing my technical passion with sports, fitness, and entertainment.""",
    "keywords": ["misconception", "misunderstand", "wrong about", "assume", "think about you"]
},
       {
    "topic": "pushing_boundaries",
    "content": """I push my boundaries by stepping into situations that challenge me both mentally and physically. 
    In my professional life, I take on projects that require me to learn new skills or explore unfamiliar technologies. 
    Outside of work, I challenge myself by hitting the gym, playing competitive sports like table tennis and cricket, 
    and continuously improving my fitness. I also make an effort to connect with new people and step out of my comfort zone socially, 
    because I believe growth comes not just from technical challenges, but also from building relationships and new perspectives.""",
    "keywords": ["boundaries", "limits", "challenge", "push yourself", "comfort zone"]
}
]


def initialize_knowledge_base():
    """Initialize the knowledge base if not already done"""
    if not st.session_state.knowledge_base:
        st.session_state.knowledge_base = DEFAULT_KNOWLEDGE_BASE

def simple_embed(text: str) -> List[float]:
    """Simple embedding using character frequency (replace with proper embeddings in production)"""
    # Normalize text
    text = text.lower()
    # Create a simple frequency vector
    chars = 'abcdefghijklmnopqrstuvwxyz '
    vector = [text.count(c) / max(len(text), 1) for c in chars]
    return vector

def find_relevant_context(query: str, knowledge_base: List[Dict]) -> str:
    """Find the most relevant context from knowledge base using simple similarity"""
    query_lower = query.lower()
    
    # Score each knowledge entry
    scores = []
    for entry in knowledge_base:
        score = 0
        # Check keyword matches
        for keyword in entry['keywords']:
            if keyword in query_lower:
                score += 2
        
        # Check topic relevance
        if entry['topic'].replace('_', ' ') in query_lower:
            score += 3
            
        scores.append(score)
    
    # Get the entry with highest score
    if max(scores) > 0:
        best_idx = scores.index(max(scores))
        return knowledge_base[best_idx]['content']
    
    # If no good match, return all contexts
    return "\n\n".join([entry['content'] for entry in knowledge_base])

def call_openrouter_api(prompt: str, context: str, api_key: str) -> str:
    """Call OpenRouter API with RAG context"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = f"""You are a personal voice assistant representing someone. 
Answer questions naturally and conversationally based on the following information about the person:

{context}

Keep responses concise (2-4 sentences) and natural, as if speaking. 
If the question isn't covered in the context, politely say you don't have that specific information."""

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error calling API: {str(e)}"

def text_to_speech(text: str) -> str:
    """Convert text to speech and return file path"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def speech_to_text() -> str:
    """Convert speech to text using microphone"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.info("Processing speech...")
            text = recognizer.recognize_google(audio)
            return text
    except sr.WaitTimeoutError:
        st.error("No speech detected. Please try again.")
        return None
    except sr.UnknownValueError:
        st.error("Could not understand audio. Please try again.")
        return None
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
        return None

# Main UI
st.title("🎤 Personal Voice Bot with RAG")
st.markdown("Ask me questions about myself! I'll respond using voice or text.")

# Hardcoded API Key
API_KEY = st.secrets["API_KEY"]

# Initialize knowledge base
initialize_knowledge_base()

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Text input
    user_input = st.text_input("💬 Type your question:", key="text_input")

with col2:
    # Voice input button
    if st.button("🎤 Use Voice", type="primary"):
        voice_input = speech_to_text()
        if voice_input:
            user_input = voice_input
            st.session_state.text_input = voice_input

# Process input
if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Find relevant context using RAG
    with st.spinner("🔍 Searching knowledge base..."):
        relevant_context = find_relevant_context(user_input, st.session_state.knowledge_base)
    
    # Get response from API
    with st.spinner("🤔 Thinking..."):
        response = call_openrouter_api(user_input, relevant_context, API_KEY)
    
    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Generate speech
    with st.spinner("🔊 Generating speech..."):
        audio_file = text_to_speech(response)
    
    # Display response
    st.success("✅ Response generated!")
    
    # Play audio
    if audio_file:
        st.audio(audio_file, format='audio/mp3')
        # Clean up temp file after a delay
        try:
            os.unlink(audio_file)
        except:
            pass

# Display chat history
st.markdown("---")
st.subheader("💬 Conversation History")

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

# Clear chat button
if st.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()
