import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import threading
import pygame
import io
from neo4j import GraphDatabase
import requests
import os

# Initialize recognizer and pygame mixer for audio playback
r = sr.Recognizer()
pygame.mixer.init()

# Language code mapping
LANGUAGES = {
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "ml": "Malayalam",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ur": "Urdu",
    "as": "Assamese",
    "or": "Odia",
    "kok": "Konkani",
    "mai": "Maithili",
    "sat": "Santali",
    "ne": "Nepali",
    "sd": "Sindhi",
    "ks": "Kashmiri",
    "doi": "Dogri",
    "mni": "Manipuri (Meitei)",
    "sa": "Sanskrit",
    "brx": "Bodo",
    "tcy": "Tulu",
    "en": "English"
}

# Reverse mapping for language code lookup
LANGUAGE_CODES = {v: k for k, v in LANGUAGES.items()}

class GeminiAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def ask(self, prompt):
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}

        response = requests.post(f"{self.base_url}?key={self.api_key}", headers=headers, json=data)

        if response.status_code == 200:
            try:
                json_response = response.json()
                return json_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response received.")
            except requests.exceptions.JSONDecodeError:
                return "Error: Received an invalid response from the Gemini API."
        else:
            return f"Error: API request failed with status {response.status_code}: {response.text}"

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query):
        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = [record.data() for record in result]
                return records if records else None
        except Exception as e:
            return None  # Prevents breaking flow

class LegalAidAIAssistant:
    def __init__(self, gemini_api_key, neo4j_uri, neo4j_user, neo4j_password):
        self.gemini = GeminiAPI(gemini_api_key)
        self.neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)

    def generate_answer(self, question):
        """Fetch legal data from Neo4j if available; otherwise, answer using AI."""
        system_prompt = """
        [Your system prompt here...]
        """
        # Step 1: Generate a Cypher query for Neo4j
        cypher_prompt = f"""
        Convert the following legal question into a Cypher query for Neo4j.
        Ensure that the label 'Legal' and property 'topic' exist before generating the query.
        Question: {question}
        Cypher Query:
        """
        cypher_query = self.gemini.ask(cypher_prompt)

        # Step 2: Attempt to fetch knowledge from Neo4j
        knowledge = self.neo4j.query(cypher_query)

        # Step 3: Construct AI prompt based on Neo4j data availability
        if knowledge:
            final_prompt = f"{system_prompt}\nUser: {question}\nLegal Knowledge from Neo4j: {knowledge}\nAssistant:"
        else:
            final_prompt = f"{system_prompt}\nUser: {question}\n(Note: No relevant legal data found in Neo4j. Answering with pre-trained knowledge.)\nAssistant:"

        return self.gemini.ask(final_prompt)

    def close(self):
        self.neo4j.close()

def play_audio(audio_data):
    """ Play audio from a byte stream using pygame. """
    temp_file = "temp_audio.mp3"
    with open(temp_file, "wb") as f:
        f.write(audio_data)
    pygame.mixer.music.load(temp_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    # Clean up temp file after playback
    try:
        os.remove(temp_file)
    except:
        pass

def play_tts_in_chunks(text, lang_code):
    """Split long text into smaller chunks for faster TTS processing."""
    if len(text) < 500:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)
        play_audio(audio_data.read())
        return
        
    # For longer text, split into sentences and process in smaller groups
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk)
    
    for chunk in chunks:
        tts = gTTS(text=chunk, lang=lang_code, slow=False)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)
        play_audio(audio_data.read())

def get_selected_language_code():
    """Get the language code from the selected language in the dropdown."""
    selected_language = st.session_state.get("selected_language", "English")
    return LANGUAGE_CODES.get(selected_language, "en")  # Default to English if not found

def recognition_and_response_thread():
    """Full pipeline: Speech recognition -> Translation -> LLM -> Translation -> Speech."""
    st.session_state["status"] = "Listening for speech..."
    
    # Get selected input and output languages
    input_lang_code = get_selected_language_code()
    output_lang_code = get_selected_language_code()
    
    try:
        # 1. Listen and recognize speech
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            st.session_state["status"] = f"Listening in {LANGUAGES.get(input_lang_code, 'English')}..."
            audio = r.listen(source)

        # 2. Try to recognize with Google (with specified language if not English)
        if input_lang_code != "en":
            text = r.recognize_google(audio, language=f"{input_lang_code}-IN")  # Use "-IN" for Indian languages
        else:
            text = r.recognize_google(audio)
        
        st.session_state["recognized_text"] = f"Recognized text ({LANGUAGES.get(input_lang_code, 'English')}): {text}"
        
        # 3. Translate to English for the LLM if not already in English
        if input_lang_code != "en":
            english_text = GoogleTranslator(source=input_lang_code, target="en").translate(text)
            st.session_state["translated_text"] = f"Translated to English: {english_text}"
        else:
            english_text = text
        
        # 4. Send to Gemini LLM
        st.session_state["status"] = "Sending to Legal AI Assistant..."
        llm_response = assistant.generate_answer(english_text)
        st.session_state["llm_response"] = f"AI Response (English): {llm_response}"
        
        # 5. Translate response back to selected output language if needed
        if output_lang_code != "en":
            translated_response = GoogleTranslator(source="en", target=output_lang_code).translate(llm_response)
            st.session_state["translated_response"] = f"Translated Response ({LANGUAGES.get(output_lang_code, 'English')}): {translated_response}"
            response_text = translated_response
        else:
            response_text = llm_response
        
        # 6. Display the response in the chat
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # 7. Convert to speech in the selected output language and play in chunks
        st.session_state["status"] = "Playing response..."
        play_tts_in_chunks(response_text, output_lang_code)
        
    except sr.UnknownValueError:
        st.session_state["status"] = "Could not understand audio"
    except sr.RequestError as e:
        st.session_state["status"] = f"Could not request results; {e}"
    except Exception as e:
        st.session_state["status"] = f"Error: {e}"

# Initialize the LegalAidAIAssistant
gemini_api_key = "AIzaSyB41Ej7H9DccggK5RCzA96ChUrLjvxlExE"  
neo4j_uri = "neo4j+s://691ef1aa.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password = "ygDVA8lxN2Dn-ZuDeawDjZaivNOUO973-sS9vPD5_D0"

assistant = LegalAidAIAssistant(gemini_api_key, neo4j_uri, neo4j_user, neo4j_password)

# Streamlit UI
st.title("Nyaya AI Chatbot")

# Chat-like interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Language selection
selected_language = st.selectbox("Select Language:", sorted(LANGUAGES.values()), key="selected_language")

# User input
if prompt := st.chat_input("Type your legal query..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        response = assistant.generate_answer(prompt)
        st.markdown(response)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Convert the response to speech
    output_lang_code = get_selected_language_code()
    play_tts_in_chunks(response, output_lang_code)

# Speech-to-text button
if st.button("Start Voice Recognition"):
    threading.Thread(target=recognition_and_response_thread, daemon=True).start()

# Display status and results
if "status" in st.session_state:
    st.write(st.session_state["status"])
if "recognized_text" in st.session_state:
    st.write(st.session_state["recognized_text"])
if "translated_text" in st.session_state:
    st.write(st.session_state["translated_text"])
if "llm_response" in st.session_state:
    st.write(st.session_state["llm_response"])
if "translated_response" in st.session_state:
    st.write(st.session_state["translated_response"])

# Clean up when done
assistant.close()