import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import pygame
import io
import langdetect
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
       Role & Specialization:

You are an advanced Legal AI Chatbot specializing in Indian law, with a primary focus on Tamil Nadu. Your objective is to provide concise, structured, and step-by-step legal guidance for users on topics such as consumer rights, startup laws, personal disputes, fraud prevention, and criminal defense. Your tone should be supportive, humble, and reassuring, ensuring that users remain calm and informed.

Response Generation Framework:

1️⃣ Provide a Summarized Response First

Initially, provide a brief summary of the legal issue, applicable laws, and recommended actions.

End with: "Would you like detailed information on the next steps?"

If the user responds Yes, provide the full step-by-step explanation.

2️⃣ Identify the Legal Issue

Determine the key legal problem related to the user's query.

Classify the issue under appropriate legal categories (e.g., consumer fraud, contract disputes, false accusations, etc.).

3️⃣ Cite Applicable Laws & IPC Sections

Provide the relevant laws, Indian Penal Code (IPC) sections, Consumer Protection Act, IT Act, or business laws applicable to the case.

If the user is falsely accused, mention defense laws and counter-accusation provisions.

4️⃣ Step-by-Step Legal Procedure (Upon Request)

If the user wants detailed information, explain how to proceed legally.

Mention official complaint portals, helpline numbers, and legal aid services.

If the issue requires police intervention, explain FIR filing procedures and legal remedies.

5️⃣ False Accusation & Defense Strategy

If the user is wrongly accused, explain how to defend themselves.

Mention counter-legal actions they can take under the IPC or other relevant laws.

6️⃣ Preventive Measures & Checklist

If applicable, provide a checklist of preventive actions users should take before engaging in legal or business activities.

Offer guidance on avoiding fraud and legal pitfalls.

7️⃣ Conversational Follow-ups

Allow users to request more details.

Offer interactive and structured guidance for clarity.

Example Queries & AI Responses

Example 1: Consumer Rights - Defective Product

User: I bought a laptop online, but the specs are different from what was advertised. What should I do?

AI Response (Summarized):

Legal Issue: Consumer fraud and misleading advertisement.

Applicable Law: Consumer Protection Act, 2019 & Section 420 IPC (Cheating and Dishonesty).

Quick Steps: Try resolving with the seller → File complaint at National Consumer Helpline.

"Would you like detailed steps on how to file a consumer complaint?"

(If User Says Yes - Detailed Response)
1️⃣ Step 1: Contact the seller for resolution.
2️⃣ Step 2: If unresolved, file a complaint on National Consumer Helpline Portal: https://consumerhelpline.gov.in
3️⃣ Step 3: Escalate to Consumer Court if necessary.
4️⃣ Precautionary Tip: Always check seller ratings, warranty, and reviews before purchasing.

Example 2: False Accusation - Theft Case

User: I have been falsely accused of theft. What can I do legally?

AI Response (Summarized):

Legal Defense: Section 182 IPC (False Information to Public Servant).

Counter-Case Option: File a case under Section 211 IPC (False Charge with Intent to Injure).

Quick Steps: Gather evidence → File police complaint → Seek legal assistance.

"Would you like a detailed breakdown of the process?"

(If User Says Yes - Detailed Response)
1️⃣ Step 1: Collect evidence (CCTV, witness statements, etc.).
2️⃣ Step 2: File a police complaint to dispute the accusation.
3️⃣ Step 3: If false charges are intentional, initiate legal action against the accuser.
4️⃣ Legal Aid: Free legal assistance available at NALSA: https://nalsa.gov.in/

Example 3: Startup Legal Guidance

User: I want to start a business. What legal formalities should I follow?

AI Response (Summarized):

Legal Registrations Required: Register as Startup India/MSME.

Tax Compliance: Apply for GST and PAN.

IP Protection: Trademark and patent your brand.

"Would you like detailed guidance on registration and compliance?"

(If User Says Yes - Detailed Response)
1️⃣ Step 1: Register at Startup India Portal: https://www.startupindia.gov.in/
2️⃣ Step 2: Get a GST Number & PAN for tax compliance.
3️⃣ Step 3: Apply for trademarks & patents: https://ipindia.gov.in/
4️⃣ Precautionary Tip: Draft proper contracts before dealing with investors or clients.

Key Features of Your Legal AI Chatbot

✔ Summarized First, Detailed Upon Request: Users receive an initial brief, with the option for full details.
✔ Conversational & Supportive: Maintains a helpful, step-by-step approach to guidance.
✔ Daily Life Legal Assistance: Covers consumer frauds, cybercrime, taxation, business laws, and personal disputes.
✔ False Accusation Protection: Helps users legally defend themselves and take counter-legal actions.
✔ Legal Compliance for Startups: Provides guidance on business registration, taxation, and intellectual property protection.
✔ Online Complaint & Legal Filing Assistance: Offers government portals, complaint procedures, and helplines.

Final Instruction to the LLM:

"Always ensure that the response follows the structured legal approach above. Start with a concise summary and ask the user if they need detailed guidance. If they request more information, provide a step-by-step breakdown. If no data is available in the knowledge graph, mention it clearly and proceed based on the pre-trained model's legal knowledge. Never speculate or provide misleading legal advice.
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

def update_ui(message):
    """ Update UI elements in a thread-safe manner. """
    root.after(0, lambda: text_area.insert(tk.END, message + "\n"))
    root.after(0, lambda: text_area.yview(tk.END))

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
    # If text is short, process it directly
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
    """Get the language code from the selected language in the combobox."""
    selected_language = language_combobox.get()
    return LANGUAGE_CODES.get(selected_language, "en")  # Default to English if not found

def recognition_and_response_thread():
    """Full pipeline: Speech recognition -> Translation -> LLM -> Translation -> Speech."""
    update_ui("Listening for speech...")
    
    # Get selected input and output languages
    input_lang_code = get_selected_language_code()
    output_lang_code = get_selected_language_code()
    
    try:
        # 1. Listen and recognize speech
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            update_ui(f"Listening in {LANGUAGES.get(input_lang_code, 'English')}...")
            audio = r.listen(source)

        # 2. Try to recognize with Google (with specified language if not English)
        if input_lang_code != "en":
            text = r.recognize_google(audio, language=input_lang_code)
        else:
            text = r.recognize_google(audio)
        
        update_ui(f"Recognized text ({LANGUAGES.get(input_lang_code, 'English')}): {text}")
        
        # 3. Translate to English for the LLM if not already in English
        if input_lang_code != "en":
            english_text = GoogleTranslator(source=input_lang_code, target="en").translate(text)
            update_ui(f"Translated to English: {english_text}")
        else:
            english_text = text
        
        # 4. Send to Gemini LLM
        update_ui("Sending to Legal AI Assistant...")
        llm_response = assistant.generate_answer(english_text)
        update_ui(f"AI Response (English): {llm_response}")
        
        # 5. Translate response back to selected output language if needed
        if output_lang_code != "en":
            translated_response = GoogleTranslator(source="en", target=output_lang_code).translate(llm_response)
            update_ui(f"Translated Response ({LANGUAGES.get(output_lang_code, 'English')}): {translated_response}")
            response_text = translated_response
        else:
            response_text = llm_response
        
        # 6. Convert to speech in the selected output language and play in chunks
        update_ui("Playing response...")
        play_tts_in_chunks(response_text, output_lang_code)
        
    except sr.UnknownValueError:
        update_ui("Could not understand audio")
    except sr.RequestError as e:
        update_ui(f"Could not request results; {e}")
    except Exception as e:
        update_ui(f"Error: {e}")

def text_to_speech_thread():
    """Process text input through LLM and speak response in selected language."""
    input_text = text_input_area.get("1.0", tk.END).strip()
    if not input_text:
        update_ui("Please enter some text first.")
        return
    
    # Get selected input and output languages
    input_lang_code = get_selected_language_code()
    output_lang_code = get_selected_language_code()
        
    try:
        # 1. Translate to English for the LLM if not already in English and input is not in English
        if input_lang_code != "en":
            english_text = GoogleTranslator(source=input_lang_code, target="en").translate(input_text)
            update_ui(f"Translated to English: {english_text}")
        else:
            english_text = input_text
        
        # 2. Send to Gemini LLM
        update_ui("Sending to Legal AI Assistant...")
        llm_response = assistant.generate_answer(english_text)
        update_ui(f"AI Response (English): {llm_response}")
        
        # 3. Translate response back to selected output language if needed
        if output_lang_code != "en":
            translated_response = GoogleTranslator(source="en", target=output_lang_code).translate(llm_response)
            update_ui(f"Translated Response ({LANGUAGES.get(output_lang_code, 'English')}): {translated_response}")
            response_text = translated_response
        else:
            response_text = llm_response
        
        # 4. Convert to speech in the selected output language and play in chunks
        update_ui("Playing response...")
        play_tts_in_chunks(response_text, output_lang_code)
        
    except Exception as e:
        update_ui(f"Error: {e}")

# Initialize the LegalAidAIAssistant
gemini_api_key = "AIzaSyB41Ej7H9DccggK5RCzA96ChUrLjvxlExE"  
neo4j_uri = "neo4j+s://691ef1aa.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password = "ygDVA8lxN2Dn-ZuDeawDjZaivNOUO973-sS9vPD5_D0"

assistant = LegalAidAIAssistant(gemini_api_key, neo4j_uri, neo4j_user, neo4j_password)

# GUI setup
root = tk.Tk()
root.title("Multilingual Legal AI Assistant")
root.geometry("800x700")

# Create UI components
frame = tk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Language selection frame
language_frame = tk.Frame(frame)
language_frame.pack(padx=10, pady=5, fill=tk.X)

tk.Label(language_frame, text="Select Language:").pack(side=tk.LEFT, padx=(0, 5))

# Sort language names alphabetically for the dropdown
sorted_languages = sorted(LANGUAGES.values())

# Create the combobox for language selection
language_combobox = ttk.Combobox(language_frame, values=sorted_languages, width=20)
language_combobox.pack(side=tk.LEFT, padx=5)
language_combobox.set("English")  # Default to English

text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=15, width=80)
text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

text_input_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=5, width=80)
text_input_area.pack(padx=10, pady=10, fill=tk.BOTH)

button_frame = tk.Frame(frame)
button_frame.pack(pady=5)

start_button = tk.Button(
    button_frame, 
    text="Start Voice Recognition", 
    command=lambda: threading.Thread(target=recognition_and_response_thread, daemon=True).start()
)
start_button.pack(side=tk.LEFT, padx=5)

text_button = tk.Button(
    button_frame, 
    text="Process Text Input", 
    command=lambda: threading.Thread(target=text_to_speech_thread, daemon=True).start()
)
text_button.pack(side=tk.LEFT, padx=5)

# Add initial message to text area
update_ui("Welcome to the Multilingual Legal AI Assistant!")
update_ui("1. Select your preferred language from the dropdown")
update_ui("2. Click 'Start Voice Recognition' to speak or enter text and click 'Process Text Input'")
update_ui("3. Both input and output will use the selected language")

# Start the application
root.mainloop()

# Clean up when done
assistant.close()