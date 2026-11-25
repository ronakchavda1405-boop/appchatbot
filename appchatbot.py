
import streamlit as st
import os
import io

from groq import Groq
from dotenv import load_dotenv
from gtts import gTTS

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Voice AI Assistant",
    page_icon="🎙️",
    layout="centered"
    
)

# --------------------------------
# 2. LOAD API KEY
# --------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key not found. Check your .env file.")
    st.stop()

client = Groq(api_key=api_key)

# --------------------------------
# 3. CUSTOM UI DESIGN
# --------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #60A5FA;
}

.subtitle {
    text-align: center;
    color: #B8C4D9;
    font-size: 16px;
}

div.stButton > button {
    border-radius: 12px;
    background-color: #2563EB;
    color: white;
    border: none;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# 4. CHAT HEADER
# --------------------------------

st.markdown(
    '<div class="main-title">🎙️ Voice AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by Groq, Whisper and gTTS</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# 5. INITIALIZE CHAT HISTORY
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful and friendly AI assistant."
        }
    ]

# --------------------------------
# 6. SPEECH TO TEXT
# --------------------------------

def speech_to_text(audio_bytes):

    transcription = client.audio.transcriptions.create(
        file=("recording.wav", audio_bytes),
        model="whisper-large-v3",
        response_format="text",
        temperature=0.0
    )

    return str(transcription).strip()


# --------------------------------
# 7. AI CHATBOT RESPONSE
# --------------------------------

def chatbot_response(user_text):

    st.session_state.messages.append({
        "role": "user",
        "content": user_text
    })

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=st.session_state.messages,
        temperature=0.7,
        max_completion_tokens=500
    )

    answer = response.choices[0].message.content

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# --------------------------------
# 8. TEXT TO SPEECH
# --------------------------------

def text_to_speech(text):

    audio_buffer = io.BytesIO()

    tts = gTTS(
        text=text,
        lang="en",
        slow=False
    )

    tts.write_to_fp(audio_buffer)

    audio_buffer.seek(0)

    return audio_buffer.getvalue()


# --------------------------------
# 9. DISPLAY CHAT HISTORY
# --------------------------------

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant":
            audio = text_to_speech(message["content"])
            st.audio(audio, format="audio/mp3")


# --------------------------------
# 10. SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    st.write("Your AI voice assistant")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a helpful and friendly AI assistant."
            }
        ]

        st.rerun()

    st.divider()

    st.write("**AI Models**")
    st.caption("Speech Recognition: Whisper")
    st.caption("AI Response: Groq Llama")
    st.caption("Voice Generation: gTTS")


# --------------------------------
# 11. VOICE INPUT
# --------------------------------

st.subheader("🎤 Speak with your AI assistant")

audio_input = st.audio_input(
    "Click to record your voice",
    sample_rate=16000
)

if audio_input is not None:

    if st.button("🚀 Process Voice"):

        try:

            with st.spinner("Converting your voice into text..."):

                user_text = speech_to_text(
                    audio_input.getvalue()
                )

            if not user_text:

                st.warning("No speech detected. Please try again.")

            else:

                st.success("Voice converted successfully!")

                st.markdown("**You said:**")
                st.info(user_text)

                with st.spinner("AI is thinking..."):

                    answer = chatbot_response(user_text)

                st.rerun()

        except Exception as e:

            st.error(f"Error: {e}")


# --------------------------------
# 12. TEXT INPUT OPTION
# --------------------------------

st.divider()

st.subheader("💬 Or type your message")

prompt = st.chat_input("Ask me anything...")

if prompt:

    try:

        with st.spinner("AI is thinking..."):

            answer = chatbot_response(prompt)

        st.rerun()

    except Exception as e:

        st.error(f"Error: {e}")
