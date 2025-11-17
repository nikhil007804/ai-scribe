# REQUIREMENTS:
# streamlit==1.28.1
# requests==2.31.0
# python-dotenv==1.0.0
#
# Install with: pip install -r requirements.txt
# Run with: streamlit run app.py

import os
import streamlit as st
import requests
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get API keys from Streamlit secrets or environment variables
try:
    ASSEMBLE_API_KEY = st.secrets.get("ASSEMBLE_API_KEY") or os.getenv("ASSEMBLE_API_KEY")
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
except:
    ASSEMBLE_API_KEY = os.getenv("ASSEMBLE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="AI Scribe — Windsurf", layout="wide")
st.title("AI Medical Scribe (AssembleAI + Gemini)")

def upload_to_assemblyai(file_bytes, filename):
    upload_url = "https://api.assemblyai.com/v2/upload"
    headers = {"authorization": ASSEMBLE_API_KEY}

    response = requests.post(
        upload_url,
        headers=headers,
        data=file_bytes
    )
    if response.status_code != 200:
        st.error("Error uploading to AssemblyAI")
        st.write(response.text)
        return None

    return response.json()['upload_url']

def transcribe_with_assemblyai(audio_url: str):
    headers = {"authorization": ASSEMBLE_API_KEY}

    config = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "format_text": True,
        "punctuate": True,
        "speech_model": "universal",
        "language_detection": True
    }

    response = requests.post("https://api.assemblyai.com/v2/transcript", json=config, headers=headers)
    transcript_id = response.json()['id']
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    with st.status("Transcribing audio with AssemblyAI…", expanded=True) as status:
        while True:
            result = requests.get(polling_endpoint, headers=headers).json()

            if result['status'] == 'completed':
                status.update(label="Transcription completed!", state="complete")
                return result['text']

            elif result['status'] == 'error':
                status.update(label="Transcription failed", state="error")
                raise RuntimeError(result['error'])

            else:
                time.sleep(2)

def call_gemini(prompt: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    response = requests.post(url, json=payload)

    try:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        st.write("Gemini API Error:", response.text)
        return ""

windsurf_prompt_template = """
You are an expert medical scribe. Based on the following medical consultation transcript, generate comprehensive medical documentation in the following format:

## SOAP NOTE

### Subjective
- Chief Complaint
- History of Present Illness
- Past Medical History
- Medications
- Allergies
- Social History

### Objective
- Vital Signs
- Physical Examination
- Lab Results (if mentioned)

### Assessment
- Primary Diagnosis
- Differential Diagnoses
- Clinical Impression

### Plan
- Treatment Plan
- Medications
- Follow-up
- Patient Education

## HISTORY & PHYSICAL (H&P)

### History of Present Illness
### Past Medical History
### Medications
### Allergies
### Social History
### Review of Systems
### Physical Examination
### Assessment and Plan

---

Transcript to process:
{{TRANSCRIPT_HERE}}

Generate professional medical documentation based on this transcript. Use proper medical terminology and formatting.
"""

# UI Layout
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Input")
    input_method = st.radio("Choose input method:", ["Upload Audio", "Paste Transcript"], horizontal=False)
    
    if input_method == "Upload Audio":
        audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
        transcript_text = ""
    else:
        audio = None
        transcript_text = st.text_area("Paste your transcript here:", height=300, placeholder="Enter medical consultation transcript...")

with col2:
    st.subheader("⚙️ Settings")
    st.info("🔧 Configuration")
    include_soap = st.checkbox("Generate SOAP Note", value=True)
    include_hp = st.checkbox("Generate H&P", value=True)
    
    st.markdown("---")
    st.subheader("🚀 Generate")
    run = st.button("Generate Documentation", use_container_width=True, type="primary")

if run:
    final_transcript = transcript_text.strip() if transcript_text else ""

    if audio and not final_transcript:
        st.info("📤 Uploading audio to AssemblyAI...")
        file_bytes = audio.read()
        audio_url = upload_to_assemblyai(file_bytes, audio.name)

        if audio_url:
            final_transcript = transcribe_with_assemblyai(audio_url)

    if not final_transcript:
        st.error("❌ No transcript found. Please upload audio or paste a transcript.")
        st.stop()

    full_prompt = windsurf_prompt_template.replace("{{TRANSCRIPT_HERE}}", final_transcript)

    with st.spinner("🤖 Generating medical documentation..."):
        result = call_gemini(full_prompt)

    if result:
        st.success("✅ Documentation generated successfully!")
        st.markdown("---")
        st.subheader("📄 Generated Medical Documentation")
        st.markdown(result)
        
        # Download button
        st.download_button(
            label="📥 Download as Markdown",
            data=result,
            file_name="medical_documentation.md",
            mime="text/markdown"
        )
    else:
        st.error("❌ Failed to generate documentation. Please try again.")