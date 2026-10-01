import os
import json
import re
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pypdf import PdfReader

load_dotenv()

st.set_page_config(
    page_title="AI Flashcard Generator",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.card {
    padding: 28px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.25);
    margin: 12px 0;
    min-height: 150px;
}
.question { font-size: 22px; font-weight: 700; }
.answer { font-size: 18px; margin-top: 16px; }
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(100,100,100,.12);
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

DEFAULT_MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-120b:fastest")


def extract_pdf(file):
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def clean_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return text


def generate_flashcards(notes, count, difficulty, model):
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add your Hugging Face token to the .env file."
        )

    client = InferenceClient(
        api_key=token,
        provider="auto",
    )

    system_prompt = """You are an expert educational flashcard generator.
Create accurate study flashcards ONLY from the supplied study material.
Return ONLY valid JSON, with no markdown and no extra text.
The JSON must be an array of objects using exactly these keys:
question, answer, difficulty.
Keep each question and answer concise.
Do not invent facts that are not supported by the study material.
"""

    user_prompt = f"""Study material:
{notes}

Generate exactly {count} flashcards.
Difficulty: {difficulty}

Requirements:
- One clear question per card.
- Answer should be short but sufficient.
- Prefer definitions, key concepts, differences, formulas, processes, and important facts.
- difficulty must be one of: Easy, Medium, Hard.
- Output only a JSON array.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=max(1000, count * 180),
    )

    content = response.choices[0].message.content
    data = json.loads(clean_json(content))

    if not isinstance(data, list):
        raise ValueError("The model did not return a JSON array.")

    cards = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        d = str(item.get("difficulty", "Medium")).strip().title()
        if q and a:
            cards.append({
                "question": q,
                "answer": a,
                "difficulty": d if d in {"Easy", "Medium", "Hard"} else "Medium",
            })

    return cards[:count]


def cards_to_csv(cards):
    df = pd.DataFrame(cards)
    return df.to_csv(index=False).encode("utf-8")


st.title("🧠 AI Flashcard Generator")
st.caption("Generate study flashcards from your notes using a Hugging Face model.")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Hugging Face model", value=DEFAULT_MODEL)
    count = st.slider("Number of cards", 5, 30, 10)
    difficulty = st.selectbox(
        "Difficulty",
        ["Mixed", "Easy", "Medium", "Hard"],
        index=0,
    )
    st.divider()
    st.markdown("### Hugging Face setup")
    st.write("Put your token in `.env` as `HF_TOKEN=...`.")
    st.caption("Use a token with permission to make Inference Providers calls.")

if "cards" not in st.session_state:
    st.session_state.cards = []

tab1, tab2 = st.tabs(["Generate", "Flashcards"])

with tab1:
    st.subheader("1. Add your study material")

    uploaded = st.file_uploader(
        "Upload a TXT or PDF file",
        type=["txt", "pdf"],
    )

    notes = st.text_area(
        "Or paste your notes here",
        height=300,
        placeholder="Example: The OSI model has seven layers...",
    )

    if uploaded:
        if uploaded.name.lower().endswith(".pdf"):
            extracted = extract_pdf(uploaded)
        else:
            extracted = uploaded.getvalue().decode("utf-8", errors="ignore")

        if extracted.strip():
            notes = extracted
            st.success(f"Loaded {len(extracted):,} characters from {uploaded.name}.")
        else:
            st.warning("No readable text was found in the uploaded file.")

    if st.button("✨ Generate Flashcards", type="primary", use_container_width=True):
        if not notes.strip():
            st.error("Please paste notes or upload a TXT/PDF file.")
        else:
            with st.spinner("Generating flashcards with Hugging Face..."):
                try:
                    cards = generate_flashcards(
                        notes=notes,
                        count=count,
                        difficulty=difficulty,
                        model=model.strip(),
                    )
                    st.session_state.cards = cards
                    st.success(f"Generated {len(cards)} flashcards.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.info(
                        "Check HF_TOKEN, model availability, internet access, "
                        "and that your Hugging Face token has Inference Providers permission."
                    )

with tab2:
    st.subheader("2. Review your flashcards")

    cards = st.session_state.cards

    if not cards:
        st.info("Generate flashcards first.")
    else:
        st.write(f"**{len(cards)} cards ready**")

        for i, card in enumerate(cards, start=1):
            with st.expander(
                f"Card {i}: {card['question']}",
                expanded=False,
            ):
                st.markdown(f"<span class='badge'>{card['difficulty']}</span>", unsafe_allow_html=True)
                st.markdown("**Answer**")
                st.write(card["answer"])

        st.download_button(
            "⬇️ Download CSV",
            data=cards_to_csv(cards),
            file_name="flashcards.csv",
            mime="text/csv",
            use_container_width=True,
        )

        json_bytes = json.dumps(cards, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "⬇️ Download JSON",
            data=json_bytes,
            file_name="flashcards.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.button("🗑️ Clear cards"):
            st.session_state.cards = []
            st.rerun()

st.divider()
st.caption("Built with Streamlit + Hugging Face Inference Providers.")
