import streamlit as st
import streamlit.components.v1 as components
import logging
import sys
import os
import cv2
import traceback   
import pandas as pd 
import altair as alt# ✅ add this line
from predict_fixed import predict_image_streamlit
from PIL import Image




# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the prediction function
from predict_fixed import predict_image_streamlit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Page Setup
# -------------------------------------------------------
st.set_page_config(
    page_title="Digital Screening Aid - Down Syndrome Detection",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------------------------------
# Custom CSS — DNA Helix Background + Gradient Overlay
# -------------------------------------------------------
st.markdown("""
<style>
.stApp {
    position: relative !important;
    background: linear-gradient(120deg, #d0e7f9, #f0f7ff) !important;
    min-height: 100vh;
    font-family: 'Poppins', sans-serif !important;
}

/* Magical floating bubbles */
.stApp:before, .stApp:after {
    content: "";
    position: absolute;
    z-index: 0;
    border-radius: 50%;
}
.stApp:before {
    top: 5%;
    left: 70%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, #bbe3f7 65%, #e3edf7 100%);
    opacity: 0.2;
}
.stApp:after {
    bottom: 10%;
    right: 65%;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, #a8d0ff 65%, #e3edf7 100%);
    opacity: 0.15;
}

/* Text styles */
body, .stApp {
    color: #233248;
    font-size: 17px;
}
h1,h2,h3 {
    color: #165282;
    font-family: 'Nunito', sans-serif;
    font-weight: 900;
    letter-spacing: -1.2px;
    margin-bottom: 10px;
}

/* Cards */
.upload-card, .result-card {
    background: rgba(255,255,255,0.93);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    text-align: center;
    margin-top: 18px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #81D4FA, #F8BBD0);
    color: #263146;
    font-weight: 700;
    border: none;
    border-radius: 14px;
    padding: 0.7em 1.2em;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #4FC3F7, #F48FB1);
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# Top navigation bar
# -------------------------------------------------------
nav_options = ["🏠 Home", "📤 Upload Image", "ℹ️ About", "❓ Help"]
if "page" not in st.session_state:
    st.session_state.page = nav_options[0]

cols = st.columns(len(nav_options))
for i, name in enumerate(nav_options):
    if cols[i].button(name):
        st.session_state.page = name
page = st.session_state.page

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
if page == "🏠 Home":
    home_html = """
    <div class="hero-bubble fade-in" style="display:flex; flex-direction:column; align-items:center; justify-content:center; gap:30px; padding-top:40px;">

        <!-- Centered white card with heading and intro -->
        <div class="hero-card" style="
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 25px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            max-width: 650px;
            text-align:center;
        ">
            <h1>👶 Digital Screening Aid for<br>
                <span style="color:#56a5f7;">Down</span> <span style="color:#58d7c8;">Syndrome</span>
                <span style="color:#f7b788;">Detection</span>
            </h1>
            <p style="font-size:1.18em; margin-top:16px; color:#213347;">
                Welcome to the <b>Digital Screening Aid</b>, an AI-powered healthcare tool designed 
                for the preliminary screening of <b>Down Syndrome</b> features using facial image analysis.
            </p>
        </div>

        <!-- Two centered paragraphs below the card -->
        <div style="width:100%; text-align:center; margin-top:30px;">
            <p style="font-size:1.1em; color:#444; max-width:700px; margin:auto; margin-top:20px; margin-bottom:0.5em;">
                Empowering early detection through the compassion of AI.<br>
                Supporting families with accessible, non-invasive, preliminary screening technology.
            </p>

            <div style="text-align:center; margin-top:10px; color:#3762a8;">
                <p><b>Fast, gentle, and confidential detection for every child and family.</b></p>
            </div>
        </div>

    </div>
    """
    components.html(home_html, height=650, scrolling=False)




# -------------------------------------------------------
# UPLOAD PAGE
# -------------------------------------------------------
elif page == "📤 Upload Image":
    st.markdown("<h2>📸 Upload Infant Image</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555;'>Please upload a clear infant face photo for analysis.</p>", unsafe_allow_html=True)

    with st.container():
        uploaded_file = st.file_uploader("Choose an image (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            # Resize image to 224x224 to reduce memory usage
            image_resized = image.resize((224, 224))
            st.image(image_resized, caption="🖼 Uploaded Image (Resized)", width=340)

            # Save temporarily
            assets_dir = "assets"
            os.makedirs(assets_dir, exist_ok=True)
            temp_name = uploaded_file.name if hasattr(uploaded_file, "name") else "uploaded.png"
            temp_path = os.path.join(assets_dir, temp_name)
            image_resized.save(temp_path)

            analyze_button = st.button("🔍 Analyze Image")

            if analyze_button:
                with st.spinner("🧠 Analyzing facial features..."):
                    try:
                        result = predict_image_streamlit(temp_path)
                    except Exception as e:
                        st.error("⚠️ Error running prediction function.")
                        st.text(traceback.format_exc())
                        result = {"status": "error", "error": str(e)}

                    if result.get("status") == "error":
                        st.error(result.get("error", "Unknown error during prediction."))
                    else:
                        down_prob = float(result.get("down_prob", 0))
                        normal_prob = float(result.get("normal_prob", 0))
                        model_accuracy = result.get("model_accuracy", 95)
                        color = "#FCE4EC" if down_prob > 60 else "#E8F5E9"
                        border = "#F06292" if down_prob > 60 else "#66BB6A"

                        st.markdown(f"""
                        <div class="fade-in result-card" style="background:{color}; border-left:8px solid {border};">
                            <h3>🧩 Screening Result</h3>
                            <p style="font-size:1.1em;">
                            {"<b style='color:#C2185B;'>⚠️ High likelihood of Down Syndrome features detected.</b>" if down_prob > 60 else "<b style='color:#2E7D32;'>✅ Likely Normal features detected.</b>"}
                            </p>
                            <p style='color:#555;'>Confidence: <b>{max(down_prob, normal_prob):.2f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.subheader("📊 Confidence Overview")
                        data = pd.DataFrame({
                            'Category': ['Down Syndrome', 'Normal'],
                            'Probability (%)': [down_prob, normal_prob]
                        })
                        chart = (
                            alt.Chart(data)
                            .mark_bar(size=50, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                            .encode(
                                x=alt.X('Probability (%):Q', scale=alt.Scale(domain=[0, 100])),
                                y=alt.Y('Category:N', sort='-x'),
                                color=alt.Color('Category:N', scale=alt.Scale(
                                    domain=['Down Syndrome', 'Normal'],
                                    range=['#F48FB1', '#81C784']
                                )),
                                tooltip=['Category', 'Probability (%)']
                            )
                            .properties(width=600, height=200)
                        )
                        st.altair_chart(chart, use_container_width=True)
                        st.caption(f"Model accuracy (validation set): **{model_accuracy}%**")

        except Exception as e:
            st.error(f"⚠️ Error processing uploaded image: {e}")
            st.text(traceback.format_exc())
    else:
        st.info("👆 Upload an image above to begin analysis.")

# -------------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------------
elif page == "ℹ️ About":
    st.markdown("<h2>🧬 Understanding Down Syndrome</h2>", unsafe_allow_html=True)
    
    about_html = """
<div class='fade-in' style='max-width:900px; margin:auto; font-size:1.05em; color:#444; text-align:justify;'>
    <p>🧬 <b>What is Down Syndrome?</b><br>
       A genetic condition caused by an extra chromosome 21, affecting facial features, muscle tone, and sometimes development. Early awareness supports timely care.
    </p>

    <p>🌈 <b>How It Happens</b><br>
       It occurs naturally due to an error in cell division (<b>nondisjunction</b>) and is not caused by parental actions.
    </p>

    <p>💖 <b>Early Diagnosis & Support</b><br>
       Screening helps identify potential markers early, enabling therapies and guidance for better developmental outcomes.
    </p>

    <p>🌼 <b>Living Well</b><br>
       With love, education, and healthcare support, children with Down Syndrome can thrive and participate actively in life.
    </p>

    <p>🤖 <b>Our Digital Approach</b><br>
       This AI tool provides preliminary, non-invasive facial analysis and probability scores, assisting families and professionals with early insights.
    </p>
</div>
"""
    components.html(about_html, height=500, scrolling=False)

# -------------------------------------------------------
# HELP PAGE
# -------------------------------------------------------
elif page == "❓ Help":
    st.markdown("<h2>🩹 Help & Guidance</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='fade-in' style='max-width:750px; margin:auto; font-size:1.05em; color:#444;'>
        <ul>
            <li>📤 Upload a clear infant photo (front-facing) in the <b>Upload Image</b> section.</li>
            <li>⚙️ The AI analyzes facial markers quickly and non-invasively for Down Syndrome features.</li>
            <li>📊 Receive a probability score and an easy-to-read confidence chart.</li>
            <li>⚠️ This is a <b>preliminary screening</b> only; consult healthcare professionals for confirmation.</li>
            <li>👩‍⚕️ Supports awareness and early intervention alongside professional guidance.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# FOOTER
# ----------------------------it ---------------------------
st.markdown("""
<footer>
    <hr>
    Developed as part of <b>Academic Mini Project</b> | AI & Computer Vision 
</footer>
""", unsafe_allow_html=True)


