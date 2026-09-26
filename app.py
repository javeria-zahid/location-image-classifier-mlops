import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(
    page_title="TerraLens | Location Image Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "TerraLens classifies location images across six scene categories using a TensorFlow transfer-learning model."
    },
)

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

class_icons = {
    'buildings': '🏙️',
    'forest': '🌲',
    'glacier': '🧊',
    'mountain': '⛰️',
    'sea': '🌊',
    'street': '🚏',
}

class_details = {
    'buildings': ('🏙️', 'Urban form', '#d97552'),
    'forest': ('🌲', 'Dense canopy', '#4f8a58'),
    'glacier': ('🧊', 'Ice & snow', '#55a6b2'),
    'mountain': ('⛰️', 'High terrain', '#8a795e'),
    'sea': ('🌊', 'Open water', '#4f83a4'),
    'street': ('🚏', 'City streets', '#bd9147'),
}

theme_palettes = {
    'Light': {
        'ink': '#1b3128', 'muted': '#66766d', 'green': '#25694f',
        'green_hover': '#174b39', 'accent': '#d97552', 'page': '#f1f4ed',
        'panel': '#fcfdf9', 'sidebar': '#e7eee5', 'input': '#ffffff',
        'line': '#d6e0d5', 'hero': '#dce9dc', 'contour': 'rgba(31, 82, 60, 0.13)',
        'tile': '#f4f7f0', 'shadow': 'rgba(29, 56, 41, 0.08)', 'chart': '#25694f',
    },
    'Dark': {
        'ink': '#eaf0e7', 'muted': '#a8b5ab', 'green': '#a4cf8f',
        'green_hover': '#c0e5a7', 'accent': '#f1a17b', 'page': '#111916',
        'panel': '#1b2520', 'sidebar': '#18221d', 'input': '#202b25',
        'line': '#34433a', 'hero': '#21362c', 'contour': 'rgba(202, 231, 190, 0.13)',
        'tile': '#202c25', 'shadow': 'rgba(0, 0, 0, 0.22)', 'chart': '#a4cf8f',
    },
}

active_theme = 'Dark' if st.session_state.get('appearance_toggle', False) else 'Light'
palette = theme_palettes[active_theme]
theme_variables = '\n'.join(
    f'        --terra-{name.replace("_", "-")}: {value};'
    for name, value in palette.items()
)

styles = """
<style>
:root {
THEME_VARIABLES
    color-scheme: THEME_COLOR_SCHEME;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--terra-page);
    color: var(--terra-ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: var(--terra-sidebar);
    border-right: 1px solid var(--terra-line);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label { color: var(--terra-ink); }
[data-testid="stMarkdownContainer"] { color: var(--terra-ink); }
.hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 220px;
    align-items: center;
    gap: 2rem;
    overflow: hidden;
    min-height: 250px;
    padding: 2.5rem 3rem;
    margin: 0.6rem 0 1.4rem;
    border: 1px solid var(--terra-line);
    border-left: 5px solid var(--terra-accent);
    border-radius: 12px;
    background-color: var(--terra-hero);
    background-image: repeating-radial-gradient(
        ellipse at 96% 46%, transparent 0 20px, var(--terra-contour) 21px 22px,
        transparent 23px 36px
    );
    box-shadow: 0 12px 34px var(--terra-shadow);
}
.hero-copy { position: relative; z-index: 1; }
.eyebrow, .section-label {
    color: var(--terra-green);
    font-size: 0.73rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.hero h1 {
    color: var(--terra-ink);
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 4.1rem;
    line-height: 1.02;
    margin: 0.45rem 0 0;
    letter-spacing: 0;
}
.hero p {
    color: var(--terra-muted);
    font-size: 1.05rem;
    line-height: 1.65;
    max-width: 640px;
    margin: 0.85rem 0 0;
}
.hero-stat {
    position: relative;
    z-index: 1;
    justify-self: end;
    min-width: 175px;
    padding: 1rem 0 1rem 1.5rem;
    border-left: 1px solid var(--terra-line);
}
.hero-stat strong {
    display: block;
    color: var(--terra-accent);
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1;
}
.hero-stat span {
    display: block;
    color: var(--terra-muted);
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    margin-top: 0.65rem;
    text-transform: uppercase;
}
.section-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.4rem 0 0.75rem;
}
.section-note { color: var(--terra-muted); font-size: 0.82rem; }
.category-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.65rem;
    margin: 0 0 1.6rem;
}
.category-tile {
    min-height: 82px;
    padding: 0.8rem 0.85rem;
    border: 1px solid var(--terra-line);
    border-top: 3px solid var(--tile-accent);
    border-radius: 8px;
    background: var(--terra-tile);
    transition: transform 160ms ease, box-shadow 160ms ease;
}
.category-tile:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px var(--terra-shadow);
}
.category-icon { font-size: 1.25rem; line-height: 1; }
.category-name {
    color: var(--terra-ink);
    font-size: 0.84rem;
    font-weight: 750;
    margin-top: 0.45rem;
}
.stForm {
    padding: 1.15rem 1.25rem 0.8rem;
    border: 1px solid var(--terra-line);
    border-radius: 10px;
    background: var(--terra-panel);
    box-shadow: 0 8px 24px var(--terra-shadow);
}
[data-testid="stTextInputRootElement"] {
    border-color: var(--terra-line);
    background: var(--terra-input);
}
[data-testid="stTextInputRootElement"] input {
    color: var(--terra-ink);
    background: var(--terra-input);
}
.stForm button, .stButton > button {
    min-height: 2.8rem;
    padding: 0.55rem 1.25rem;
    border: 0;
    border-radius: 8px;
    background: var(--terra-green);
    color: var(--terra-page);
    font-weight: 750;
    transition: background 160ms ease, transform 160ms ease;
}
.stForm button:hover, .stButton > button:hover {
    background: var(--terra-green-hover);
    color: var(--terra-page);
    transform: translateY(-1px);
}
[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--terra-line);
    border-radius: 10px;
    background: var(--terra-panel);
    box-shadow: 0 8px 24px var(--terra-shadow);
}
.result-label {
    color: var(--terra-muted);
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.result-name {
    color: var(--terra-ink);
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 2.3rem;
    font-weight: 700;
    line-height: 1.15;
    text-transform: capitalize;
}
[data-testid="stProgressBar"] > div > div { background-color: var(--terra-green); }
.footer {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    border-top: 1px solid var(--terra-line);
    color: var(--terra-muted);
    font-size: 0.82rem;
    margin-top: 3rem;
    padding: 1rem 0 1.5rem;
}
.footer a { color: var(--terra-green); font-weight: 750; text-decoration: none; }
@media (max-width: 900px) {
    .category-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
    .hero { grid-template-columns: 1fr; gap: 1rem; padding: 1.5rem; }
    .hero h1 { font-size: 3rem; }
    .hero-stat { justify-self: start; min-width: 0; padding: 0.6rem 0 0; border: 0; }
    .hero-stat strong { display: inline; font-size: 2rem; }
    .hero-stat span { display: inline; margin-left: 0.55rem; }
    .category-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .footer { flex-direction: column; }
}
</style>
"""
styles = styles.replace('THEME_VARIABLES', theme_variables)
styles = styles.replace('THEME_COLOR_SCHEME', active_theme.lower())
st.markdown(styles, unsafe_allow_html=True)

# Load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('location_classifier_final.keras')

model = load_model()

with st.sidebar:
    st.markdown("## TerraLens")
    st.caption("LOCATION SCENE CLASSIFIER")
    st.toggle("Dark mode", key="appearance_toggle")
    st.markdown("---")
    st.markdown("### About this project")
    st.write(
        "A location-scene classifier built with TensorFlow, MobileNetV2 transfer learning, "
        "Streamlit, and Docker."
    )
    st.markdown("### Model classes")
    for class_name in class_names:
        st.write(f"{class_icons[class_name]}  {class_name.capitalize()}")
    st.markdown("---")
    st.markdown("**Validation accuracy**  \n~92%")

st.markdown(
    """
    <div class="hero">
        <div class="hero-copy">
            <div class="eyebrow">TerraLens &nbsp; / &nbsp; Visual scene intelligence</div>
            <h1>Read the landscape.</h1>
            <p>Turn a location image into a scene classification. TerraLens identifies the visual signature of urban places, wild terrain, ice, and open water.</p>
        </div>
        <div class="hero-stat"><strong>06</strong><span>scene categories</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading"><span class="section-label">The scene index</span>'
    '<span class="section-note">Six environments, one image at a time</span></div>',
    unsafe_allow_html=True,
)
category_tiles = ''.join(
    f'<div class="category-tile" style="--tile-accent:{class_details[name][2]}">'
    f'<div class="category-icon">{class_details[name][0]}</div>'
    f'<div class="category-name">{name.capitalize()}</div></div>'
    for name in class_names
)
st.markdown(f'<div class="category-grid">{category_tiles}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">01 &nbsp; / &nbsp; Image source</div>', unsafe_allow_html=True)
with st.form("classification_form"):
    image_url = st.text_input(
        "Image URL",
        placeholder="https://example.com/your-location-image.jpg",
        help="Enter a direct, publicly accessible URL to an image.",
        label_visibility="collapsed",
    )
    classify_clicked = st.form_submit_button("Classify image")

if classify_clicked:
    st.session_state.pop('classification_result', None)
    if image_url:
        try:
            with st.spinner("Analyzing the scene..."):
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content)).convert('RGB')

                # Preprocess
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                predictions = model.predict(img_array)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions) * 100

            st.session_state['classification_result'] = {
                'image': img,
                'predictions': predictions,
                'predicted_class': predicted_class,
                'confidence': confidence,
            }
        except Exception as e:
            st.error(f"Couldn't load or classify image: {e}")
    else:
        st.warning("Enter an image URL to start classification.")

result = st.session_state.get('classification_result')
if result:
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    predictions = result['predictions']
    image_column, results_column = st.columns([1.05, 0.95], gap="large")
    with image_column:
        st.markdown('<div class="section-label">02 &nbsp; / &nbsp; Input image</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.image(result['image'], use_container_width=True)
    with results_column:
        st.markdown('<div class="section-label">03 &nbsp; / &nbsp; Classification</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="result-label">Predicted scene</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-name">{class_icons[predicted_class]} {predicted_class}</div>',
                unsafe_allow_html=True,
            )
            st.write(f"Confidence · {confidence:.1f}%")
            st.progress(float(confidence / 100))
            st.markdown("#### Class probabilities")
            st.bar_chart(
                {
                    "Class": class_names,
                    "Probability": (predictions[0] * 100).tolist(),
                },
                x="Probability",
                y="Class",
                horizontal=True,
                height=280,
                color=palette['chart'],
            )

st.markdown(
    '<div class="footer"><span>TerraLens &nbsp; · &nbsp; Built with TensorFlow and Streamlit</span>'
    '<a href="https://github.com/javeria-zahid/location-image-classifier-mlops" target="_blank">'
    'View source on GitHub ↗</a></div>',
    unsafe_allow_html=True,
)