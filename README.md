# 🌍 TerraLens — Location Image Classifier

An end-to-end MLOps project that classifies location images into six scene categories — **buildings, forest, glacier, mountain, sea, and street** — using a fine-tuned deep learning model, served through a Streamlit web app, containerized with Docker, and deployed with an automated CI/CD pipeline.

**🔗 Live Demo:** [location-image-classifier-mlops.streamlit.app](https://location-image-classifier-mlops-cnieeiltrebo6c8qwmpqvz.streamlit.app)

---

## 📖 Overview

TerraLens takes an image URL as input and predicts which of six scene categories it belongs to, along with a confidence score and a full probability breakdown across all classes. The project was built to demonstrate a complete, production-style ML deployment workflow — not just training a model, but automating how it gets built, tested, and shipped.

### Model Classes

| Class | Description |
|---|---|
| 🏢 Buildings | Urban structures, architecture |
| 🌲 Forest | Wooded, densely vegetated terrain |
| ❄️ Glacier | Ice fields, snow-covered mountains |
| ⛰️ Mountain | Rocky peaks, highland terrain |
| 🌊 Sea | Oceans, coastlines, open water |
| 🛣️ Street | Roads, urban pathways |

---

## 🧠 Model

- **Architecture:** MobileNetV2 (pretrained on ImageNet) as a frozen feature extractor, with a custom classification head (`GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(6, Softmax)`)
- **Dataset:** [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (~14,000 training images, ~3,000 test images across 6 balanced classes)
- **Approach:** Transfer learning — rather than training a CNN from scratch, the project fine-tunes only the new classification head on top of a frozen, pretrained backbone. This keeps training fast while leveraging features already learned from millions of general images.
- **Performance:** ~92% validation accuracy after 5 epochs
- **Framework:** TensorFlow 2.20.0 / Keras 3.13.2

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Model training | TensorFlow, Keras, Google Colab |
| Web app | Streamlit |
| Containerization | Docker |
| Version control | Git & GitHub |
| CI (continuous integration) | GitHub Actions |
| Deployment | Streamlit Community Cloud |

---

## 🏗️ Architecture & Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌───────────────┐
│   Colab     │────▶│  Streamlit   │────▶│   Docker   │────▶│    GitHub     │
│  (Training) │     │     App      │     │ (Container)│     │  (Version     │
└─────────────┘     └──────────────┘     └────────────┘     │  Control)     │
                                                              └───────┬───────┘
                                                                      │
                                                                      ▼
                                                          ┌───────────────────────┐
                                                          │   GitHub Actions      │
                                                          │  (Auto build & test   │
                                                          │   on every push)      │
                                                          └───────────┬───────────┘
                                                                      │
                                                                      ▼
                                                          ┌───────────────────────┐
                                                          │  Streamlit Community  │
                                                          │  Cloud (Auto-deploy   │
                                                          │  live public app)     │
                                                          └───────────────────────┘
```

Every push to `main` automatically triggers:
1. **GitHub Actions** — rebuilds the Docker image and runs a smoke test (starts the container, confirms it's healthy) to catch broken builds before they matter
2. **Streamlit Community Cloud** — detects the push and redeploys the live public app with the latest code

No manual rebuild or redeploy step is required after the initial setup — that's the core of the CI/CD loop this project demonstrates.

---

## 📂 Project Structure

```
location-classifier/
├── app.py                          # Streamlit application
├── location_classifier_final.keras # Trained model
├── requirements.txt                 # Python dependencies (pinned versions)
├── Dockerfile                       # Container build instructions
├── .gitignore
├── .github/
│   └── workflows/
│       └── docker-build.yml         # GitHub Actions CI workflow
└── README.md
```

---

## 🚀 Running Locally

### Option 1: Direct (Python + Streamlit)

```bash
git clone https://github.com/javeria-zahid/location-image-classifier-mlops.git
cd location-image-classifier-mlops
pip install -r requirements.txt
streamlit run app.py
```

App will be available at `http://localhost:8501`.

### Option 2: Docker

```bash
docker build -t location-classifier .
docker run -p 8501:8501 location-classifier
```

App will be available at `http://localhost:8501`.

---

## ⚙️ CI/CD Workflow

The GitHub Actions workflow (`.github/workflows/docker-build.yml`) runs automatically on every push to `main`:

1. Checks out the latest code
2. Builds the Docker image
3. Starts the container and verifies it runs without crashing
4. Reports success/failure directly on GitHub (visible in the **Actions** tab)

This mirrors what a Cloud Build + Kubernetes pipeline would do in a full GCP setup, using entirely free tooling (GitHub Actions + Streamlit Community Cloud) instead.

---

## 🧩 Key Engineering Challenges Solved

- **Keras/TensorFlow version mismatch:** The model was trained in Colab using TensorFlow 2.20.0 / Keras 3.13.2. The Docker container initially used an older, unpinned Keras version, causing deserialization errors on load. Fixed by explicitly pinning both versions in `requirements.txt` and upgrading the Docker base image to `python:3.11-slim` (required for Keras 3.13.2 compatibility).
- **Streamlit Cloud Python version:** Streamlit Community Cloud defaulted to a newer Python version incompatible with the pinned TensorFlow version. Resolved by explicitly setting the Python version in the app's deployment settings.
- **Network-constrained builds:** Built and debugged this entire pipeline over a slow, occasionally unstable internet connection — required switching to `tensorflow-cpu` (smaller install), using `--progress=plain` for build visibility, and adding retry/timeout flags to pip installs.

---

## 📈 Possible Future Improvements

- Add automated unit tests for the prediction pipeline (not just a container health check)
- Add model versioning/experiment tracking (e.g., MLflow or Weights & Biases)
- Expand to true geolocation estimation (predicting real-world location from an image) as a separate, more advanced model
- Add batch image upload support instead of single URL input

---

## 👤 Author

**Javeria Zahid**
[GitHub](https://github.com/javeria-zahid) · [LinkedIn](https://linkedin.com/in/javeria-zahid-)

---

## 📄 License

This project is open source and available for educational and portfolio purposes.
