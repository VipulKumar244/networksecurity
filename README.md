---
title: Network Security Classifier
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Network Security Classifier

Upload a CSV of network records to classify them with the trained phishing detection model. The Streamlit interface is intended for the project presentation; the FastAPI service is also available at `/docs` when run with Docker.

## Free presentation deployment

Deploy `streamlit_app.py` through [Streamlit Community Cloud](https://share.streamlit.io):

1. Push the project to a GitHub repository that Streamlit Community Cloud can access.
2. Create an app from that repository and select `streamlit_app.py` as the entrypoint.
3. The app expects `final_model/model.pkl` in the repository. `.gitignore` allows this model file while continuing to ignore other generated model files.

The model is a Python pickle, so keep the deployed scikit-learn version aligned with the version used to train it. Do not commit `.env` or credentials. Streamlit Community Cloud documents free deployment for educational apps; public apps require a public GitHub repository. See its [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Docker API deployment

This repo also contains a FastAPI service. Create a Docker Space and push the repository to its Git remote. The service provides `/predict` and `/docs`; `/train` runs synchronously and is intended for local/demo use. Hugging Face currently says Docker Spaces on CPU Basic require a paid plan, even though that hardware has no hourly compute cost.

## Run locally with Docker

```sh
docker build -t network-security .
docker run --rm -p 7860:7860 network-security
```

Then open `http://localhost:7860/docs`.
