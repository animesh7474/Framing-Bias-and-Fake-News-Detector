# Fake News Detection Project Architecture & Flow

This document explains the architecture, execution flow, and complete working logic of the Fake News Detection Project. 

The project operates as a **Hybrid AI Pipeline** that leverages both a traditional Machine Learning (ML) model and a Large Language Model (LLM) to analyze news text for framing bias, manipulation, and factual accuracy. It consists of a backend Flask API and a clean, responsive frontend dashboard.

---

## 1. High-Level Architecture & Flow

The system processes user input through a 4-stage pipeline. When a user submits a news snippet via the frontend, the following happens:

1. **Stage 1 (ML Model)**: The text is passed to a pre-trained ML model (`framing_bias_model.pkl`) which classifies the "Frame" of the text (Economic, Political, Social, Security, Environment). It also runs a Threat Analyzer to detect manipulation and propaganda keywords, generating a confidence score and a threat level.
2. **Stage 2 (Live News Fetcher)**: The system takes the input text and searches the live web (via DuckDuckGo) to fetch the top 4 related real news articles.
3. **Stage 3 (LLM Analysis)**: The original text, the ML scores (Frame, Confidence, Threat Level), and the fetched real news context are all combined into a structured prompt and sent to an LLM (Groq API, running LLaMA 3.3 70B). The LLM analyzes the text, cross-references it with the real news, and generates a structured JSON report detailing the fake news likelihood, emotional manipulation, hidden agendas, and factual gaps.
4. **Stage 4 (Combined Output)**: The ML results, the news context, and the LLM analysis are aggregated into a single JSON response and sent back to the frontend.

---

## 2. Backend Logic (Flask API)

The backend is built using **Python and Flask**. It serves as the orchestrator for the entire pipeline.

**Key Files:**
*   `app.py`: The entry point for the Flask server. It defines the API endpoints, specifically `/api/analyze`. It loads the ML model into memory on startup to ensure fast inference.
*   `pipeline.py`: Contains the `run_full_analysis()` function, which defines the 4-stage flow described above. It orchestrates the handover between the ML model, the news fetcher, and the LLM client.
*   `config.py`: Centralized configuration file storing paths, API settings, and model hyperparameters.

**Endpoint Flow (`/api/analyze`):**
1.  Receives a POST request with a JSON payload containing the `text`.
2.  Calls `run_full_analysis(text)` from `pipeline.py`.
3.  Returns the comprehensive JSON result to the frontend.

---

## 3. Machine Learning (ML) Integration

The ML component is responsible for fast, initial classification and heuristics-based threat detection.

**Key Files:**
*   `framing_bias_model.pkl`: The serialized, pre-trained scikit-learn pipeline (typically combining TF-IDF vectorization with a classifier like Random Forest or Logistic Regression).
*   `threat_analyzer.py`: A rule-based engine that scans the text for known propaganda keywords, emotional triggers, and formatting anomalies (like excessive capitalization) to calculate a `manipulation_score` and assign a `threat_level` (LOW, MEDIUM, HIGH, CRITICAL).

**How it works inside the pipeline:**
*   `_model.predict_proba([text])` is called to get the probabilities for each of the 5 frames. The final frame is the one with the highest probability.
*   It generates a lightweight approximation of word importance (similar to LIME) to highlight keywords associated with the detected frame.

---

## 4. Large Language Model (LLM) Integration

The LLM component provides the deep, semantic analysis and fact-checking that traditional ML struggles with.

**Key Files:**
*   `llm_client.py`: Manages the connection to the Groq API.
*   `news_fetcher.py`: Scrapes real-time context.

**How it works:**
1.  **Context Preparation:** `news_fetcher.py` formats the fetched DDG articles into a readable string.
2.  **Prompt Engineering:** `llm_client.py` constructs a highly specific prompt. This involves a `SYSTEM_PROMPT` (defining the persona as an expert media analyst) and an `ANALYSIS_PROMPT`.
3.  **Data Injection:** The `ANALYSIS_PROMPT` is injected with the user's text, the ML scores (`frame`, `confidence`, `threat_level`), and the `news_context`.
4.  **Structured Generation:** The prompt strictly instructs the LLM to output a JSON object with specific keys (`fake_news_likelihood`, `verdict`, `factual_gaps`, etc.). The Groq client is configured with `response_format={"type": "json_object"}` to guarantee this structure.
5.  **Fallback Mechanism:** If the Groq API key is missing or the call fails, a `_fallback()` method generates a basic, heuristics-based analysis using only the ML scores to ensure the application doesn't crash.

---

## 5. Frontend Logic

The frontend is a single-page HTML application (`simulation.html`) with embedded vanilla CSS and JavaScript, eliminating the need for complex build tools while still providing a modern UI.

**Key Features:**
*   **Design System:** Uses CSS variables for consistent theming (dark mode, specific accent colors).
*   **State Management (Vanilla JS):** The JavaScript handles DOM manipulation, showing/hiding loading spinners (`.stage-loader`), and rendering the results sequentially as they arrive.
*   **Asynchronous Fetch:** When the user clicks "Analyze", an `async function analyzeText()` runs:
    1.  Shows the loading state.
    2.  Sends a `fetch()` POST request to `http://localhost:5000/api/analyze`.
    3.  Awaits the JSON response.
*   **Dynamic Rendering:** Once the response is received, the JS updates the DOM elements:
    *   Fills in the LLM analysis text blocks (Verdict, Agenda, Factual Gaps).
    *   Populates the ML confidence bars visually by updating CSS widths.
    *   Renders the "LIME" word highlights (coloring words positively or negatively based on their fake-news association score).
    *   Generates the News Cards iterating over the `related_news` array.

This seamless data handoff from the Flask backend to the vanilla JS frontend ensures a highly responsive and interactive user experience.
