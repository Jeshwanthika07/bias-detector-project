# Bias Detector for a Pretrained Text Model

## Project overview
This project compares the outputs of a locally loaded pretrained text-generation model when two matched prompts differ only by a demographic reference. It shows the generated text, a comparison table, a simple tone-score metric, a bar chart, and a downloadable experiment report.

## Technology used
- Python
- Streamlit (frontend)
- Hugging Face Transformers (pretrained model)
- DistilGPT-2 (`distilgpt2`) loaded locally
- Pandas and Matplotlib

## Requirements
- Python 3.10 or newer recommended
- Internet connection for the first download of the model weights
- About 1–2 GB of free disk space is recommended for dependencies and model cache
- CPU works; a compatible GPU is optional

## Run locally
1. Extract the ZIP file.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (optional but recommended).

   Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. A browser tab should open with the application. Choose a demographic dimension and role, then click **Run bias comparison**.

## How it works
1. The frontend creates two matched prompts.
2. The backend loads the pretrained DistilGPT-2 model through Transformers.
3. The same generation settings are used for both prompts.
4. The app displays responses and compares a simple word-list tone score.
5. The user can download a text report.

## Metric and limitations
The tone score is a transparent, simple proxy calculated from a small list of positive and negative words. It is not a validated fairness metric. It may miss context, sarcasm, negation, and subtle stereotypes. One pair of prompts cannot prove bias or fairness. For a stronger study, use many matched prompt pairs, repeat tests, manually review outputs, and report limitations.

## Suggested screenshots for submission
1. Main screen showing the matched prompts.
2. Generated outputs for both groups.
3. Comparison table and bar chart.
4. Downloadable report / limitations section.

## Troubleshooting
- The first run downloads model files, so it can take a few minutes.
- If loading fails, check the internet connection and rerun `streamlit run app.py`.
- If installation fails, use Python 3.10 or newer and update pip with `python -m pip install --upgrade pip`.
