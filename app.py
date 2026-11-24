import gradio as gr
import torch
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH ='D:/OneDrive/Desktop/Consumer_Complaint_Classification/distilbert_best_model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer_bert = AutoTokenizer.from_pretrained(MODEL_PATH)
bert_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
bert_model.to(device)
bert_model.eval()

CLASSES = ['credit_card', 'credit_reporting', 'debt_collection',
           'mortgages_and_loans', 'retail_banking']

LABEL_DISPLAY = {
    'credit_card'        : '💳 Credit Card',
    'credit_reporting'   : '📊 Credit Reporting',
    'debt_collection'    : '📬 Debt Collection',
    'mortgages_and_loans': '🏠 Mortgages & Loans',
    'retail_banking'     : '🏦 Retail Banking',
}

MAX_LENGTH = 256

def predict_complaint(text):
    if not text or not text.strip():
        return "⚠️ Please enter a complaint.", "", None

    inputs = tokenizer_bert(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs     = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    class_idx = int(np.argmax(probs))
    label     = CLASSES[class_idx]
    confidence= float(probs[class_idx])

    predicted_label = LABEL_DISPLAY[label]
    confidence_str  = f"{confidence * 100:.2f}%"

    chart_data = pd.DataFrame({
        'Category': [LABEL_DISPLAY[cls] for cls in CLASSES],
        'Score'   : [float(probs[i]) for i in range(len(CLASSES))]
    })

    return predicted_label, confidence_str, chart_data


css = """
body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    font-family: 'Inter', sans-serif !important;
}
.main-card {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 20px !important;
    padding: 32px !important;
    backdrop-filter: blur(12px) !important;
}
.header-box {
    text-align: center;
    padding: 20px 0 10px 0;
}
.header-box h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 6px !important;
}
.header-box p {
    color: #94a3b8 !important;
    font-size: 1rem !important;
}
textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    padding: 14px !important;
}
textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}
.predict-btn button {
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
}
.clear-btn button {
    background: transparent !important;
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
}
.output-text textarea {
    color: #38bdf8 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
"""

with gr.Blocks(title="Consumer Complaint Classifier") as app:

    gr.HTML("""
        <div class="header-box">
            <h1>🛡️ Consumer Complaint Classifier</h1>
            <p>Paste a customer complaint below — the model will predict its category and confidence score.</p>
        </div>
    """)

    with gr.Row():

        
        with gr.Column(scale=3, elem_classes="main-card"):
            complaint_input = gr.Textbox(
                lines=8,
                placeholder="e.g. I have been trying to dispute a charge on my credit card for months but the bank keeps ignoring me...",
                label="Customer Complaint"
            )

            with gr.Row():
                clear_btn   = gr.Button("🗑️ Clear",   elem_classes="clear-btn")
                predict_btn = gr.Button("🔍 Predict", elem_classes="predict-btn")

            gr.Examples(
                examples=[
                    ["I was charged twice for the same transaction on my credit card and the bank refuses to refund me."],
                    ["My mortgage company is reporting incorrect information to the credit bureaus."],
                    ["The debt collector keeps calling me multiple times a day after I told them to stop."],
                    ["My bank account was closed without any prior notice and I cannot access my funds."],
                    ["There is a fraudulent account on my credit report that I never opened."],
                ],
                inputs=complaint_input,
                label="📋 Example Complaints"
            )

        
        with gr.Column(scale=2, elem_classes="main-card"):

            predicted_label = gr.Textbox(
                label="Predicted Category",
                interactive=False,
                elem_classes="output-text"
            )
            confidence_score = gr.Textbox(
                label="Confidence Score",
                interactive=False,
                elem_classes="output-text"
            )

            gr.HTML("<br>")

            confidence_chart = gr.BarPlot(
                x="Category",
                y="Score",
                title="Confidence per Category",
                x_title="",
                y_title="Probability",
                color="Category",
                height=260,
                label="Score Breakdown"
            )

    
    predict_btn.click(
        fn=predict_complaint,
        inputs=complaint_input,
        outputs=[predicted_label, confidence_score, confidence_chart]
    )
    complaint_input.submit(
        fn=predict_complaint,
        inputs=complaint_input,
        outputs=[predicted_label, confidence_score, confidence_chart]
    )
    clear_btn.click(
        fn=lambda: ("", "", None),
        outputs=[predicted_label, confidence_score, confidence_chart]
    )

app.launch(css=css, show_error=True)