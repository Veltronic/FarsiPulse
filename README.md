# FarsiPulse
FarsiPulse by Voltronic: AI-driven sentiment analysis for Persian. Understand emotions (positive, negative, neutral) in Farsi texts. Unlocks insights into public sentiment. Open-source Persian NLP.

FarsiPulse: Advanced Sentiment Analysis Tool for the Persian Language
FarsiPulse is an innovative project by the Voltronic team, focusing on sentiment analysis in Persian texts. Leveraging advanced Artificial Intelligence models, this tool can identify and categorize the emotions behind Persian comments, reviews, and general texts (positive, negative, neutral). FarsiPulse helps businesses, researchers, and developers gain deeper insights into Persian public opinion. This project represents a significant step in the development of Persian Natural Language Processing (NLP) and is made available to the community as an open-source initiative.

Features
Persian Sentiment Analysis: Ability to detect positive, negative, and neutral sentiments in Farsi texts.

AI Model: Utilizes pre-trained Transformer models (e.g., HooshvareLab/bert-fa-base-uncased) fine-tuned on Persian sentiment data.

Simple Web Interface (Under Development): Planned web-based user interface to allow text input and display sentiment analysis results.

Open-Source: Project code is publicly available, encouraging community contributions and collaboration.

Installation and Setup
To set up the FarsiPulse project locally, follow these steps:

Clone the Repository:

git clone https://github.com/VoltronicTeam/FarsiPulse.git
cd FarsiPulse

Create a Virtual Environment (Optional but Recommended):

python -m venv venv
source venv/bin/activate  # For Linux/macOS
# venv\Scripts\activate   # For Windows

Install Dependencies:

pip install -r requirements.txt

(Your requirements.txt file should include: pandas, scikit-learn, datasets, transformers, torch, accelerate.)

Usage
Once installed, you can use the trained model for sentiment analysis.

Loading the Model and Tokenizer:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./voltronic_sentiment_model" # Path where your model is saved
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

Analyzing Sentiment of a Text:
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    # Map numerical labels to text labels
    # IMPORTANT: Adjust this map based on the actual labels (0, 1, 2) in your dataset.
    # From previous debugging, it seems your dataset only has 0 and 1.
    sentiment_map = {0: 'positive', 1: 'negative', 2: 'no_idea'} # Adjust if 'no_idea' is not present
    
    return sentiment_map[predictions.item()]

# Example Usage
text_to_analyze = "این غذا واقعا خوشمزه بود و از کیفیتش راضی بودم." # This food was really delicious and I was satisfied with its quality.
sentiment = predict_sentiment(text_to_analyze)
print(f"Text: \"{text_to_analyze}\"\nSentiment: {sentiment}")

text_to_analyze_negative = "متاسفانه کیفیت غذا خیلی پایین بود و سرد به دستم رسید." # Unfortunately, the food quality was very low and it arrived cold.
sentiment_negative = predict_sentiment(text_to_analyze_negative)
print(f"Text: \"{text_to_analyze_negative}\"\nSentiment: {sentiment_negative}")

Dataset
This model has been trained on the Snappfood Persian Sentiment Analysis dataset. This dataset includes user comments from Snappfood along with their sentiment labels (positive, negative, neutral).

Dataset Source: Snappfood - Sentiment Analysis Dataset on Kaggle

Model
The model used in FarsiPulse is based on the BERT architecture, specifically the Persian pre-trained model (HooshvareLab/bert-fa-base-uncased), which has been fine-tuned on the Snappfood sentiment analysis dataset. The final model is saved in the safetensors format.

Contributors
This project is developed by the Voltronic team:

Alan Jafari (Founder, AI & Deep Learning Specialist, Project Manager)

Mohammad (Frontend Developer, HTML, CSS, Tailwind CSS)

Abbas (Frontend Developer, HTML, CSS Specialist)

License
This project is released under the MIT License. See the LICENSE file for more details.

Acknowledgments
We extend our gratitude to Soheil Tehranipour for providing the Snappfood Persian Sentiment Analysis dataset and to the HooshvareLab team for their pre-trained Persian BERT model.
