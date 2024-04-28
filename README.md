# CAGE - Country and Gender Exploration

## Overview
CAGE is an interactive web application built with Streamlit that leverages machine learning to classify text by gender and country. This project includes text preprocessing, tokenization, and the use of pre-trained BERT models for prediction.

![Gender Prediction](/Demo/demo1.jpg)
![Country Prediction](/Demo/demo2.jpg)

## Features
- Text classification by gender and country
- Beautiful visualization of class probabilities using matplotlib
- Simple and intuitive user interface

## Getting Started
To run the application locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/EmotionEngineer/CAGE.git
   ```
2. Navigate to the project directory:
   ```
   cd CAGE
   ```
3. Download the `models.zip` file from the [Releases](https://github.com/EmotionEngineer/CAGE/releases) page and extract the `gender_fp16.tflite` and `country_fp16.tflite` files into the repository folder
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the application:
   ```
   streamlit run app.py
   ```

## Model Training
Learn more about how the models were trained on Kaggle:
- [Gender by Text Explainer](https://www.kaggle.com/code/arpeccop/gender-by-text-explainer)
- [Country by Tweet Explainer](https://www.kaggle.com/code/arpeccop/country-by-tweet-explainer)

## Technologies
- Streamlit
- TensorFlow Lite
- Transformers
- Matplotlib

## License
This project is licensed under the MIT License - see the `LICENSE` file for details
