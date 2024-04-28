import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast
import re, string

# Define labels for gender and country
gender_labels = ['female', 'male']
country_labels = ['Australia', 'Canada', 'Ghana', 'India', 'Ireland', 'Kenya',
                  'New Zealand', 'Nigeria', 'Singapore', 'South Africa',
                  'United Kingdom', 'United States']

# Load the TFLite models
gender_interpreter = tf.lite.Interpreter(model_path="gender_fp16.tflite")
country_interpreter = tf.lite.Interpreter(model_path="country_fp16.tflite")
gender_interpreter.allocate_tensors()
country_interpreter.allocate_tensors()

# Get input and output details
gender_input_details = gender_interpreter.get_input_details()
gender_output_details = gender_interpreter.get_output_details()
country_input_details = country_interpreter.get_input_details()
country_output_details = country_interpreter.get_output_details()

# Load the tokenizer (ensure it's the same used during training)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Define text preprocessing functions
def preprocess_text(text):
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    banned_list = string.punctuation + 'Ã±¼â»§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = re.sub("\s\s+" , " ", text)
    return text

# Define tokenization function
def tokenize_text(text, tokenizer, max_length=128):
    tokenized = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        add_special_tokens=True,
    )
    return tokenized

# Define prediction function
def predict_fn(interpreter, input_details, output_details, tokenized_text, attention_first=True):
    # Set the correct order of tensors for each model
    if attention_first:
        interpreter.set_tensor(input_details[0]['index'], tokenized_text['attention_mask'])
        interpreter.set_tensor(input_details[1]['index'], tokenized_text['input_ids'])
    else:
        interpreter.set_tensor(input_details[0]['index'], tokenized_text['input_ids'])
        interpreter.set_tensor(input_details[1]['index'], tokenized_text['attention_mask'])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Define functions to make predictions
def predict_gender(text):  
    tokenized_text = tokenize_text(text, tokenizer)
    output_data = predict_fn(gender_interpreter, gender_input_details, gender_output_details, tokenized_text, attention_first=True)
    return output_data, gender_labels[np.argmax(output_data)]

def predict_country(text):
    tokenized_text = tokenize_text(text, tokenizer)
    output_data = predict_fn(country_interpreter, country_input_details, country_output_details, tokenized_text, attention_first=False)
    return output_data, country_labels[np.argmax(output_data)]

# Function to plot probabilities
def plot_probabilities(probabilities, labels, title):
    # Create a bar chart
    fig, ax = plt.subplots()
    ax.barh(labels, probabilities, color='orange')
    ax.set_xlabel('Probability')
    ax.set_title(title)
    ax.set_xlim(0, 1)  # Probabilities range from 0 to 1
    plt.tight_layout()

    # Render the plot in the Streamlit app
    st.pyplot(fig)

# Streamlit app
st.title("Text Gender/Country Classification")

# Input text box
text_input = st.text_input("Enter text:")

# Predict button
if st.button("Predict"):
    if text_input:
        # preprocessed_text = preprocess_text(text_input)
        proba_gender, predicted_gender = predict_gender(text_input)
        proba_country, predicted_country = predict_country(text_input)
        st.write(f"Predicted gender: **{predicted_gender}**")
        plot_probabilities(proba_gender[0], gender_labels, "Gender Prediction Probabilities")
        st.write(f"Predicted country: **{predicted_country}**")
        plot_probabilities(proba_country[0], country_labels, "Country Prediction Probabilities")
    else:
        st.write("Input text is empty")