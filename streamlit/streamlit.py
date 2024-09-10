import streamlit as st
import requests
import os
from PIL import Image
import io
import seaborn as sns
import matplotlib.pyplot as plt

# API_URL = "http://localhost:8080"
API_URL = os.getenv("API_URL", "http://localhost:8080")

def main():
    st.title("Machine Learning Model Rakuten II - Streamlit")

    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ["Health Check", "Model Summary", "Predict", "Fine-Tune", "Evaluate"])

    if options == "Health Check":
        st.header("Health Check")
        if st.button("Check API Health"):
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                st.success("API is healthy")
            else:
                st.error("API is not healthy")

    elif options == "Model Summary":
        st.header("Model Summary")
        if st.button("Get Model Summary"):
            response = requests.get(f"{API_URL}/model-summary")
            if response.status_code == 200:
                st.text(response.json()["summary"])
            else:
                st.error("Failed to get model summary")

    elif options == "Predict":
        st.header("Predict")
        uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Prepare the file for the API request
            uploaded_file.seek(0)  # Reset file pointer to the beginning
            files = {"file": uploaded_file}
            response = requests.post(f"{API_URL}/predict", files=files)
            if response.status_code == 200:
                st.success(f"Class predite: {response.json()['predicted_class'][1:-1]}")
            else:
                st.error("Prediction failed")

    elif options == "Fine-Tune":
        st.header("Fine-Tune Model")
        uploaded_file = st.file_uploader("Upload a CSV file with 'filePath' and 'labels' columns", type=["csv"])
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.9, value=0.33)
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5)
        if uploaded_file is not None:
            if st.button("Fine-Tune Model"):
                files = {"csv_file": uploaded_file}
                data = {"test_size": test_size, "epochs": epochs}
                response = requests.post(f"{API_URL}/fine-tune", files=files, data=data)
                if response.status_code == 200:
                    st.success("Model fine-tuned successfully")
                    st.json(response.json())
                    st.success("Model fine-tuned successfully")
                    response_data = response.json()
                    
                    # Plot loss and accuracy
                    epochs_range = range(1, epochs + 1)
                    loss = response_data["history"]["loss"]
                    accuracy = response_data["history"]["accuracy"]

                    st.write(loss)
                    st.write(accuracy) 

                    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

                    sns.lineplot(x=epochs_range, y=loss, ax=ax[0])
                    ax[0].set_xlabel('Epochs')
                    ax[0].set_ylabel('Loss')
                    ax[0].set_title('Training Loss')

                    sns.lineplot(x=epochs_range, y=accuracy, ax=ax[1])
                    ax[1].set_xlabel('Epochs')
                    ax[1].set_ylabel('Accuracy')
                    ax[1].set_title('Training Accuracy')

                    st.pyplot(fig)
                else:
                    st.error("Fine-tuning failed")

    elif options == "Evaluate":
        st.header("Evaluate Model")
        uploaded_file = st.file_uploader("Upload a CSV file with 'filePath' and 'labels' columns", type=["csv"])
        if uploaded_file is not None:
            files = {"csv_file": uploaded_file}
            response = requests.post(f"{API_URL}/evaluate", files=files)
            if response.status_code == 200:
                st.success("Model evaluated successfully")
                st.json(response.json())
            else:
                st.error("Evaluation failed")

if __name__ == "__main__":
    main()