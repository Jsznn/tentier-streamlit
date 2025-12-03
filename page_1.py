import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Load Model and Tokenizer (Cached)
@st.cache_resource
def load_assets():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'sentiment_model.h5')
    tokenizer_path = os.path.join(base_dir, 'models', 'tokenizer.pickle')
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Model or Tokenizer not found. Please run the training notebook first.")

    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    max_len = 100
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len)
    prediction = model.predict(padded)
    
    output_shape = model.output_shape
    sentiment = "Unknown"
    
    if output_shape[-1] == 1:
        score = prediction[0][0]
        sentiment = "Positif" if score > 0.5 else "Negatif"
    else:
        # Assuming 2 classes: Negatif, Positif
        classes = ['Negatif', 'Positif']
        class_idx = np.argmax(prediction)
        if class_idx < len(classes):
            sentiment = classes[class_idx]
            
    return sentiment

@st.fragment
def display_header():
    st.title("Product Review Sentiment Analysis")
    st.markdown("Masukkan ulasan secara manual untuk dianalisis sentimennya.")

@st.fragment
def display_input_section(model, tokenizer):
    # Initialize session state for storing queries if not exists
    if 'history' not in st.session_state:
        st.session_state.history = []

    tab1, tab2 = st.tabs(["Input Manual", "Upload CSV"])

    with tab1:
        with st.form("sentiment_form", clear_on_submit=True):
            user_input = st.text_area("Masukkan Ulasan:", placeholder="Contoh: Barangnya bagus banget, pengiriman cepat!")
            submitted = st.form_submit_button("Analisis")
            
            if submitted and user_input:
                sentiment = predict_sentiment(user_input, model, tokenizer)
                st.session_state.history.append({
                    "Ulasan": user_input,
                    "Sentimen": sentiment
                })
                st.success(f"Hasil: {sentiment}")

    with tab2:
        uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write("Preview Data:")
                st.dataframe(df_upload.head())
                
                text_col = st.selectbox("Pilih kolom ulasan:", df_upload.columns)
                
                if st.button("Proses CSV"):
                    progress_bar = st.progress(0, text="Menganalisis...")
                    total = len(df_upload)
                    results = []
                    
                    for i, text in enumerate(df_upload[text_col]):
                        # Ensure text is string
                        text_str = str(text) if pd.notna(text) else ""
                        if text_str.strip():
                            sentiment = predict_sentiment(text_str, model, tokenizer)
                            results.append({
                                "Ulasan": text_str,
                                "Sentimen": sentiment
                            })
                        
                        # Update progress every 10 items or at the end
                        if (i + 1) % 10 == 0 or (i + 1) == total:
                            progress_bar.progress((i + 1) / total, text=f"Menganalisis {i+1}/{total}")
                    
                    # Add to history
                    st.session_state.history.extend(results)
                    progress_bar.empty()
                    st.success(f"Berhasil memproses {len(results)} ulasan!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

    # Display Data and Chart if history exists
    if st.session_state.history:
        st.divider()
        st.subheader("Riwayat Analisis")
        
        # Create DataFrame
        df = pd.DataFrame(st.session_state.history)
        
        # Layout: Table on left, Chart on right
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.dataframe(df, use_container_width=True)
            
            if st.button("Hapus Riwayat"):
                st.session_state.history = []
                st.rerun()

        with col2:
            # Pie Chart
            sentiment_counts = df['Sentimen'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentimen', 'Jumlah']
            
            fig = px.pie(
                sentiment_counts, 
                values='Jumlah', 
                names='Sentimen', 
                title='Distribusi Sentimen',
                color='Sentimen',
                color_discrete_map={'Positif': 'green', 'Negatif': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    try:
        model, tokenizer = load_assets()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
    display_header()
    display_input_section(model, tokenizer)

if __name__ == "__main__":
    main()
