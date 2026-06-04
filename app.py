import streamlit as st
import pandas as pd
import os
import wandb
from data_handler import load_imdb_sample, process_upload
from nlp_pipeline import SentimentPipeline
from visuals import plot_sentiment_distribution, plot_confusion_matrix, display_metrics
import io

# Set page config with modern default
st.set_page_config(
    page_title="Universal Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS INJECTION FOR PREMIUM LOOK ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: #f8fafc;
    }
    
    /* Custom Title Style */
    .main-title {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Styled custom card container */
    .custom-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    
    /* Adjust standard streamit tabs */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #64748b !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4f46e5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for Pipeline and Data
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = SentimentPipeline()
if 'df' not in st.session_state:
    st.session_state.df = None
if 'text_col' not in st.session_state:
    st.session_state.text_col = None
if 'label_col' not in st.session_state:
    st.session_state.label_col = None

# --- SIDEBAR: MANUAL GUIDE & SAMPLE TEMPLATE DOWNLOAD ---
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/brain.png", width=64)
    st.markdown("### 🤖 Panduan Penggunaan")
    st.info(
        "Aplikasi ini membantu Anda melatih model Machine Learning untuk mengklasifikasikan sentimen teks menjadi **Positif** atau **Negatif**."
    )
    
    with st.expander("📖 4 Langkah Mudah", expanded=True):
        st.markdown("""
        1. **Tab 1: Data Discovery**
           * Upload file data Anda sendiri (Excel/CSV), **ATAU**
           * Klik **Load IMDB Sample** untuk langsung memakai data uji coba bawaan.
        
        2. **Tab 2: Training Model**
           * Klik **Train Model** untuk melatih model Logistic Regression.
           * Lihat hasil evaluasi akurasi dan confusion matrix.
        
        3. **Tab 3: Error Analysis**
           * Cari tahu teks mana saja yang gagal diprediksi dengan benar oleh model untuk bahan evaluasi.
        
        4. **Tab 4: Playground**
           * Ketik teks buatan Anda sendiri untuk dites sentimennya secara langsung.
        """)
        
    st.markdown("### 📥 Download Template Data")
    st.markdown(
        "Ingin mencoba menggunakan data sendiri? Buat file CSV di Excel/Google Sheets dengan isi seperti template berikut:"
    )
    
    # Generate CSV template
    template_data = (
        "review,sentiment\n"
        "\"Film ini sangat luar biasa dan aktingnya keren sekali!\",positive\n"
        "\"Alur ceritanya membosankan dan aktingnya jelek.\",negative\n"
        "\"Biasa saja, tidak ada yang spesial dari film ini.\",negative\n"
    )
    
    st.download_button(
        label="📄 Download Template CSV",
        data=template_data,
        file_name="template_data_pengujian.csv",
        mime="text/csv",
        help="Gunakan file ini sebagai contoh format untuk membuat data pengujian Anda sendiri di Excel."
    )

# --- MAIN APP LAYOUT ---
st.markdown('<div class="main-title">Universal Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modular NLP Pipeline featuring Logistic Regression & TF-IDF</div>', unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Data Discovery", 
    "⚙️ Training & Evaluation", 
    "🔍 Error Analysis", 
    "🎮 Live Playground"
])

# --- TAB 1: Data Discovery ---
with tab1:
    st.subheader("Langkah 1: Siapkan & Muat Dataset")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("### 📤 Upload Dataset Kustom")
            st.markdown("Unggah file CSV atau Excel Anda di sini. Pastikan minimal memiliki satu kolom teks dan satu kolom label.")
            uploaded_file = st.file_uploader(
                "Pilih file CSV / Excel", 
                type=['csv', 'xls', 'xlsx'],
                help="Ukuran file disarankan di bawah 10MB agar pemrosesan cepat."
            )
            if uploaded_file is not None:
                try:
                    st.session_state.df = process_upload(uploaded_file)
                    st.success("File kustom berhasil diunggah!")
                except Exception as e:
                    st.error(f"Error memproses file: {e}")
                
    with col2:
        with st.container(border=True):
            st.markdown("### 📦 Gunakan Data Contoh Bawaan")
            st.markdown("Belum punya data? Klik tombol di bawah untuk otomatis mengunduh 10.000 data ulasan film (IMDB) dari internet sebagai bahan latihan.")
            if st.button("📥 Muat Data Contoh IMDB (10k)", type="secondary", use_container_width=True):
                with st.spinner("Mengunduh data IMDB dari GitHub..."):
                    df_imdb = load_imdb_sample()
                    if df_imdb is not None:
                        st.session_state.df = df_imdb
                        st.success("Data contoh IMDB berhasil dimuat!")

    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("Pratinjau Data & Pemetaan Kolom")
        
        cols = st.session_state.df.columns.tolist()
        
        col_text, col_label = st.columns(2)
        with col_text:
            text_col = st.selectbox(
                "Pilih Kolom Teks (Review):", 
                options=cols, 
                index=cols.index('review') if 'review' in cols else 0,
                help="Pilih kolom yang berisi kalimat/ulasan yang ingin dianalisis."
            )
        with col_label:
            label_col = st.selectbox(
                "Pilih Kolom Target (Sentiment/Label):", 
                options=cols,
                index=cols.index('sentiment') if 'sentiment' in cols else 0,
                help="Pilih kolom yang berisi label kebenaran (misal: positive, negative)."
            )
            
        st.session_state.text_col = text_col
        st.session_state.label_col = label_col
        
        st.markdown("##### 5 Data Teratas:")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Visualisasi Distribusi Sentimen Awal")
        fig_pie, fig_bar = plot_sentiment_distribution(st.session_state.df, label_col)
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with chart_col2:
            st.plotly_chart(fig_bar, use_container_width=True)


# --- TAB 2: Training & Evaluation ---
with tab2:
    st.subheader("Langkah 2: Latih Model AI")
    
    if st.session_state.df is None:
        st.warning("⚠️ Silakan upload data atau muat data contoh di tab '📂 Data Discovery' terlebih dahulu.")
    else:
        st.info("ℹ️ Model menggunakan algoritma **Logistic Regression** & pembobotan kata **TF-IDF** (10k fitur, n-gram 1 hingga 2). Teks akan dibersihkan terlebih dahulu menggunakan NLTK (Lemmatization & Stopwords removal).")
        
        if st.button("🚀 Mulai Latih Model", type='primary', use_container_width=True):
            with st.spinner("Pembersihan teks sedang berjalan dan model sedang dilatih... Mohon tunggu sebentar."):
                # W&B Integration
                wandb_api_key = os.environ.get("WANDB_API_KEY") or (st.secrets.get("WANDB_API_KEY") if hasattr(st, 'secrets') else None)
                use_wandb = bool(wandb_api_key)
                
                if use_wandb:
                    try:
                        wandb.login(key=wandb_api_key)
                        wandb.init(project="universal-sentiment-dashboard", 
                                   config={"model": "Logistic Regression", "vectorizer": "TF-IDF"})
                    except Exception as e:
                        st.warning(f"Gagal menghubungkan ke W&B: {e}")
                        use_wandb = False
                
                # Train
                acc, f1, cm, evaluated_df, classes = st.session_state.pipeline.train(
                    st.session_state.df, 
                    st.session_state.text_col, 
                    st.session_state.label_col
                )
                
                # Update dataframe with predictions
                st.session_state.df = evaluated_df
                
                # Log metrics
                if use_wandb:
                    wandb.log({"accuracy": acc, "f1_score": f1})
                    wandb.finish()
                    
                st.success("🎉 Model berhasil dilatih dengan sukses!")
                
                # Display Results
                st.markdown("### 📊 Hasil Evaluasi Model (Data Uji 20%)")
                display_metrics(acc, f1)
                
                st.markdown("---")
                st.subheader("Matriks Kebingungan (Confusion Matrix)")
                fig_cm = plot_confusion_matrix(cm, classes)
                st.plotly_chart(fig_cm, use_container_width=True)
                
        elif st.session_state.pipeline.is_trained:
            st.success("✅ Model sudah dilatih dan siap digunakan untuk analisis.")
            st.info("Jika Anda memuat data baru, silakan klik tombol '🚀 Mulai Latih Model' lagi untuk melatih ulang.")


# --- TAB 3: Error Analysis ---
with tab3:
    st.subheader("Langkah 3: Analisis Kesalahan Prediksi")
    if st.session_state.df is None or not st.session_state.pipeline.is_trained:
        st.warning("⚠️ Silakan latih model Anda terlebih dahulu di tab '⚙️ Training & Evaluation' untuk mengaktifkan analisis error.")
    else:
        st.write("Di bawah ini adalah data-data ulasan yang gagal diprediksi secara tepat oleh model. Halaman ini sangat penting untuk mengevaluasi kata-kata apa saja yang membingungkan model.")
        
        df_errors = st.session_state.df[st.session_state.df['is_error'] == True]
        st.metric("Total Sampel Salah Prediksi", len(df_errors))
        
        if len(df_errors) > 0:
            # Filterable table
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                true_label_filter = st.multiselect("Filter berdasarkan Label Asli", options=df_errors[st.session_state.label_col].unique())
            with col_f2:
                pred_label_filter = st.multiselect("Filter berdasarkan Hasil Prediksi Model", options=df_errors['prediction'].unique())
            
            filtered_errors = df_errors.copy()
            if true_label_filter:
                filtered_errors = filtered_errors[filtered_errors[st.session_state.label_col].isin(true_label_filter)]
            if pred_label_filter:
                filtered_errors = filtered_errors[filtered_errors['prediction'].isin(pred_label_filter)]
                
            display_cols = [st.session_state.text_col, st.session_state.label_col, 'prediction', 'cleaned_text']
            st.dataframe(filtered_errors[display_cols], use_container_width=True)
        else:
            st.success("Sempurna! Tidak ada kesalahan prediksi pada data evaluasi saat ini.")

        st.markdown("---")
        st.subheader("Unduh Hasil Prediksi Lengkap")
        st.write("Anda dapat mengunduh seluruh data Anda lengkap dengan kolom hasil prediksi model (`prediction`) dalam format CSV.")
        
        # Convert df to csv for download
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Prediksi (.CSV)",
            data=csv,
            file_name='hasil_prediksi_sentimen.csv',
            mime='text/csv',
        )


# --- TAB 4: Playground ---
with tab4:
    st.subheader("Langkah 4: Coba Prediksi Sentimen Teks Kustom")
    
    if not st.session_state.pipeline.is_trained:
        st.warning("⚠️ Silakan latih model Anda terlebih dahulu di tab '⚙️ Training & Evaluation' agar playground ini dapat digunakan.")
    else:
        st.write("Ketik kalimat/ulasan film kustom Anda di bawah ini untuk melihat bagaimana model mendeteksi sentimennya secara langsung (real-time).")
        
        user_input = st.text_area("Masukkan teks ulasan di sini:", height=150, placeholder="Contoh: The movie was absolutely stunning, the visuals were breathtaking...")
        
        if st.button("🔮 Prediksi Sentimen Sekarang", type="primary"):
            if user_input.strip() == "":
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                with st.spinner("Menganalisis kalimat..."):
                    prediction = st.session_state.pipeline.predict_single(user_input)
                    
                    # Styled output
                    if prediction.lower() in ['positive', 'pos', '1']:
                        st.balloons()
                        st.success(f"### 🎉 **Hasil Analisis:** Sentimen **POSITIF**")
                    else:
                        st.error(f"### 👎 **Hasil Analisis:** Sentimen **NEGATIF**")

