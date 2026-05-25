import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import joblib

st.set_page_config(
    page_title="Prediksi Risiko Gangguan Tidur",
    layout="wide"
)

st.markdown("""
<style>
iframe, img {
    max-height: 450px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    padding-top: 3rem;
}

div[data-testid="stHorizontalBlock"] {
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

from streamlit_option_menu import option_menu

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder
)

# =====================================================
# LOAD MODEL DAN DATASET
# =====================================================

import os
import gdown

MODEL_FILE = "sleep_disorder_model.pkl"

if not os.path.exists(MODEL_FILE):

    file_id = "1-9cpXEtw2bNqIEOdHt1hGCxBXJflQOaM"

    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        MODEL_FILE,
        quiet=False
    )

model = joblib.load(MODEL_FILE)

df = pd.read_csv(
    "sleep_health_dataset.csv"
)
# =====================================================
# FITUR YANG DIGUNAKAN MODEL BARU
# =====================================================

X = df[
[
    'age',
    'gender',
    'occupation',
    'bmi',
    'sleep_duration_hrs',
    'caffeine_mg_before_bed',
    'alcohol_units_before_bed',
    'screen_time_before_bed_mins',
    'exercise_day',
    'steps_that_day',
    'stress_score',
    'work_hours_that_day',
    'chronotype',
    'mental_health_condition',
    'heart_rate_resting_bpm',
    'shift_work',
    'weekend_sleep_diff_hrs'
]
]

y = df['sleep_disorder_risk']

# =====================================================
# ENCODE TARGET
# =====================================================

le = LabelEncoder()

y = le.fit_transform(y)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# PREDIKSI UNTUK HALAMAN EVALUASI
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# MENU NAVIGASI
# =====================================================

selected = option_menu(
    menu_title=None,
    options=[
        "Beranda",
        "Visualisasi Data",
        "Prediksi Risiko",
        "Evaluasi Model"
    ],
    icons=[
        "house",
        "bar-chart",
        "activity",
        "clipboard-data"
    ],
    orientation="horizontal",
)

# =====================================================
# HALAMAN BERANDA
# =====================================================

if selected == "Beranda":

    st.title(
        "Prediksi Risiko Gangguan Tidur Menggunakan Random Forest"
    )

    st.header("Selamat Datang")

    st.write("""
    Halo, Perkenalkan nama saya Hanjaya Hartono dengan NIM 23.11.5449.
    """)

    st.subheader("Tujuan Project")

    st.write("""
    Tujuan dari project ini adalah memprediksi risiko gangguan tidur
    berdasarkan pola tidur, kondisi kesehatan, dan gaya hidup menggunakan
    algoritma Random Forest.

    Faktor yang digunakan meliputi usia, BMI, durasi tidur,
    tingkat stres, aktivitas fisik, konsumsi kafein,
    screen time sebelum tidur, kondisi kesehatan mental,
    dan berbagai faktor pendukung lainnya.
    """)

    st.subheader("Algoritma")

    st.write("""
    Random Forest Classifier digunakan sebagai algoritma machine learning
    untuk melakukan klasifikasi risiko gangguan tidur berdasarkan pola tidur,
    kondisi kesehatan, dan gaya hidup pengguna.

    Algoritma ini dipilih karena mampu menangani dataset berukuran besar,
    memiliki performa yang baik dalam klasifikasi multi-kelas,
    serta menghasilkan tingkat akurasi sebesar 81,98%.
    """)

    st.subheader("Dataset")

    st.write("""
    Dataset yang digunakan berisi sekitar 100.000 data observasi yang
    mencakup informasi mengenai pola tidur, kesehatan fisik,
    kesehatan mental, aktivitas harian, konsumsi kafein,
    screen time sebelum tidur, serta faktor gaya hidup lainnya
    yang berhubungan dengan risiko gangguan tidur.
    """)

# =====================================================
# HALAMAN VISUALISASI
# =====================================================

elif selected == "Visualisasi Data":

    st.header("Visualisasi Data")

    # =====================================================
    # BARIS 1
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Distribusi Risiko Gangguan Tidur")

        fig1, ax1 = plt.subplots(figsize=(5,4))

        sns.countplot(
            x='sleep_disorder_risk',
            data=df,
            ax=ax1
        )

        plt.tight_layout()

        st.pyplot(fig1)

        st.info("""
Mayoritas data berada pada kategori Healthy, diikuti kategori Mild.
Hal ini menunjukkan bahwa sebagian besar individu dalam dataset
memiliki risiko gangguan tidur yang relatif rendah.
""")

    with col2:

        st.subheader("Distribusi Durasi Tidur")

        fig2, ax2 = plt.subplots(figsize=(5,4))

        sns.histplot(
            df['sleep_duration_hrs'],
            bins=20,
            kde=True,
            ax=ax2
        )

        plt.tight_layout()

        st.pyplot(fig2)

        st.info("""
Sebagian besar individu memiliki durasi tidur pada rentang
6 hingga 8 jam per hari. Durasi tidur merupakan salah satu
faktor utama dalam menentukan risiko gangguan tidur.
""")

    # =====================================================
    # BARIS 2
    # =====================================================

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Correlation Matrix")

        numeric_features = [
            'age',
            'bmi',
            'sleep_duration_hrs',
            'caffeine_mg_before_bed',
            'alcohol_units_before_bed',
            'screen_time_before_bed_mins',
            'exercise_day',
            'steps_that_day',
            'stress_score',
            'work_hours_that_day',
            'heart_rate_resting_bpm',
            'shift_work',
            'weekend_sleep_diff_hrs'
        ]

        fig3, ax3 = plt.subplots(
            figsize=(7,5)
        )

        sns.heatmap(
            df[numeric_features].corr(),
            cmap='coolwarm',
            annot=False,
            ax=ax3
        )

        plt.tight_layout()

        st.pyplot(fig3)

        st.info("""
Correlation Matrix digunakan untuk melihat hubungan antar
fitur numerik. Nilai korelasi yang mendekati 1 menunjukkan
hubungan positif yang kuat, sedangkan nilai yang mendekati
-1 menunjukkan hubungan negatif yang kuat.
""")

    with col4:

        st.subheader("Distribusi Tingkat Stres")

        fig4, ax4 = plt.subplots(
            figsize=(5,4)
        )

        sns.boxplot(
            x='sleep_disorder_risk',
            y='stress_score',
            data=df,
            ax=ax4
        )

        plt.tight_layout()

        st.pyplot(fig4)

        st.info("""
Tingkat stres yang lebih tinggi cenderung ditemukan pada
kelompok dengan risiko gangguan tidur yang lebih tinggi.
Hal ini menunjukkan bahwa stres memiliki hubungan yang kuat
dengan kualitas tidur seseorang.
""")

# =====================================================
# HALAMAN PREDIKSI
# =====================================================

elif selected == "Prediksi Risiko":

    st.header("Prediksi Risiko Gangguan Tidur")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Usia",
            min_value=1,
            max_value=100,
            value=25
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=50.0,
            value=22.0
        )

        sleep_duration = st.number_input(
            "Durasi Tidur (Jam)",
            min_value=0.0,
            max_value=24.0,
            value=7.0
        )

        stress_score = st.slider(
            "Tingkat Stres",
            1,
            10,
            5
        )

    with col2:

        heart_rate = st.number_input(
            "Detak Jantung Istirahat",
            min_value=40,
            max_value=150,
            value=70
        )

        screen_time = st.number_input(
            "Screen Time Sebelum Tidur (Menit)",
            min_value=0,
            max_value=600,
            value=120
        )

        work_hours = st.number_input(
            "Jam Kerja per Hari",
            min_value=0,
            max_value=24,
            value=8
        )

        steps = st.number_input(
            "Jumlah Langkah Harian",
            min_value=0,
            max_value=50000,
            value=5000
        )

    # =====================================================
    # PREDIKSI
    # =====================================================

    if st.button("Prediksi Risiko"):

        input_data = pd.DataFrame([{
            'age': age,
            'gender': 'Male',
            'occupation': 'Student',
            'bmi': bmi,
            'sleep_duration_hrs': sleep_duration,
            'caffeine_mg_before_bed': 50,
            'alcohol_units_before_bed': 0,
            'screen_time_before_bed_mins': screen_time,
            'exercise_day': 3,
            'steps_that_day': steps,
            'stress_score': stress_score,
            'work_hours_that_day': work_hours,
            'chronotype': 'Intermediate',
            'mental_health_condition': 'None',
            'heart_rate_resting_bpm': heart_rate,
            'shift_work': 0,
            'weekend_sleep_diff_hrs': 1
        }])

        prediction = model.predict(input_data)

        label_map = {
            0: "Healthy",
            1: "Mild",
            2: "Moderate",
            3: "Severe"
        }

        result = label_map[prediction[0]]

        st.subheader("Hasil Prediksi")

        if result == "Healthy":

            st.success(
                "Risiko gangguan tidur rendah (Healthy)"
            )

        elif result == "Mild":

            st.warning(
                "Risiko gangguan tidur ringan (Mild)"
            )

        elif result == "Moderate":

            st.warning(
                "Risiko gangguan tidur sedang (Moderate)"
            )

        else:

            st.error(
                "Risiko gangguan tidur tinggi (Severe)"
            )

        st.info(
            f"Hasil prediksi model menunjukkan kategori: **{result}**"
        )

# =====================================================
# HALAMAN EVALUASI MODEL
# =====================================================

elif selected == "Evaluasi Model":

    st.header("Evaluasi Model")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Accuracy Testing",
            value=f"{accuracy*100:.2f}%"
        )

    with col2:

        st.metric(
            label="Cross Validation",
            value="81.94%"
        )

    st.markdown("---")

    st.info("""
Model Random Forest dievaluasi menggunakan data testing
dan Cross Validation.

Hasil evaluasi menunjukkan bahwa model memperoleh
akurasi sekitar 81,98% dengan nilai Cross Validation
81,94%.

Perbedaan yang sangat kecil antara kedua nilai tersebut
menunjukkan bahwa model mampu melakukan generalisasi
dengan baik dan tidak mengalami overfitting.
""")

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    class_names = [
        "Healthy",
        "Mild",
        "Moderate",
        "Severe"
    ]

    fig5, ax5 = plt.subplots(
        figsize=(7,5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='white',
        ax=ax5
    )

    ax5.set_title(
        "Confusion Matrix Random Forest"
    )

    ax5.set_xlabel(
        "Predicted Label"
    )

    ax5.set_ylabel(
        "Actual Label"
    )

    plt.tight_layout()

    st.pyplot(fig5)

    st.caption("""
Confusion Matrix digunakan untuk membandingkan hasil
prediksi model dengan data aktual. Semakin banyak
nilai pada diagonal utama maka semakin baik performa
model dalam melakukan klasifikasi.
""")

    st.markdown("---")

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader(
        "Top Feature Importance"
    )

    rf_model = model.named_steps[
        'classifier'
    ]

    importances = rf_model.feature_importances_

    feature_names = (
        [
            'age',
            'bmi',
            'sleep_duration_hrs',
            'caffeine_mg_before_bed',
            'alcohol_units_before_bed',
            'screen_time_before_bed_mins',
            'exercise_day',
            'steps_that_day',
            'stress_score',
            'work_hours_that_day',
            'heart_rate_resting_bpm',
            'shift_work',
            'weekend_sleep_diff_hrs'
        ]
        +
        list(
            model.named_steps[
                'preprocessor'
            ]
            .transformers_[1][1]
            .named_steps['onehot']
            .get_feature_names_out(
                [
                    'gender',
                    'occupation',
                    'chronotype',
                    'mental_health_condition'
                ]
            )
        )
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig6, ax6 = plt.subplots(
        figsize=(10,6)
    )

    sns.barplot(
        data=importance_df.head(15),
        x="Importance",
        y="Feature",
        ax=ax6
    )

    plt.tight_layout()

    st.pyplot(fig6)

    st.caption("""
Feature Importance menunjukkan tingkat kontribusi
masing-masing fitur terhadap hasil prediksi.

Semakin tinggi nilai importance maka semakin besar
pengaruh fitur tersebut terhadap keputusan model
Random Forest.
""")

    st.dataframe(
        importance_df.head(15),
        use_container_width=True
    )
