import streamlit as st
import requests


st.set_page_config(page_title="AI Churn Predictor", page_icon="🚀", layout="centered")

st.title(" Yapay Zeka Destekli Müşteri Analiz Paneli")
st.markdown(
    "Bu panel, arka planda çalışan **FastAPI** ve **Llama 3.1** destekli makine öğrenmesi modeline bağlanarak müşterilerin ayrılma riskini analiz eder.")

st.header("👤 Müşteri Bilgilerini Girin")


with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Müşteri Süresi (Ay)", min_value=0, max_value=72, value=4)
        monthly_charges = st.number_input("Aylık Fatura Tutarı", min_value=0.0, value=120.0)
        total_services = st.number_input("Kullanılan Hizmet Sayısı", min_value=1, max_value=9, value=4)

    with col2:
        has_family = st.selectbox("Aile Durumu (Eş/Çocuk var mı?)", ["Hayır", "Evet"])
        bill_spike = st.selectbox("Son Faturada Şok Yaşadı mı?", ["Hayır", "Evet"])
        high_risk = st.selectbox("Riskli Kontrat (Aylık + Çeksiz)?", ["Hayır", "Evet"])


    total_charges = tenure * monthly_charges


    submit_button = st.form_submit_button(label="🤖 Yapay Zeka ile Analiz Et")


if submit_button:

    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Total_Services": total_services,
        "Has_Family": 1 if has_family == "Evet" else 0,
        "Bill_Spike": 1 if bill_spike == "Evet" else 0,
        "High_Risk_Profile": 1 if high_risk == "Evet" else 0
    }

    with st.spinner("Llama 3 müşteri profilini inceliyor ve strateji üretiyor..."):
        try:

            response = requests.post("http://127.0.0.1:8000/predict", json=payload)

            if response.status_code == 200:
                result = response.json()

                prob = result["churn_probability"]
                is_churn = result["churn_prediction"] == 1
                ai_message = result["ai_analysis_and_action"]

                st.divider()

                # 4. Sonuçları Ekrana Yazdırma
                colA, colB = st.columns([1, 2])

                with colA:
                    st.metric(label="Ayrılma Riski", value=f"%{prob}")
                    if is_churn:
                        st.error("🚨 RİSKLİ MÜŞTERİ")
                    else:
                        st.success("✅ GÜVENDE")

                with colB:
                    st.subheader("🧠 Llama 3 Analizi ve Aksiyon Önerisi")
                    if is_churn:
                        st.warning(ai_message)
                    else:
                        st.info(ai_message)

            else:
                st.error(f"API Hatası: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Bağlantı Hatası! Lütfen FastAPI sunucusunun (uvicorn) arka planda çalıştığından emin ol.")