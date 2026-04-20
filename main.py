from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from groq import Groq


model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')


client = Groq(api_key="api_key")

app = FastAPI(title="AI Churn Predictor API", description="ML Modeli + Llama 3 Müşteri Danışmanı")


class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Total_Services: int
    Has_Family: int
    Bill_Spike: int
    High_Risk_Profile: int
    other_features: dict = {}


@app.post("/predict")
def predict_churn(data: CustomerData):
    
    input_data = {
        "tenure": data.tenure, "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges, "Total_Services": data.Total_Services,
        "Has_Family": data.Has_Family, "Bill_Spike": data.Bill_Spike,
        "High_Risk_Profile": data.High_Risk_Profile
    }
    input_data.update(data.other_features)
    df = pd.DataFrame([input_data])

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    probability = model.predict_proba(df)[0][1]
    prediction = int(model.predict(df)[0])


    if prediction == 1:
        fatura_durumu = "Evet" if data.Bill_Spike == 1 else "Hayır"

        
        prompt = f"""
        Müşterinin bizi terk etme ihtimali %{round(probability * 100, 1)}.
        Müşteri Profili:
        - Bizimle geçirdiği süre: {data.tenure} ay
        - Faturasında ani artış yaşadı mı: {fatura_durumu}
        - Kullandığı toplam hizmet sayısı: {data.Total_Services}

        Lütfen bu müşterinin neden riskli olduğunu 1 cümleyle açıkla.
        Ardından bu müşteriyi ikna edip sistemde tutmak için müşteri temsilcisine 1 adet yaratıcı ve uygulanabilir teklif öner. 
        Kısa, profesyonel ve Türkçe yanıt ver.
        """


        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Sen uzman bir yapay zeka iş stratejistisin."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.5,
        )

        llm_tavsiyesi = chat_completion.choices[0].message.content
    else:
        llm_tavsiyesi = "Müşteri güvende görünüyor, standart sadakat programına devam edilebilir."


    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability * 100, 2),
        "ai_analysis_and_action": llm_tavsiyesi
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
