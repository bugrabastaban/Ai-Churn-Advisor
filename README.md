AI Churn Predictor (Musteri Kaybetme Tahmincisi)
Selamlar,

Bu proje temelde sunu yapiyor: Elimizde bir sirketin musteri verileri var. Klasik makine ogrenmesi modeliyle "bu musteri aboneligi iptal edip rakiplere kacacak mi?" diye tahmin yapiyoruz. Eger model "bu musteri riskli, kacabilir" derse, topu Llama 3 yapay zekasina atiyoruz. Llama 3 musteri profiline bakip "bu adami ikna etmek icin soyle bir teklif sunun" diye bize taktik veriyor.

Ozetle sadece tahmin yapip birakan degil, ayni zamanda cozum de ureten full-stack bir yapay zeka projesi.

Hangi Araclari Kullandik?
Model Egitimi: Scikit-Learn (Random Forest)

Backend / API: FastAPI

LLM (Yapay Zeka): Groq API uzerinden Llama 3.1

Frontend (Arayuz): Streamlit
