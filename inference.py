import streamlit as st
import joblib
import pandas as pd
import numpy as np 

# load the trained model

@st.cache_resource
def load_model():
    return joblib.load(r"sentiment_model.joblib")

model=load_model()


label_map={
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

st.title("Sentiment Analysis App")
st.write("Enter text to analyze its sentiment")

user_input=st.text_area("Enter text here","Type Here")

if st.button("Predict Sentiment"):
    #Predicting the sentiment
    prediction=model.predict([user_input])[0]
    prob=model.predict_proba([user_input])[0]

    #Displaying the prediction and probabilities
    sentiment_label=label_map.get(prediction,f"Class{prediction}")
    st.success(f"Predicted Sentiment: {sentiment_label}")

    st.write("Confidenct by class")

    df=pd.DataFrame({
        "Sentiment":[label_map[i] for i in range(len(prob))],
        "Probability":[np.round(prob[i],3) for i in range(len(prob))]
    })

    st.bar_chart(df.set_index("Sentiment"))

    with st.expander("Prediction Probabilities"):
        st.dataframe(df)

else:
    st.warning("Please enter text and click the 'Predict Sentiment' button.")