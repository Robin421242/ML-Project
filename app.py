import streamlit as st
import pickle

# Load the trained vectorizer and models
with open('ticket_models.pkl', 'rb') as f:
    vectorizer, dept_model, urg_model = pickle.load(f)

st.set_page_config(page_title="E-Commerce Ticket Router", page_icon="🎫", layout="centered")

st.title("🎫 Automated E-Commerce Customer Support Ticket Routing")
st.write("An NLP-powered system that automatically classifies incoming customer complaints into the correct department and assigns an urgency level.")

st.markdown("---")

# User Input Text Area
st.subheader("✍️ Enter Customer Support Ticket Text")
sample_text = st.text_area(
    "Type or paste a customer complaint below:",
    placeholder="e.g., My order hasn't arrived yet and tracking is completely stuck for 4 days!",
    height=120
)

if st.button("🚀 Route Ticket", type="primary"):
    if sample_text.strip() == "":
        st.warning("Please enter a ticket description first.")
    else:
        # Transform text using the trained TF-IDF vectorizer
        text_vectorized = vectorizer.transform([sample_text])
        
        # Predict Department and Urgency
        predicted_dept = dept_model.predict(text_vectorized)[0]
        predicted_urgency = urg_model.predict(text_vectorized)[0]
        
        st.success("Ticket Successfully Analyzed & Routed!")
        
        # Display Results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="📂 Assigned Department", value=predicted_dept)
            
        with col2:
            # Color code urgency display
            if predicted_urgency == "High":
                st.metric(label="⚡ Urgency Level", value="🔴 High")
            elif predicted_urgency == "Medium":
                st.metric(label="⚡ Urgency Level", value="🟡 Medium")
            else:
                st.metric(label="⚡ Urgency Level", value="🟢 Low")
                
        st.info(f"**Analysis Summary:** The NLP pipeline processed the text using TF-IDF feature extraction and routed it to **{predicted_dept}** with a priority level of **{predicted_urgency}**.")

st.markdown("---")
st.caption("MCA Project | NLP & Text Classification Module")