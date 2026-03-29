import streamlit as st
import pickle
import os

# 1. Page Header
st.set_page_config(page_title="Gender Classifier", page_icon="👤")

st.sidebar.markdown(f"""
### Student Details
**Name:** Anosha Fatima  
**Roll No:** [Your Roll Number]
""")

st.title("👤 Gender Classification App")

# 2. HELPER FUNCTION TO FIND FILES
def find_file(name):
    """Search for the file in the current directory and subdirectories."""
    if os.path.exists(name):
        return name
    # Search subdirectories just in case
    for root, dirs, files in os.walk("."):
        if name in files:
            return os.path.join(root, name)
    return None

# 3. LOAD MODELS
model_file = find_file('gender_model.pkl')
tfidf_file = find_file('tfidf_vectorizer.pkl')

if model_file and tfidf_file:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(tfidf_file, 'rb') as f:
        tfidf = pickle.load(f)
    
    # --- Prediction UI ---
    user_input = st.text_area("Enter Text:", placeholder="Type a tweet or bio here...")
    
    if st.button("Predict"):
        if user_input.strip():
            vect = tfidf.transform([user_input])
            prediction = model.predict(vect)
            st.success(f"### Result: {prediction[0]}")
            st.balloons()
        else:
            st.warning("Please enter some text.")
else:
    # 4. ERROR DIAGNOSIS (This helps you see what is wrong)
    st.error("❌ Model Files Not Found!")
    st.write("Current files in your repository:")
    # List all files so you can see exactly what GitHub sees
    all_files = []
    for root, dirs, files in os.walk("."):
        for f in files:
            all_files.append(os.path.join(root, f))
    st.code("\n".join(all_files))
    st.info("Check the list above. If you see 'gender_model.pkl.pkl', rename it on GitHub!")

st.markdown("---")
st.caption("Deployment Task - Streamlit Cloud")
