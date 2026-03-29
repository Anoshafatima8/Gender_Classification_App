{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5a88a8e-5988-45ea-93cd-2ef7dd2e6902",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "\n",
    "# 1. Page Header with Student Info\n",
    "st.set_page_config(page_title=\"Gender Classifier\")\n",
    "\n",
    "st.sidebar.markdown(\"\"\"\n",
    "### Student Information\n",
    "**Name:** Anosha Fatima  \n",
    "**Roll No:** SP23-BCS-133 \n",
    "\"\"\")\n",
    "\n",
    "st.title(\"Gender Classification App\")\n",
    "st.info(\"This model predicts gender based on text input (e.g., Tweets or Bios).\")\n",
    "\n",
    "# 2. Load Models\n",
    "try:\n",
    "    model = pickle.load(open('Trained Models/gender_model.pkl', 'rb'))\n",
    "    tfidf = pickle.load(open('Trained Models/tfidf_vectorizer.pkl', 'rb'))\n",
    "except:\n",
    "    # Fallback if folder structure is different\n",
    "    model = pickle.load(open('gender_model.pkl', 'rb'))\n",
    "    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))\n",
    "\n",
    "# 3. Input Area\n",
    "user_input = st.text_area(\"Enter text to classify:\", placeholder=\"Type something here...\")\n",
    "\n",
    "if st.button(\"Predict Gender\"):\n",
    "    if user_input.strip():\n",
    "        vect = tfidf.transform([user_input])\n",
    "        prediction = model.predict(vect)\n",
    "        \n",
    "        # Display Result\n",
    "        st.subheader(f\"Result: {prediction[0]}\")\n",
    "        st.balloons()\n",
    "    else:\n",
    "        st.error(\"Please enter some text first!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (NLP)",
   "language": "python",
   "name": "nlp_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
