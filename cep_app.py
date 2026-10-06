import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- DATASET ----------------
data = {
    "Role": [
        "AI/ML Intern",
        "Data Science Intern",
        "Web Development Intern",
        "Software Development Intern",
        "Cyber Security Intern",
        "Healthcare AI Intern",
        "Sports Data Analyst Intern",
        "Space Technology Intern",
        "Finance Data Analyst Intern",
        "IoT Intern"
    ],

    "Sector": [
        "Technology",
        "Technology",
        "Technology",
        "Technology",
        "Cyber Security",
        "Medical",
        "Sports",
        "Space",
        "Finance",
        "Technology"
    ],

    "Skills": [
        "python machine learning artificial intelligence data science",
        "python statistics machine learning pandas data analysis",
        "html css javascript web development bootstrap",
        "c cpp python java programming software development",
        "networking cyber security linux ethical hacking",
        "python machine learning healthcare medical data analysis",
        "python statistics data analysis sports analytics",
        "python mathematics data analysis space technology",
        "python statistics data analysis finance",
        "python electronics sensors arduino iot"
    ]
}

df = pd.DataFrame(data)

# ---------------- NLP ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Skills"])

# ---------------- KNN MODEL ----------------
model = NearestNeighbors(n_neighbors=3, metric="cosine")
model.fit(X)


# ---------------- PAGE DESIGN ----------------
st.set_page_config(
    page_title="AI Internship Recommendation System",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI-Based Internship Recommendation System")

st.write(
    "This system recommends suitable internships based on "
    "student skills and interests using NLP and KNN."
)

st.divider()

# ---------------- STUDENT INPUT ----------------
st.header("👩‍🎓 Student Information")

name = st.text_input("Enter your name")

skills = st.text_input(
    "Enter your skills",
    placeholder="Example: Python, Machine Learning, HTML, CSS"
)

interest = st.text_input(
    "Enter your area of interest",
    placeholder="Example: AI, Medical, Sports, Space"
)

# ---------------- RECOMMENDATION ----------------
if st.button("🔍 Recommend Internships"):

    if skills == "" and interest == "":
        st.warning("Please enter your skills or area of interest.")

    else:

        student_text = skills + " " + interest

        student_vector = vectorizer.transform([student_text])

        distances, indices = model.kneighbors(student_vector)

        st.success("Recommendations generated successfully!")

        st.subheader("🌟 Recommended Internships")

        for i, index in enumerate(indices[0]):

            similarity = (1 - distances[0][i]) * 100

            st.markdown(
                f"""
                ### {i + 1}. {df.iloc[index]["Role"]}

                **Sector:** {df.iloc[index]["Sector"]}

                **Match Score:** {similarity:.1f}%

                **Required Skills:** {df.iloc[index]["Skills"].title()}
                """
            )

            st.divider()


# ---------------- DATASET ----------------
with st.expander("📊 View Internship Dataset"):

    st.dataframe(df, use_container_width=True)


st.caption(
    "Developed as a CEP Project using NLP (TF-IDF) and K-Nearest Neighbors (KNN)."
)