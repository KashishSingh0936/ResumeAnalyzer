import streamlit as st
from model import analyze_resume
st.title("📄 Resume Analyzer")

resume = st.file_uploader("Upload Resume (PDF)")
job = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if resume and job:

        # Save uploaded file
        with open("resume.pdf", "wb") as f:
            f.write(resume.read())

        # Run model
        result = analyze_resume("resume.pdf",job)

        # Display results
        st.success("Analysis Complete")

        st.subheader("📌 Domain")
        st.write(result["domain"])

        st.subheader("📊 Match Score")
        st.write(f"{result['match_score']} %")

        st.subheader("✅ Matched Skills")
        st.write(result["matched_skills"] if result["matched_skills"] else "None")

        st.subheader("❌ Missing Skills")
        st.write(result["missing_skills"] if result["missing_skills"] else "None")

        st.subheader("🧠 Section Scores")
        st.write("Skills:", result["skills_score"])
        st.write("Experience:", result["exp_score"])
        st.write("Projects:", result["proj_score"])

        st.subheader("🎓 Education Score")
        st.write(result["edu_score"])

        st.subheader("🚀 Suggestions")

        if result["missing_skills"]:
            st.write("🔴 Critical Skills Missing:", result["missing_skills"])

        if result["weak_skills"]:
            st.write("🟡 Skills to Strengthen:", result["weak_skills"])

        if result["improvement_skills"]:
            st.write("🔵 Recommended Skills:", result["improvement_skills"])

    else:
        st.error("Upload file + enter JD")
