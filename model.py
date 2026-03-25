import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load SBERT
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# Skill Database
# -------------------------------
skills_db = {
    "IT": [
        "python","java","c++","sql","machine learning","deep learning","nlp",
        "data science","pandas","numpy","tensorflow","pytorch","api","docker","git"
    ],
    "Finance": [
        "accounting","financial analysis","excel","tax","audit","banking","investment"
    ],
    "HR": [
        "recruitment","onboarding","payroll","employee relations","training"
    ],
    "Marketing": [
        "seo","content marketing","social media","branding","advertising","google ads"
    ],
    "Healthcare": [
        "patient care","clinical research","nursing","medical"
    ],
    "Sales": [
        "sales","lead generation","crm","negotiation","business development"
    ]
}

# -------------------------------
# Synonyms
# -------------------------------
synonym_map = {
    "ms excel": "excel",
    "spreadsheet": "excel",
    "machine learning": "ml",
    "deep learning": "dl",
    "natural language processing": "nlp",
    "customer relationship management": "crm"
}

def normalize_text(text):
    for k,v in synonym_map.items():
        text = text.replace(k,v)
    return text

# -------------------------------
# Extract text
# -------------------------------
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# -------------------------------
# Clean text
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# -------------------------------
# Extract section
# -------------------------------
def extract_section(text, section):
    pattern = section + r"(.*?)(education|experience|skills|projects|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else ""

# -------------------------------
# Extract skills
# -------------------------------
def extract_skills(text, skills_list):
    return list(set([
        s for s in skills_list
        if re.search(r"\b"+re.escape(s)+r"\b", text)
    ]))

# -------------------------------
# Skill frequency
# -------------------------------
def skill_frequency(text, skills):
    freq = {}
    for skill in skills:
        count = len(re.findall(r"\b" + re.escape(skill) + r"\b", text))
        if count > 0:
            freq[skill] = count
    return freq

# -------------------------------
# Domain detection
# -------------------------------
def detect_domain(text):
    scores = {}
    for d, skills in skills_db.items():
        domain_text = " ".join(skills)
        score = cosine_similarity(
            [model.encode(text)],
            [model.encode(domain_text)]
        )[0][0]
        scores[d] = score
    return max(scores, key=scores.get)

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def analyze_resume(resume_path, job_description):

    # Extract + preprocess
    resume_text = normalize_text(clean_text(extract_text_from_pdf(resume_path)))
    job_text = normalize_text(clean_text(job_description))

    # Domain detection
    domain = detect_domain(resume_text)
    relevant_skills = skills_db[domain]

    # Sections
    skills_section = extract_section(resume_text, "skills")
    exp_section = extract_section(resume_text, "experience")
    proj_section = extract_section(resume_text, "projects")

    # SBERT similarity
    resume_vec = model.encode(resume_text)
    job_vec = model.encode(job_text)

    base_score = cosine_similarity([resume_vec],[job_vec])[0][0]

    skills_score = cosine_similarity([model.encode(skills_section)],[job_vec])[0][0] if skills_section else 0
    exp_score = cosine_similarity([model.encode(exp_section)],[job_vec])[0][0] if exp_section else 0
    proj_similarity = cosine_similarity([model.encode(proj_section)],[job_vec])[0][0] if proj_section else 0

    # Skills
    resume_skills = extract_skills(resume_text, relevant_skills)
    job_skills = extract_skills(job_text, relevant_skills)

    matched_skills = list(set(job_skills) & set(resume_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))

    skill_boost = (len(matched_skills)/len(job_skills)) if job_skills else 0

    # Skill frequency (weak skills)
    skill_freq = skill_frequency(resume_text, relevant_skills)
    weak_skills = [skill for skill, count in skill_freq.items() if count == 1]

    # Industry-level suggestions
    top_domain_skills = relevant_skills[:5]
    improvement_skills = list(set(top_domain_skills) - set(resume_skills))

    # Project strength
    strong_terms = ["developed","built","implemented","designed"]
    proj_strength = sum(1 for w in strong_terms if w in proj_section)
    proj_strength_score = min(proj_strength/4,1)

    proj_score = (0.6 * proj_similarity) + (0.4 * proj_strength_score)

    # Education
    edu_keywords = ["btech","mtech","bachelor","master","phd"]
    edu_score = 1 if any(e in resume_text for e in edu_keywords) else 0

    # Final score
    final_score = (
        0.35 * base_score +
        0.25 * skill_boost +
        0.2 * exp_score +
        0.15 * proj_score +
        0.05 * edu_score
    )

    match_percentage = round(final_score * 100,2)



    if missing_skills:
        print("\n🔴 Critical Skills Missing:")
        for skill in missing_skills:
            print(f"- {skill}")

    if weak_skills:
        print("\n🟡 Skills to Strengthen:")
        for skill in weak_skills:
            print(f"- {skill}")

    if improvement_skills:
        print("\n🔵 Recommended Skills for Growth:")
        for skill in improvement_skills:
            print(f"- {skill}")
            return {
    "domain": domain,
    "match_score": match_percentage,
    "matched_skills": matched_skills,
    "missing_skills": missing_skills,
    "skills_score": skills_score,
    "exp_score": exp_score,
    "proj_score": proj_score,
    "edu_score": edu_score,
    "weak_skills": weak_skills,
    "improvement_skills": improvement_skills
}
