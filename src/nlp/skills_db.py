# skills_db.py
# Database of skills organized by category
# Used to extract skills from resume text

SKILLS_BY_CATEGORY = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "go", "rust", "ruby", "php", "scala", "r", "matlab", "swift",
        "kotlin", "perl", "sql", "html", "css", "bash", "powershell"
    ],

    "Web & Frameworks": [
        "react", "angular", "vue.js", "node.js", "django", "flask",
        "spring boot", "express.js", "fastapi", "next.js", "bootstrap",
        "tailwind css", "jquery", "asp.net", "laravel", "ruby on rails"
    ],

    "Data Science & ML": [
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "scikit-learn", "pandas", "numpy", "data analysis", "data visualization",
        "statistics", "nlp", "natural language processing", "computer vision",
        "neural networks", "artificial intelligence", "large language models",
        "generative ai", "regression", "classification", "clustering",
        "feature engineering", "data mining", "predictive modeling", "a/b testing"
    ],

    "Databases": [
        "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "sqlite", "oracle", "firebase", "cassandra", "dynamodb",
        "sql server"
    ],

    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ci/cd", "jenkins", "linux", "git", "github actions",
        "ansible", "nginx", "shell scripting"
    ],

    "Business & Finance": [
        "financial analysis", "budgeting", "forecasting", "accounting",
        "excel", "power bi", "tableau", "sap", "erp", "crm",
        "salesforce", "business development", "market research",
        "strategic planning", "risk management", "compliance"
    ],

    "HR & Management": [
        "talent acquisition", "recruitment", "employee relations",
        "performance management", "compensation and benefits",
        "training and development", "onboarding", "hris",
        "workforce planning", "succession planning", "payroll",
        "labor laws", "organizational development"
    ],

    "Sales & Marketing": [
        "sales strategy", "lead generation", "digital marketing",
        "seo", "sem", "social media marketing", "content marketing",
        "email marketing", "google analytics", "hubspot",
        "customer relationship management", "negotiation",
        "cold calling", "account management"
    ],

    "Soft Skills": [
        "communication", "problem solving", "teamwork", "leadership",
        "project management", "critical thinking", "time management",
        "analytical thinking", "presentation", "stakeholder management",
        "collaboration", "adaptability", "attention to detail",
        "decision making", "conflict resolution"
    ]
}

# Flat list of all skills (used for matching)
ALL_SKILLS = []
for category_skills in SKILLS_BY_CATEGORY.values():
    ALL_SKILLS.extend(category_skills)
