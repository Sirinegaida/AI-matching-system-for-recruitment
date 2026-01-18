# AI-matching-system-for-recruitment
 Our contribution consists of developing two microservices : an AI-  based matching microservice that identifies suitable candidates for a job offer and suitable  job offers for a candidate, and an analytics microservice that aims to provide analytical  dashboards for performance monitoring. 
 This work is the subject of an end-of-studies project for obtaining a Professional 
Master’s degree in Data Science. It was carried out at Tekboot Solutions as part of the 
development of the TalentCloud digital recruitment platform. The latter is composed of 
several microservices. Our contribution consists of developing two microservices : an AI- 
based matching microservice that identifies suitable candidates for a job offer and suitable 
job offers for a candidate, and an analytics microservice that aims to provide analytical 
dashboards for performance monitoring.
CV PROCESSING SYSTEM ARCHITECTURE

┌─ FRONTEND ─────────────────────────────────────┐
│ Streamlit → Candidate/Client/Admin Dashboards  │
└─────────────────┬───────────────────────────────┘
                  │
┌─ PROCESSING ────▼───────────────────────────────┐
│ File Upload → Document Parser → LangChain       │
│ → OpenAI GPT-4 → Vector Embeddings             │
│ → Matching Engine → Job Recommendations        │
└─────────────────┬───────────────────────────────┘
                  │
┌─ STORAGE ───────▼───────────────────────────────┐
│ AWS S3 (Files) + PostgreSQL (Data)             │
│ Tables: candidates, job_offers, job_matches     │
│ Apache Spark (Analytics & Fast Processing)     │
└─────────────────────────────────────────────────┘

EXTERNAL: OpenAI API + AWS Services
