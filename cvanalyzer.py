
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv
from collections import defaultdict
from typing import List
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
import re

load_dotenv()  # loads .env file

def get_files(directory_path):
    # STEP 1: Collect all .pdf files using DirectoryLoader (doesn't load yet)

    loader = DirectoryLoader(
        path=directory_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
        use_multithreading=True,

    )
    docs = loader.load()

    return docs

def merge_pages(docs):
    # Step 1: Group all text chunks by source
    combined_texts = defaultdict(str)

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        combined_texts[source] += doc.page_content.strip() + "\n"

    # Step 2: Recreate Document objects, one per source
    merged_docs = [
        Document(page_content=text, metadata={"source": source})
        for source, text in combined_texts.items()
    ]

    print(f"✅ Merged documents count: {len(merged_docs)}")

    return merged_docs

def get_embedding(merged_docs):
    embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(merged_docs, embeddings)
    return vector_store

# Convert distance to similarity score
def distance_to_score(distance: float) -> float:
    """Convert L2 distance to a normalized similarity score (0–100%)"""
    similarity = 1 / (1 + distance)         # Normalized between 0 and 1
    return round(similarity * 100, 2)       # As percentage



def ret_chain(vector_store):
    retriever_chain = RunnableLambda(
        lambda query: [
            _add_score_to_doc(doc, score)
            for doc, score in vector_store.similarity_search_with_score(query)
        ]
    )
    return retriever_chain

def _add_score_to_doc(doc: Document, score: float) -> Document:
    score_percent = distance_to_score(score)
    doc.metadata["score_percent"] = score_percent
    doc.metadata["distance"] = score
    return doc

def label_score_by_distance(distance: float) -> str:
    if distance <= 0.8:
        return "Strong match"
    elif distance <= 1.2:
        return "Moderate match"
    else:
        return "Weak match"


class Education(BaseModel):
    university_name: str = Field(..., description="Name of the university or institution")
    degree: str = Field(..., description="Degree earned by the candidate")
    gpa: Optional[str] = Field(None, description="GPA or grade, if available")


class Experience(BaseModel):
    company_name: str = Field(..., description="Name of the company or organization")
    n_years: Optional[str] = Field(None, description="Duration in years at this company")
    project_name: Optional[str] = Field(None, description="Name or title of a key project")
    project_description: Optional[str] = Field(None, description="Brief description of the project")
    tech_stack: List[str] = Field(..., description="Technologies and tools used in the project")

    @field_validator("tech_stack", mode="before")
    def handle_not_found_tech_stack(cls, v):
        if isinstance(v, str) and v.strip().lower() == "not found":
            return []
        return v


class ContactInfo(BaseModel):
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")


class ResumeReport(BaseModel):
    name: str = Field(..., description="Full name of the candidate")
    # age: Optional[int] = Field(None, ge=0, description="Age of the candidate, if available")
    # native_languages: List[str] = Field(default_factory=list, description="List of native languages spoken by the candidate")
    # match_score: int = Field(..., ge=0, le=100, description="Match score between 0–100")
    # match_score: str = Field(..., description="Match score between 0–100")
    education: Optional[List[Education]] = Field(..., description="List of education details")
    experience: Optional[List[Experience]] = Field(..., description="List of professional experience")
    skills: Optional[List[str]] = Field(..., description="List of key skills and strengths")
    employment_gaps: Optional[str] = Field(None, description="Employment gaps if found")
    contact_info: ContactInfo = Field(..., description="Contact information")
    summary: str = Field(..., description="Short summary about the candidate's fit for the job")

    @field_validator("skills", mode="before")
    def convert_skills_not_found(cls, v):
        if isinstance(v, str) and v.strip().lower() == "not found":
            return []
        return v

    @field_validator("education", "experience", mode="before")
    def convert_list_fields_not_found(cls, v):
        if isinstance(v, str) and v.strip().lower() == "not found":
            return []
        return v

def get_metadata(doc):
  distance = round(float(doc.metadata['distance']),4)
  score = round(float(doc.metadata['score_percent']),4)
  source = doc.metadata['source']
  return {
      'distance': distance,
      'score': score,
      'source': source
  }

def extract_years_from_text(text: str) -> float:
    """
    Extracts a numeric value (float/int) from a string and converts months to years if needed.
    
    Args:
        text (str): e.g., '6-12 months', '0.5 years', '1 year'
    
    Returns:
        float: Duration in years
    """
    match = re.search(r"\d+(\.\d+)?", text)
    if not match:
        return 0.0

    value = float(match.group())

    if 'month' in text.lower():
        return round(value / 12, 2)
    else:
        return round(value, 2)



def get_all_cv_results(retriever_docs, main_chain, job_description):
  RESULTS = []
  for retriever in retriever_docs:
    res = main_chain.invoke({"job_description": job_description, "resume": retriever.page_content}).dict()
    metadata = get_metadata(retriever)
    res['metadata'] = metadata
    res['label'] = label_score_by_distance(metadata['distance'])
    res['match_score'] = metadata['score']
    total = 0 
    for exp in res['experience']:
      yrs = extract_years_from_text(exp['n_years']) 
      total += yrs
    res['total_experience'] = total 
    RESULTS.append(res)
  return RESULTS

def get_results(directory_path, job_description):
    docs = get_files(directory_path)
    merged_docs = merge_pages(docs)
    vector_store = get_embedding(merged_docs)
    retriever_chain = ret_chain(vector_store)

    llm = ChatGroq(temperature = 0, model="llama3-8b-8192")

    retriever = retriever_chain.invoke(job_description)

    resume_parser = PydanticOutputParser(pydantic_object=ResumeReport)

    resume_prompt = PromptTemplate(
    template="""
        You are a smart and precise Resume Analyzer.

        You are given a **Job Description** and a candidate's **Resume** (plain text).
        Your task is to analyze how well the resume matches the job requirements and return a structured report as a VALID JSON object.

        ONLY use the resume content for your answers.  
        If any information is missing from the resume, write "Not Found".

        ---

        Job Description:
        {job_description}

        Resume:
        {resume}

        ---

        Return ONLY a JSON object in the format below:
        {format_instructions}
        """,
            input_variables=["job_description", "resume"],
            partial_variables={"format_instructions": resume_parser.get_format_instructions()}
        )
    
    main_chain =  resume_prompt | llm | resume_parser
    
    RESULTS = get_all_cv_results(retriever, main_chain, job_description)

    return RESULTS


if __name__ == "__main__" :
#    C:\Users\Talha\Desktop\RAG\AI-CV-Analyzer\uploads\session_20250630_202803
   directory_path = "C:/Users/Talha/Desktop/RAG/AI-CV-Analyzer/uploads/session_20250630_153024"
   job_description = '''
        Job Title: Data Scientist
        We are seeking a Data Scientist with a strong foundation in data analysis, machine learning, and statistical modeling. The ideal candidate will be responsible for extracting insights from complex datasets, building predictive models, and communicating findings that drive business decisions.

        Key Responsibilities
        Analyze large datasets to discover trends, patterns, and actionable insights.

        Build and deploy machine learning models for classification, regression, and clustering tasks.

'''
   res = get_results(directory_path, job_description)
   print(res)
    
    



