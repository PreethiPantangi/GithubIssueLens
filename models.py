from pydantic import BaseModel
from typing import Literal

class Intent(BaseModel):
    intent: Literal["analyze_issue", "find_issues", "explain_issue"]

class IssueExplanation(BaseModel):
    explanation: str

class RepositoryDetails(BaseModel):
    repo_found: bool
    repo: str | None = None
    owner: str | None = None

class IssueDetails(BaseModel):
    issue_title: str
    owner: str
    issue_number: int
    project: str
    issue_type: str
    difficulty: str
    technologies: list[str]
    summary: list[str]    
    link: str

class IssueList(BaseModel):
    issues: list[IssueDetails]

class IssueAnalysis(BaseModel):
    summary: str
    issue_type: str
    difficulty: str
    technologies: list[str]
    likely_work_required: list[str]    

class Identifier(BaseModel):
    owner: str
    repo: str
    issue_number: int
