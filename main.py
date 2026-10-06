from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json

from tools import (
    get_issue,
    get_issues_by_repo,
    get_comments_on_issue,
    get_languages_in_repo,
)

from models import (
    Identifier,
    IssueList,
    IssueExplanation,
    IssueDetails,
    Intent,
    IssueAnalysis,
    RepositoryDetails,
)


# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------

class InputState(TypedDict):
    user_message: str


class State(TypedDict):
    user_message: str
    intent: str

    # analyze / explain
    identifiers: Identifier
    issue_data: dict

    # find
    repo_details: RepositoryDetails
    repo_issues: list[dict]

    # final outputs
    analysis: IssueAnalysis
    issues_found: list[IssueDetails]
    explanation: IssueExplanation
    response: str


llm = ChatOpenAI(model="gpt-4o")


# ---------------------------------------------------------
# INTENT
# ---------------------------------------------------------

def parse_user_message(state):
    """
    Determine whether user wants to:
    - analyze an issue
    - explain an issue
    - find issues
    """

    print("Parsing user intent")

    messages = [
        SystemMessage(
            content="""
            Classify the user's GitHub/open-source request.

            analyze_issue:
            User wants an assessment of a specific GitHub issue,
            including difficulty, technologies, or likely work.

            explain_issue:
            User wants a specific GitHub issue explained.

            find_issues:
            User wants issues to work on in a repository.
            """
        ),
        HumanMessage(content=state["user_message"]),
    ]

    structured_llm = llm.with_structured_output(Intent)
    result = structured_llm.invoke(messages)

    return {"intent": result.intent}


def route_intent(state):
    return state["intent"]


# ---------------------------------------------------------
# ANALYZE / EXPLAIN PATH
# ---------------------------------------------------------

def fetch_identifiers(state):
    """
    Extract owner, repository and issue number for
    analyze/explain requests.
    """

    print("Extracting issue identifiers")

    messages = [
        SystemMessage(
            content="""
            Extract the GitHub owner, repository name, and issue number
            from the user's request.

            Do not invent values.
            """
        ),
        HumanMessage(content=state["user_message"]),
    ]

    structured_llm = llm.with_structured_output(Identifier)
    result = structured_llm.invoke(messages)

    return {"identifiers": result}


def fetch_issue_data(state):
    """
    Fetch the actual GitHub issue and comments.
    This node contains NO LLM reasoning.
    """

    print("Fetching actual GitHub issue")

    identifiers = state["identifiers"]

    issue = get_issue(owner=identifiers.owner,repo=identifiers.repo,issue_number=identifiers.issue_number)

    comments = get_comments_on_issue(owner=identifiers.owner,repo=identifiers.repo,issue_number=identifiers.issue_number)

    languages = get_languages_in_repo(owner=identifiers.owner,repo=identifiers.repo)

    issue_data = {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "body": issue.get("body"),
        "state": issue.get("state"),
        "labels": [
            label.get("name")
            for label in issue.get("labels", [])
        ],
        "url": issue.get("html_url"),
        "author": issue.get("user", {}).get("login"),
        "comments": [
            {
                "author": comment.get("user", {}).get("login"),
                "body": comment.get("body"),
            }
            for comment in comments
        ],
        "repository_languages": languages,
    }

    return {"issue_data": issue_data}


def route_specific_issue_request(state):
    """
    We already fetched the issue.
    Now determine whether to analyze or explain it.
    """

    return state["intent"]


def analyze_issue(state):
    """
    LLM reasons over actual GitHub data.
    """

    print("Analyzing issue")

    messages = [
        SystemMessage(
            content="""
            You are an open-source expert helping a developer evaluate
            whether they should contribute to a GitHub issue.

            Base your analysis ONLY on the GitHub issue information
            provided.

            Do not invent repository details, technologies, requirements,
            or implementation work that are not supported by the data.

            If something cannot be determined, say so.
            """
        ),
        HumanMessage(
            content=f"""
            User request:
            {state["user_message"]}

            GitHub issue:
            {json.dumps(state["issue_data"], indent=2)}
            """
        ),
    ]

    structured_llm = llm.with_structured_output(IssueAnalysis)
    result = structured_llm.invoke(messages)

    return {
        "analysis": result
    }


def explain_issue(state):
    """
    Explain actual issue data in simpler language.
    """

    print("Explaining issue")

    messages = [
        SystemMessage(
            content="""
            You are an open-source expert.

            Explain the supplied GitHub issue clearly and unambiguously.

            Base your explanation ONLY on the supplied GitHub data.

            Do not invent implementation details.
            """
        ),
        HumanMessage(
            content=f"""
            User request:
            {state["user_message"]}

            GitHub issue:
            {json.dumps(state["issue_data"], indent=2)}
            """
        ),
    ]

    structured_llm = llm.with_structured_output(IssueExplanation)
    result = structured_llm.invoke(messages)

    return {"explanation": result}


# ---------------------------------------------------------
# FIND ISSUES PATH
# ---------------------------------------------------------

def extract_repo_details(state):
    """
    Find out which repository the user wants searched.
    """

    print("Extracting repository details")

    messages = [
        SystemMessage(
            content="""
            Extract the GitHub repository owner and repository name only
            if they are explicitly provided or can be unambiguously
            identified from the user's request.

            NEVER guess or choose a repository for the user.

            If a specific repository is not provided:

            repo_found = false
            owner = null
            repo = null
            """
        ),
        HumanMessage(content=state["user_message"]),
    ]

    structured_llm = llm.with_structured_output(RepositoryDetails)
    result = structured_llm.invoke(messages)

    return {
        "repo_details": result
    }


def route_repo(state):
    if state["repo_details"].repo_found:
        return "repo_found"

    return "repo_missing"


def ask_for_repo(state):
    """
    We cannot search until the user tells us which repository.
    No LLM required.
    """

    return {
        "response": (
            "Please specify the GitHub repository you want me to search. "
            "For example: opensearch-project/OpenSearch-Dashboards."
        )
    }


def fetch_repo_issues(state):
    """
    Fetch real issues from the requested repository.
    """

    print("Fetching repository issues")

    repo_details = state["repo_details"]

    raw_items = get_issues_by_repo(owner=repo_details.owner,repo=repo_details.repo)

    # GitHub's /issues endpoint can also contain pull requests.
    actual_issues = [
        item
        for item in raw_items
        if "pull_request" not in item
    ]

    # Keep only useful information before sending to LLM.
    repo_issues = [
        {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "body": issue.get("body"),
            "labels": [
                label.get("name")
                for label in issue.get("labels", [])
            ],
            "url": issue.get("html_url"),
            "author": issue.get("user", {}).get("login"),
            "state": issue.get("state"),
        }
        for issue in actual_issues
    ]

    return {"repo_issues": repo_issues}


def find_issues(state):
    """
    Rank/select from issues that actually came from GitHub.
    """

    print("Selecting matching issues")

    messages = [
        SystemMessage(
            content="""
            You are an open-source contribution advisor.

            Select up to 3 GitHub issues from the supplied candidate
            issues that best match the user's request.

            IMPORTANT:
            - Only select issues from the supplied candidate list.
            - Never invent an issue.
            - Never invent a URL.
            - Do not claim technologies or difficulty unless they can
              reasonably be inferred from the supplied issue information.
            """
        ),
        HumanMessage(
            content=f"""
            User request:
            {state["user_message"]}

            Repository:
            {state["repo_details"].owner}/{state["repo_details"].repo}

            Candidate issues:
            {json.dumps(state["repo_issues"], indent=2)}
            """
        ),
    ]

    structured_llm = llm.with_structured_output(IssueList)
    result = structured_llm.invoke(messages)

    return {
        "issues_found": result.issues
    }


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

def formatted_result(state):
    return state


# ---------------------------------------------------------
# GRAPH
# ---------------------------------------------------------

builder = StateGraph(State,input_schema=InputState)


builder.add_node("parse_user_message",parse_user_message)

builder.add_node("fetch_identifiers",fetch_identifiers)

builder.add_node("fetch_issue_data",fetch_issue_data)

builder.add_node("analyze_issue",analyze_issue)

builder.add_node("explain_issue",explain_issue)

builder.add_node("extract_repo_details",extract_repo_details)

builder.add_node("fetch_repo_issues",fetch_repo_issues)

builder.add_node("find_issues",find_issues)

builder.add_node("ask_for_repo",ask_for_repo)

builder.add_node("formatted_result",formatted_result)


# START
builder.add_edge(START,"parse_user_message")


# First routing decision
builder.add_conditional_edges(
    "parse_user_message",
    route_intent,
    {
        "analyze_issue": "fetch_identifiers",
        "explain_issue": "fetch_identifiers",
        "find_issues": "extract_repo_details",
    },
)


# ---------------------------------------------------------
# ANALYZE / EXPLAIN
# ---------------------------------------------------------

builder.add_edge("fetch_identifiers","fetch_issue_data")

builder.add_conditional_edges(
    "fetch_issue_data",
    route_specific_issue_request,
    {
        "analyze_issue": "analyze_issue",
        "explain_issue": "explain_issue",
    },
)


# ---------------------------------------------------------
# FIND ISSUES
# ---------------------------------------------------------

builder.add_conditional_edges(
    "extract_repo_details",
    route_repo,
    {
        "repo_found": "fetch_repo_issues",
        "repo_missing": "ask_for_repo",
    },
)

builder.add_edge("fetch_repo_issues","find_issues")


# ---------------------------------------------------------
# FINISH
# ---------------------------------------------------------

builder.add_edge("analyze_issue","formatted_result")

builder.add_edge("explain_issue","formatted_result")

builder.add_edge("find_issues","formatted_result")

builder.add_edge("ask_for_repo","formatted_result")

builder.add_edge("formatted_result",END)


graph = builder.compile()