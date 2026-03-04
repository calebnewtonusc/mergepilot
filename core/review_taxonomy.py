"""
Review Taxonomy — 25-category taxonomy for code review classification

Used for:
  - Tagging training data
  - Structuring model outputs
  - MergeBench evaluation by category
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReviewCategory:
    """A category in the review taxonomy."""
    id: str
    name: str
    description: str
    severity: str   # "blocking", "suggestion", "optional"
    keywords: list[str]
    examples: list[str]


TAXONOMY: dict[str, ReviewCategory] = {
    "correctness_bug": ReviewCategory(
        id="correctness_bug",
        name="Correctness Bug",
        description="Logic error that produces wrong output for valid input",
        severity="blocking",
        keywords=["bug", "incorrect", "wrong", "error", "off-by-one", "logic"],
        examples=["Off-by-one in loop bounds", "Incorrect base case in recursion"],
    ),
    "null_pointer": ReviewCategory(
        id="null_pointer",
        name="Null/None Dereference",
        description="Accessing member of potentially null/None value",
        severity="blocking",
        keywords=["null", "none", "undefined", "dereference", "npex"],
        examples=["user.name when user could be None"],
    ),
    "security_injection": ReviewCategory(
        id="security_injection",
        name="Injection Vulnerability",
        description="SQL injection, command injection, SSTI, etc.",
        severity="blocking",
        keywords=["injection", "sql", "command", "exec", "eval", "format string"],
        examples=["f-string interpolated into SQL query"],
    ),
    "security_auth": ReviewCategory(
        id="security_auth",
        name="Authentication/Authorization Issue",
        description="Missing auth check, privilege escalation, insecure session",
        severity="blocking",
        keywords=["auth", "permission", "privilege", "access control", "bypass"],
        examples=["Missing @login_required decorator"],
    ),
    "security_secrets": ReviewCategory(
        id="security_secrets",
        name="Hardcoded Secrets",
        description="API keys, passwords, tokens committed to code",
        severity="blocking",
        keywords=["secret", "password", "token", "key", "credential", "hardcoded"],
        examples=["API key hardcoded as string literal"],
    ),
    "security_xss": ReviewCategory(
        id="security_xss",
        name="XSS Vulnerability",
        description="User input rendered as HTML without escaping",
        severity="blocking",
        keywords=["xss", "innerHTML", "dangerouslySetInnerHTML", "unescaped", "sanitize"],
        examples=["innerHTML set to user-controlled value"],
    ),
    "performance_n_plus_1": ReviewCategory(
        id="performance_n_plus_1",
        name="N+1 Query Problem",
        description="Loop that executes N database queries instead of 1",
        severity="blocking",
        keywords=["n+1", "select_related", "prefetch_related", "eager loading", "lazy"],
        examples=["for user in users: user.profile  (ORM)"],
    ),
    "performance_complexity": ReviewCategory(
        id="performance_complexity",
        name="Algorithmic Complexity",
        description="O(n²) or worse where O(n) or O(n log n) is achievable",
        severity="suggestion",
        keywords=["O(n^2)", "nested loop", "quadratic", "optimization", "complexity"],
        examples=["Nested loop over same list"],
    ),
    "performance_memory": ReviewCategory(
        id="performance_memory",
        name="Memory Leak / Excessive Allocation",
        description="Memory not freed, or unnecessary large allocations",
        severity="suggestion",
        keywords=["memory leak", "allocation", "gc", "garbage collection", "buffer"],
        examples=["File handle opened but not closed"],
    ),
    "api_naming": ReviewCategory(
        id="api_naming",
        name="Poor Naming",
        description="Function/variable names that don't describe what they do",
        severity="suggestion",
        keywords=["naming", "name", "rename", "descriptive", "clarity"],
        examples=["def do_thing(): vs def process_payment():"],
    ),
    "api_breaking_change": ReviewCategory(
        id="api_breaking_change",
        name="Breaking Change",
        description="API change that breaks existing callers",
        severity="blocking",
        keywords=["breaking", "backwards compat", "deprecate", "semver", "migration"],
        examples=["Removed required parameter from public function"],
    ),
    "api_abstraction": ReviewCategory(
        id="api_abstraction",
        name="Wrong Abstraction Level",
        description="Implementation leaks through abstraction or wrong layer",
        severity="suggestion",
        keywords=["abstraction", "layer", "separation", "concern", "leaky"],
        examples=["Business logic in view layer"],
    ),
    "test_missing": ReviewCategory(
        id="test_missing",
        name="Missing Test Coverage",
        description="New code not covered by tests",
        severity="suggestion",
        keywords=["test", "coverage", "untested", "missing test", "spec"],
        examples=["New error path has no test"],
    ),
    "test_weak_assertion": ReviewCategory(
        id="test_weak_assertion",
        name="Weak Test Assertion",
        description="Test passes even when behavior is wrong",
        severity="suggestion",
        keywords=["assert", "assertEqual", "verify", "mock", "assert_called"],
        examples=["assert result is not None  (too weak)"],
    ),
    "error_handling": ReviewCategory(
        id="error_handling",
        name="Missing Error Handling",
        description="Exception could propagate uncaught or silently swallowed",
        severity="blocking",
        keywords=["exception", "try/catch", "error handling", "except", "panic"],
        examples=["try: ... except: pass  (swallows all exceptions)"],
    ),
    "type_safety": ReviewCategory(
        id="type_safety",
        name="Type Safety Issue",
        description="Missing type annotations or unsafe type operations",
        severity="suggestion",
        keywords=["any", "cast", "type", "typing", "mypy", "typescript", "as any"],
        examples=["Function returns Any instead of specific type"],
    ),
    "concurrency": ReviewCategory(
        id="concurrency",
        name="Concurrency Issue",
        description="Race condition, deadlock, or improper synchronization",
        severity="blocking",
        keywords=["race condition", "deadlock", "thread", "mutex", "lock", "atomic"],
        examples=["Shared mutable state accessed without lock"],
    ),
    "docs_missing": ReviewCategory(
        id="docs_missing",
        name="Missing Documentation",
        description="Public API lacks docstring or complex logic lacks comment",
        severity="optional",
        keywords=["docstring", "comment", "documentation", "explain", "readme"],
        examples=["Public function has no docstring"],
    ),
    "code_duplication": ReviewCategory(
        id="code_duplication",
        name="Code Duplication",
        description="Same logic repeated instead of extracted to function",
        severity="suggestion",
        keywords=["duplicate", "copy-paste", "dry", "refactor", "extract"],
        examples=["Same validation logic in 3 places"],
    ),
    "dependency_issue": ReviewCategory(
        id="dependency_issue",
        name="Dependency Management",
        description="Unnecessary dependency, version pinning, or circular import",
        severity="suggestion",
        keywords=["dependency", "import", "circular", "version", "require"],
        examples=["Importing heavy library for single utility function"],
    ),
    "logging": ReviewCategory(
        id="logging",
        name="Logging Issue",
        description="Missing logging, wrong log level, or sensitive data logged",
        severity="optional",
        keywords=["log", "logging", "debug", "print", "sensitive"],
        examples=["Printing passwords to debug log"],
    ),
    "style_formatting": ReviewCategory(
        id="style_formatting",
        name="Style/Formatting",
        description="Violates project style guide or linter rules",
        severity="optional",
        keywords=["style", "format", "lint", "pep8", "prettier", "whitespace"],
        examples=["Inconsistent indentation"],
    ),
    "architecture": ReviewCategory(
        id="architecture",
        name="Architecture Issue",
        description="Structural problem: wrong component, circular dependency, violation of design pattern",
        severity="suggestion",
        keywords=["architecture", "design pattern", "solid", "coupling", "cohesion"],
        examples=["Controller directly accessing database"],
    ),
    "migration": ReviewCategory(
        id="migration",
        name="Migration/Upgrade Issue",
        description="Database migration missing or incorrect for schema change",
        severity="blocking",
        keywords=["migration", "schema", "database", "upgrade", "rollback"],
        examples=["Column added without corresponding migration"],
    ),
    "general": ReviewCategory(
        id="general",
        name="General Feedback",
        description="Positive feedback, general suggestions, or hard to categorize",
        severity="optional",
        keywords=[],
        examples=["LGTM with minor suggestions"],
    ),
}


def classify_review_comment(comment: str) -> str:
    """Classify a review comment into a taxonomy category."""
    comment_lower = comment.lower()

    # Check categories in priority order (blocking first)
    priority_order = [
        "correctness_bug", "null_pointer", "security_injection",
        "security_auth", "security_secrets", "security_xss",
        "api_breaking_change", "error_handling", "concurrency",
        "migration", "performance_n_plus_1", "performance_complexity",
        "test_missing", "test_weak_assertion", "type_safety",
        "api_naming", "code_duplication", "docs_missing",
        "logging", "style_formatting", "general",
    ]

    for cat_id in priority_order:
        cat = TAXONOMY.get(cat_id)
        if cat is None:
            continue
        if any(kw in comment_lower for kw in cat.keywords):
            return cat_id

    return "general"


def get_blocking_categories() -> list[str]:
    """Get all blocking review categories."""
    return [cat_id for cat_id, cat in TAXONOMY.items() if cat.severity == "blocking"]


def get_category(cat_id: str) -> Optional[ReviewCategory]:
    """Get a category by ID."""
    return TAXONOMY.get(cat_id)
