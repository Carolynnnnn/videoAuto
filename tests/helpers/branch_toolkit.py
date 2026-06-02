"""Branch workflow helper utilities for test assertions.

Provides small deterministic functions for parsing git branch status
and validating branch workflow policies.
"""

from __future__ import annotations


def parse_ahead_behind(status_line: str) -> tuple[int, int]:
    """Parse ahead/behind counts from git status output.

    Args:
        status_line: A single line from git status output containing branch info.
                     Example: "Your branch is ahead of 'origin/main' by 3 commits."
                     Example: "Your branch is behind 'origin/main' by 2 commits."
                     Example: "Your branch and 'origin/main' have diverged,"

    Returns:
        Tuple of (ahead_count, behind_count).
        Returns (0, 0) if no ahead/behind info found.
    """
    ahead = 0
    behind = 0

    if "ahead" in status_line:
        parts = status_line.split("by")
        if len(parts) >= 2:
            try:
                ahead = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    if "behind" in status_line:
        parts = status_line.split("by")
        if len(parts) >= 2:
            try:
                behind = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    return ahead, behind


def is_on_protected_branch(branch_name: str) -> bool:
    """Check if branch name is a protected branch (main/master).

    Args:
        branch_name: Name of the branch.

    Returns:
        True if branch is main or master, False otherwise.
    """
    return branch_name in {"main", "master"}


def is_clean_working_tree(status_output: str) -> bool:
    """Check if working tree is clean from git status output.

    Args:
        status_output: Full output from git status --porcelain.

    Returns:
        True if no changes detected, False otherwise.
    """
    return len(status_output.strip()) == 0


def extract_branch_name(branch_line: str) -> str:
    """Extract branch name from git branch --show-current output.

    Args:
        branch_line: Output from git branch --show-current.

    Returns:
        Stripped branch name.
    """
    return branch_line.strip()


def can_force_push_safely(ahead: int, behind: int, on_protected: bool) -> bool:
    """Determine if force-push is safe based on branch state.

    Args:
        ahead: Number of commits ahead of upstream.
        behind: Number of commits behind upstream.
        on_protected: Whether currently on protected branch (main/master).

    Returns:
        True if force-push is safe (not on protected, ahead > 0, behind == 0).
        False otherwise.
    """
    if on_protected:
        return False
    if ahead == 0:
        return False
    if behind > 0:
        return False
    return True


def requires_rebase(behind: int) -> bool:
    """Check if branch requires rebase based on behind count.

    Args:
        behind: Number of commits behind upstream.

    Returns:
        True if behind > 0.
    """
    return behind > 0


def parse_commit_count_from_log(log_output: str) -> int:
    """Count number of commits from git log --oneline output.

    Args:
        log_output: Output from git log --oneline.

    Returns:
        Number of non-empty lines (commit count).
    """
    lines = [line for line in log_output.strip().split("\n") if line.strip()]
    return len(lines)


def is_valid_sha(sha_candidate: str) -> bool:
    """Check if string looks like a valid git SHA (40 hex chars or 7+ hex chars).

    Args:
        sha_candidate: String to validate.

    Returns:
        True if valid SHA format, False otherwise.
    """
    sha = sha_candidate.strip()
    if len(sha) < 7:
        return False
    if len(sha) > 40:
        return False
    try:
        int(sha, 16)
        return True
    except ValueError:
        return False
