#!/bin/bash
# Branch Toolkit Wrapper - Deterministic Git Branch State Checks
# Provides machine-parseable status checks for branch workflow validation
# Exit codes: 0=pass, 1=fail, 2=usage error

set -euo pipefail

VERSION="1.0.0"

show_help() {
    cat <<EOF
branch_toolkit_check.sh - Git Branch Workflow Validation

USAGE:
    bash scripts/branch_toolkit_check.sh [OPTIONS] <MODE>

MODES:
    --protected-check      Check if current branch is protected (main/master)
    --clean-check          Check if working tree is clean (no uncommitted changes)
    --ahead-behind         Report commits ahead/behind tracking branch
    --force-push-safe      Check if force-push is safe (no upstream or local-only)
    --rebase-needed        Check if branch needs rebase from base
    --sha-validate SHA     Validate SHA format (7-40 hex chars)

OPTIONS:
    --help                 Show this help message
    --version              Show version

EXIT CODES:
    0 = Check passed / Validation successful
    1 = Check failed / Validation failed
    2 = Usage error / Invalid arguments

EXAMPLES:
    # Check if on protected branch
    bash scripts/branch_toolkit_check.sh --protected-check

    # Verify working tree is clean before rebase
    bash scripts/branch_toolkit_check.sh --clean-check

    # Check if force-push is safe
    bash scripts/branch_toolkit_check.sh --force-push-safe

    # Validate commit SHA format
    bash scripts/branch_toolkit_check.sh --sha-validate abc1234

OUTPUT FORMAT:
    All outputs follow "STATUS: <result>" format for machine parsing
    Additional context printed to stderr (errors, warnings)
EOF
}

check_protected_branch() {
    local current_branch
    current_branch=$(git branch --show-current 2>/dev/null || echo "")
    
    if [[ -z "$current_branch" ]]; then
        echo "STATUS: ERROR - detached HEAD state" >&2
        return 1
    fi
    
    if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
        echo "STATUS: PROTECTED - branch '$current_branch' is protected"
        return 1
    else
        echo "STATUS: SAFE - branch '$current_branch' is not protected"
        return 0
    fi
}

check_clean_working_tree() {
    local status_output
    status_output=$(git status --porcelain 2>/dev/null || echo "")
    
    if [[ -z "$status_output" ]]; then
        echo "STATUS: CLEAN - no uncommitted changes"
        return 0
    else
        local file_count
        file_count=$(echo "$status_output" | wc -l)
        echo "STATUS: DIRTY - $file_count file(s) with uncommitted changes" >&2
        return 1
    fi
}

check_ahead_behind() {
    local status_line
    status_line=$(git status --branch --porcelain=v1 2>/dev/null | head -n1 || echo "")
    
    if [[ ! "$status_line" =~ \[(ahead|behind) ]]; then
        echo "STATUS: IN_SYNC - branch is up-to-date with tracking"
        return 0
    fi
    
    local ahead=0
    local behind=0
    
    if [[ "$status_line" =~ ahead\ ([0-9]+) ]]; then
        ahead="${BASH_REMATCH[1]}"
    fi
    
    if [[ "$status_line" =~ behind\ ([0-9]+) ]]; then
        behind="${BASH_REMATCH[1]}"
    fi
    
    echo "STATUS: DIVERGED - ahead $ahead, behind $behind"
    
    # Fail if behind (needs rebase)
    if [[ $behind -gt 0 ]]; then
        return 1
    fi
    return 0
}

check_force_push_safe() {
    local current_branch
    current_branch=$(git branch --show-current 2>/dev/null || echo "")
    
    # Fail if on protected branch
    if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
        echo "STATUS: UNSAFE - cannot force-push to protected branch '$current_branch'" >&2
        return 1
    fi
    
    # Check if branch has upstream
    local upstream
    upstream=$(git rev-parse --abbrev-ref @{upstream} 2>/dev/null || echo "")
    
    if [[ -z "$upstream" ]]; then
        echo "STATUS: SAFE - no upstream tracking (local branch only)"
        return 0
    fi
    
    # Check if commits exist on remote
    local status_line
    status_line=$(git status --branch --porcelain=v1 2>/dev/null | head -n1 || echo "")
    
    if [[ "$status_line" =~ ahead\ ([0-9]+) ]] && [[ ! "$status_line" =~ behind ]]; then
        echo "STATUS: SAFE - commits are local-only (ahead ${BASH_REMATCH[1]})"
        return 0
    fi
    
    echo "STATUS: UNSAFE - branch has shared history with upstream" >&2
    return 1
}

check_rebase_needed() {
    local base_branch="${1:-main}"
    local current_branch
    current_branch=$(git branch --show-current 2>/dev/null || echo "")
    
    if [[ -z "$current_branch" ]]; then
        echo "STATUS: ERROR - detached HEAD state" >&2
        return 2
    fi
    
    # Get merge-base
    local merge_base
    merge_base=$(git merge-base HEAD "$base_branch" 2>/dev/null || echo "")
    
    if [[ -z "$merge_base" ]]; then
        echo "STATUS: ERROR - cannot find merge-base with $base_branch" >&2
        return 2
    fi
    
    # Get base branch HEAD
    local base_head
    base_head=$(git rev-parse "$base_branch" 2>/dev/null || echo "")
    
    if [[ "$merge_base" == "$base_head" ]]; then
        echo "STATUS: UP_TO_DATE - no rebase needed"
        return 0
    else
        echo "STATUS: NEEDS_REBASE - branch diverged from $base_branch"
        return 1
    fi
}

validate_sha() {
    local sha="$1"
    
    # SHA must be 7-40 hex characters
    if [[ "$sha" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
        echo "STATUS: VALID - SHA format correct"
        return 0
    else
        echo "STATUS: INVALID - SHA must be 7-40 hex characters" >&2
        return 1
    fi
}

# Main execution
main() {
    if [[ $# -eq 0 ]]; then
        echo "ERROR: No mode specified" >&2
        show_help
        return 2
    fi
    
    case "$1" in
        --help|-h)
            show_help
            return 0
            ;;
        --version|-v)
            echo "branch_toolkit_check.sh version $VERSION"
            return 0
            ;;
        --protected-check)
            check_protected_branch
            ;;
        --clean-check)
            check_clean_working_tree
            ;;
        --ahead-behind)
            check_ahead_behind
            ;;
        --force-push-safe)
            check_force_push_safe
            ;;
        --rebase-needed)
            check_rebase_needed "${2:-main}"
            ;;
        --sha-validate)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --sha-validate requires SHA argument" >&2
                return 2
            fi
            validate_sha "$2"
            ;;
        *)
            echo "ERROR: Unknown mode '$1'" >&2
            show_help
            return 2
            ;;
    esac
}

main "$@"
