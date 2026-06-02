# Branch Toolkit Documentation

## Purpose
The Branch Toolkit provides a set of utilities and a shell wrapper for validating Git branch states and workflow policies. It ensures deterministic checks for branch protection, working tree cleanliness, and safe history rewriting.

## Available Checks

### Shell Wrapper (`scripts/branch_toolkit_check.sh`)
The shell wrapper provides machine-parseable status checks.

- `--protected-check`: Verifies if the current branch is a protected branch (`main` or `master`).
- `--clean-check`: Verifies if the working tree is clean (no uncommitted changes).
- `--ahead-behind`: Reports commit counts ahead/behind the tracking branch.
- `--force-push-safe`: Determines if a force-push is safe (not on protected branch, no shared history).
- `--rebase-needed [base]`: Checks if the current branch has diverged from the specified base branch (default: `main`).
- `--sha-validate <SHA>`: Validates if a string follows the Git SHA format (7-40 hex characters).

### Python Helper (`tests/helpers/branch_toolkit.py`)
The Python module provides granular functions for test assertions.

- `parse_ahead_behind(status_line)`: Extracts ahead/behind counts from git status.
- `is_on_protected_branch(branch_name)`: Boolean check for `main`/`master`.
- `is_clean_working_tree(status_output)`: Boolean check for clean `--porcelain` output.
- `extract_branch_name(branch_line)`: Strips branch name from output.
- `can_force_push_safely(ahead, behind, on_protected)`: Logic for safe force-pushing.
- `requires_rebase(behind)`: Simple check for behind count.
- `parse_commit_count_from_log(log_output)`: Counts commits from `--oneline` log.
- `is_valid_sha(sha_candidate)`: Validates SHA string format.

## Exit Codes
The shell wrapper uses standard exit codes for automation:

- `0`: Check passed / Validation successful.
- `1`: Check failed / Validation failed.
- `2`: Usage error / Invalid arguments.

## Example Commands

```bash
# Check if current branch is protected
bash scripts/branch_toolkit_check.sh --protected-check

# Verify clean working tree before operations
bash scripts/branch_toolkit_check.sh --clean-check

# Check if rebase from main is needed
bash scripts/branch_toolkit_check.sh --rebase-needed main

# Validate a commit SHA
bash scripts/branch_toolkit_check.sh --sha-validate abc1234
```

## Output Parsing Contract
All shell wrapper outputs follow a strict format for machine parsing:

`STATUS: <RESULT> [- <Context>]`

Examples:
- `STATUS: SAFE - branch 'feature-x' is not protected`
- `STATUS: DIRTY - 3 file(s) with uncommitted changes`
- `STATUS: NEEDS_REBASE - branch diverged from main`
- `STATUS: VALID - SHA format correct`

Additional context or error details are printed to `stderr`.
