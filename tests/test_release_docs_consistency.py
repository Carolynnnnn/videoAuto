import os
import pytest
import re


def test_operational_docs_exist():
    """Verify that all required operational documents exist."""
    docs = [
        "pixelle_snapshot/PRODUCTION_RUNBOOK.md",
        "pixelle_snapshot/INCIDENT_PLAYBOOK.md",
        "pixelle_snapshot/RELEASE_CHECKLIST.md",
        "pixelle_snapshot/TROUBLESHOOTING_RUNBOOK.md"
    ]
    for doc in docs:
        assert os.path.exists(doc), f"Operational doc missing: {doc}"


def test_referenced_scripts_exist():
    """Verify that scripts referenced in operational docs exist."""
    scripts = [
        "pixelle_snapshot/validate_config_profiles.py",
        "pixelle_snapshot/audit_boundaries.py",
        "build.py",
        "build_incremental.py"
    ]
    for script in scripts:
        assert os.path.exists(script), f"Referenced script missing: {script}"


def test_doc_content_consistency():
    """Verify that docs contain key sections and commands."""
    with open("pixelle_snapshot/RELEASE_CHECKLIST.md", "r") as f:
        content = f.read()
        assert "## 2. Rollout Procedure" in content
        assert "## 4. Rollback Procedure" in content
        assert "PIXELLE_TEST_MODE=1" in content

    with open("pixelle_snapshot/INCIDENT_PLAYBOOK.md", "r") as f:
        content = f.read()
        assert "## 2. Response Procedures" in content
        assert "## 3. Fallback & Degradation Policy" in content
        assert "PIXELLE_CONCURRENCY_LIMIT" in content

    with open("pixelle_snapshot/PRODUCTION_RUNBOOK.md", "r") as f:
        content = f.read()
        assert "## 1. Operational Modes" in content
        assert "## 2. Normal Operations" in content
        assert "PIXELLE_CONCURRENCY_LIMIT" in content


def test_duration_policy_documentation():
    """Verify that duration policy (1/2/3 minutes) is documented correctly."""
    # Check README.md
    with open("README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
        # Assert duration-minutes policy exists with options 1, 2, 3
        assert re.search(r"duration-minutes.*\(.*1.*2.*3.*\)", readme_content, re.IGNORECASE), \
            "README.md must document --duration-minutes with options 1, 2, 3"
        # Assert default is 1
        assert re.search(r"duration-minutes.*默认.*1", readme_content) or \
               re.search(r"默认.*1.*分钟", readme_content), \
            "README.md must document default duration as 1 minute"

    # Check workflow doc
    with open("docs/video-generation-workflow.md", "r", encoding="utf-8") as f:
        workflow_content = f.read()
        # Assert duration-minutes appears with 1, 2, 3 options
        assert re.search(r"duration-minutes.*\(.*1.*2.*3.*\)", workflow_content, re.IGNORECASE), \
            "video-generation-workflow.md must document --duration-minutes (1, 2, 3)"


def test_ai_only_policy_cap_and_fallback():
    """Verify ai_only mode documents 6-clip cap and template fallback semantics."""
    # Check README.md
    with open("README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
        
        # Assert 6-clip cap is documented
        assert re.search(r"ai_only.*AI.*6.*片段", readme_content, re.IGNORECASE | re.DOTALL) or \
               re.search(r"AI.*额度.*6", readme_content, re.DOTALL), \
            "README.md must document ai_only 6-clip cap"
        
        # Assert template fallback for over-cap segments
        assert re.search(r"超过.*6.*模板|Template.*降级|回退.*模板", readme_content, re.IGNORECASE | re.DOTALL), \
            "README.md must document template fallback for segments exceeding 6-clip cap"
        
        # Assert strict exhaustion semantics for in-cap failures
        assert re.search(r"ai_only_exhausted.*Provider.*耗尽|技术故障.*ai_only_exhausted", readme_content, re.IGNORECASE | re.DOTALL), \
            "README.md must document ai_only_exhausted for provider failures within cap"

    # Check workflow doc
    with open("docs/video-generation-workflow.md", "r", encoding="utf-8") as f:
        workflow_content = f.read()
        
        # Assert 6-clip cap
        assert re.search(r"ai_only.*6.*片段|AI.*额度.*6", workflow_content, re.IGNORECASE | re.DOTALL), \
            "video-generation-workflow.md must document 6-clip AI cap"
        
        # Assert template fallback
        assert re.search(r"超过.*6.*模板|Template.*降级", workflow_content, re.IGNORECASE | re.DOTALL), \
            "video-generation-workflow.md must document template fallback for over-cap"


def test_ai_only_policy_no_outdated_strict_wording():
    """Guard against regression to outdated ai_only strict-only wording (no fallback mentioned)."""
    # This test detects if docs revert to old wording that implied ai_only mode
    # had NO template fallback at all (strict exhaustion for all segments)
    
    with open("README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
        
        # Extract ai_only section (semantic block extraction via markers)
        ai_only_section_match = re.search(
            r"ai_only.*?(?=^-\s+\*\*|\n##|\Z)",
            readme_content,
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        
        if ai_only_section_match:
            ai_only_section = ai_only_section_match.group(0)
            
            # If section mentions "严格" or "仅允许", it MUST also mention fallback/template/cap
            if re.search(r"严格.*AI.*模式|仅允许.*AI", ai_only_section, re.IGNORECASE):
                assert re.search(r"模板|Template|降级|回退|cap|额度|6", ai_only_section, re.IGNORECASE), \
                    "README ai_only section mentions strict semantics but missing cap/fallback policy - possible regression to outdated wording"
    
    with open("docs/video-generation-workflow.md", "r", encoding="utf-8") as f:
        workflow_content = f.read()
        
        # Same check for workflow doc
        ai_only_section_match = re.search(
            r"ai_only.*?(?=^###|\n##|\Z)",
            workflow_content,
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        
        if ai_only_section_match:
            ai_only_section = ai_only_section_match.group(0)
            
            if re.search(r"严格.*AI.*模式|仅允许.*AI", ai_only_section, re.IGNORECASE):
                assert re.search(r"模板|Template|降级|回退|cap|额度|6", ai_only_section, re.IGNORECASE), \
                    "workflow doc ai_only section mentions strict semantics but missing cap/fallback policy - possible regression"
