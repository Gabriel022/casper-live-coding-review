# Casper Studios Live Coding Interview - Review Copy

This repository is a review copy of the solution I built during my Casper Studios live coding interview on July 14, 2026.

The purpose of this repository is to make the finished implementation, tests, runtime outputs, and supporting evidence easy to review.

## Important note about Git history

This review repository was created on August 11, 2026 for convenience and was intentionally committed with the current date.

The original interview repository has been preserved separately and has not been modified.

The preserved original Git history shows that at 11:02:05 AM on July 14, 2026, I created a "Base commit" where:

- main.py was 0 bytes
- requirements.txt was 0 bytes
- tests_main.py was 0 bytes

The implementation was then built during the live interview.

Supporting Git evidence is available in the evidence/ directory. The complete original repository state is preserved separately in the evidence archive.

## What I built

The solution is an end-to-end Hacker News research pipeline:

1. Fetch the Hacker News front page.
2. Parse stories into structured records.
3. Filter for external articles.
4. Skip articles processed during earlier runs.
5. Download and extract usable article text.
6. Validate source records before passing them to the LLM.
7. Use an LLM to generate structured article analysis.
8. Validate the LLM output against the known source set.
9. Render a readable report.
10. Save the report as a Markdown artifact.
11. Persist processed Hacker News IDs so later runs avoid duplicates.

## Architecture

The implementation is intentionally divided into clear stages:

Discovery
-> Candidate Filtering
-> Deduplication
-> Article Extraction
-> Record Validation
-> LLM Analysis
-> Application Validation
-> Report Rendering
-> Persistent Seen-ID State

My goal was to keep each stage understandable and testable rather than treating the AI model as the architecture of the application.

## AI-assisted development approach

AI coding tools were used to accelerate implementation, as explicitly encouraged by the interview instructions.

I used the AI tool as an implementation accelerator rather than as the decision maker.

During the session I:

- Defined the pipeline architecture before implementation.
- Selected the libraries and data contracts.
- Identified scraping and LLM failure cases.
- Defined validation boundaries.
- Reviewed the generated implementation.
- Ran and inspected the application.
- Reproduced the repeated-article issue raised by the interviewers.
- Designed a lightweight persistence solution.
- Added a test for the new behavior.
- Reran the complete application and verified the result.

## Tests

tests_main.py contains focused tests for important boundaries in the system:

- Hacker News parsing and external-link filtering.
- Rejection of records with unknown source IDs.
- Persistent deduplication across executions.

Run the tests with:

pytest -v tests_main.py

## Runtime evidence

The output/ directory contains reports generated during the July 14 interview session.

The first runs:

- research_brief_20260714_111011.md
- research_brief_20260714_111259.md

contain the same source articles and demonstrate the repeated-processing behavior discussed during the interview.

After the deduplication change, later reports contain different article sets.

Examples:

- research_brief_20260714_112037.md
- research_brief_20260714_112115.md
- research_brief_20260714_112132.md

output/seen_articles.json contains the Hacker News source IDs remembered across runs.

## Evidence directory

The evidence/ directory contains:

- casper_git_history.txt
- casper_git_reflog.txt
- casper_base_commit_tree.txt
- casper_interview_final_vs_base.patch
- casper_evidence_sha256.txt

These files were generated from the preserved original repository before this review repository was created.

## Original evidence package

The complete original July 14 repository, including its .git directory and final uncommitted interview working tree, has been preserved separately.

That archive is being provided separately so the original evidence remains distinct from this convenience review copy.
