# GitHub Workflows Documentation

This directory contains GitHub Actions workflows and custom actions for the Agent-First-Organization project.

## Structure Overview

```
.github/
├── workflows/           # GitHub workflow definitions
│   ├── reusable/       # Reusable workflows (called by others)
│   │   ├── ruff_code_linting.yml
│   │   ├── taskgraph_generation_validation.yml
│   │   ├── diff_based_test_coverage.yml
│   │   └── readme_coverage_badge_update.yml
│   ├── *.yml           # Trigger workflows (PR, schedule, etc.)
│   └── nightly_*.yml   # Scheduled workflows
├── actions/            # Custom composite actions
│   ├── display-coverage-comment/
│   ├── run-coverage-tests/
│   ├── update-badge/
│   └── upload-coverage-report/
├── CODEOWNERS         # Code ownership rules
└── pull_request_template.md  # PR template
```

## Workflow Architecture

### Reusable Workflows (Core Logic)

These contain the main business logic and are called by trigger workflows:

- **`reusable/ruff_code_linting.yml`**: Ruff linting with configurable Python version and file scope
- **`reusable/taskgraph_generation_validation.yml`**: TaskGraph generation and validation with configurable config path
- **`reusable/diff_based_test_coverage.yml`**: Diff-based coverage checking with PR integration
- **`reusable/readme_coverage_badge_update.yml`**: README badge updates based on coverage data

### Trigger Workflows (Event Handlers)

These respond to GitHub events and call reusable workflows:

- **`pr_code_linting.yml`**: Triggers on PR changes to Python files
- **`pr_taskgraph_validation.yml`**: Triggers on PR events for taskgraph validation
- **`pr_diff_coverage_check.yml`**: Triggers on PR labels for coverage checks
- **`pr-check.yml`**: Validates PR descriptions
- **`test_coverage.yml`**: Full coverage testing on labeled PRs
- **`test-resources.yml`**: Integration testing on labeled PRs
- **`release.yml`**: PyPI publishing on releases
- **`nightly_regression.yml`**: Scheduled nightly testing

### Custom Actions (Reusable Steps)

These encapsulate common step patterns:

- **`display-coverage-comment/`**: Posts coverage comments on PRs
- **`run-coverage-tests/`**: Runs tests and validates coverage thresholds
- **`update-badge/`**: Updates README coverage badges
- **`upload-coverage-report/`**: Uploads coverage artifacts

## Modularity Principles

1. **Single Responsibility**: Each reusable workflow handles one specific task
2. **Configuration via Inputs**: Reusable workflows accept parameters for flexibility
3. **Event Separation**: Trigger workflows handle events, reusable workflows handle logic
4. **Action Composition**: Custom actions encapsulate common step patterns
5. **Consistent Naming**: Use descriptive names for reusable workflows, hyphens for actions

## Usage Examples

### Adding a New Test Type

1. Create a reusable workflow (e.g., `reusable/integration_tests.yml`)
2. Create a trigger workflow that calls it (e.g., `pr_integration_tests.yml`)
3. Add the trigger to appropriate events (PR, schedule, etc.)

### Adding a New Custom Action

1. Create a new directory in `actions/`
2. Define `action.yml` with inputs and composite steps
3. Use the action in workflows that need the functionality

## Maintenance Guidelines

- Keep reusable workflows focused on single concerns
- Use consistent input/output patterns across reusable workflows
- Document new workflows and actions in this README
- Test workflow changes in PRs before merging to main
- Use semantic versioning for custom actions when they change
