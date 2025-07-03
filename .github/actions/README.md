# GitHub Actions - Reusable Actions

This directory contains reusable GitHub Actions (composite actions) that can be used as steps within workflows.

## Available Actions

### 1. `run-coverage-tests/`

Composite action to run tests with coverage and check against minimum threshold.

**Inputs:**

- `coverage-command`: Command to run tests with coverage (default: pytest with coverage)
- `min-coverage-threshold`: Minimum coverage percentage to pass (default: 99.1)
- `checkout-repo`: Whether to checkout the repository (default: true)

### 2. `display-coverage-comment/`

Composite action to display coverage results as a comment on PRs.

**Inputs:**

- `coverage-percentage`: Coverage percentage to display
- `min-threshold`: Minimum coverage threshold
- `pr-number`: Pull request number

### 3. `upload-coverage-report/`

Composite action to upload coverage reports as artifacts.

**Inputs:**

- `coverage-files`: Coverage files to upload (default: coverage.xml,htmlcov/)

### 4. `update-badge/`

Composite action to update coverage badge in README.

**Inputs:**

- `coverage-percentage`: Coverage percentage to display in badge
- `badge-color`: Color of the badge (default: green)

## Usage Examples

### Using Composite Actions in Workflows

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: pip install -e '.[milvus,shopify,hubspot]'
        
      - name: Run coverage tests
        uses: ./.github/actions/run-coverage-tests
        with:
          min-coverage-threshold: '99.1'
          
      - name: Display coverage comment
        uses: ./.github/actions/display-coverage-comment
        with:
          coverage-percentage: ${{ steps.coverage-check.outputs.coverage }}
          min-threshold: '99.1'
          pr-number: ${{ github.event.pull_request.number }}
```

## Differences from Reusable Workflows

- **Actions** are composite steps that run within a single job
- **Workflows** are complete workflows that can be called by other workflows
- Actions are more lightweight and focused on specific tasks
- Workflows can contain multiple jobs and more complex logic

## Notes

- These actions are designed to be reusable across different workflows
- They handle common CI/CD tasks like testing, coverage reporting, and badge updates
- For notification functionality, see the reusable workflows in `.github/workflows/`
