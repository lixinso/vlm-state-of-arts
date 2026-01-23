# Contributing to VLM State-of-Arts

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## How to Contribute

### Reporting Issues

If you find errors, outdated information, or have suggestions:

1. Check existing issues first
2. Create a new issue with:
   - Clear description of the problem
   - Specific file and section affected
   - Suggested correction (if applicable)
   - Sources for any factual claims

### Updating Information

The VLM field moves quickly. Help us stay current by:

1. **New Models**: Add documentation for significant new models
2. **Benchmark Updates**: Update scores when new results are published
3. **Paper Additions**: Add important new papers to the papers directory
4. **Resource Updates**: Add helpful new tools and resources

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b add-new-model`)
3. Make your changes
4. Ensure markdown formatting is correct
5. Submit a pull request with:
   - Clear description of changes
   - References/sources for new information
   - Any related issues

## Content Guidelines

### Model Documentation

When adding new model documentation:

```markdown
# Model Name

## Overview
Brief description of the model and its significance.

## Architecture
- Vision encoder details
- Language model details
- Projection mechanism
- Include architecture diagram if helpful

## Benchmark Performance
Include table with relevant benchmarks.

## Usage
Code examples for common use cases.

## Resources
Links to official resources.

## Citation
BibTeX entry if available.
```

### Benchmark Information

When adding benchmark details:

- Official name and acronym
- What it measures
- Dataset size and composition
- Evaluation metrics
- Current top performers
- Link to official benchmark

### Paper Summaries

When adding papers:

- Full citation (authors, title, venue, year)
- arXiv or publication link
- Key contributions (2-3 sentences)
- Impact on the field
- BibTeX entry

## Style Guidelines

### Markdown

- Use proper heading hierarchy (H1 for title, H2 for sections, etc.)
- Include blank lines before and after lists
- Use code blocks with language specification
- Keep tables aligned and readable

### Content

- Be factual and cite sources
- Use present tense for current capabilities
- Avoid marketing language
- Include both strengths and limitations
- Keep information concise but complete

### Code Examples

- Include working code that can be copy-pasted
- Specify required packages and versions
- Use comments to explain non-obvious parts
- Test code before submitting

## Areas Needing Contributions

High-priority areas for contributions:

1. **Benchmark scores**: Keep performance tables current
2. **New models**: Document significant releases
3. **Tutorials**: Add practical how-to guides
4. **Comparisons**: Add comparison tables across models
5. **Translations**: Translate content to other languages

## Questions?

If you have questions about contributing, please open an issue for discussion.

Thank you for helping keep this resource accurate and useful!
