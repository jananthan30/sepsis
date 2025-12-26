# Workflow Preferences

## Development Approach
**Standard Development**

Write implementation code first, then add tests for critical paths as needed. This approach prioritizes rapid iteration while ensuring essential functionality is verified.

### Guidelines
- Implement features incrementally with working code at each step
- Add unit tests for critical business logic (prediction pipeline, data processing)
- Add integration tests for API endpoints
- Use notebooks for exploratory analysis, then formalize into production scripts

## Commit Strategy
**Conventional Commits with Semantic Versioning**

Use structured commit messages for clarity and automated changelog generation.

### Commit Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Commit Types
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes nor adds feature |
| `perf` | Performance improvement |
| `test` | Adding tests |
| `chore` | Maintenance, dependencies |
| `data` | Data pipeline or model training changes |

### Scopes
- `model` - ML model architecture or training
- `api` - FastAPI endpoints
- `dashboard` - Streamlit application
- `data` - Data processing pipeline
- `eda` - Exploratory data analysis
- `deploy` - Docker, deployment configs

### Examples
```
feat(api): add patient batch prediction endpoint
fix(model): correct temperature unit conversion in preprocessing
docs(readme): update deployment instructions
perf(dashboard): cache model predictions for repeated queries
```

## Code Review
**Self-Review with Documentation**

- Primary development is solo; self-review before merge
- Document significant decisions in commit messages or code comments
- For major architectural changes, create a design doc in `maestro/tracks/<track>/`

## Testing Requirements

### Coverage Guidelines
| Component | Coverage Target | Test Type |
|-----------|-----------------|-----------|
| Data Pipeline | High (critical) | Unit + Integration |
| Model Training | Medium | Validation metrics |
| API Endpoints | High | Integration |
| Dashboard | Low | Manual testing |

### Test Approach
1. **Unit Tests**: Critical data transformations, feature engineering
2. **Integration Tests**: API endpoint response validation
3. **Model Validation**: AUROC, sensitivity, specificity on held-out test set
4. **Manual Testing**: Dashboard UI/UX verification

### Testing Tools
- `pytest` for Python tests (when added)
- Model metrics computed during training
- FastAPI TestClient for API testing

## Documentation Standards
**Comprehensive**

### Required Documentation
1. **Docstrings**: All public functions and classes
   - Google-style docstrings preferred
   - Include parameter types, return types, and examples

2. **README Updates**: Keep README.md current with:
   - Installation instructions
   - Usage examples
   - Model performance metrics

3. **Inline Comments**: For complex logic only
   - Clinical domain knowledge explanations
   - Non-obvious algorithmic choices
   - Data transformation rationale

4. **API Documentation**:
   - FastAPI auto-generates OpenAPI/Swagger docs
   - Keep Pydantic models well-documented

5. **Architecture Documentation**:
   - Model architecture diagrams
   - Data flow documentation
   - Clinical validation protocols

### Clinical Context Documentation
Given the healthcare domain, document:
- Clinical rationale for feature selection
- Physiological bounds and their medical significance
- Sepsis-3 criteria implementation details
- Potential clinical workflow integration points

## Branch Strategy

### Branch Naming
```
<type>/<short-description>
```

Examples:
- `feature/batch-prediction-api`
- `fix/temperature-conversion`
- `experiment/transformer-model`

### Main Branches
- `main` - Production-ready code
- `hf-deploy` - HuggingFace Spaces deployment

## Pull Request Guidelines

### PR Description Template
```markdown
## Summary
[Brief description of changes]

## Changes
- [Bullet list of specific changes]

## Testing
- [ ] Unit tests pass
- [ ] Model metrics validated
- [ ] API endpoints tested
- [ ] Dashboard manually verified

## Clinical Impact
[Any clinical workflow implications]
```
