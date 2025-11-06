---
name: create-adr
description: Create a new Architectural Decision Record (ADR) with automatic numbering and standard template. Use when documenting significant architectural or design decisions in the mlir-sem project.
---

# Create Architectural Decision Record (ADR)

This skill guides you through creating a well-structured ADR to document architectural decisions in the mlir-sem project.

## Overview

ADRs capture important decisions made during project development:
- **What** decision was made
- **Why** it was made
- **What alternatives** were considered
- **What consequences** result from the decision

## Workflow Steps

### Step 1: Gather Decision Information

Ask the user:
- **Title**: Brief description of the decision (e.g., "Use ITrees for Semantics", "Translation Validation Framework")
- **Context**: What is the issue or problem that prompted this decision?
- **Decision**: What was decided?
- **Alternatives Considered**: What other options were evaluated?
- **Consequences**: What are the trade-offs? Benefits? Drawbacks?

### Step 2: Determine ADR Number

Check existing ADRs to assign the next number:

1. List files in `docs/adr/`
2. Find highest existing ADR number (format: `ADR-XXXX-title.md`)
3. Assign next sequential number (zero-padded to 4 digits)
4. Example: If `ADR-0003-*` exists, create `ADR-0004-*`

### Step 3: Generate ADR File

Create the ADR file in `docs/adr/` with this template:

```markdown
# ADR-XXXX: [Title]

**Status**: [Proposed | Accepted | Deprecated | Superseded by ADR-YYYY]

**Date**: YYYY-MM-DD

**Authors**: [Name(s)]

## Context

[Describe the context and problem that necessitated a decision]

- What is the architectural or design problem?
- What constraints exist?
- What requirements must be satisfied?

## Decision

[State the decision clearly and concisely]

We will [decision statement].

## Alternatives Considered

### Alternative 1: [Name]

[Description of alternative]

**Pros**:
- [Benefit 1]
- [Benefit 2]

**Cons**:
- [Drawback 1]
- [Drawback 2]

**Why not chosen**: [Explanation]

### Alternative 2: [Name]

[Similar structure...]

## Consequences

### Positive

- [Benefit 1]
- [Benefit 2]

### Negative

- [Trade-off 1]
- [Trade-off 2]

### Neutral

- [Implication 1]
- [Implication 2]

## Implementation Notes

[Optional: Implementation guidance, references to code, etc.]

- Key files affected: [List files]
- Related design docs: [Links]

## References

- [Link to relevant resources]
- [Related ADRs]
- [External documentation]

## Revision History

| Date | Author | Change |
|------|--------|--------|
| YYYY-MM-DD | [Name] | Initial version |
```

### Step 4: Populate the Template

Fill in the template with information gathered in Step 1:

1. Replace `XXXX` with the assigned ADR number
2. Replace `[Title]` with the decision title
3. Set status (typically "Proposed" for new ADRs)
4. Add today's date
5. Fill in all sections based on user input

### Step 5: Link Related Documents

If this ADR relates to:
- **Existing ADRs**: Add references in "References" section
- **Design docs**: Link to `docs/design/` files
- **Code**: Reference file paths using `src/...` notation
- **Issues**: Link to GitHub issues if applicable

### Step 6: Review and Refine

Before finalizing:
1. Check that all sections are complete
2. Ensure alternatives are fairly evaluated
3. Verify consequences are realistic
4. Confirm links are valid
5. Ask user for final review

### Step 7: Create Follow-up Tasks (if needed)

If the ADR requires implementation:
1. Create GitHub issue tracking implementation
2. Link issue to ADR in "References" section
3. Add ADR link to issue description

## ADR Lifecycle

ADRs go through status changes:

- **Proposed**: Initial state, under review
- **Accepted**: Decision is approved and being implemented
- **Deprecated**: Decision no longer applies
- **Superseded by ADR-YYYY**: Replaced by a newer decision

**Update status** when decision status changes.

## When to Create an ADR

Create ADRs for:
- ✓ Choice of core technologies (ITrees, Coq version)
- ✓ Architectural patterns (semantics framework structure)
- ✓ Dialect design decisions (how to model operations)
- ✓ Proof strategies (equivalence relations to use)
- ✓ Extraction policies (what to extract, how)
- ✓ Testing approaches (property-based testing strategy)

**Don't** create ADRs for:
- ✗ Routine bug fixes
- ✗ Simple refactorings
- ✗ Documentation typos
- ✗ Trivial implementation details

## Example ADR Titles

Good titles are specific and outcome-focused:

- ✓ "ADR-0001: Use ITrees for Compositional Semantics"
- ✓ "ADR-0002: Translation Validation Framework Design"
- ✓ "ADR-0003: Extraction Strategy for Reference Interpreter"
- ✗ "ADR-0004: About Semantics" (too vague)
- ✗ "ADR-0005: Various Changes" (not specific)

## Best Practices

1. **Be Specific**: Vague ADRs are not useful
2. **Show Trade-offs**: Every decision has consequences
3. **Evaluate Fairly**: Don't strawman alternatives
4. **Link Widely**: Connect to related docs, issues, code
5. **Keep Updated**: Update status as decisions evolve
6. **Write Clearly**: Future you (and teammates) will thank you

## Template Variations

For different types of decisions, adjust emphasis:

### Technology Choice
- Focus on: Capabilities, ecosystem, learning curve
- Example: Choosing between Coq tactics

### Design Pattern
- Focus on: Modularity, maintainability, extensibility
- Example: Handler composition strategy

### Proof Strategy
- Focus on: Proof burden, automation, maintainability
- Example: Using eutt vs eqit

### Process Decision
- Focus on: Team workflow, efficiency, quality
- Example: Testing strategy

## Common Issues

- **ADR too long**: Break into multiple ADRs if covering multiple decisions
- **No alternatives**: Consider at least 2-3 alternatives (including "do nothing")
- **Missing consequences**: Every decision has trade-offs; document them
- **Outdated ADRs**: Periodically review and update status

## Integration with Other Workflows

ADRs complement:
- **Design docs**: ADRs capture "why", design docs capture "how"
- **Issues**: ADRs provide context for implementation work
- **Code comments**: Reference ADRs in code for rationale
- **Team discussions**: Use ADRs to structure and document decisions

## After Creation

Once ADR is created:
1. Commit to version control
2. Share with team for feedback (if status is "Proposed")
3. Update related documentation to reference the ADR
4. Move status to "Accepted" once decision is approved
5. Implement decision and link to implementation commits/PRs
