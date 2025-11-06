# add-pass

Guide through verifying an MLIR optimization pass using Translation Validation - proving semantic equivalence between programs before and after optimization. Use when the user wants to verify an MLIR pass (e.g., sccp, constant folding, DCE) or add pass verification support.

## Steps

1. **Identify the Pass**
   - Determine which MLIR optimization pass to verify
   - Find test programs that exercise the pass
   - Run mlir-opt to get before/after versions

2. **Convert to Coq AST**
   - Use the existing OCaml driver to parse both versions
   - Export the ASTs as Coq definitions
   - Import into Coq verification module

3. **Set Up Verification Module**
   - Create `src/TranslationValidation/{PassName}.v`
   - Import the ITree semantics framework
   - Define equivalence relation (prog_equiv using eutt)

4. **Generate Test Cases**
   - Process MLIR test suite examples
   - Create Coq modules with before/after ASTs
   - Organize by pass and test complexity

5. **Prove Equivalence**
   - Use ITree tactics to prove semantic equivalence
   - Document any assumptions or limitations
   - Add to CI golden test suite

6. **Update Documentation**
   - Create ADR if introducing new verification approach
   - Update pass verification status in docs
   - Add to test oracle comparison

## Example Usage

When user says "verify the SCCP pass", this skill will:
1. Find SCCP test cases in MLIR test suite
2. Run mlir-opt --sccp on each test
3. Convert both versions to Coq AST using OCaml driver
4. Create TranslationValidation/SCCP.v with equivalence proofs
5. Integrate with oracle testing framework

## Key Files
- `driver/run.ml` - OCaml interpreter entry point
- `driver/transformer.ml` - MLIR to OCaml AST conversion
- `src/Semantics/Interp.v` - ITree-based interpreter
- `test/oracle/` - Golden test infrastructure