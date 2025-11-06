# How to Write Property-Based Tests

This guide explains how to write property-based tests using QuickChick for the MLIR-Sem project.

## Prerequisites

- Familiarity with Coq
- Understanding of property-based testing concepts
- QuickChick installed (`opam install coq-quickchick`)

## Quick Start

### 1. Choose What to Test

Good candidates for property-based testing:

- **Parsers/Printers**: Roundtrip properties
- **Optimizations**: Semantic preservation
- **Type systems**: Type safety invariants
- **Data structures**: Invariant maintenance

### 2. Write Generators

Create generators for your types in `test/property/YourModule_Properties.v`:

```coq
From QuickChick Require Import QuickChick.
Import QcDefaultNotation.
Open Scope qc_scope.

(* Simple generator using elements *)
Definition gen_comparison : G comparison :=
  elements (Lt :: Eq :: Gt :: nil).

(* Composite generator using liftGen *)
Definition gen_pair : G (nat * nat) :=
  liftGen2 pair arbitrary arbitrary.

(* Conditional generator *)
Definition gen_positive : G nat :=
  suchThatMaybe arbitrary (fun n => 0 <? n).

(* Sized generator for recursive types *)
Fixpoint gen_tree (size : nat) : G tree :=
  match size with
  | O => ret Leaf
  | S size' =>
      oneOf [
        ret Leaf;
        liftGen2 Node (gen_tree size') (gen_tree size')
      ]
  end.
```

### 3. Define Properties

Express your invariants as decidable propositions:

```coq
(* Simple boolean property *)
Definition prop_addition_commutes : Checker :=
  forAll arbitrary (fun x : nat =>
  forAll arbitrary (fun y : nat =>
    (x + y =? y + x) ?
  )).

(* Property with custom generator *)
Definition prop_parse_print_roundtrip : Checker :=
  forAllShrink gen_ast shrink (fun ast =>
    match parse (print ast) with
    | Some ast' => ast =? ast'
    | None => false
    end ?
  ).

(* Property with preconditions *)
Definition prop_div_positive : Checker :=
  forAll arbitrary (fun x : nat =>
  forAllShrink gen_positive shrink (fun y : nat =>
    (x / y * y <=? x) ?
  )).

(* Property with whenFail for debugging *)
Definition prop_sorted_preserves_elements : Checker :=
  forAll arbitrary (fun l : list nat =>
    let sorted := sort l in
    whenFail "Lists don't have same elements"
      (permutation l sorted) ?
  ).
```

### 4. Create Test Suite

Combine properties into a test suite:

```coq
Definition test_my_module : Checker :=
  conjoin [
    whenFail "Addition not commutative" prop_addition_commutes;
    whenFail "Parse/print not roundtrip" prop_parse_print_roundtrip;
    whenFail "Division incorrect" prop_div_positive;
    whenFail "Sort doesn't preserve elements" prop_sorted_preserves_elements
  ].

(* Optionally set number of tests *)
Extract Constant defNumTests => "1000".
```

### 5. Run Tests

#### Interactive Mode

```coq
QuickChick test_my_module.
```

#### Command Line

Add to your `dune` file:

```lisp
(test
 (name my_property_tests)
 (libraries quickchick)
 (action (run quickchick test_my_module)))
```

## Best Practices

### Generators

1. **Start Simple**: Begin with `arbitrary` or `elements`
2. **Control Size**: Use sized generators for recursive types
3. **Use Shrinking**: Always provide shrink functions for debugging
4. **Avoid Preconditions**: Prefer `suchThatMaybe` over filtering

```coq
(* Good: Generates only valid values *)
Definition gen_valid_index (l : list A) : G nat :=
  choose (0, length l - 1).

(* Bad: Filters, may fail *)
Definition gen_valid_index' (l : list A) : G nat :=
  suchThat arbitrary (fun i => i <? length l).
```

### Properties

1. **Keep Properties Simple**: One invariant per property
2. **Make Properties Decidable**: Use `?` for boolean properties
3. **Add Debugging Info**: Use `whenFail` with descriptive messages
4. **Test Edge Cases**: Include properties for boundary conditions

```coq
(* Good: Clear, focused property *)
Definition prop_map_length : Checker :=
  forAll arbitrary (fun l : list nat =>
  forAll arbitrary (fun f : nat -> nat =>
    (length (map f l) =? length l) ?
  )).

(* Bad: Tests multiple things *)
Definition prop_map_everything : Checker :=
  forAll arbitrary (fun l : list nat =>
  forAll arbitrary (fun f : nat -> nat =>
    (length (map f l) =? length l) &&
    (map id l =? l) &&
    (map (fun x => x) l =? l) ?
  )).
```

### Test Organization

```
test/property/
├── README.md                  # Overview and instructions
├── dune                       # Build configuration
├── AST_Properties.v          # AST property tests
├── Semantics_Properties.v    # Semantic property tests
└── Pass_Properties.v         # Optimization pass tests
```

## Common Patterns

### Roundtrip Properties

```coq
Definition prop_roundtrip : Checker :=
  forAllShrink gen_value shrink (fun v =>
    decode (encode v) =? Some v ?
  ).
```

### Preservation Properties

```coq
Definition prop_optimization_preserves_semantics : Checker :=
  forAllShrink gen_program shrink (fun p =>
    eval (optimize p) =? eval p ?
  ).
```

### Invariant Properties

```coq
Definition prop_tree_invariant : Checker :=
  forAllShrink gen_tree shrink (fun t =>
    invariant (insert x t) ?
  ).
```

### Equivalence Properties

```coq
Definition prop_implementations_equivalent : Checker :=
  forAll arbitrary (fun x : input =>
    impl1 x =? impl2 x ?
  ).
```

## Debugging Failed Tests

When QuickChick finds a counterexample:

```coq
*** Failed after 42 tests and 3 shrinks (0 discards)

Counterexample:
  x := 5
  y := 0

Error message: Division by zero
```

### Steps to Debug

1. **Reproduce Manually**:
   ```coq
   Compute (my_property 5 0).
   ```

2. **Check Generator**:
   ```coq
   Sample gen_my_type.
   ```

3. **Simplify Property**:
   ```coq
   (* Test with concrete values *)
   QuickChick (5 / 0 <> 0) ?
   ```

4. **Add Preconditions**:
   ```coq
   forAllShrink gen_nonzero shrink (fun y => ...)
   ```

## Example: Testing MLIR Operations

Here's a complete example for testing MLIR constant folding:

```coq
(* Generator for constant operations *)
Definition gen_const_op : G operation :=
  liftGen OpConstant (choose (-1000, 1000)%Z).

(* Property: Constant folding is correct *)
Definition prop_const_fold_correct : Checker :=
  forAllShrink gen_const_op shrink (fun op =>
    match eval op, eval (const_fold op) with
    | Some v1, Some v2 => v1 =? v2
    | None, None => true
    | _, _ => false
    end ?
  ).

(* Property: Constant folding never increases code size *)
Definition prop_const_fold_size : Checker :=
  forAllShrink gen_program shrink (fun p =>
    size (const_fold p) <=? size p ?
  ).

(* Test suite *)
Definition test_const_fold : Checker :=
  conjoin [
    whenFail "Constant folding changed semantics"
      prop_const_fold_correct;
    whenFail "Constant folding increased size"
      prop_const_fold_size
  ].
```

## Integration with CI

Add to `.github/workflows/ci.yml`:

```yaml
- name: Run Property Tests
  run: |
    opam install coq-quickchick
    dune test test/property
```

## Further Reading

- [QuickChick Manual](https://github.com/QuickChick/QuickChick)
- [Software Foundations: QuickChick](https://softwarefoundations.cis.upenn.edu/qc-current/)
- [Vellvm Property Tests](../../../deps/vellvm/doc/intern/vellvm-quickchick-overview.org)
- [Testing Strategy](../CLAUDE.md#testing-strategy)
