(** ITree Basics - Learning by Doing

    A hands-on tutorial for ITree (Interaction Trees).
    All proofs complete with Qed!

    Goal: Understand ITree reasoning for MLIR semantics.
*)

From ITree Require Import ITree Eq.
From Stdlib Require Import ZArith String.

Import ITreeNotations.
Open Scope itree_scope.

(** * Part 1: What are ITrees?

    ITrees represent computations with three constructors:
    - Ret r   : Return value r (done)
    - Tau t   : Silent step, continue with t
    - Vis e k : Visible effect e, continuation k

    We use eutt (≈) for equivalence up to Tau steps.
*)

Section WithoutEffects.
(** For learning, we use void1 - the empty effect type *)
Context {E : Type -> Type}.

(** * Part 2: Basic Equivalences *)

(** Example 1: Ret is reflexive *)
Lemma ret_refl :
  forall (n : nat),
    @eutt E nat nat eq (Ret n) (Ret n).
Proof.
  intros.
  reflexivity.
Qed.

(** Example 2: Tau is transparent *)
Lemma tau_is_invisible :
  forall (n : nat),
    @eutt E nat nat eq (Tau (Ret n)) (Ret n).
Proof.
  intros.
  rewrite tau_eutt.
  reflexivity.
Qed.

(** Example 3: Multiple Taus *)
Lemma many_taus :
  @eutt E nat nat eq (Tau (Tau (Ret 42))) (Ret 42).
Proof.
  repeat rewrite tau_eutt.
  reflexivity.
Qed.

(** * Part 3: Bind (>>=) - Sequencing *)

(** Example 4: Bind with Ret (left identity) *)
Lemma bind_ret_left :
  forall (v : nat) (k : nat -> itree E nat),
    (x <- Ret v ;; k x) ≈ k v.
Proof.
  intros.
  rewrite bind_ret_l.
  reflexivity.
Qed.

(** Example 5: Bind with Ret (right identity) *)
Lemma bind_ret_right :
  forall (m : itree E nat),
    (x <- m ;; Ret x) ≈ m.
Proof.
  intros.
  rewrite bind_ret_r.
  reflexivity.
Qed.

(** Example 6: Bind is associative *)
Lemma bind_is_associative :
  forall (m : itree E nat) (f g : nat -> itree E nat),
    (x <- (y <- m ;; f y) ;; g x) ≈
    (y <- m ;; x <- f y ;; g x).
Proof.
  intros.
  rewrite bind_bind.
  reflexivity.
Qed.

(** * Part 4: Simple Computations *)

(** Example 7: Computing a sum *)
Definition sum_10_20 : itree E nat :=
  x <- Ret 10 ;;
  y <- Ret 20 ;;
  Ret (x + y).

Lemma sum_equals_30 :
  sum_10_20 ≈ Ret 30.
Proof.
  unfold sum_10_20.
  rewrite bind_ret_l.
  rewrite bind_ret_l.
  reflexivity.
Qed.

(** Example 8: Computing a product *)
Definition product_6_7 : itree E nat :=
  a <- Ret 6 ;;
  b <- Ret 7 ;;
  Ret (a * b).

Lemma product_equals_42 :
  product_6_7 ≈ Ret 42.
Proof.
  unfold product_6_7.
  rewrite bind_ret_l.
  rewrite bind_ret_l.
  reflexivity.
Qed.

(** * Part 5: The KEY Insight for Translation Validation!

    This is exactly what we do for constant folding:
    - Original program: compute at runtime (bind operations)
    - Optimized program: pre-computed constant
    - Proof: Show they're equivalent using bind_ret_l
*)

Definition compute_at_runtime : itree E nat :=
  x <- Ret 5 ;;
  y <- Ret 3 ;;
  Ret (x + y).

Definition precomputed : itree E nat :=
  Ret 8.

Theorem constant_fold_correct :
  compute_at_runtime ≈ precomputed.
Proof.
  unfold compute_at_runtime, precomputed.
  repeat rewrite bind_ret_l.
  reflexivity.
Qed.

(** * Part 6: Nested Computations *)

Definition nested_calc : itree E nat :=
  x <- (a <- Ret 2 ;; Ret (a + 3)) ;;
  y <- (b <- Ret 4 ;; Ret (b * 2)) ;;
  Ret (x + y).

Lemma nested_result :
  nested_calc ≈ Ret 13.  (* (2+3) + (4*2) = 5 + 8 = 13 *)
Proof.
  unfold nested_calc.
  repeat rewrite bind_ret_l.
  reflexivity.
Qed.

(** * Part 7: Using Associativity *)

Definition assoc_left : itree E nat :=
  x <- (y <- Ret 10 ;; Ret (y + 5)) ;;
  Ret (x * 2).

Definition assoc_right : itree E nat :=
  y <- Ret 10 ;;
  x <- Ret (y + 5) ;;
  Ret (x * 2).

Lemma assoc_equiv :
  assoc_left ≈ assoc_right.
Proof.
  unfold assoc_left, assoc_right.
  rewrite bind_bind.
  reflexivity.
Qed.

(** * Part 8: Tau Handling *)

Definition with_taus : itree E nat :=
  Tau (x <- Tau (Ret 5) ;; Tau (Ret (x + 3))).

Lemma remove_all_taus :
  with_taus ≈ Ret 8.
Proof.
  unfold with_taus.
  rewrite tau_eutt.            (* Remove outer Tau *)
  rewrite tau_eutt.            (* Remove Tau in bind left *)
  rewrite bind_ret_l.          (* Simplify bind *)
  rewrite tau_eutt.            (* Remove Tau before Ret *)
  reflexivity.
Qed.

End WithoutEffects.

(** * Summary

    Key Learnings:
    1. ✅ eutt (≈) is equivalence up to Tau
    2. ✅ Tau is transparent (tau_eutt)
    3. ✅ Ret is identity for bind (bind_ret_l, bind_ret_r)
    4. ✅ Bind is associative (bind_bind)
    5. ✅ These are the SAME patterns we use for translation validation!

    Pattern for Optimization Proofs:
    ```
    unfold definitions
    repeat rewrite bind_ret_l
    repeat rewrite tau_eutt
    reflexivity
    ```

    Next Steps:
    - Apply these to MLIR programs
    - Handle effects (LocalE, FunctionE, etc.)
    - Complete real translation validation proofs!

    All proofs in this file are complete with Qed! 🎉
*)
