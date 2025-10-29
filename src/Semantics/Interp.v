(* src/Semantics/Interpreter.v *)
From Stdlib Require Import String ZArith List.
From Stdlib.Structures Require Import OrderedTypeEx.
From ExtLib Require Import Structures.Monads Data.Monads.EitherMonad StateMonad.
From ITree Require Import ITree Events.State Core.ITreeDefinition.
From Stdlib.FSets Require Import FMapWeakList.
From MlirSem Require Import Syntax.AST Semantics.Events Semantics.Denotation.

Import MonadNotation.
Import ListNotations.

Open Scope monad_scope.
Open Scope itree_scope.

Module ZStringMap := FMapWeakList.Make(String_as_OT).

(** The local environment for a single function call. *)
Definition call_frame := ZStringMap.t mlir_value.

(** The program context holds all function definitions. *)
Definition function_def := region.
Definition program_context := ZStringMap.t function_def.

(** The interpreter's state. *)
Record interpreter_state := {
  prog_ctx : program_context;
  call_stack : list call_frame;
}.

(** The main event handler for the interpreter. It handles non-control-flow events. *)
Definition handle_event : MlirSemE ~> Monads.stateT interpreter_state (itree MlirSemE) :=
  fun T (e: MlirSemE T) =>
    match e with
    | inl1 loc_e => (* LocalE *)
        fun s : interpreter_state =>
          match s.(call_stack) with
          | [] => trigger (inr1 (inr1 (inr1 (Throw "call stack is empty"))))
          | frame :: rest =>
            match loc_e in (LocalE _ _ T0) return itree MlirSemE (interpreter_state * T0) with
            | @LocalRead _ _ id =>
                let v := match ZStringMap.find id frame with
                         | Some v => v
                         | None => IntVal 0 (* Default value for now *)
                         end in
                Ret ({| prog_ctx := s.(prog_ctx); call_stack := frame :: rest |}, v)
            | @LocalWrite _ _ id val =>
                let new_frame := ZStringMap.add id val frame in
                Ret ({| prog_ctx := s.(prog_ctx); call_stack := new_frame :: rest |}, tt)
            end
          end
    | inr1 (inl1 _func_e) => (* FunctionE *)
        Monads.liftState (trigger e)
    | inr1 (inr1 (inl1 _ctrl_e)) => (* ControlE *)
        Monads.liftState (trigger e)
    | inr1 (inr1 _) => (* FailureE *)
        Monads.liftState (trigger e)
    end.

(** [interpret] handles all non-control-flow events in an itree. *)
Definition interpret {T} (t : itree MlirSemE T)
  : Monads.stateT interpreter_state (itree MlirSemE) T :=
  interp_state handle_event t.

(** Build a program_context (name -> function body region) from an mlir_program. *)
Definition build_program_context (prog : mlir_program) : program_context :=
  List.fold_right
    (fun f ctx =>
       match f with
       | FuncOp name _ body => ZStringMap.add name body ctx
       end)
    (ZStringMap.empty function_def)
    prog.


(** [run_program] is the top-level execution function.
    It sets up the initial state and returns the initial, interpreted ITree
    to be executed by an external runner. *)
Definition run_program (prog: mlir_program) (main_fn: string)
  : option (itree MlirSemE (interpreter_state * unit)) :=
  let ctx := build_program_context prog in
  match ZStringMap.find main_fn ctx with
  | None => None (* Main function not found *)
  | Some main_body =>
      let main_func := FuncOp main_fn (FunctionType [] []) main_body in
      let t := denote_func main_func in
      let initial_frame := ZStringMap.empty mlir_value in
      let s0 := {| prog_ctx := ctx; call_stack := [initial_frame] |} in
      Some ((interpret t) s0)
  end.
