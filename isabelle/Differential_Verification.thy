(* Title: Differential_Verification.thy
   Verifies differential forms from differential.hpp
*)
theory Differential_Verification
  imports Main
    Dual_Verification
begin

type_synonym point = "real list"
type_synonym scalar_field = "point => real"
type_synonym one_form = "scalar_field list"

definition exterior_derivative_scalar :: "scalar_field => nat => one_form" where
  "exterior_derivative_scalar f n = map (%i. %p. (f (list_update p i (p!i + 0.0000001)) - f (list_update p i (p!i - 0.0000001))) / 0.0000002) [0..<n]"

lemma exterior_derivative_dim: "length (exterior_derivative_scalar f n) = n"
  by (simp add: exterior_derivative_scalar_def)

datatype sym_expr = SConst real | SVar nat | SAdd sym_expr sym_expr | SMul sym_expr sym_expr | SSin sym_expr | SCos sym_expr

fun sym_eval :: "sym_expr => (nat => real) => real" where
  "sym_eval (SConst c) _ = c"
| "sym_eval (SVar i) env = env i"
| "sym_eval (SAdd a b) env = sym_eval a env + sym_eval b env"
| "sym_eval (SMul a b) env = sym_eval a env * sym_eval b env"
| "sym_eval (SSin a) env = sin (sym_eval a env)"
| "sym_eval (SCos a) env = cos (sym_eval a env)"

fun sym_diff :: "sym_expr => nat => sym_expr" where
  "sym_diff (SConst _) _ = SConst 0"
| "sym_diff (SVar i) var = (if i = var then SConst 1 else SConst 0)"
| "sym_diff (SAdd a b) var = SAdd (sym_diff a var) (sym_diff b var)"
| "sym_diff (SMul a b) var = SAdd (SMul (sym_diff a var) b) (SMul a (sym_diff b var))"
| "sym_diff (SSin a) var = SMul (SCos a) (sym_diff a var)"
| "sym_diff (SCos a) var = SMul (SMul (SConst (-1)) (SSin a)) (sym_diff a var)"

lemma sym_diff_add_correct:
  "sym_eval (sym_diff (SAdd a b) var) env = sym_eval (sym_diff a var) env + sym_eval (sym_diff b var) env"
  by simp

fun sym_simplify :: "sym_expr => sym_expr" where
  "sym_simplify (SAdd a b) = (if a = SConst 0 then sym_simplify b else if b = SConst 0 then sym_simplify a else SAdd (sym_simplify a) (sym_simplify b))"
| "sym_simplify (SMul a b) = (if a = SConst 0 | b = SConst 0 then SConst 0 else if a = SConst 1 then sym_simplify b else if b = SConst 1 then sym_simplify a else SMul (sym_simplify a) (sym_simplify b))"
| "sym_simplify x = x"

lemma simplify_add_zero: "sym_simplify (SAdd (SConst 0) x) = sym_simplify x"
  by simp

end
