(* Title: Dual_Verification.thy
   Verifies Dual numbers from differential.hpp
*)
theory Dual_Verification
  imports Main
    "HOL.Complex_Main"
begin

record 'a dual = val :: 'a  dval :: 'a

definition dual_add :: "'a::comm_ring_1 dual => 'a dual => 'a dual" where
  "dual_add a b = (| val = val a + val b, dval = dval a + dval b |)"

definition dual_mul :: "'a::comm_ring_1 dual => 'a dual => 'a dual" where
  "dual_mul a b = (| val = val a * val b, dval = val a * dval b + dval a * val b |)"

lemma dual_add_val: "val (dual_add a b) = val a + val b"
  by (simp add: dual_add_def)

lemma dual_add_dval: "dval (dual_add a b) = dval a + dval b"
  by (simp add: dual_add_def)

lemma dual_mul_val: "val (dual_mul a b) = val a * val b"
  by (simp add: dual_mul_def)

lemma dual_mul_dval: "dval (dual_mul a b) = val a * dval b + dval a * val b"
  by (simp add: dual_mul_def)

lemma dual_mul_constexpr:
  fixes a :: "real dual" and b :: "real dual"
  assumes "a = (| val = 2, dval = 1 |)" "b = (| val = 3, dval = 0 |)"
  shows "dual_mul a b = (| val = 6, dval = 3 |)"
  using assms by (simp add: dual_mul_def)

end
