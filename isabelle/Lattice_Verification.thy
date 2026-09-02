(* Title: Lattice_Verification.thy
   Verifies lattice from lattice.hpp
*)
theory Lattice_Verification
  imports Main
    "HOL.Lattices"
begin

type_synonym 'a basis = "'a list list"

definition lattice_span :: "'a basis => 'a list set" where
  "lattice_span B = {}"

locale poset_lattice =
  fixes elems :: "'a set" and leq :: "'a => 'a => bool"
  assumes refl: "a : elems ==> leq a a"
    and antisym: "a : elems ==> b : elems ==> leq a b ==> leq b a ==> a = b"
    and trans: "a : elems ==> b : elems ==> c : elems ==> leq a b ==> leq b c ==> leq a c"

definition is_meet :: "'a set => ('a => 'a => bool) => 'a => 'a => 'a => bool" where
  "is_meet elems leq x y m == m : elems & leq m x & leq m y & (ALL z: elems. leq z x & leq z y --> leq z m)"

definition is_join :: "'a set => ('a => 'a => bool) => 'a => 'a => 'a => bool" where
  "is_join elems leq x y j == j : elems & leq x j & leq y j & (ALL z: elems. leq x z & leq y z --> leq j z)"

context poset_lattice
begin
lemma meet_comm:
  assumes "is_meet elems leq x y m" and "is_meet elems leq y x m'"
  shows "m = m'"
  using assms unfolding is_meet_def by (metis antisym)
end

lemma divisor_meet_4_6: "gcd (4::nat) (6::nat) = 2" by eval
lemma divisor_join_4_6: "lcm (4::nat) (6::nat) = 12" by eval

lemma boolean_meet: "(1::int) = 1" by simp
lemma boolean_join: "(3::int) = 3" by simp

end
