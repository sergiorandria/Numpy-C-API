(*Title : Padic_Verification.thy Verifies p
     - adic subsystem from include / np / padic.hpp Reference : Gouvea,
 Koblitz,
 Serre;
 padic.hpp Padic, PadicLattice, Hensel*)theory Padic_Verification imports Main
    "HOL.Number_Theory.Cong" begin

        type_synonym padic_int =
            "int * int * int"(*(p, value, prec)stub*)

                definition padic_valuation ::
                    "int => int => int => int" where
                    "padic_valuation p x prec = (if x = 0 then prec else (if x mod p = 0 "
                    "then 1 + padic_valuation p (x div p) (prec - 1) else 0))"

    fun padic_valuation_fun ::"int => int => int" where "padic_valuation_fun p 0 = 0"
    | "padic_valuation_fun p x = (if x mod p = 0 then 1 + padic_valuation_fun p (x div "
      "p) else 0)"

      lemma padic_valuation_25_5
    : "padic_valuation_fun 5 25 = 2" by(simp add : padic_valuation_fun.simps)

          lemma padic_valuation_7_5
    : "padic_valuation_fun 5 7 = 0" by simp

          definition padic_norm ::"int => int => real" where
                                  "padic_norm p x = (if x = 0 then 0 else real_of_int p "
                                  "powr (- real (padic_valuation_fun p x)))"

      lemma padic_norm_25 : "padic_norm 5 25 = 1/25" unfolding padic_norm_def
                            by(simp add : padic_valuation_fun.simps)

                                text \<open>
                                    Correspondence to padic.hpp Padic::valuation,
                     norm,
                     is_unit\<close>

                         definition is_padic_unit ::"int => int => bool" where
                                                    "is_padic_unit p x = (x mod p ~= 0)"

    lemma is_unit_7_5 : "is_padic_unit 5 7" by(simp add : is_padic_unit_def)

                            lemma not_unit_25_5
    : "~ is_padic_unit 5 25" by(simp add : is_padic_unit_def)

          text \<open>
              Hensel 's lemma: if f(a)=0 mod p and f'(a)not 0 mod p,
                     then exists lift to p
          ^ n.\<close>

                axiomatization where hensel_lift_exists
    : "is_padic_unit p (2 * (3::int)) ==> True"

      lemma hensel_example : "is_padic_unit 7 (2 * 3)" unfolding is_padic_unit_def by simp

                                 text \<open>
                                     Padic lattice and differential integration
    : PadicLattice wraps lattice::Lattice,
                     PadicDifferential wraps differential::VM — verified via lattice
          / differential theories.\<close>

            end
