module

import LearnedTGReference
import LearnedTGAngles
import LearnedTGLosses
import LearnedTGPreconditioning
import LearnedTGDiscretePerturbation
import Lean.Util.CollectAxioms

/-! Reject proof holes and nonstandard axioms in every LearnedTG theorem and definition,
including auxiliary declarations and all transitive dependencies. -/

run_cmd do
  let obsolete : Array Lean.Name := #[`LearnedTG.AnalysisSetup, `LearnedTG.ReferenceData, `LearnedTG.Covers,
    `LearnedTG.EnergyDefect, `LearnedTG.end_to_end_operator_bound,
    `LearnedTG.uniform_convergence_on_common_event, `LearnedTG.gmres_field_of_values_bound,
    `LearnedTG.preconditioned_inverse_bound, `LearnedTG.referenceFromSpectralCut]
  for name in obsolete do
    if (← Lean.getEnv).contains name then
      throwError "Obsolete continuous/contraction declaration is still imported: {name}"
  let allowed : Array Lean.Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let names ← (← Lean.getEnv).constants.foldM (init := #[]) fun names name info => do
    if Lean.Name.isPrefixOf `LearnedTG name then
      match info with
      | .thmInfo _ | .defnInfo _ | .axiomInfo _ => return names.push name
      | _ => return names
    else
      return names
  let names := names.qsort (fun a b => a.toString < b.toString)
  for name in names do
    let axioms ← Lean.collectAxioms name
    for axiomName in axioms do
      unless allowed.contains axiomName do
        throwError "Unexpected axiom {axiomName} in {name}"
    Lean.logInfo m!"{name}: {axioms}"
  Lean.logInfo m!"Axiom audit passed for {names.size} LearnedTG theorem/definition declarations."
