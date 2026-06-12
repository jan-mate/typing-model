#set page(
  numbering: none,
)
#set heading(numbering: none)

#align(center)[
  #v(4em)
  #text(size: 11pt)[Bachelor Thesis, 15 ECTS — Machine Learning and Data Science]

  #v(3em)
  #text(size: 20pt, weight: "bold")[A Machine-Learning Model of Inter-Keystroke Intervals for Evaluating Typing Systems]

  #v(3em)
  #text(size: 13pt)[█████ ██████]

  #text(size: 11pt)[KU-ID: ██████]

  #v(3em)
  #text(size: 11pt)[Main supervisor: █████████ ████████]

  #text(size: 11pt)[Co-supervisor: ██████ ██████]

  #v(2em)
  #text(size: 11pt)[University of Copenhagen, DIKU]

  #text(size: 11pt)[2026-06-12]
  #v(2em)
]

#include "abstract.typ"

#pagebreak()
#outline(depth: 2)

#pagebreak()
#set page(numbering: "1 / 1")
#counter(page).update(1)
#set heading(numbering: "1.1")

#include "introduction.typ"
#include "background.typ"

#include "methodology/overview.typ"
#include "methodology/data.typ"
#include "methodology/data_preparation.typ"
#include "methodology/data_enrichment.typ"
#include "methodology/data_splitting.typ"
#include "methodology/models.typ"
#include "methodology/feature_selection.typ"
#include "methodology/hpo.typ"
#include "methodology/final_training.typ"
#include "methodology/evaluation.typ"

#include "results.typ"
#include "discussion.typ"

#include "applications/overview.typ"
#include "applications/oneshot_shift.typ"
#include "applications/repeat_key.typ"
#include "applications/abbr_dict.typ"
#include "applications/dvorak.typ"
#include "conclusion.typ"

#set heading(numbering: "A.1", supplement: [Appendix])
#counter(heading).update(0)
#include "appendix.typ"

#bibliography("sources.yml")