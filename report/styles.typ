// Shared table styling. Tweak these lines to restyle every table at once.
#let header-fill = luma(220) // header row (darker, so light zebra stays distinct)
#let stripe-fill = luma(239) // zebra band on alternate data rows

// Zebra fill: darker header on row 0, first data row white, then stripe every other row.
#let zebra = (col, row) => if row == 0 { header-fill } else if calc.even(row) { stripe-fill }

// Flat fill: grey header only, plain white data rows (no zebra).
#let flat = (col, row) => if row == 0 { header-fill }

// Stacked cell: a value with a small grey sub-line beneath it (used for 95% CIs).
#let mae(v, c) = [#v \ #text(size: 0.78em, fill: luma(95))[#c]]
