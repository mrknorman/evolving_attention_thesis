#let vectorn(symbol) = {
  $bold(accent(symbol, arrow))$
}

#let dvectorn(symbol) = {
  $bold(Delta accent(symbol, arrow))$
}

#let uvectorn(symbol) = {
  $bold(accent(symbol, hat))$
}

#let udvectorn(symbol) = {
  $bold(Delta accent(symbol, hat))$
}

#let matrixn(symbol) = {
  //$bold(mat(symbol;))$
  //$bold(accent(accent(symbol, macron), macron))$
  //$bold(underline(underline(symbol)))$

  math.bold(
    underline(extent: 2pt,
      underline(extent: 2pt,
        symbol
      )
    )
  )
}

