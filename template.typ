// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "", subtitle: "", authors: (), logo: none, body) = {
  // Set the document's basic properties.
  set document(author: authors, title: title)

  // Save heading and body font families in variables.
  let body-font = "Avenir"
  let sans-font = "Avenir"
  
  // Set body font family.
  //set text(font: body-font, lang: "en")
  //show heading: set text(font: sans-font)
  set heading(numbering: "1.1.1")

  // Title page.
  // The page can contain a logo if you pass one with `logo: "logo.png"`.
  v(0.6fr)
  if logo != none {
    align(right, image(logo, width: 26%))
  }
  v(9.5fr)
  text(2em, weight: 700, title)

  v(0.5fr)
  text(1.5em, weight: 400, subtitle)
  
  v(0.3fr)
  // Author information.
  pad(
    top: 0.7em,
    right: 20%,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(start, strong(author))),
    ),
  )

  v(4.5fr)

  text(1.2em, weight: 300, "Submitted for the degree of Doctor of Philosophy School of Physics and Astronomy
Cardiff University")

  v(0.3fr)
  text(1.0em, weight: 300, "2023-12-30")

  v(2.4fr)
  pagebreak()

  pagebreak()

  align(center + horizon, text(2.0em, weight: 600, title))
  align(center, text(1.5em, weight: 300, subtitle))

  align(center, image("02_gravitation/waves.png", width: 70%))
  
  // Author information.
  align(center, text(strong("Michael R K Norman")))

  pagebreak()

  include "preface.typ"

  // Display inline code in a small box
  // that retains the correct baseline.
  show raw.where(block: false): box.with(
    fill: luma(240),
    inset: (x: 3pt, y: 0pt),
    outset: (y: 3pt),
    radius: 2pt,
  )
  
  // Display block code in a larger block
  // with more padding.
  show raw.where(block: true): block.with(
    fill: luma(240),
    inset: 10pt,
    radius: 4pt,
  )

  show figure: it => align(center)[
    #it.body
    #align(left)[
      #strong[
      #it.supplement
      #it.counter.display(it.numbering)
      ] | #it.caption.body
    ]
    #v(25pt, weak: true)
  ]

  // Main body.
  set par(justify: true)

  let recursive_count(_body) = {
    let r(cont) = {
      let _C = 0
    
      if type(cont) == content {

        for key in cont.fields().keys() {

          if key == "children" {
            for _child in cont.fields().at("children") {
              let resp = r(_child)
              _C += resp
            }
            
          } else if key == "body" {
            _C += r(cont.fields().at("body"))
          } else if key == "text" {
            _C += cont.fields().at("text").split(" ").len()
          } else if key == "child" {
            // return r(cont.at("child"))
            _C += r(cont.at("child"))
            // return [#cont - a]
          } else if key == "block" {
            if cont.fields().keys().contains("text") {
              _C += cont.fields().at("text").split(" ").len()
            }
          } else if key == "caption" {
            //_C += r(cont.fields().at("body"))
            
          } else if key == "label" {
            _C += r(cont.fields().at("body"))
                        
          } else if key == "supplement" {
            _C += r(cont.fields().at("body")) 
          } else if ("func", "double", "key", "keys", "update", "base").contains(key) {
            // we can skip those
            // return [#cont]
          } else if key == "t" {
            // math output - idk if I should count it
            // return [#cont]
          } else if key == "b" {
            // math output - idk if I should count it
            // return [#cont]
          } else if key == "path" {
            // image
            // return [#cont]
          } else if key == "data" {

          } else if key == "accent" {
            // return [#cont]
          } else if key == "num" {
            // return [#cont]
          } else if key == "denom" {
            
          } else if key == "dest" {
            // return [#cont]
            
          } else if key == "level" {
            // return [#cont]
          }   
        }
      } else if type(cont) == array {
        
        for item in cont {
          _C += r(item)
        } 
      }
      return _C
    }
    return r(_body)     
  }

  body

  //[#recursive_count(body)]

  counter(page).update(0)
  set page(numbering: "a", number-align: center)

  bibliography("bibliography.yml")
}