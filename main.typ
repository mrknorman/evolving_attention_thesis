#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "Evolving Attention",
  subtitle: "Exploring the Use of Genetic Algorithms and Attention for Gravitational Wave Data Science",
  authors: (
    "Michael R K Norman",
  ),
  logo: "cardiff_logo.png"
)

// We generated the example code below so you can see how
// your document will look. Go ahead and replace it with
// your own content!

#set page(numbering: "1", number-align: center)

#include "01_intro/01_introduction.typ"
#pagebreak()
#include "02_gravitation/02_gravitational_waves.typ"
#pagebreak()
#include "03_machine_learning/03_machine_learning.typ"
#pagebreak()
#include "04_application/04_application.typ"
#pagebreak()
#include "05_parameters/05_the_problem_with_parameters.typ"
#pagebreak()
#include "06_skywarp/06_skywarp.typ"
#pagebreak()
#include "07_crosswave/07_crosswave.typ"
#pagebreak()