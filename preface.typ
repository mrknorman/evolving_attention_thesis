#set page(numbering: "i", number-align: center)
#counter(page).update(1)
#set heading(numbering: "I.a")

= Summary

This thesis investigates the application of machine learning to gravitational-wave data analysis. Primarily, it focuses on artificial neural networks, but it also presents work to optimize the design and application of these networks with genetic algorithms, another machine learning method. This method of hyperparameter optimisation was utilised to design models for a low-latency burst search pipeline, MLy. Along with the use of genetic algorithms for hyperparameter optimisation, work is also performed to test the performance of attention-based networks on two gravitational wave data analysis tasks, compact binary coalescence detection, and the estimation of the parameters of overlapping pairs of compact binary coalescences.

@gravitational-waves-sec introduces gravitational wave science, in order to contextualize the data analysis problems examined throughout. 

@machine-learning-sec examines the underlying principles of artificial neural networks and presents a simple example which demonstrates the effectiveness of artificial neural networks as a data analysis method. 

@application-sec then explores the intersection between the two introduced fields. It presents the methodology used for training dataset generation throughout the thesis and introduces the custom software developed in order to enable rapid dataset generation and iteration. This section also contains a review of previous work that has been done to use artificial neural networks for gravitation wave data analysis, as well as a series of experiments to demonstrate the ineffectiveness of unspecialized artificial neural networks for the task. This section concludes with recreations of some important results from the literature which act as a comparative baseline for the rest of the thesis.

@dragonn-sec presents Dragonn, a genetic algorithm that can act as a general optimisation method for neural networks in gravitational wave data science. Dragonn can optimize the network architecture, the training dataset, and the training procedure simultaneously, and easily allows for the inclusion of new hyperparameters.

@skywarp-sec presents experiments to test the effectiveness of attention-based networks for gravitational-wave analysis, more specifically, compact binary coalescence detection. It demonstrates a marginal improvement over recreated convolutional neural networks from the literature presented in @application-sec. 

Finally, @crosswave-sec expands the exploration of attention-based models to investigate cross-attention between multiple gravitational-wave detector outputs. We use this novel approach to examine the problem of parameter estimation on overlapping signals. We find that a standard convolutional neural network adapted from the literature is sufficient to distinguish between single gravitational wave signals and pairs of overlapping signals. The larger cross-attention architecture demonstrates the ability to extract parameters from multiple signals simultaneously, providing results that, in the future, may be used to aid more developed parameter estimation techniques.

#pagebreak()

// Table of contents.
#show outline: set heading(numbering: "I.a")
#outline(depth: 3, indent: true)
#pagebreak()

#let figureList(fig_type) = {
  locate(loc => {
    let elems = query(figure, loc)

    let notTypeCount = 0

    for fig in elems {

      let figLoc = query(fig.label, loc).first().location()
      let chapterCount = counter(heading).at(figLoc).first()
      let figureCount = counter(figure).at(figLoc).first()
      
      if figureCount == 1 {
        notTypeCount = 0
        counter(figure).update(0)
      }

      if fig.supplement.text == fig_type {

        let figureTypeCount = figureCount - notTypeCount

        let supp = fig.supplement 
        let body = fig.caption.body
        let label = link(fig.label, strong(
          supp + " " + str(chapterCount) + "." + str(figureTypeCount)
        ))
        label + " | " + body + "\n"
      }
      else {
        notTypeCount += 1
      }
    }
    
  })
}

= List of Figures
#figureList("Figure")
#pagebreak()

= List of Tables
#figureList("Table")
#pagebreak()

= List of Listings
#figureList("Listing")
#pagebreak()

// Collaborative Work
= Collaborative Work

A small amount of the work presented in this thesis was performed in collaboration with others. Here is a list of the sections that contain collaborative work:

- Several of the figures presented in @gravitational-waves-sec were produced by Meryl Kinnear using her impressive knowledge of Mathematica. Specifically, these were: @flat, @gravitaional-potentials, and @waves which is also the image on the half-cover page. This was done as a favour to me and is greatly appreciated.
- Although the work done to train the MLy models using the genetic-algorithm-based hyperparameter optimisation method presented in this thesis was not strictly collaborative, it is described in @deployment-in-mly as a use case of the method and software that was developed. Work to optimise and train these models was performed solely by Vasileios Skliris, with whom I have collaborated in the past on development work for the MLy pipeline, but not for any of the work presented in this thesis other than what is mentioned in @deployment-in-mly.
- The work presented in @crosswave-sec was collaborative. The problem was presented, and the training and testing datasets were generated by Philip J. Relton. I performed all the work to create and train the models, although some guidance on the nature of the problem and the importance of different aspects was provided by Phil. Our collaboration extended only to the first of the models presented, Overlapnet, after which Phil left the project. The parameter estimation model, CrossWave, was a continuation of the concept, and the same training and validation datasets generated by Phil were used, however, there was no further input from Phil in the training of CrossWave. All data analysis was performed independently, although again, the importance of certain aspects of the problem had previously been highlighted by Phil.

#pagebreak()

// Acknowledgments
= Acknowledgments

A great number of people, fortuitous events, and cups of coffee were required to create the conditions necessary for the production of this thesis. It would be a hopeless task to try and name them all (although perhaps I could train a neural network to do it). Nonetheless, I will attempt to highlight and express my overwhelming gratitude toward the most crucial contributors. 

Firstly, I would like to thank my supervisor Patrick Sutton. He has helped improve my skills as a scientist in innumerable ways and managed to direct me toward my goal whilst still providing me with the freedom to explore my ideas. I imagine achieving this balance must be one of the most difficult parts of being a Ph.D. supervisor, but he has excelled at the task.

Of course, I need to cover the basics, although thankfully in my case it is not only down to a sense of obligation. I have the most supportive family anyone could ever wish for. I am fortunate enough that they have never shown a single moment of doubt or uttered a single question as to whether this was a thing I could do and whether was a thing I should do. They have always been there, in the background, to support me and let me know that even if things didn't work out it would be okay in the end. I give special thanks to my father, Mark, who has always shown an interest in my work and even attempted to read this thesis, my mother Caroline, who sends me an advent calendar every year, and my siblings, Vanessa, Chris, and Adam.

Speaking of support, I would be remiss not to mention my source of funding, AIMLAC (it's a terrible acronym that I won't do the dignity of expanding), and the wonderful people who run it. I have made many hopefully enduring connections through the myriad of conferences and events they have put on for us, both academically and socially. Through AIMLAC, I have met many people whom I now consider friends, including Tom, Hattie, Ben, Cory, Maciek, Sophie, Robbie, and Tonichia.

Perhaps the largest contribution to the ideas behind this thesis, and the intersection between gravitational waves and machine learning research at Cardiff, comes from Vasileios Skliris. He was Patrick's student prior to me and paved much of the path that this work follows. Despite having to deal with me and my constant stream of new ideas, he continues to push for real applications of machine learning methods with his development of the MLy pipeline.

Next, we come to those who have supported me beyond an academic sense, but whose roles have been of equal importance. Without them, the thought of four years of Ph.D. work is almost incomprehensible (although maybe I could have got it done in two if I didn't have anyone in the office that I could distract). There are a great many people who fit into this category and I will certainly forget some of them (the thesis is long enough as it is). Firstly, Phil, Zoltan, and Sander, you kept me sane with many opportunities for terrible coffees, shots of gin, and rants about AI destroying the world. I already miss you all. I'd also be remiss not to mention all those who came before me, including Ali, Johnathan, and Virginia, who are hopefully enjoying the sunshine in California; and all those who will remain here after I'm gone (I swear I'm perfectly healthy), including Abhinav, Debatri, Jordan, Alex, Wasim, and of course Sama, who promised to read my thesis when it's done (I'm sorry it's so long. Good luck). I hope that she will continue to work on our joint unstarted project, Crocodile. I will also mention friends I have somehow managed to retain from the world outside gravitational waves, all of whom are very dear to me: Harvey, Luke, Patrick, and Elliot. With special thanks to Elliot, who has been my agony aunt through many situations. Oh, and I should probably thank some astronomers too, including Jacob, who helped develop the best game that can be played on the surface of a table; Andy, who continues the noble tradition of gin shots; and a myriad of others who are too numerous to mention.

Lastly, (I put these three in a separate paragraph because I think it'll be funny if they think I've missed them) thank you to Meryl, Terri, and Eva. Thank you for encouraging me to write an acknowledgments section, for supplying me with a good proportion of my current wardrobe, and for your unsuccessful attempts to help me get over my fear of sauce. You've made this last year a lot less stressful than it could have been.