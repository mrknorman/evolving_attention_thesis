= Conclusion <conclusion-sec> 

Gravitational wave science will soon graduate from its infancy and into the routine. One could be forgiven for thinking that the most exciting days are behind us and that all that remains is an exercise in stamp collecting --- perhaps it will be a useful project to map the features of the universe, but not one that will reveal any deep fundamental truths. I think such a statement would be premature. Given the sensitivity that may be within our grasp with next-generation detectors as well as promising upgrade paths for current-generation interferometers, I think there will be ample opportunity for us to stumble upon something unexpected, or to find something we always anticipated would be there if we just looked hard enough. That being said, if we wish to one day make these discoveries, we need to continually improve the toolset with which we look, both in hardware and software. We must improve all aspects of our search pipelines with a focus on efficiency, in order that we do not lose momentum and with it public and government support. I believe that the use of machine learning will be crucial to this end, due to its flexibility and its power to be employed in domains wherein exact specifications of the problem are difficult. That being said, caution is advised when applying machine learning techniques to problems. More effort should be employed to review the possibilities within traditional data analysis techniques before jumping into machine learning solutions.

In recent years, there has been a scramble to utilise artificial neural networks for gravitational-wave data analysis, and many papers have applied many different techniques and utilised many different, sometimes flawed, validation schemes which are oftentimes divorced from the reality of detection and parameter estimation problems within a real live detection environment. This vast multitude of techniques presented by the literature ensures that it is difficult to build a picture of what specific techniques are effective, and which are not. The field suffers from a lack of standardised pipelines and comparative metrics with which to compare methods. This PhD had been an attempt to slightly ameliorate that problem, primarily by developing the GravyFlow library, and using it to attempt a more robust hyperparameter optimisation framework, as well as investigate the application of the attention mechanism for two problems in gravitational wave data analysis, CBC detection and overlapping waveform detection and parameter estimation. This chapter serves as a summary of the main results of each chapter of this thesis, and concludes with suggested future work that could further advance the ideas presented here.

== Chapter Summaries

In this subsection, we will go over the main results of each chapter and explain the scope of the work performed. @intro-sec, @gravitational-waves-sec, and @machine-learning-sec are omitted as these are introductory chapters and do not contain any results. They present a general introduction to the thesis, an introduction to gravitational wave science, and an introduction to machine learning techniques respectively.

=== #link(<application-sec>)[Application Summary]

@application-sec does not present novel work. Rather, it attempts an explanation of the methodology used throughout the thesis, a review of existing techniques, and recreates some of those existing techniques to act as a baseline for comparison of other architectures. In this chapter, we show that dense-layer neural networks are not sufficient to solve the detection problem within the criteria outlined for live detection pipelines by the LIGO-VIRGO-Kagra collaboration. We introduce Convolutional Neural Networks (CNNs) and review the literature that has previously applied this architecture to the detection. We then recreate some standout results from the literature and show that these models come much closer to approaching that of matched filtering, the current standard detection method. We comment that the need for a low false alarm rate remains the most significant barrier to the conversion of such methods into a working pipeline.

=== #link(<dragonn-sec>)[Dragonn Summary]

@dragonn-sec presents original work. In this chapter, we utilise genetic algorithms in order to optimise the hyperparameters of CNNs used for the CBC detection problem. Due to time constraints, we are not able to run the optimiser for as many generations as was hoped, and thus we do not achieve significant results other than a demonstration of the application of the optimisation method. Dispite the dissapointing performance, the optimisation does offer



=== #link(<skywarp-sec>)[Skywarp Summary]

=== #link(<crosswave-sec>)[CrossWave Summary]

== Future Work

NoiseNet
UniCron

