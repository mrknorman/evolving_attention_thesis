#set page(numbering: "1", number-align: center)

#set math.equation(numbering: it => {[5.#it]})
#counter(math.equation).update(0)

= Dragonn: Exploring Deep Gravitational-Wave Classifier Hyperparameter Space with Genetic Algorithms <dragonn-sec>

== The Problem with Parameters <hyperparameters-section>
\
An applicable machine-learning approach can be found for almost every problem in gravitational-wave data science @gw_machine_learning_review. That does not mean that machine learning should be applied to every problem in gravitational-wave data science. We must be careful with a liberal application of machine learning approaches and always keep the goal in mind: what exactly are we trying to achieve by applying this particular method? As described in the "No free lunch theorem" @no_free_lunch, for every possible algorithm, there are advantages and disadvantages, and there is no algorithm that completely supersedes another in all cases. This means a rigorous and systematic method for comparing different techniques is required. This problem is confounded with deep learning techniques, as the number of free parameters when designing and optimising your neural network is vast --- technically infinite in the non-real case where network size is not a constraint. 

There are a huge number of adaptations that can be applied to a network @architecture_review @deep_learning_review @conv_review @attention_is_all_you_need, and the number of developed layer types and model architectures is considerable and increasing in an almost exponential fashion year on year @exponential_growth. Even ignoring the number of different types of network modifications, most modifications have multiple associated parameters, including parameters specifying the design of individual network layers @cnn_review @attention_is_all_you_need @dropout_ref @batch_normalisation_ref. We label any parameter to do with the model design or any parameter that is not optimized during the model training process as a *hyperparameter* @hyperparameter_optimisation_review @cnn_hyperparameters.

Hyperparameters include values to do with the greater structure of the network, such as the type and number of layers in a network, the configuration of the layers themselves, such as the number of neurons in a dense layer, or the number of filters in the convolutional layer; the training of the network, such as the learning rate, the number of epochs, and the optimiser; as well as all the parameters associated with the training dataset @hyperparameter_optimisation_review @cnn_hyperparameters. Essentially, hyperparameters encompass all parameters that must be determined before the initiation of model training.

This chapter will first give a brief overview of available hyperparameter optimisation methods, then discuss why evolutionary population-based methods were chosen as the hyperparameter optimisation technique of choice, followed by a demonstration of the effectiveness of hyperparameter optimisation to improve model performance on various gravitational-wave data analysis problems. We will conclude by discussing how this work has been pivotal in developing MLy @MLy, a machine learning pipeline currently preparing for live deployment in the latter half of the fourth joint observing run.

The goal of any given hyperparameter optimisation process is to maximise the model's performance given a specific *objective function* @hyperparameter_optimisation_review @cnn_hyperparameters. This objective function could be as simple as minimising the model loss, but other performance metrics might also be important to us, such as model inference time or memory usage --- or, as is the case for gravitational wave transient detection, minimising values that it would not necessarily make sense to have as part of the loss function, like the False Alarm Rate (FAR) @false_alarm_rate_ref. If we naively gave models a loss function that only allows a once-in-one-hundred-year FAR, they might never produce a positive result at all @false_alarm_rate_ref. It would be hard to balance such a low FAR requirement with other terms in the loss function, and balancing loss function terms is always a difficult challenge that can lead to training instability.

If one is to compare two different sets of architectures, for example, comparing fully connected networks @perceptron_and_neural_network_chapter to networks with some convolutional layers @deep_learning_review @conv_review, a method must be used to determine all of these hyperparameters. Many, if not most, of these hyperparameters, will have some effect, somewhere between significant and small, on the model's overall performance @hyperparameters_model_performance. Thus, just like the tunable parameters of the model itself, the vector space defined by these hyperparameters comprises regions of different model performances, and indeed model performance can be measured in multiple ways. Presumably, given the task at hand, there will be some region within this parameter space that maximises desired performance goals. In the optimal scenario, the comparison of two sets of architectures will occur between these regions. Thus, a method must find approximate values for these optimal hyperparameters. 

We might now see the recursion that has started. We are applying an optimisation method to an optimisation that will introduce its own set of hyperparameters. Such hyperparameters will, in turn, need to be at least selected if not optimised. However, it can be shown that the selection of network hyperparameters can make a profound impact @hyperparameters_model_performance on the performance of the model. It is hoped that with each optimisation layer, the effects are considerably diminished, meaning that roughly tuned hyperparameters for the hyperparameter optimiser are sufficient to find comparably optimised solutions.

We can use a similar example parameter space to the one that we that we generated in @gradient-descent-sec, except this time it is being used to represent the hyperparameter space against the model objective function, rather than parameter space against the model loss. See @hyperparameter_space.

#figure(
    image("hyperparameter_space.png",  width: 90%), 
    caption: [An example arbitrary hyperparameter space generated from a random mixture of Gaussians. The space presented here is 2D. In actuality, the space is likely to have a much larger dimensionality. Unlike in gradient descent where we are trying to minimize our loss, here we are trying to maximize our objective function, whatever we have determined that to be.]
) <hyperparameter_space>

Perhaps unsurprisingly, hyperparameter optimisation is an area of considerable investigation and research in machine learning @hyperparameter_optimisation_review. However, similar to the rest of the field, it would be incorrect to call it well-understood. Whilst there are several effective methods for hyperparameter optimisation, there is no universally accepted set of criteria for which method to use for which problems. What follows is a brief non-comprehensive review of currently available hyperparameter optimisation techniques.

=== Human-guided trial and error

The most straightforward and obvious method to find effective model hyperparameters relies on human-guided trial and error. This method, as might be expected, involves a human using their prior assumptions about the nature of the problem, the dataset, and the model structure, to roughly guide them towards an acceptable solution, using multiple trials to rule out ineffective combinations and compare the results to the human's hypothesised intuitions. Evidently, whilst this technique is simple to implement and can be time efficient, it suffers from several deficiencies. The results of this method can vary in effectiveness depending on the previous experience of the guiding human; if they have a lot of experience with prior optimisation tasks, they are likely to have more effectively tuned priors. It is also possible that an experienced optimiser might have overly tuned priors, and this bias might cause them to miss possible new solutions that were either previously overlooked or are only relevant to the particular problem being analysed. The results of this method also suffer from a lack of consistency; even the most experienced human optimiser is unlikely to perform precisely the same optimisation technique across multiple problems. Despite these weaknesses, this method is commonly used throughout gravitational wave machine-learning papers @gabbard_messenger_cnn @george_huerta_cnn and can still be an effective solution for isolated optimisation.

=== Grid Search

A more methodical approach is to perform a grid search across the entirety or a specified subsection of the available parameter space @hyperparameter_optimisation_review. In this method, a grid of evenly spaced points is distributed across the selected parameter space. A trial is performed at each grid point, and the performance results of those trials are then evaluated. Depending on the computing power and time available, this process can be recursed between high-performing points. This method has the advantage of performing a much more rigorous search over the entirety of the parameter space. However, it can be highly computationally expensive if your parameter space has large dimensionality, which is often the case. A grid search can also be ineffective at finding an optimal solution if the objective function is non-linear and highly variable with minor changes, or evidently, if its solution lies outside of the range of initial boundaries. See @grid_search for an example grid search.

#figure(
    image("grid_search.png",  width: 90%), 
    caption: [ An example of the samples a grid search might use to find an optimal hyperparameter solution.]
) <grid_search>

=== Random Search

Random search is very similar to a grid search; however, instead of selecting grid points evenly spaced across the parameter space, it randomly selects points from the entirety of the space @hyperparameter_optimisation_review. It has similar advantages and disadvantages to grid search, and with infinite computing resources, both would converge on the ground truth value for the objective function. However, random search has some benefits over grid search that allow it to more efficiently search the parameter space with fewer evaluations. When performing a grid search, the separation of grid points is a user-defined parameter, which both introduces a free parameter and creates possible dimensional bias. A grid search will also search the same value for any given hyperparameter many times, as along the grid axis, it will appear many times, whereas a random search should rarely repeat samples on any hyperparameter. It should also be noted that some statistical uncertainty will be introduced, which would not be present in the case of a grid search and might limit the comparability of different approaches. Both the random and grid search techniques have the disadvantage that all samples are independently drawn, and unless the processes are recursed, no information from the performance results can influence the selection of new points. See @random_search.

#figure(
    image("random_search.png",  width: 90%), 
    caption: [ An example of the samples a random search might use to find an optimal hyperparameter solution.]
) <random_search>

=== Bayesian Optimisation

A Bayesian optimisation approach makes use of our initial beliefs, our priors, about the structure of the objective function @hyperparameter_optimisation_review. For example, you might expect the objective function to be continuous and that closer points in the parameter space might have similar performance. The objective function is estimated probabilistically across the parameter space. It is updated as more information is gathered by new samples, which can be tested either in batches or one at a time. The information obtained by these new samples is incorporated into the estimated objective function in an effort to move it closer to the ground truth objective function. 

The placement of samples is determined by a combination of the updated belief and a defined acquisition function, which determines the trade-off between exploration and exploitation. The acquisition function assigns each point in the parameter space a score based on its expected contribution to the optimisation goal, effectively directing the search process. A standard method for modeling the objective function in Bayesian optimisation is Gaussian Processes, but other techniques are available, such as Random Forests and Bayesian Neural Networks, among others. This optimisation technique is often employed when evaluating the objective function is expensive or time-consuming, as it aims to find the optimal solution with as few evaluations as possible. See @bayesian_descent_hp_optimisation.

#figure(
    image("bayesian_descent.png",  width: 90%), 
    caption: [ An example of the samples a Bayesian optimization might use to find an optimal hyperparameter solution. The descent method shown here has used a Gaussian Process to attempt to find the objective function maximum but has not done so particularly successfully. The method was not tuned to try and increase performance, as it was just for illustratory purposes.]
) <bayesian_descent_hp_optimisation>

=== Gradient-Based Optimisation

In some rare cases, it is possible to find optimal model hyperparameters using a similar method to the one we used to determine model parameters during model training @hyperparameter_optimisation_review. We can treat the hyperparameter space as a surface and perform gradient-descent (or in this case ascent, which follows the same principles but in reverse). Since gradient descent was already discussed in some detail in @gradient-descent-sec we will not repeat ourselves here. The advantage of gradient-based optimisation is that it can utilize extremely powerful gradient descent mechanisms that we have seen are potent optimisiers. The major disadvantage, however, is that for most hyperparameters, it is not possible to calculate the gradient. There are workarounds in some specific scenarios and much research has gone into making gradients available, but such work is still in early development and not applicable in many scenarios, thus we limit our discussion to this paragraph.

=== Population-Based Methods

The final category of hyperparameter optimization methods that we will discuss, and the one that we have chosen to employ in our search for more optimal classifiers, are population-based methods @hyperparameter_optimisation_review_2. These come in a variety of different subtypes, most prominent of which perhaps are evolution-based methods, such as genetic algorithms. Population-based methods are any methods that trial several solutions before iterating, or iterate several solutions in parallel, as opposed to trialing one solution, and then iterating the next solution on the results of the previous. Technically, since they trial a number of solutions before iteration, both random and grid searches could be considered population-based methods with only one step, although they are not usually included. Since we have chosen to adopt a method from this group, will will review some of the subtypes.

==== Genetic Algorithms

For our hyperparameter search, we have chosen to implement genetic algorithms, a population-based evolutionary method @genetic_algotrithm_review @hyperparameter_optimisation_review @hyperparameter_optimisation_review_2. Genetic algorithms are inspired by the principle of survival of the fittest found in nature within Darwinian evolution @genetic_algotrithm_review. They require the ability to list and freely manipulate the parameters we wish to optimize (in our case our hyperparameters). Continuing with the biological analogy these parameters are a given solution's genes, $g_i$ the full set of which is a solution genome $G_i$. We must also be able to test any genome and how well a solution generated with that genome satisfies our objective function. We must be able to condense these measurements into a single performance metric --- the fitness of that solution. Any optimization problem that fits these wide criteria can be attempted with genetic algorithms, meaning they are a flexible optimization solution. Our problem, the hyperparameter optimization of deep learning model, fits both criteria, thus, genetic algorithms are an applicable for the task. 

Initially, a number of genomes, $N$, are randomly generated within predefined parameter space limits @genetic_algotrithm_review. All possible gene combinations must produce a viable genome or a mechanism must be in place to return a fitness function of zero if a solution is attempted with an invalid genome. A solution (in our case, a model) is generated for each of the $N$ genomes. This set of solutions forms your population. Every member of the population is trialed (in our case, the model is trained) either sequentially or in parallel depending on your computational resources and the scope of the problem. In the basic genetic algorithm case, each trial within a generation is independent and cannot affect another member of the population until the next generation, and is validated (the model is validated) in order to produce a fitness function. This process of generating a set of genomes that defines a population of solutions, and then testing each member of the population to measure its effectiveness is known as a generation. Multiple generations will be iterated through, but each generation after the first is based on the fitnesses and the genomes of the previous generation rather than just being randomly generated as in the first generation. Genes and gene combinations that are found in highly-scoring members of the population are more likely to be selected for use in the next generation. After the algorithm has run for a number of generations, possibly determined by some cut-off metric, in theory, you should have produced a very highly-scoring population. You can then select the best-performing model from the entirety of your evolutionary history.

It is the selection process between generations that gives the genetic algorithm its optimising power @genetic_algotrithm_review, rather than grid or random methods, each generation uses information from the previous generation to guide the current generation's trials. There are multiple slightly different variations, we use one of the most common techniques, which is described in more detail in @dragonn-method. 

As mentioned genetic algorithms are very flexible, they can be applied to a wide variety of optimization problems @genetic_algotrithm_review. They can handle almost any objective function and operate in any kind of parameter space, including discrete, continuous, or mixed search spaces @genetic_algotrithm_review. They are also quite robust, unlike many optimization solutions which, sometimes rapidly, single out a small area of the parameter space for searching, genetic algorithms perform a more global search over the parameter space. Despite these advantages, they have the large disadvantage of requiring a large number of trials before converging on a high-performing solution, for this reason, they are less often used for hyperparameter optimization as each trial requires model training and validation @hyperparameter_optimisation_review_2.

==== Particle Swarm Optimisation

For completion, we will also discuss several other population-based optimization techniques. One of which is particle swarm optimization, which is inspired by the emergent behavior found in swarms of insects, flocks of birds, and schools of fish @hyperparameter_optimisation_review_2. Seemingly without coordination or central intelligence, large numbers of individually acting agents can arrive at a solution to a problem using information from their nearest neighbours @flocks.

In particle swarm optimisation, akin to genetic algorithms, an initial population is randomly generated and trialed. In this case, each member of the population is called a particle, forming the elements of a swarm. Rather than waiting for the end of each generation in order to update the parameters of each solution, each solution is given a parameter-space velocity which is periodically, or continuously updated by the performance of the other members of the population. Some variations aim to imitate real animal swarms more closely by limiting each particle's knowledge to certain regions, or to improve convergence rates by weighting some particles more highly than others @hyperparameter_optimisation_review_2.

Particle swarms can have much quicker convergence than genetic algorithms, due to the continual updates to their trajectory in parameter space. However, effective employment of particle swarms requires that your solutions can adjust their parameters quickly, which is not the case for many deep learning hyperparameters, most prominently structural hyperparameters which would often require retraining the model from scratch after only small changes. 

==== Differential Evolution

Like genetic algorithms, differential evolution methods are a form of evolutionary algorithm, but rather than generating a new population based on a selection of genes from the previous generation, differential evolution instead generates new parameters based on differentials between current solutions @hyperparameter_optimisation_review_2 @differential_evoloution. This means that genes in the current generation are not necessarily anything like genes in the previous generation --- parameters are treated in a vector-like manner rather than discreetly.

Differential evolution can work well for continuous parameter spaces, and shares many of the advantages of genetic algorithms as well as sometimes converging more quickly, however, it deals less well with discrete parameters than genetic algorithms and is less well studied, so understanding of their operation is not as well developed @differential_evoloution.

== Dragon Method <dragonn-method>

As our attempt to apply genetic algorithms to the problem of deep learning model optimisation in gravitational waves, we introduce Dragonn (Dynamic Ranking And Genetic Optimisation of Neural Networks). Originally a standalone software library developed in C, Dragonn was rewritten in Python utilizing other recent advances made in the GravyFlow pipeline. A previous version which was used to optimise the core MLy models was existent, but data from those early experiments was lost, so a decision was made to remake the experiments with the updated Dragonn tools. In the following subsection, we will justify our selection of genetic algorithms as the hyperparameter optimisation method of choice, explain in detail the operation of genetic algorithms, and discuss the choice of optimiser parameters selected for tests of Dragonn's optimization ability. 

=== Why Genetic Algorithms?

Genetic algorithms are an unusual choice for artificial neural network hyperparameter optimization and have fallen somewhat out of fashion in recent years, with Bayesian methods taking the limelight. Genetic algorithms typically require many trials before they converge on an acceptable solution, and although they are extremely flexible and adaptable methods, which are easy to implement and fairly straightforward to understand, the computational expense of individual trials of neural network architectures can often be prohibitively expensive to the application of genetic algorithms. Many of the hyperparameters of artificial neural networks are immutable without completely restarting training. While it is possible to adjust the training dataset and training hyperparameters such as the learning rate during model training, there are many hyperparameters related to the network architecture for which training would have to be completely restarted should they be altered, typically reinitializing the model's tunable parameters in the process. This means that for each trial during our optimization campaign, we will have to train a model from scratch, which can be a computationally expensive endeavour especially if the models are large. More computationally hungry layers, such as the attention-based layers that are discussed in future chapters, would require even more time and resources per trial, making genetic algorithms even more costly.

Unfortunately, most of this was not known at the initiation of the project. It should be noted, that at that time, hyperparameter optimization methods were less developed. As the project developed, however, there were new ideas for how genetic algorithms could be better adapted for their task. We can imagine some methods to alter model architectural hyperparameters without entirely resetting the tunable weights of the model in the process. For example, we could add an extra convolutional filter to a convolutional layer, randomly initializing only the new parameters, and keeping existing parameters the same, similarly, we could remove a convolutional kernel. It might also be possible to add and deduct entire layers from the model without completely resetting the tunable parameters every time. A method to reuse existing trained parameters was envisioned. Unfortunately, performing such surgery on models compiled using one of the major machine-learning libraries, in our case TensorFlow, is fairly difficult. So although many alternative methods were conceived, none progressed to the point where they were ready for testing.

With all that said, population-based methods are far from dead, and there are still some significant advantages over other methods. For extremely complex spaces, with many parameters to optimize, genetic algorithms can be the best solutions possible, though as noted they can take many trials to reach this optimum solution. It should also be noted, that although hyperparameter optimization can be very highly dimensional, it is usual, in artificial neural network design, for the number of dimensions that are important to model performance to be quite low, meaning that the search space is considerably lessened. There are big players in AI who use population-based methods similar to genetic algorithms for model optimization, including Google DeepMind @deepmind_population_based, so it is hoped that further development of this method could result in a highly adaptable population-based method for the optimization of neural networks for use in gravitational-wave research. Much of the software has already been developed, and although it would be a complex task, it would be a rewarding one.

We have some things in our favour: our input data is not particularly highly dimensional, and the models we are attempting to train, are simple Convolutional Neural Networks (CNNs) that are not especially memory or resource intensive, meaning that it should be possible for us to run a relatively large number of trials. The method and software developed for this research have also seen use in a gravitational wave detection pipeline, MLy @MLy, so it has already been useful to the scientific community. The developed genetic algorithm software included as part of GravyFlow makes it very easy to add new hyperparameters to the optimisation task, those parameters can be continuous, discrete, or boolean. The range of hyperparameters set up for optimisation with Dragonn is already extensive, as is demonstrated in @hyperparameter-seclection-sec.

There has been at least one attempt to use genetic algorithms for hyperparameter optimisation within gravitational wave data science in the past @ga_graviational_waves. Deighan _et al._ share an interest in developing a consistent method for generating hyperparameter solutions, and they use a similar approach to the method described here. They demonstrate that genetic algorithms can indeed generate models with high performance. The work of Deighan _et al._ optimizes a reasonable, but limited number of hyperparameters, predefining several structural elements of the network. We have allowed our optimiser considerably more freedom, although we note that this could also lead to an increased convergence time.

=== Selection of Mutable Hyperparameters <hyperparameter-seclection-sec>

Genetic Algorithms are optimisation methods that can be used to find a set of input parameters that maximise a given fitness function. Often, this fitness function measures the performance of a certain process. In our case the process being measured is the training and testing of a given set of hyperparameters --- the hyperparameters then, form the parameters that are optimization method will adjust to find a performant solution. The GravyFlow optimisation model allows us to optimize a wide range of hyperparameters, and whilst it was wished to perform a large optimisation run over all possible hyperparameters, we elected to use only a subset in order to improve convergence speeds due to time constraints.

Model hyperparameters can be split into three categories, with the last category divisible into two subcategories: *Dataset Hyperparameters*, *Training Hyperparameters*, and *Structural Hyperparameters.* 

- *Dataset hyperparameters* control the structure and composition of the training dataset, including its size, the number of each class of example within the dataset, and the properties of each example. In our case, the properties of the noise, the signals injected into that noise, and any additional obfuscations we wish to add to the data like the injection of simulated glitches. It's important that our method cannot also adjust the properties of the validation and testing dataset. It would be very easy for the genetic algorithm to find a solution wherein it makes the difference between classes in the validation set as easy as possible to identify, or make the validation dataset incredibly short, or perhaps remove all but one class of example. If we restrict our optimisation to the training dataset, however, this can be a good way to find optimal hyperparameters. The composition of the training dataset can often be a crucial part of optimising model performance. Unfortunately, we did not run the genetic algorithm on any dataset parameters, since we attempted to optimise for time. The set of possible genes for dataset hyperparameters is shown in @datset-hyperparameters.

#figure(
    table(
        columns: (auto, auto, auto, auto),
        [*Hyperparameters Name (gene)*], [*Type*], [*Optimised*], [*Range*],
        [Sample Rate (Hz)], [Integer], [No], [-],
        [Onsource Duration (s)], [Integer], [No], [-],
        [Offsource Duration (s)#super($plus$)], [Integer], [No], [-],
        [Total Num Examples], [Integer], [No], [-],
        [Percent Signal], [Float], [No], [-],
        [Percent Noise], [Float], [No], [-],
        [Percent Glitch], [Float], [No], [-],
        [Noise Type], [Discrete], [No], [-],
        [Whiten Noise? ($plus$)], [Boolean], [No], [-],
        [_For each feature type_], [], [], [],
        [SNR Min\*], [Float], [No], [-],
        [SNR Max\*], [Float], [No], [-],
        [SNR Mean\*], [Float], [No], [-],
        [SNR Median\*], [Float], [No], [-],
        [SNR Distribution (\*)], [Discrete], [No], [-],
    ),
    caption: [Possible Dataset Hyperparameters. These are parameters that alter the structure and composition of the dataset used to train or model. None of these parameters were selected for inclusion in our hyperparameter optimization test, in order to decrease convergence time. Parameters with a superscript symbol become active or inactive depending on the value of another parameter in which that symbol is contained within brackets. Range entries are left black for Hyperparameters not included in optimisation, as no ranges were selected for these values. ]
) <datset-hyperparameters>

- *Training hyperparameters* are parameters used by the gradient descent algorithm, which dictate the training procedure of the neural network. These include things like the learning rate, batch size, and optimization choice. As with the dataset hyperparameters, these are fairly easy to alter after training has begun without first resetting all of the model's tunable parameters, so could easily be incorporated into a more complex population-based method. None of these parameters were selected for optimization. The set of possible genes for dataset hyperparameters is shown in @training-hyperparaneters.

#figure(
    table(
        columns: (auto, auto, auto, auto),
        [*Hyperparameters Name (gene)*], [*Type*], [*Optimised*], [*Range*],
        [Batch Size], [Integer], [No], [-],
        [Learning Rate], [Float], [No], [-],
        [Choice of Optimiser(\*)], [Discrete], [No], [-],
        [Various Optimiser Parameters\*], [Discrete], [No], [-],
        [Num Training Epochs], [Float], [No], [-],
        [Patience], [Discrete], [No], [-],
        [Choice of Loss function], [Discrete], [No], []
    ),
    caption: [Possible Training hyperparameters. These are parameters that alter the training procedure of the model. None of these parameters were selected for inclusion in our hyperparameter optimization test, in order to decrease convergence time. Parameters with a superscript symbol become active or inactive depending on the value of another parameter in which that symbol is contained within brackets. There are different optimiser parameters that could also be optimized depending on your choice of optimiser, for example, values for momentum and decay. It is not typical to optimise your choice of loss function for most tasks, but some are possible with a range of loss functions, such as regression, which could benefit from optimisation of this parameter. Range entries are left black for Hyperparameters not included in optimisation, as no ranges were selected for these values.] 
) <training-hyperparaneters>

- *Architecture hyperparameters* are parameters that control the number and type of layers in a network. This is by far the most extensive category of hyperparameters since many of the layers that themselves are controlled by hyperparameters contain hyperparameters. For example, a layer in a network could be any of several types, dense, convolutional, or pooling. If convolutional were selected by the optimizer as the layer type of choice, then the optimizer must also select how many filters to give that layer, the size of those filters, and whether any dilation or stride is used. Each layer also comes with a selection of possible activation functions. This increases the number of hyperparameters considerably. In order to allow the optimizer maximal freedom, no restrictions on the order of layers in the network were imposed, any layer in a generated solution could be any of the possible layer types. Another independent hyperparameter selected how many of those layers would be used in the generation of the network in order to allow for various network depths. The output layer was fixed as a dense layer with fixed output size, to ensure compatibility with label dimensions. The set of possible genes for dataset hyperparameters is shown in @architecture-hyperparameters

#figure(
    table(
        columns: (auto, auto, auto, auto),
        [*Hyperparameters Name (gene)*], [*Type*], [*Optimised*], [*Range*],
        [Nummber of Hidden Layers], [Integer], [Yes], [0 to 10],
        [_One each for each active layer_], [], [], [],
        [Layer Type], [Discrete], [Yes], [Dense(\*, $plus$), Convolutional(\*, $times$), Pooling($diamond$), Dropout($square$)],
        [Activation Function\*], [Discrete], [Yes], [ReLU, ELU, Sigmoid, TanH, SeLU, GeLU, Swish, SoftMax],
        [Num Dense Neurons #super($plus$)], [Integerr], [Yes], [1 to 128 (all values)],
        [Num Filters #super($times$)], [Integer], [Yes], [1 to 128 (all values)],
        [Kernel Size #super($times$)], [Integer], [Yes], [1 to 128 (all values)],
        [Kernel Stride #super($times$)], [Integer], [Yes], [1 to 128 (all values)],
        [Kernel Dilation #super($times$)], [Integer], [Yes], [0 to 64 (all values)],
        [Pool Size #super($diamond$)], [Integer], [Yes], [1 to 32 (all values)],
        [Pool Stride #super($diamond$)], [Integer], [Yes], [1 to 32 (all values)],
        [Dropout Value #super($square$)], [Float], [Yes], [0 to 1 (all values)],
    ),
    caption: [Possible architecture hyperparameters. These are parameters that alter the architectural structure of the model, or the internal structure of a given layer. All these parameters were selected for optimisation. Parameters with a superscript symbol become active or inactive depending on the value of another parameter in which that symbol is contained within brackets. For each of the $N$ layers, where $N$ is the value of the number of hidden layers genome, a layer type gene determines the type of that layer, and other hyperparameters determine the internal structure of that layer. ]
) <architecture-hyperparameters>

These parameters are called genes ($g$), and each set of genes is called a genome *genomes* ($G$). $G = [g_1, g_i ... g_{x}]$, where $x$ is the number of input parameters. Each genome should map to a single fitness score ($F$) via the fitness function. 

=== Genetic Algorithms in Detail

Genetic algorithms operate under the following steps, note that this describes the procedure as performed in this paper, slight variations on the method are common:

+ *Generation:* First, an initial population of genomes, $P$ is generated. $P = [G_1, G_i, ... G_N]$, where $N$ is the number of genomes in the population. Each genome is randomised, with each gene limited within a search space defined by $g_(i"min")$ and $g_(i"max")$.
+ *Evaluation:* Next, each genome is evaluated by the fitness function to produce an initial fitness score. In our case, this means that each genome is used to construct a CNN model which is trained and tested. The result of each test is used to generate a fitness score for that genome.
+ *Selection:* These fitness scores are used to select which genomes will continue to the next generation. There are a few methods for doing this, however, since we do not expect to need any special functionality in this area we have used the most common selection function - "the Roulette Wheel" method. In this method, the fitness scores are normalised so that the sum of the scores is unity. Then the fitness scores are stacked into bins with each bin width determined by that genome's fitness score. $N$ random numbers between 0 and 1 are generated, and each genome is selected by the number of random numbers that fall into its bin. Any given genome can be selected multiple or zero times.
+ *Crossover and Mutation:* The genomes that have been selected are then acted upon by two genetic operators, crossover and mutation. Firstly, genomes are randomly paired into groups of two, then two new genomes are created by randomly selecting genes from each parent. A "mutation "is then performed on each of the new genomes with a certain mutation probability $M$. Mutation and Crossover create genomes that share elements of both parents but with enough differences to continue exploring the domain space.
+ *Termination:* If the desired number of generations has been reached the process ends and the highest-performing solution is returned. Else-wise the process loops back to step 2 and the newly created genomes are evaluated.

==== Choice of Fitness Function

There are multiple possible variants on the standard genetic algorithm model but for the most part, we have kept to the generic instantiation. It is a common choice to use the model loss as the fitness metric for optimisation, this makes sense in many ways, as the goal of training a model is to reduce its loss function, a better loss indicates a better model. However, the model loss function often fails to map exactly for our requirements to the model. The form of the loss function affects the model's training dramatically, so we cannot just use any function we wish as the loss function, and some things we are trying to optimise for might be too expensive to compute during every training iteration, or impossible to compute directly in this manner. We have chosen to use the area under a FAR-calibrated efficiency curve. Only values above an SNR of 8 were included in the sum, and the FAR chosen was 0.01 Hz, with the assumption that performance at this FAR would translate to performance at a lower FAR. A lower FAR was not directly used because it would be computationally expensive to compute for every trial. This objective function was chosen as it is representative of the results we look for to determine whether our model is performant or not. If those are the results we will be examining, we may as well attempt to optimise them directly.

==== Choice of Crossover Method

There are several potential choices for crossover methods, one-point crossover, k-point crossover, or uniform crossover. In one-point crossover we treat our two genomes, one from each parent, as long arrays, like two DNA strands, the crossover mechanism randomly cuts both strands in two and selects half from one strand, and the second half from the other genome, generating a new genome by splicing the old, this the simplest approach which in some cases can lead to faster convergence, but it can reduce the possibly for mixing genomes in interesting ways, reducing the total search space. K-point crossover is similar, but selects multiple places to cut, and splices the gene in a more complex manner, this can increase mixing possibilities but can decrease convergence, as the new genome is more likely to gain combinations of genes that perform poorly. The final possibility is uniform mixing, which effectively equates to cutting before and after every genome. Each genome in the new genome is randomly selected between parent a and parent b, this maximizes mixing but can increase convergence time. We selected to use uniform crossover in order to maximise the possible search space, although we were concerned about increasing convergence times, we wanted to ensure that we explored a wide area of the parameter space effectively.

=== Choice of Mutation Method

As well as crossover, mutation was also performed at the inception of every new genome. Mutation ensures that the population keeps exploring new areas of parameter space even as certain traits dominate the population, by introducing a small chance that a gene can randomly change value. Our method for performing mutation is dependent on whether the value of that gene is an integer, continuous, discrete, or boolean. For all cases, there is a 5% chance for mutation to occur in any given gene after crossover has taken place. For continuous and integer values, the value of the gene is mutated either negatively or positively by an amount drawn from a Gaussian distribution, in the case of integer parameters this is then rounded to the nearest parameter. For discrete and boolean values, a new value is drawn from the possible selection, with all values being equally likely -- this is different from the integer case as choices in the discrete category are not ordered.

=== Datasets

The GravyFlow data @gwflow_ref and training pipeline were used to generate the datasets used in each of the trial solutions. We are attempting to detect BBH IMRPhenomD signals generated with cuPhenom @cuphenom_ref and obfuscated by real LIGO interferometer noise drawn from the LIGO Livingston detector, Although GravyFlow lends itself well for use in hyperparameter optimization methods due to its rapid generation of datasets and lack of requirement for pre-generated datasets, we elected not to optimize dataset parameters in an attempt to decrease the time till model convergence.  Instead, we used identical dataset parameters to those used for the perceptron experiments, but we decreased the training patience to a single epoch, meaning if any epoch has a validation loss higher than the epoch previous, training halts. This was done in order to reduce the time taken for each trial. The parameters used for the training and dataset can be seen in @perceptron-training-parameters.

== Dragonn Results

The genetic algorithm work presented in this chapter has been in development for a long time, but this particular iteration only reached its full capabilities in recent months. What this has meant is that time pressure did not allow for a large number of generations to be run. Optimization was performed over four generations, which is very low for a genetic algorithm optimization run. Nonetheless, we can explore the results, and we have made some intriguing discoveries, even if they were somewhat accidental.

== Dragonn Training

First, we can examine the training histories of our models, and note the difference in performance between generations. @dragon_training_results displays the training results, demonstrating that most of the networks fail to achieve any classification ability. This is expected. As we have allowed complete genetic freedom for layer order and parameters, many nonsensical arrangements of layers are possible which will inhibit any possibility of classification performance. 

With disappointment, we note that even in the later generations no models reach accuracies above around 95%, this could, in part, be a result of our reduced training patience halting training early before extra performance can be extracted, although we note that even in cases where more epochs were reached, the accuracy seems to flatline. Setting the value of patience to one has other consequences, a great number of the somewhat models across the generations were stopped as they reached epoch two where their model loss dropped below the loss for epoch one. It is unknown exactly why this is the case since all models were trained on exactly the same training dataset generated with the same random see, it could be that the training data in that epoch is particularly unhelpful to the model in some way though a statistical fluke. The validation datasets are consistent across epochs, so there could not be a variation in validation difficulty causing this hurdle.

Even with the chaotic diagrams, it is easy to see that the number of performant models increases with each generation, so we have verified that our optimiser works --- we will examine average metrics later for verification of this. However this is not a particularly interesting result, it is known that genetic algorithms work. We do however have an interesting result that demonstrates the importance of a future, wide hyperparameter search.

performances close to gabbard small. 
#figure(
    grid(
        columns: 1,
        rows:    4,
        gutter: 1em,
        [ #image("accuracy_generation_1.png", width: 100%) ],
        [ #image("accuracy_generation_2.png", width: 100%) ],
        [ #image("accuracy_generation_2.png", width: 100%) ],
        [ #image("accuracy_generation_4.png", width: 100%) ],
    ),
    caption: [Dragonn model training histories from each of the four generations. All models were trained with identical training datasets and validated with epoch-consistent validation data. After each epoch, a new population was generated by applying the genetic algorithms mechanism to select perfomant genes in previous generations. In all generations many models lack any classification ability, this is anticipated because, because of the scope of the hyperparameter search, many of the models generated will be nonsensical, with extremely small data channels or near complete dropout layers. However, we also see that our population size was enough for a considerable number of performance models. With increasing generations, we see increasing numbers of performant models, demonstrating that our genetic optimiser is operating as intended.]
) <dragon_training_results>

Next, we can examine the average metrics from each epoch. In @dragon_averages we examine four metrics of interest; the average maximum model accuracy, that is, the average of all the highest accuracies models archived across their training run; the average lowest model validation loss, the average number of epochs a training run lasted for, and finally the average model fitness. The model fitness is the percentage of correctly classified validation examples with an optimal SNR greater than 8 when using a detection threshold calibrated to a far of 0.01. The metrics show us what we anticipated, increasing average performance in all metrics across generations. The average number of epochs increases as the number of performant models increases since performant models are more likely to reduce their validation loss over the previous epoch.

As expected an increase in model fitness correlates with an increase in accuracy and a decrease in model loss, suggesting that better-performing models when measured with uncalibrated FAR thresholds and loss functions, in general act as better-performing methods in low FAR regimes, although this is not always the case, as we will explore when we examine the best performing models of the generation.

#figure(
    image("averages.png", width: 100%),
    caption: [Dragonn average metrics from each of the four generations. The blue line is the average best model accuracy across its training run, The red line is the average model loss, the purple line is the average number of epochs in a model's training history, and the green line is the average model fitness. These results are mostly as all average metrics improve with increasing generation count, the drop in loss is particularly impressive, but this probably corresponds to the shedding of extremely poorly designed models after the first epoch. Accuracy is slowly improving, as the number of performant models increases, and with it the average number of epochs in a model's training history. Within increasing numbers of performant models comes increasing numbers of models that can perform better than their last epoch after further training.]
) <dragon_averages>

The result of an optimization algorithm is only as good as the highest-performing model. So we shall examine the population and extract the most performant models for inspection. Luckily our choice of objective function --- the percentage of validation examples over an optimal SNR of 8 that are correctly classified when calibrated to a FAR of 0.02 Hz, encapsulates most of what we desire out of our models, so we can use this to guide our search to the best models. 

We have extracted the top ten scoring models in @top_model_perfomances. Interestingly, and perhaps worryingly for the effectiveness of our optimization method, the top three models are all in the first generation. This tells us that although the average fitness was increasing along with other metrics of interest, that does not necessarily equate to generating more highly scoring models, which seems counterintuitive. However, examining the validation results of the top-scoring model more closely can lead us toward a reason for this discrepancy, and perhaps an interesting area of further investigation.

#figure(
    table(
        columns: (auto, auto, auto),
        [*Rank*], [*Generation*],  [*Fitness*],
        [1], [1],  [0.9423],
        [2], [1], [0.9337], 
        [3], [1], [0.9215],
        [4], [4], [0.888], 
        [5], [3], [0.887], 
        [6], [2], [0.870], 
        [7], [2], [0.868], 
        [8], [4], [0.860], 
        [9], [1], [0.841],
        [10], [4], [0.841],
    ),
    caption: [The top ten models in any of the populations throughout the genetic optimisation process, out of a total of 800 trial solutions, 200 at each epoch. Unexpectedly, the three top-scoring models when ranked by fitness, the very metric our optimization method is attempting to optimise are in the first generation. The first generation of a genetic optimisation search will alone act as a random search, so it is perhaps not unsurprising that it has some ability to find good solutions, however, we would expect better solutions to arise out of on average better-performing populations. This could perhaps be a result of our very low epoch count, or a statistical fluke. If it were the latter, however, it would seem very unlikely that the top three spots were taken by a first-generation model. The other option is that there was some asymmetry between the generations.]
) <top_model_perfomances>

In @top_model_perfomance we examine the efficiency plot used to generate the fitness scores for the highest-scoring model, a model from the first generation. These efficiency plots show extremely strong performance in the lower FAR regimes at medium SNRs, but this appears to come at the cost of some of the performance at the higher SNRs, which do not perform as well as the CNN models from the literature.


#figure(
    grid(
        columns: 1,
        rows:    3,
        gutter: 1em,
        [ #image("best_model_efficiency_0_1.PNG", width: 100%) ],
        [ #image("best_model_efficiency_0_01.PNG", width: 100%) ],
        [ #image("best_model_efficiency_0_001.PNG", width: 100%) ],
    ),
    caption: [Efficiency curves of the top performing model from the population of Dragonn trials. The curves maintain high accuracy at low FARs, though their performance a high SNRs above 10 is worse, never reaching a 100% accuracy, their performance at an SNR of 6 is considerably greater. It is hypothesized that this is due to an inoculation effect generated by the erroneous injection of WNB glitches into the dataset during the first generation.  _Top:_ Efficiency curve using a threshold calibrated to a FAR of 0.1 Hz. _Middle:_ Efficiency curve generated using a threshold calibrated to a FAR of 0.01 Hz. _Bottom:_ Efficiency curve generated using a threshold calibrated to a FAR of 0.001 Hz. ]
) <top_model_perfomance>

We hypothesize that this effect arises from a mistake made during the first generation. Initially, the plan had been to optimize the dataset parameters at the same time as the model parameters, and Dragonn was set up to allow the optimiser to adjust the percentage of training examples that contained CBC signals, this defaulted to 50%. However, it was also envisioned that the network could add its own synthetic glitches to the dataset, in order to act as counterexamples, this was also set to inject simulated WNBs into 50% of injections by default, including ones that also contained a CBC. It was not realised that this had been left in this state until partway through the first generation, where it was rectified, however, due to time pressure, the first trials were not repeated. Considering the three most performant results were all in the first 50 values, this oversight likely seems the cause.

The initial idea behind allowing the optimiser to inject simulated glitches was to allow it to act as an inoculant against particularly structured background noise, it would force the network to use the signal morphology because excess power could also come from glitches which would not correlate to a signal class. Due to the way the software was set up, these WNBs were also erroneously injected into the validation examples. In situations where a very high WNB is injected over the top of a signal it could obfuscate it even if that signal had quite a high SNR, this effect could be causing the reduction in ability at the higher SNRs. 

== Discussion

Our attempt to expand the range of the hyperparameter search was admirable but overambitious. The time taken to perform such an expansive search was underestimated, and time pressure led to mistakes in the final optimisation run, and an insufficient number of generations to gain any real insight into the optimisation power of genetic algorithms with this degree of optimisation freedom.  

Our mistakes did, however, lead us to an interesting discovery that could certainly warrant further investigation. There do not seem to be any existent investigations within the literature into such a method of using fake glitches to innoculate CBC detection models from structured background noise. The closest to such a method is perhaps the training procedure used to train the models used in the MLy pipeline @MLy. MLy is a coherence detection pipeline so relies on the machine learning models being able to detect coherence between different detectors rather than specific signal morphologies. In order to train the model to distinguish between glitches and real signals, it is trained with counterexamples consisting both of coincident and incoherent glitches across multiple detectors and single detector glitches.

Without a deeper investigation, it is difficult to know whether these glitches were indeed the source of improved performance. If this does turn out to be the case it is very exciting. We can perhaps remove the degradation at high SNRs by only injecting glitches into noise examples, it is unclear whether this would maintain the impressive performance at low SNRs and FARs, but it seems reasonable to think that it might. If this investigation has unveiled anything, it is that a wide search of the parameter space of both models and datasets could reveal unpredicted and useful results. 

== Deployment in MLy <deployment-in-mly>

Whilst this attempt to demonstrate genetic algorithms for optimizing CBC detection models has fallen short, they were used to generate models for the MLy pipeline, which consists of two models. One that is designed to detect coincidence @mly_coincidence, and a second model that is trained to detect coherence @mly_cohernece. Since these were both relatively unknown problems compared to the CBC search, not much was known about the ideal structure of artificial neural networks for these problems.

Optimizing models by hand can be time-consuming and generates many opportunities to miss interesting areas of the parameter search space. A previous version of the Dragonn optimiser was used to develop the models that are today in use by MLy @MLy.

#figure(
    image("mly_coincidence_diagram.png",  width: 75%), 
    caption: [MLy Coincidence Model developed with Dragonn @MLy. ]
) <mly_coincidence>

#figure(
    image("mly_coherence_diagram.png",  width: 100%), 
    caption: [MLy Coherence Model developed with Dragonn @MLy. ]
) <mly_cohernece>