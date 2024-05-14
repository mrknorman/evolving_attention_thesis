#set page(numbering: "1", number-align: center)

#set math.equation(numbering: it => {[3.#it]})
#import "../notation.typ": vectorn, uvectorn, dvectorn, udvectorn, matrixn
#counter(math.equation).update(0)

= Machine Learning <machine-learning-sec>

Machine learning techniques can be applied to almost any area of gravitational-wave data science; therefore, an exhaustive list would be difficult to compile and quickly outdated. However, here are some current areas of investigation: transient detection @george_huerta_cnn @gabbard_messenger_cnn @gebhard_conv_only_cnn and parameterisation @george_huerta_cnn @bbh_pe_1 @vitamin, including compact binary coalesces @george_huerta_cnn @gabbard_messenger_cnn @gebhard_conv_only_cnn, bursts @supernovae_cnn_1 @supernovae_cnn_2 @MLy @semi-supervised, and detector glitches @glitch_detection_1 @gravity_spy; continuous waveform detection @continious_1 @continious_2 @continious_3 and parameterisaction @continious_clustering; stochastic background detection and parameterisation @stocastic_1; detector noise characterisation @noise_characterisation and cleaning @deepclean; detector control and calibration @detector_control_1 @detector_control_2; and approximant generation @aproximant_generation_1. This thesis will focus on the application of machine learning to transients, including compact binary coalesces and burst events. To contextualise this research, this chapter will serve as a brief introduction to machine learning. 

Many ambiguous, sometimes contradictory definitions exist within machine learning and artificial intelligence. The definitions used throughout this thesis will be discussed here, attempting to use the most technically correct, or failing that, most commonly used definitions available.

*Artificial Intelligence* is perhaps the broadest of the terms associated with machine learning and perhaps also the vaguest. It has various, sometimes conflicting, definitions but is often defined as a property of human-designed intelligent agents --- systems that take, as input, information about the world and process that data, along with any internal state, to produce an output that maximises the chance of achieving a specific goal @ai_modern. This broad definition can be applied to an extensive range of artificial devices, from a toaster, which takes as an input the twist of a dial and tries to maximise its goal of applying heat for an amount of time relating to the position of the dial, to a chess engine with the all-consuming goal of checkmating its opponent. Most people would probably not consider a toaster artificially intelligent, and indeed, in the years since DeepBlue first defeated Garry Kasparov @deep_blue, many have come to consider chess engines in much the same light. This phenomenon is known as the 'A.I. effect', wherein a task is only considered something requiring intelligence until it has been successfully demonstrated by a machine @ai_effect. At that point, it is pushed out of the realm of intellectual endeavour and into the mundane, therefore preserving human supremacy over their cognitive dominion. I fear that with the rise of large language models, a few years is all that separates the act of writing a thesis such as this from the same relegation @scientific_writing. This transience can make artificial intelligence a tricky definition to use in a technical sense, so the term will, where possible, be avoided.

*Machine Learning* is somewhat easier to define. Depending on your definition of artificial intelligence, it could be considered either a subset of that field or merely at an intersection with it @machine_learning_intersection. It is loosely defined as the study of agents who can gain competency at a task without explicit human instruction @machine_learning. This is achieved through the use of specialised algorithms and statistical methods @machine_learning. Since, for the context of this thesis, it is probably more helpful to think of these agents as statistical techniques rather than actors that react to the world, the rest of this thesis will use the term *model* to refer to these agents, as they often model the relationship between a specific distribution of input data and a specific distribution of output data.

Machine learning can be subdivided in multiple ways, but one of the most common distinctions separates it into three basic paradigms: supervised learning, unsupervised learning, and reinforcement learning @pattern_matching_and_machine_learning. 

*Supervised Learning* refers to any machine learning task wherein the model attempts to match its outputs with preexisting values labelled by humans or another technique @pattern_matching_and_machine_learning. Training a model through supervised learning requires datasets of labelled training data from which the model learns. After which, if successful, the model should be able to approximate the desired output given new unseen input data.

*Unsupervised learning*, on the other hand, does not provide the model with any preexisting values to attempt to match its outputs with @pattern_matching_and_machine_learning. This can include techniques that use the input data as the desired output data, such as in autoencoders @autoencoder_unsupervised, or techniques that attempt to divine patterns within the dataset previously unknown to the model and, often, the model user. For example, clustering tasks look for similar latent features between groups of training examples @unsupervised_clustering. 

*Semi-supervised learning* lies, perhaps unsurprisingly, in the lacuna between supervised and unsupervised learning @semi_supervised. Whilst training under this paradigm, some of the training data is labelled and some unlabeled. This can be used when the labels are too computationally expensive to compute for the entirety of the training dataset or when some of the labels are intractable by other techniques or simply unknown.

*Reinforcement Learning* is a paradigm based on slightly different principles. Instead of using extensive data sets to train an agent, reinforcement learning utilises algorithms that try to maximise themselves against an externally defined reward function @pattern_matching_and_machine_learning. While training a model using reinforcement learning, the model can take actions that affect the state of the environment in which the model is allowed to act. The state of its environment will then be mapped to a score; this score is used to update the model. Through an iterative process, the model is updated to improve its ability to maximise the score of its environment.

Reinforcement learning is commonly used in scenarios where huge training datasets are not available, and the model is primarily designed to interact with an environment (virtual or real), such as training a robot to walk @robot_walk or training a virtual car to drive around a virtual track @robot_drive. Though this has proved a powerful technique for many machine learning applications, it has not been investigated in this thesis and thus will not be discussed in detail.

== The Artificial Neural Network

The Artificial Neural Network is a machine-learning technique that has seen rapid innovation, development, and adoption over the last decade @ann_history. They've shown the ability to solve many long-standing problems in artificial intelligence, including image, audio, and text classification, captioning, and generation, @image_classification @audio_classification @text_classification @image_captioning @audio_captioning @text_summarisation @image_generation @audio_generation @text_generation, as well as producing game-playing algorithms that have attained superhuman performance in previously human-superior games like Go @alpha_go. They can teach themselves the rules from scratch in a matter of hours @alpha_zero --- compared to the many years of development required for previous game-playing engines. They can compete in complex, highly-dimensional computer games like Starcraft 2 @starcraft and League of Legends @league and they have achieved large-scale adoption across many industrial sectors, managing power grids @power_grid_management, performing quality control @quality_control, and paving the way, albeit slowly, toward fully autonomous self-driving cars @self_driving_cars. Artificial neural networks have also been applied to many scientific problems, such as AlphaFold @alpha_fold, a method that, to some extent, solved the protein folding problem.

With their rampant and rapid success across many domains previously thought intractable or at least many decades away from a solution, it is easy to ascribe to artificial neural networks more than what they are, but it is also easy to underestimate their potential to solve previously unthinkable problems. Artificial neural networks are little more than simple statistical structures compiled into complex architectures, which allow them to perform intricate tasks @artifical_neurons @deep_learning_review @perceptron_and_neural_network_chapter.

They are loosely inspired by the structures of biological neurons inside animal brains @biological_inspiration @perceptron_and_neural_network_chapter. Although they indeed show a greater likeness to the workings of biological systems than most computers, this analogy should not be taken too literally. Biological brains are far more complex than current artificial neural networks, and there is much about them we do not yet understand. There may still be something missing from state-of-the-art models that prevents them from the full range of computation available to a biological brain @biological_differences. Having said that, there are still ample further developments that can be made with artificial neural networks, even considering their possible limits. We do not yet seem close to unlocking their full potential @future_improvements.

There is no universally agreed-upon definition of *deep learning*, but one of the most widely accepted definitions is that it must have a Credit Assignment Path (CAP) depth greater than two. This means that there must be more than two data transformations from input to output @deep_learning_3. This equates to a dense artificial neural network with more than two layers, or in other words, one or more hidden layers. This enables *representation learning*, where a network can learn to identify hierarchical features in the model @deep_learning_2. It is proven that models with a CAP of two can act as universal function approximators @universal_aproximators, so adding more layers beyond this act only improves convergence on a parameter solution by reducing training difficulty. In practice, almost all contemporary applications of artificial neural networks are more than two layers deep. The hierarchical relationship between A.I. and machine learning is illustrated by @ai_relationships.

#figure(
  image("ai_relationships.png", width: 80%),
  caption: [The loose hierarchical relationship between different umbrella terms used in artificial intelligence @deep_learning_review.],
) <ai_relationships>

There are a plethora of different types and arrangements of artificial neural networks, often known as architectures @deep_learning_review. The following sections will introduce the main concepts surrounding artificial neural networks.

=== The Artificial Neuron <artificial_neuron_sec>

As mentioned previously, artificial neural networks are loosely inspired by biological neural networks @biological_inspiration @perceptron_and_neural_network_chapter, and as one might expect, their base unit is analogous to the biological base unit, the neuron @artifical_neurons. Artificial neurons form the basic building block of all artificial neural networks, though their form and design can vary between architectures @artifical_neurons.

The artificial neuron takes a number, $N$, of continuous numerical inputs $ vectorn(x) = [x_1, ..., x_i, ..., x_N]$ and outputs a single numerical output $A(vectorn(x))$ @artifical_neurons. Each neuron has a number of tunable parameters associated with it, $vectorn(theta)$. A single neuron has many weight values $vectorn(w) = [w_1, ..., w_i, ..., w_N] $ and a single bias value $b$. Suppose these parameters, $vectorn(theta)$, are selected correctly. In that case, the artificial neuron can, in some simple cases, act as a binary classifier that can correctly sort input vectors, $vectorn(x)$, drawn from a limited distribution into two classes @perceptron_and_neural_network_chapter @artifical_neurons. This kind of single-neuron classifier is often known as a perceptron, the original name given to this kind of classifier @perceptron. 

#figure(
  image("artificial_neuron.png", width: 60%),
  caption: [_Upper_: The Artificial Neuron. This figure illustrates the operations that compose the archetypical artificial neuron, where $vectorn(x)$ is the input vector, $f$ is the activation function, $vectorn(w)$ is the weights vector, and b is the neuron bias. An artificial neuron takes an input vector, $vectorn(x)$, and performs some useful calculations (hopefully). Both the weights vector, $vectorn(w)$, and bias value, $b$, comprise the neuron's adjustable parameters, $vectorn(theta)$, that must be tuned for the neuron to perform any useful operations @artifical_neurons. _Note_: During computation, the bias, $b$, is not normally added in a separate operation; instead, it is added as an extra $x_0$ term included in the same calculation as the summation of the product of the weights, $vectorn(w)$, and input values, $vectorn(x)$. _Lower_: An abstraction of the more complicated interior structure of the artificial neuron. Abstraction is common and necessary when describing artificial neural networks as networks are often comprised of thousands if not millions of artificial neurons.],
) <artificial_neuron_diagram>

As can be seen in @artificial_neuron_diagram, the standard artificial neuron is comprised of several consecutive mathematical operations. First, the input vector, $vectorn(x)$, is multiplied by the weights vector, $vectorn(w)$, and then the result of this operation is summed along with the bias value, $b$ @artifical_neurons @perceptron_and_neural_network_chapter. Finally, the output is then fed into an activation function $f$; see @activation_functions_sec. This sequence of operations is given by:

$ op("A")(vectorn(x)) = f(sum_(i=1)^N w_i x_i + b) = f(vectorn(x) dot vectorn(w) + b), $ <artificial_neuron>

where N is the number of elements in the input vector. In the case of the single-layer perceptron, the output of the neuron, $op("A")(vectorn(x))$, is equivalent to the output of the perceptron, $accent(y, hat)$, where our desired ground-truth output value is $y$. Since each element of the weights vector, $vectorn(w)$, is multiplied by each component of the input vector, $vectorn(x)$, the weights can be thought of as representing the significance of their corresponding input value, $x_i$, to the desired output value, $y$. The bias, $b$, acts as a linear shift to the activation function, and tuning this value can make it more or less difficult for the neuron to activate. Having well-tuned parameters, $vectorn(theta)$, is crucial for the performance of the artificial neuron. 

The purpose of the activation function, $f$, is to coerce the distribution of the output value, $op("A")(vectorn(x))$, into a particular shape @activation_functions_ref. The intricacies of why you might want to do this will not become apparent until the model training is understood. Therefore a more detailed discussion of activation functions follows in @activation_functions_sec.

=== Training Artificial Neurons <training_arificial_neurons>

Now that the structure of the artificial neuron has been described, the question becomes, how does one go about ascertaining useful values for the neuron's tunable parameters, $vectorn(theta)$, namely the weights vector, $vectorn(w) $, and the bias, $b$. It would, in theory, be possible to approach this problem by manually discovering values for each parameter, $theta_i$, using human-guided trial and error. Whilst this would be unwise, we can use this thought experiment to arrive at the automated solution to the problem. This section will describe the step-by-step process of training an artificial neuron, or in this case, multiple neurons, and for each step, illustrate how the manual approach can be automated, displaying a Python @python function demonstrating this. @an_training_import shows the required library imports to run all subsequent code listings in this section. An iPython notebook containing the described code can be found here: https://tinyurl.com/4m4n3m7n.

#show figure: set block(breakable: true) 
#figure(
```py
# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
```,
caption : [_Python @python ._ Required imports to run subsequent code listings in this section. NumPy @numpy is used for its fast numerical CPU operations. TensorFlow @tensorflow is used for fast numerical GPU operations, machine learning functionality, and loading the Modified National Institute of Standards and Technology (MNIST) dataset @mnist. Bokeh @bokeh is used to plot figures.]
) <an_training_import>

We will attempt to train an ensemble of ten artificial neurons to classify the Modified National Institute of Standards and Technology (MNIST) example dataset @mnist correctly. The MNIST dataset consists of 70,000 black-and-white images of handwritten numbers with a resolution of 28 by 28. Pixel values range from 0 for black pixels to 255 for white pixels, with the integer values representing 253 shades of grey. 10,000 images are reserved for testing, with the remaining 60,000 used for training. See @mnist_examples for examples of the images contained within the dataset.

Though slightly confusing, this ensemble of multiple neurons is often known as a single-layer perceptron @perceptron_and_neural_network_chapter, as it consists of many neurons acting (almost) independently in a single layer; see @single_layer_perceptron. The only collaboration between neurons is the normalisation that is applied to each neuron by the softmax activation function @softmax, which ensures the produced output vector sums to one and can act as a probability; see @softmax-sec. Because we are moving from a single neuron with one bias value $b$, and a vector of weights values $vectorn(w)$, to multiple neurons, the bias value becomes a vector $vectorn(b)$, and the weights vector becomes a matrix $matrixn(W)$.

$ matrixn(W) = mat(
  w_(1,1), ..., w_(1,j), ..., w_(1, P);
  dots.v, dots.down, dots.v, dots.up, dots.v;
  w_(i, 1), ..., w_(i, j), ..., w_(i, P);
  dots.v, dots.up, dots.v, dots.down, dots.v;
  w_(N, 1), ..., w_(N, j), ..., w_(N, P);
), $

where $N$ is the number of neurons in the layer, and $P$ is the number of weights per neuron, typically determined by the number of neurons in the previous layer or the number of elements in the input vector if the layer is the input layer.

#figure(
  image("mnist_examples.png", width: 80%),
  caption: [Example MNIST data @mnist. A single example of each of the ten classes within the MNIST example dataset. As can be seen, the classes range from zero to nine inclusive. Each example consists of a grid of 28 by 28 pixels containing one float value between 0.0 and 1.0. In the above image, values near one are represented as nearly white, and values near 0.0 as black. When ingested by our single-layer perception, they will be flattened into a 1D vector; see @flatten-sec.],
) <mnist_examples>

#figure(
  grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("single_layer_perceptron.png",   width: 100%) ],
        [ #align(center)[#image("single_layer_perceptron_abstract.png", width: 100%)] ],
  ),
  caption: [Various representations of a Single-Layer Perceptron or Single-Layer Artificial Neural Network. _Upper:_ Diagram illustrating the structure and operation of a single-layer perceptron. In the example shown, a handwritten zero is fed into the single-layer perceptron. The 2D image is first flattened into a 1D vector, see @flatten-sec; then, the entire vector is fed into each neuron. If the training process has worked correctly, each neuron will have learned to identify one of the possible classes, in this case, digits. As can be seen from the output values, $uvectorn(y) = [accent(y, hat)_0, ..., accent(y, hat)_9]$, which are taken from a real trained model, this model can correctly identify this input as a zero with high confidence. _Middle:_ An abridged version of the upper diagram demonstrating the operation of feeding a handwritten one into the perceptron. This shows how future network diagrams will be abstracted for simplicity and that the perceptron outputs a different, correct value when it ingests a one rather than a zero.
  _Lower:_ A further abstraction of the network. This type of abstraction will be used commonly throughout this thesis when dealing with networks consisting of multiple layers. A dense layer, wherein all neurons are attached to all previous neurons, will be shown as a filled black rectangle, and the icon next to it represents that the activation function applied is a softmax activation function @softmax_ref; see @softmax-sec.]
) <single_layer_perceptron>

*Step 1: Dataset Acquisition:* When we train a machine learning model, we are attempting to model the relationship between an input and an output distribution. In some ways, the model can be considered a compressed version of the matched input and output distributions. After training, when you feed in a single data point from the input distribution, the model will, hopefully, be able to map that input value to the correct value in the output distribution. This makes the training data a fundamental part of the training process @training_dataset_importance. Whether naively attempting a manual solution or optimising through more efficient means, we must acquire a suitable training dataset.

In many cases, the input distribution will be very large or even continuous, so an exhaustive training dataset covering every possible value in the distribution will be either technically or literally impossible. For this reason, we have to ascertain or generate a training dataset that will appropriately sample the entire input distribution. There are many preexisting example training sets; as stated, we will use the MNIST dataset @mnist for illustrative purposes. 

Automating the process of acquiring a dataset is simple. TensorFlow @tensorflow has built-in functions to allow us to acquire the MNIST dataset @mnist easily. @an_data_aquisition below shows us how this process can be performed. The listed function also prepares the data for ingestion by the ensemble of artificial neurons. *One hot encoding* changes a single numerical class label, i.e. $0, 1, ..., 9$ into a Boolean vector where each index of the vector represents a different class; for example, $0$ becomes $[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]$, whereas $1$ becomes $[0, 1, 0, 0,0, 0, 0, 0, 0, 0]$ @one_hot_encoding. This is because each neuron will learn to distinguish a single class by returning a float value closer to $0.0$ if the input falls outside its learned distribution or closer to $1.0$ if the input falls within its learned distribution. Therefore to perform the vector operations necessary for training, one hot encoding must be performed @one_hot_encoding.

#figure(
```py
# Step 1: Load and prepare the MNIST dataset.
def load_and_prepare_data():
    
    # This data is already split into train and test datasets.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the images to feed into the neural network.
    
    x_train, x_test = x_train.reshape(-1, 784)/255.0, x_test.reshape(-1, 784)/255.0

    # Convert labels to one-hot vectors. This is necessary as our output layer will have 10 neurons, 
    # one for each digit from 0 to 9.
    y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)
    
    return x_train, y_train, x_test, y_test
```,
caption : [_Python @python ._ Function to load and prepare the MNIST dataset @mnist. The MNIST dataset @mnist consists of many examples of handwritten Arabic numerals from one to nine. The images, `x`, are reshaped, and the labels, `y`, are one-hotted @one_hot_encoding.]
) <an_data_aquisition>

*Step 2: Parameter Initialization:* For an artificial neuron to produce a result when it consumes an input vector, all parameters, $vectorn(theta)$, must be initialised to some value. One could imagine choosing these initial parameters, $vectorn(theta)_bold(0)$, in a few distinct ways. Perhaps most intuitively, you could decide on the parameters based on some prior knowledge about the dataset, aiming to get as close as possible to the optimal tunings in order to minimise the number of steps required during training. However, this option is impossible if the human tuner lacks such intuition or if the input size is too large for any human to form such an intuition. That leaves choosing a uniform value for all parameters or randomly initialising all parameters. 

In any automated process, a uniform initialisation is a bad choice. If one sets all initial parameters, $vectorn(theta)_bold(0)$, to the same value, this creates symmetry @weight_initlisation. Suppose we subsequently try to use a mathematical method to adjust these parameters. In that case, the method will have no way to choose one parameter over another, meaning all parameters will be tuned identically. We will need the parameters to be adjusted independently in order to model complex relationships. For this reason, we will initiate the weights matrix, $matrixn(W)$, randomly by sampling values from a normal distribution. This choice of random distribution will not be discussed here, but note that there is an open area of research hoping to speed up and/or improve the training process by selecting more optimal distributions for parameter initialisation @weight_initlisation. The bias values, $vectorn(b)_bold(0)$, will be initialised to zero. Since there is only one bias value per neuron, we don't have to worry about creating asymmetry, as that is provided automatically by values passed through the neuron's weights.

 @an_parameter_definition demonstrates the initialisation of two variable tensors to hold the weights and biases of our artificial neurons. Because there are ten classes of numbers in the training dataset, we will initialise ten artificial neurons --- one to recognise each class of digit. There will be a single bias value for each neuron. Hence there are $C$ bias elements in the bias tensor, `biases`, where $C = op("num_classes") = 10$, and the input size is $N = 28 times 28 = 784$, so there are $N times C = 784 times 10 = 7840$ elements in our weights tensor, now a matrix, $matrixn(W) =$`weights`, arranged in the shape `[784, 10]`. This means the total number of tunable parameters in our set of ten neurons is $7840 + 10 = 7850$.

#figure(
```py
# Step 2: Define the model
# We are using a simple single-layer perceptron model
# This is essentially a single fully-connected layer

def define_model():
    # Define weights and biases. We initialise the weights with a random normal
    # distribution.
    # There are 784 input neurons (one for each pixel in the 28x28 image) and ten output 
    # neurons. We initialise biases to zero.

    weights = tf.Variable(tf.random.normal([784, 10]), name="weights")
    biases = tf.Variable(tf.zeros([10]), name="biases")
    return weights, biases
```,
caption : [_Python @python ._ Function to initialise TensorFlow @tensorflow tensors to store the artificial neuron's parameters, $vectorn(theta)$. In the case of MNIST @mnist digit recognition, ten neurons are being trained, so we have ten bias values, $vectorn(b)$, and the input images are of dimension $28 times 28 = 784$. Therefore, our weights matrix, $matrixn(W)$, is shaped `[784, 10]`.]
) <an_parameter_definition>

*Step 3: Define the model's action:* To perform any optimisation method, there must be a way to test the model. Thus we must define the action of the model, $M(vectorn(x))$. We have already shown what form this must take in @artificial_neuron_sec and @artificial_neuron. This is very easily defined by a Python @python function, as seen in @an_computation_definition.

#figure(
```py
# Step 3: Define the model's computations:
def model(x, W, b):
    return tf.nn.softmax(tf.matmul(x, W) + b)
```,
caption : [_Python @python ._ Function to perform the computation of artificial neurons in our single-layer perceptron. Since TensorFlow @tensorflow is natively vectorised, this function will calculate the output of all our tensors simultaneously. This function performs the same operation described in @artificial_neuron, with a softmax function as the activation function, $f$. Softmax activation functions are described in @softmax-sec.]
) <an_computation_definition>

*Step 4: Define the loss function*: Now that we have set up a procedure to run the model with a set of randomised parameters, $vectorn(theta)$, we must define a measure of success so that we can see how well the model is performing whilst we perform our parameter tuning operation. If we have no performance metric, then we have no indication of how to tune the model to improve its performance. To do this, we define a loss function, $L$, a function which takes in some information about the state of the model after it has ingested data, usually including the model's output, and returns a numerical output value: the loss of the model with a given set of parameters, $L(M_vectorn(theta) (vectorn(x)), vectorn(y) )$, where $vectorn(x)$, is a particular instance, or batch, of input vectors, and $vectorn(y)$ is, in the case of classification, the data label @deep_learning_review. Note that in unsupervised learning, the loss function does not ingest a label, $vectorn(y)$, as the data is not labelled @deep_learning_review. 

By convention, a high loss value indicates that the model performance is worse than that which would be indicated by a lower loss value @deep_learning_review. Our optimisation process, therefore, should attempt to minimise the average of this loss value across all potential input vectors.

There are many possible metrics for measuring the performance of a model, a large number of which can be used as the loss function @deep_learning_review. The loss function is an important aspect of the training process, which can alter the efficiency of the training significantly. They can be highly specialised to particular scenarios, to the point where using an inappropriate loss function can completely remove any possibility of training @deep_learning_review. A more detailed description of loss functions is available in @loss_functions_sec.

For this model, we elect to use the categorical cross-entropy loss function @deep_learning_review as described in @loss_functions_sec. An implementation of that loss function is shown by @loss_function_definition.

#figure(
```py
# Step 4: Define the loss function
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=[1]))
```,
caption : [_Python @python ._ Function to compute the loss of the model. The loss function utilised in this case is categorical cross-entropy loss, a loss function commonly used for multi-class, single-label datasets. A more detailed description of the function of this loss function can be found in @cross_entropy_sec.]
) <loss_function_definition>

*Step 5: Train the model*: Finally, after we have assembled all the pieces, we can start to tune the parameters, $vectorn(theta)$, so that our perceptron can output useful values when fed input. As we have previously stated, we will initialise our weight parameters, $matrixn(W)$, randomly; this means that no matter what images we feed into the untrained model, we will get non-sensical classification values with no correlation to the ground truth labels unless by extremely unlikely fluke. Using some process, we want to move the model toward successful categorisation.

If we again move back to our analogy of attempting to perform this operation manually, what we might imagine is that we would start by feeding it an image from our training dataset. We could then examine the model's output and see which parameters we would need to tune in order to move our network, for that particular image, toward the correct answer. We could achieve this by determining how much each parameter moves the current model's output, $uvectorn(y)$, toward or away from the ground truth value, $vectorn(y)$, and then adjusting each parameter accordingly. 

If we tuned the parameters by a large amount, then the model could easily become overtuned to a particular image, so we might instead choose to move it a little bit toward the correct input value and then repeat this process over hundreds, if not thousands, of examples, moving the network slowly toward a useful configuration.

Gradient descent is an algorithmic implementation of this thought experiment @deep_learning_review. In its most simple case, the loss that is given by the loss function, $L(M_vectorn(theta) (vectorn(x)), vectorn(y))$, measures the distance between the model output, $uvectorn(y)$, and the ground truth, $vectorn(y)$. Since the model is largely defined by its parameters, $vectorn(theta)$, the loss function can be thought of as a function that takes in an input vector, $vectorn(x)$, the model parameters $vectorn(theta)$, and the ground truth label, $vectorn(y)$. So the output of the loss function for a particular input vector and set of parameters becomes

$ L(M_vectorn(theta) (vectorn(x)), vectorn(y)) = L(M(vectorn(theta), vectorn(x)), vectorn(y)) = L_M (vectorn(theta), vectorn(x), vectorn(y)), $ <loss_func_eqatuion>

where L is the model-architecture-agnostic loss function, $L_M$ is the loss function for a particular model architecture, $M_vectorn(theta)$, is a model with a fixed set of parameters, $vectorn(theta)$, $M$ is a model with a set of parameters as a functional input, $vectorn(x)$, is a particular input vector to the model, and, $vectorn(y)$, is the label vector corresponding to the model input vector.

The gradient of the model is defined as the vector of partial derivatives of the model's loss function with respect to its parameters @gradient_descent_matrix. If $L_(M vectorn(x) vectorn(y)) (vectorn(theta))$ is the loss function with a fixed model architecture, input vector, and ground-truth label, then the gradient of the model is

$ vectorn(gradient) bold(L) eq.triple vectorn(gradient) bold(L_(M vectorn(x) vectorn(y)) (vectorn(theta))) = [frac(partial L_(M vectorn(x) vectorn(y)), partial theta_1), ..., frac(partial L_(M vectorn(x) vectorn(y)), partial theta_i), ..., frac(partial L_(M vectorn(x) vectorn(y)), partial theta_N)] $ <gradient_equation>

where N is the total number of tunable parameters. 

@gradient_equation describes a vector, $vectorn(gradient) bold(L)$. Each element of the vector, $frac(partial L_(M vectorn(x) vectorn(y)), partial theta_i)$, is a gradient that describes the effect of changing the value of the corresponding parameter, $theta_i$, on the model loss. If the gradient is positive, then increasing the value of the parameter will increase the value of the loss, whereas if it's negative, increasing the value of that parameter will decrease the model loss. The magnitude of the gradient is proportional to the magnitude of that parameter's effect on the loss @gradient_descent_matrix.

Since we want to reduce the model loss, we want to move down the gradient. Therefore, for each parameter, we subtract an amount proportional to the calculated gradient @gradient_descent_matrix @deep_learning_review.

#figure(
  image("gradient.png", width: 100%),
  caption: [An illustration of gradient descent, where $vectorn(gradient) bold(L_(M vectorn(x) vectorn(y)) (vectorn(theta)))$ is the loss at a fixed model architecture, $M$, input vector $vectorn(x)$, and data label $vectorn(y)$. This simplified example of the shape of a 1D parameter space shows how the gradient of the loss function with respect to the model parameters can be used to move toward the minimum of the loss function. The shape of the loss function in this example is given by $ L_(M vectorn(x) vectorn(y)) (vectorn(theta)) = theta^2$. In almost all cases, the parameter space will be much more complex than the one depicted in both dimensionality and shape complexity. Usually, the shape of the loss function will be an N-dimensional surface, where N is the number of parameters, $vectorn(theta)$, in the model, but the principle is still the same. For a 2D example of a gradient space; see @gradient_descent_examples. This plot can be recreated with the code found here: https://tinyurl.com/3ufb5my3.]
) <gradient_example>

We need to be able to control the magnitude of the parameter adjustment because the gradient is only measured for the current parameter values, $vectorn(theta)$. Therefore we are unsure of the shape of the loss function. It's possible for the tuning process to overshoot the loss function minimum. In order to apply this control, we introduce a constant coefficient to scale the gradient, known as the learning rate, $eta$ @gradient_descent_matrix @deep_learning_review.

Therefore, if we want to find the new adjusted parameters after one optimisation step, we can use

$ vectorn(theta)_(t+1) = vectorn(theta)_t - eta vectorn(gradient) bold(L_( M vectorn(x)_bold(t) vectorn(y)_t ) (vectorn(theta)_t)) $ <gradient_decent_step>

where t is the step index, we can see this process in a Python @python form in @train_step_definition. In this function, the gradients are captured using the tf.GradientTape scope, which automatically captures the gradients of all "watched" tensors within its scope. This automatic differentiation utilises a process called back-propagation @gradient_descent_matrix @deep_learning_review, which will be discussed in more detail in @backpropagate-sec.

#figure(
```py
# Step 5: Define the training step
 @tf.function
def train_step(x, y, W, b, η):
    with tf.GradientTape() as tape:
        y_pred = model(x, W, b)
        current_loss = compute_loss(y, y_pred)
    gradients = tape.gradient(current_loss, [W, b])
    W.assign_sub(η * gradients[0]) # update weights
    b.assign_sub(η * gradients[1]) # update biases
    return current_loss
```,
caption : [_Python @python ._ Function to execute a single training step. This function runs an example, ``` x ``` $ = vectorn(x)_bold(t)$, through the model (usually multiple examples at once as explained in @gradient-descent-sec) and computes the loss, ``` loss ``` $= L_( M vectorn(x)_bold(t) vectorn(y)_bold(t)) (vectorn(theta)_bold(t))$ of the output of that model, ``` y_pred ```$= uvectorn(y)_bold(t)$ compared with the ground truth label of that example, ``` y ```$= vectorn(y)_bold(t)$. The gradients, ``` gradients``` $= vectorn(gradient) L_( M vectorn(x)_bold(t) vectorn(y)_bold(t) ) (vectorn(theta)_bold(t))$, are automatically computed for each parameter by ``` tf.GradientTape()```, which produces a list of gradients for the weights, ``` w``` $= matrixn(W)$, and biases, ``` b``` $= vectorn(b)$, which are then used multiplied by the learning rate ``` η``` $= eta$ and used to update the parameters, $vectorn(theta)$, for the next training step; see @gradient_decent_step.]
) <train_step_definition>

If we repeat this process over T steps, where T is the number of training examples in our dataset, then the model will hopefully begin to gain aptitude at the classification task. The process of tuning the model parameters once with all examples in the training dataset is called a training epoch @perceptron_and_neural_network_chapter. Oftentimes, if our training dataset is not large enough, we can improve the model performance by running for multiple epochs, hence training the model with the same examples multiple times. Between epochs, the training dataset is usually shuffled in order to explore new areas of parameter space and avoid repeating exactly the same pathway @perceptron_and_neural_network_chapter.

Pulling all the functions we have defined together; we can now implement our main training loop, @train_loop_definition.

#figure(
```py
def train_model(epochs, batch_size, η, x_train, y_train):
    # Define model
    W, b = define_model()

    # Store loss and accuracy for each epoch
    loss_per_epoch = []
    accuracy_per_epoch = []

    # Training loop
    for epoch in range(epochs):
        i = 0
        while i < len(x_train):
            start = i
            end = i + batch_size
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]
            current_loss = strategy.run(train_step, args=(x_batch, y_batch, W, b, η))
            i += batch_size

        # Compute loss and accuracy for each epoch
        y_pred = strategy.run(compute_model, args=(x_test, W, b))
        loss_per_epoch.append(current_loss)
        accuracy_per_epoch.append(compute_accuracy(y_test, y_pred))
        print(f'Epoch {epoch+1} completed')
        
    return loss_per_epoch, accuracy_per_epoch, W, b
```,
caption : [_Python @python ._ Function to execute multiple training steps across multiple epochs. This function runs the function defined in @train_step_definition for each example in the training_dataset, ``` x_train```, and repeats this process for each requested epoch, ``` num_epochs```, updating the model parameters each time. It returns the model parameters, ``` W, b```, and some metrics measuring the model's performance; see @gradient_decent_step.]
) <train_loop_definition>

=== Testing the Model <mnist-test-sec>

Once we have trained our model using the aforementioned procedure, we can evaluate its performance. Often the first step toward this is to look at the model's performance at each step during training; see @perceptron_history.

The model training progresses quickly at first but soon reaches a point of diminishing returns at about 85 per cent accuracy. Although we may be able to squeeze out a little more performance by running the training for more epochs, this can lead to overfitting, where a model becomes tailored too specifically to its training dataset and cannot generalise well, or at all, to other points in the training distribution @perceptron_and_neural_network_chapter. In most cases, we will want our model to classify new unseen data drawn from a similar distribution as the training dataset but not overlapping with any existing points, so we try to avoid this.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("perceptron_history.png",   width: 100%) ],
        [ #image("unsucessfull_clasifications.png", width: 100%) ],
    ),
    caption: [_Upper:_ The performance of the single layer perceptron model described in @training_arificial_neurons over 15 epochs, where one epoch consists of training the model on all training examples in the MNIST dataset of handwritten Arabic numerals @mnist. The model loss is defined as the categorical cross-entropy of the model's output vector, $uvectorn(y)$ and the ground-truth label, $vectorn(y)$, whereas the accuracy metric is defined as the number of examples in the test dataset that are correctly classified, where a correct classification is any output with 50 per cent or more probability in the correct class. _Lower_: Two examples of less successful classifications. The left example would still be measured as a successful classification by our accuracy metric, whereas the right example would be marked as an unsuccessful classification. ] 
) <perceptron_history>
 
We can also investigate what parameters the neurons have actually learned over this process; see @perceptron_parameters. It is often very difficult to come to much of a conclusion about the true inner workings of artificial neural networks, especially dense layers, which are the most general but also the most non-intuitive @interpetability @deep_learning_review. Network interpretability is a large and ongoing area of machine learning research for many obvious reasons. Being able to see why a model has given you the answer that it has can massively boost confidence in that answer @interpetability @deep_learning_review. However, this thesis will not focus heavily on interpretability, as that could be a whole other thesis on its own.

#figure(
  image("perceptron_parameters.png", width: 100%),
  caption: [Learned model parameters. Each artificial neuron in our single-layer perception is represented by a labelled parameter map shaped into the same dimensions as the input images. These maps show the learned weight values that correspond to each pixel of the input images. Very little structure can be made out by the human eye. Perhaps in the weight maps for the zero-classifier neuron, we can see an area toward the centre of the map that is negatively weighted. This might be expected as there are rarely high-value pixels at the centre of the circular zero. A similar but opposite effect might also be attributed to the one-classifier, where the centre of the image often contains high-value pixels. In general, unless you squint very hard, it is difficult to make out patterns in the parameters. This "black-box" effect means that after even one more layer is added to the network, it becomes very difficult to determine the action of dense layer neurons intuitively.]
) <perceptron_parameters>

Whilst it is difficult to make specific claims on how artificial neural networks are doing what they are doing, we can often speculate on general methods of operation. In the case of the single-layer perceptron, like the one we have built here, the only kinds of operations that can be learned are linear ones. The only path each neuron has available to it is to learn which pixels are often highly valued in its class of digit and which pixels are very rarely highly valued in its class which are more likely to be highly valued in another class. Then it can adjust the bias value so that the neuron only activates when a certain criterion is met. If we were distinguishing between ones and zeros, for example, which have, in general, very different highlighted pixels, then this might be enough for a high degree of classification efficiency. However, there are a multitude of digits which can share many common pixel values, which makes this problem more difficult.

In order to solve the problem with more accuracy, we must add a non-linear element to the computation and the ability for the model neurons to work collaboratively on the problem @deep_learning_review @activation_functions_ref. This allows the model to extract more complex "features" from the input vector. We, therefore, introduce the concept of multi-layered neural networks and deep learning.

=== Neurons Together Strong <together_strong>

As we have seen, a single layer of artificial neurons can be trained to perform a small amount of computation, which is often enough for many simple problems. There are, however, a great many problems which require more complex solutions. In order to do this, we can add what is known as "hidden layers" to our network @deep_learning_review @perceptron_and_neural_network_chapter They are called hidden layers because the exact computation that is performed by these additional layers is much more difficult to divine than in output layers, layers that directly output solution vectors, as we have seen in our single-layer perceptron @perceptron_and_neural_network_chapter.

For simplicity of design, artificial neural networks are usually organised into layers of neurons, which are usually ordered, and interactions are usually limited to between adjacent layers in the network @perceptron_and_neural_network_chapter. Layer one will usually only pass information to layer two, and layer two will receive information from layer one and pass information to layer three if there are three layers; see @multi-layer-perceptron. This is not always the case, and there are exceptions to all the rules mentioned in this paragraph, including skip connections @skip_connections @res_net_intro, recurrent neural networks @rnn_review @deep_learning_review, and Boltzmann machines @boltzman_machine.

Artificial neural network layers come in many varieties, the most simple of which are feed-forward dense (or sometimes linear) layers @deep_learning_review. Dense layers consist of $N$ neurons, where every neuron takes as an input vector the output of every neuron on the previous layer unless the dense layer is acting as the input layer, in which case every neuron takes in as input every element of the input vector @perceptron_and_neural_network_chapter. If the dense layer is acting as the output layer, as was the case for our single-layer perceptron where one layer was both the input and output layer, then $N$ must equal the required size of our output vector, $vectorn(y)$. In the case of a classification problem, this is equal to the number of classes, $C$. In hidden layers, the number of neurons, $N$, can be any number and is, therefore, a customisable non-trainable parameter known as a hyper-parameter that must be chosen before network training by some other method; see @hyperparameters-section.

As can be imagined, finding the gradient for networks with one or more hidden layers is a more complex problem than for a single layer. Backpropagation allows us to do this @perceptron_and_neural_network_chapter @deep_learning_review @gradient_descent_matrix and, in fact, is the tool that unlocked the true power of artificial neural networks; see @backpropagate-sec.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("multi_layer_network.png",   width: 100%) ],
        [ #align(center)[#image("multi_layer_abstracted.png", width: 100%)] ],
    ),
    caption: [_Upper:_ Diagram of a multi-layer network with one output layer and one hidden layer. The non-linear computation introduced by the ReLU activation function applied to the hidden layer allows this network to solve considerably more complex problems than the previously described single-layer perceptron model. _See @relu-sec _.  As can be seen, by the displayed output, which again is taken from a real instance of a trained model, this network has no problem classifying the previously difficult image of a five. _Lower:_ An abstraction of the same model.] 
) <multi-layer-perceptron>

This model performs considerably better than the previous model; see @multi-layer-perceptron; and seems to be complex enough to more or less solve the problem; see @multi-layer-perceptron_history. We might, in fact, even try reducing the number of neurons in the hidden layer. It is often beneficial to find the simplest possible network, by the number of parameters, that is able to achieve the desired computation, as more complex networks are more computationally expensive and time-consuming to train, require more training data, have an increased inference time (inference meaning to run the model on new unseen data), and crucially are more prone to overfitting to the training dataset @deep_learning_review. This model reaches a high accuracy within just a few epochs, and unless we are very concerned about false alarm rates, then there is no need to add extra complexity to our model. The notebook used to train this model can be found here: https://tinyurl.com/yrbwuw4e.

#figure(
    image("multi_layer_network_history.png", width: 100%),
    caption: [The performance of the multi-layer perceptron model described in @together_strong over 15 epochs. As can be seen in comparison to @perceptron_history, the training is both faster and with a better final result.] 
) <multi-layer-perceptron_history>

The addition of hidden layers to our network architectures introduces the possibility of all kinds of structural variations. For more complex problems than the recognition of simple handwritten digits, we have many tools in our arsenal to increase performance. The first thing you might do is try the addition of more than one hidden layer; in fact, there is no inherent theoretical limit to the number of hidden layers that can be added to a network. Of course, at some point, you would run into computational limits, and although the gradient can be calculated, there are problems when attempting to run gradient descent algorithms on very deep networks that are designed without careful consideration. Gradients that vanish over many layers can lead network training to become an almost impossible task @vanishing_gradients @deep_learning_review.

These problems with increasing complexity lead researchers to explore types of layers beyond the dense layer. Although the dense layer alone can be thought of as a universal function approximator, there exists no perfect training algorithm to find the ideal set of parameters to achieve every possible function, and this statement is technically only true for all possible arbitrarily complex functions as the number of layers approaches infinity. For this reason, different layer designs and network architectures can create easier environments for training @deep_learning_review, saving computational resources and allowing feasible routes to massively increase network ability. Most often, these non-dense layers are designed with some insight into the structure of the input vectors. An example of this would be the convolutional neural network, see @cnn-sec, which uses the spatial information of the input, as well as the notion that there will be transform-invariant features within the image, to create layers that can perform a similar or better job than dense layers with far fewer parameters @vanishing_gradients. 

One could also experiment by moving away from the paradigm of feed-forward networks, although this can increase solution complexity significantly. Within feed-forward neural networks, neuron connections only ever move toward neurons that have not yet causally influenced the emitting neurons. Within recurrent networks, however, signal paths can loop, taking either previous inferences as inputs or looping within the calculation itself @reccurant_neural_networks @deep_learning_review. This can allow the network memory of previous inferences, something feed-forward networks do not possess.

#figure(
    image("ann_rnn.png", width: 100%),
    caption: [_ Left: _ The generalised dense feed-forward artificial neural network. Where $T$ is the number of hidden layers in your network, $H$ is the number of neurons at that layer, $N$, is the number of elements in the input vector, $vectorn(x)$, and $O$ is the number of elements in the output vector $uvectorn(y)$. As can be seen in the diagram, the number of hidden layers in your network is unconstrained, as is the number of neurons in each of those layers, which should be noted does not have to be the same. This is opposed to the output layer, which must have the same number of neurons as is expected by your loss function. _ Right _ A very simple illustration of a recurrent neural network. This network illustrates the retroactive data flow that is possible in a recurrent neural network. In this example, the output of the network from one inference operation is added to the input of the next inference operation. It should be noted that this is a very naive implementation of a recurrent neural network. In actuality, the networks usually have a much more complex structure, such as LSTMs (Long Short Term Memory) networks.] 
) <rnn>

There will be a more detailed discussion of many of these different network layers and architectures further on in the thesis; the following few sections will explore the concepts outlined in the last two sections in more detail.

=== Activation Functions <activation_functions_sec>

There are many different activation functions; as with most things in machine learning, they are an active area of research @deep_learning_review @activation_functions_ref. As such, this small section will only give a brief introduction plus a few examples. Since the activation function normally acts on the weighted sum of the inputs plus the bias, in this section, we will define $z = sum vectorn(x) dot vectorn(w) + b$ to avoid confusion with the raw input values previously denoted $x$. Depending on the network structure, $z$ could also be generated with an artibtraty function. We will define the vector of all $z$ values in a single network layer of $N$ neurons, i.e. $vectorn(z) = [z_1, ..., z_i, ..., z_N]$.

As noted in @artificial_neuron_sec, the activation function aims to coerce an artificial neuron's output into a particular shape @deep_learning_review @activation_functions_ref @perceptron_and_neural_network_chapter. This has several purposes. Firstly, it can act as a thresholding function, which along with a specific value of bias, $b$, can activate or deactivate the neuron depending on the weighted sum of the input vector, $z$. The activation function also limits the output values to a specific range, ensuring that values within the network do not grow without bounds along favoured pathways and destabilise the network. These values can be considered in some way analogous to the maximum firing rate of a biological neuron. Without activation functions, instability can cause values to explode to infinity or vanish to zero. Finally, activation functions provide a non-linear component to the neuron. Without non-linear activation functions, neuron output, hence network outputs, could only be linear combinations of the input values and so would need to be, in general, much more complex to solve non-trivial problems. 

There are some limits to the type of function we can use within a neural network, primarily since we must be able to flow gradients through the function during backpropagation; the function must be differentiable at all points @activation_functions_ref. For example, if we tried to use a step function as an activation function, the derivative would be 0 at all points, except for at the step where it would be undefined. This would make backpropagating through this function very difficult, as it would fail to update the weights and bias of its corresponding neuron. In other non-continuously differentiable functions, like the ReLU function, we can use a trick to avoid the undefined derivative by defining the value of the derivative at that point, $z = 0$ in this case, to 0 or 1.

As well as the distinction between linear and non-linear activation functions, a few further distinctions can be made. Outside of the linear function, we can split activation functions into three types: ridge @ridge_functions, radial basis @pattern_matching_and_machine_learning, and fold @softmax_ref. 

*Ridge* functions are standard activation functions that change an input's shape based on directionality around a specific point or ridge @ridge_functions. The most common example is the ReLU function @relu_intro @activation_functions_ref and its variants described below. Ridge functions are computationally efficient and introduce non-linearity without requiring exponentiation or other computationally expensive operations.

*Radial basis* functions @pattern_matching_and_machine_learning @radial_basis_function, on the other hand, are less commonly used. They are symmetric around a specific point rather than just directional. Their value, therefore, depends entirely on the magnitude of the distance to this point rather than in ridge functions where the sign is also vital. Radial basis functions can create complex surfaces which can localise to a specific region, which can be helpful if you believe your data structure to be localised in such a manner. A typical example of a radial basis function would be the Gaussian function, which can localise a neuron's activation to a particular region. However, they can be computationally expensive and lead to overfitting due to their ability to form complex surfaces.  

*Fold* functions are complex activation functions that aggregate over multiple neuron $z$ values, such as mean or max functions in pooling layers, or even over the entirety of $vectorn(z)$, such as in softmax layers described below @softmax_ref. Calculating these can be computationally expensive, so they are used in moderation.

==== Linear <linear-sec>

The most straightforward activation function is the linear activation, represented simply by @linear below. The linear activation function will not change the shape of the data and is crucial for many applications where this is a desired feature:

$ op("linear")(z) = k z. $ <linear>

Evidently, in the case where $k = 1$, this is equivalent to not applying any activation function and thus, all the previously stated problems resulting from no activation function will apply. The derivative of the linear activation function is always a constant irrespective of the input values, so it is straightforward to compute. This simplicity brings a significant drawback when dealing with complex data. If it is the only activation function used, the entire network, regardless of complexity or number of layers, will behave as a single-layer model because of the lack of non-linearity between layers. As we have seen, single-layer perceptrons are insufficient for many tasks we wish to tackle with artificial neural networks.

One of the primary uses of linear activation functions is as the output layer of regression problems, where the output is expected to be a continuous float value not constrained within a particular distribution @deep_learning_review. The drawbacks are alleviated if the rest of the network before the output layer involves non-linear activation, leaving the output layer to combine inputs into a final output value linearly. They can also sometimes be used in straightforward networks where non-linearity is not required and computational efficiency is highly prized. Therefore, while the linear activation function has its uses, it is not commonly applied in hidden layers of deep learning models, wherein non-linear layers, such as ReLU and its variants, are more valuable; see @activation_functions for a graphical depiction.

==== Logistic <sigmoid-sec>

The logistic activation function @sigmoid_ref is a ridge function defined

$ f(z) = frac(L, 1 + e^(-k (z - z_0))), $ <logistic>

where $z_{0}$ represents the z-coordinate of the function's midpoint, $L$ signifies the maximum value that the function can reach, often referred to as the function's supremum and $k$ is a parameter that controls the steepness of the function's curve, determining the logistic growth rate. The particular case where $L = 1$, $k = 1$, and $x_0 = 0$ is known as the sigmoid function

$ sigma(z) = frac(1, 1 + e^(-k z)). $ <sigmoid>

See @activation_functions for a graphical depiction.

Since this smoothly limits the output of the neuron to between 0 and 1, it is often used on output neurons in a network designed for classification since, in this case, the ground truth vector would consist of entirely Boolean values, meaning an activation function that tends to 0 and 1 in the extreme is very useful @sigmoid_ref. The sigmoid activation function is used in multi-class, multi-label classification problems, where each class variable is independent, and an example can be in multiple classes and single-class single-label problems, where there is only one output neuron. Since each output is calculated independently, it is unsuitable for cases where an example can be in only one class and there are multiple classes; in that case, a Softmax layer, as described in @softmax-sec, is more appropriate.

The sigmoid function's derivative is at maximum at the midpoint $z = 0$ and falls off as $z$ moves in either direction $z arrow infinity or z arrow -infinity$; this is a suitable environment for backpropagation as the derivative is always defined and never 0. There are, however, some limitations since the gradient, although never quite 0, can become very small, leading to the "vanishing gradients" problem wherein the model's parameter updates can become negligible, and hence learning is very slow. Secondly, the sigmoid function is not centred at 0. This can lead to zig-zagging during gradient descent, also slowing down convergence. Finally, the sigmoid function involves the computation of exponentials, which can be computationally expensive, especially for large-scale networks.

Despite these limitations, the sigmoid function is widely used, particularly in the output layer, for multi-class, multi-label classification problems. However, for hidden layers, modern practices prefer other functions like ReLU or its variants to mitigate some of the issues related to the sigmoid function.

==== ReLU (Rectified Linear Unit) <relu-sec>

One of the most common activation functions used very widely in neural network hidden layers is the ReLU (Rectified Linear Unit) function @sigmoid_ref @activation_functions_ref, defined by

$ op("ReLU")(z) = cases(
  z "if" z > 0,
  0 "if" z <= 0
). $ <relu>

ReLU is another example of a ridge function. This function is 0 for $z <= 0$, and $z$ for $z > 0$, meaning it is equivalent to the linear function above 0. It is a very simple function but still provides the neuron with the ability to threshold values and adds a non-linear component to the neuron.

Because of its simplicity, ReLU is very computationally efficient compared to other activation functions that require expensive operations such as exponentiation or division, an essential factor when deciding on activation functions to use, especially in very large networks @deep_learning_2. The derivative is also very simple, either 1 above $z$ or 0 below $z$; hence it lacks the possibility of becoming very small. This means that the use of ReLU functions can be efficient for training. Having a large section of the domain with a derivative of 0 does, however, also lead to problems. During the training process, some neurons can "die", only able to emit 0s, and since the gradient is also 0, they can become stuck in this state, unable to reactivate @dying_neurons. Evidently, this can reduce the capacity of the network for practical computation since these dead neurons can no longer contribute valuable operations.

To ameliorate some of the downsides, there are a plethora of possible ReLU variants, most of which have a non-zero gradient below $z = 0$. These include but are not limited to Leaky ReLU (LReLU) @leaky_relu, Randomized Leaky ReLU (RReLU) @randomised_leaky_relu, Parametric ReLU (PReLU) @parameteric_relu, Exponential Linear Unit (ELU) @parameteric_relu, and Scaled Exponential Linear Unit (SELU) @parameteric_relu. The first three variants, LReLU, RReLU, and PReLU, are defined by

$ op("LeakyReLU")(z) = cases(
  z "if" z > 0,
  alpha z "if" z <= 0
), $ <lrelu>

where $alpha$ depends on the variant in question; in standard LeakyReLU, $alpha$ is a small, predefined value such as 0.05, meaning the slope is much shallower before 0 than after it; this prevents dying neurons whilst still allowing the function to threshold the input value. In the case of Randomised Leaky ReLU, $alpha$ is randomly sampled from a specified distribution during training but fixed during model inference @randomised_leaky_relu. This solves the dying neuron problem and adds robustness to the training process. Finally, in Parametric ReLU, $alpha$ is treated as a trainable parameter that the model can adjust during backpropagation, allowing it to hopefully self-optimise to a good value @parameteric_relu.

ELU and SELU are also both based on a similar definition 

$ op("SELU")(z) = cases(
  s z "if" z > 0,
  s alpha (exp(z) - 1) "if" z <= 0
). $ <selu>

For any $alpha$ value if $s = 1$, the equation defines ELU. ELU has all the death-defying properties of the previously mentioned ReLU variants whilst also introducing differentiability at $z = 0$, meaning that the redefinition trick is not required. Unlike other ReLU variants, it saturates as $z -> inf$, increasing robustness to errors. These extra benefits come at the cost of the computational simplicity of the previous ReLU variants, as the calculation of exponentials is a significant computational expense. 

If $alpha = 1.673263... and s = 1.05070...$, the equation defines SELU, a self-normalising activation function. These very specific values of $alpha$ and $s$ are designed to work in conjunction with LeCun initialization, a method that initializes neuron parameters with values drawn from distributions with mean zero and variance $1/N$, where $N$ is the number of neurons in that layer. These values of $alpha$ and $s$ massage the neurons toward outputs with a distribution centred on zero and with a variance of one. Which can help smooth the training process by avoiding vanishing or exploding gradients.

In practice, ReLU and its variants are the most commonly used activation functions in the hidden layers of deep neural networks due to their efficiency and performance. See @activation_functions for a graphical depiction.

==== Softmax <softmax-sec>

Softmax is a more complicated fold distribution and is of interest due to its use in multi-class, single-label classification problems @pattern_matching_and_machine_learning @softmax_ref. It is an extension of the Sigmoid function described above in @sigmoid-sec, which aims to convert a vector of continuous unconstrained output values, in our case $vectorn(z)$, into a vector representing probabilities, with outputs limited between 0 and 1 and a vector sum equal to exactly 1. It does this by finding the exponential of each $z$ value, then normalising by the sum of the exponential of all elements in $vectorn(z)$

$ op("softmax")(vectorn(z))_i = frac(e^(z_i), sum_(j=1)^N e^(z_j)), $ <softmax>

where N is the number of elements in $vectorn(z)$, equivalent to the number of neurons in the layer and the number of classes in the dataset, and $i$ is the index of the neuron/class whose output value is calculated. See @activation_functions for a graphical depiction.

The softmax function represents a way of mapping the non-normalized output of the network to a probability distribution over predicted output classes, making it invaluable for multi-class, single-label classification problems. It is also differentiable so that it can be used in gradient-decent methods.

Softmax can be computationally expensive, particularly in the case of a large number of classes, as each output classification requires the use of multiple expensive operations such as exponentiation and division; it can also suffer from numerical instability when the scores in the input vector are very large or small which may result in numerical overflow or underflow problems. This is not typically too much of an issue as it is usually only used in the output layer of a network.

The Softmax function remains the standard choice for multi-class classification problems due to its ability to provide a probabilistic interpretation of the outputs, handle multiple classes, and its differentiability. 

#figure(
  image("activation_functions.png", width: 100%),
  caption: [Four of the most common activation functions. _Upper Left:_ A linear activation function. In this case, the slope, k, is 1, meaning that the shape of the output is unchanged vs the input. _Upper Right:_ Sigmoid activation function, a special case of the logistic activation function, which limits the output value between 0 and 1. _Lower Left:_ ReLU (Rectified Linear Unit) activation function and its variants, an easy way to provide non-linearity to multi-layer networks. _Lower Right:_ SoftMax activation function. In the case of multi-neuron outputs, when using softmax, the output of each neuron depends on the value of each other neuron. For this reason, the simplest non-trivial case, where the length of the output vector, $N$, is 2, has been chosen, and the outputs are represented on a 3D plot. This figure can be recreated with the notebook found at: https://tinyurl.com/muppechr.],
) <activation_functions>

=== Loss Functions <loss_functions_sec>

The loss function (sometimes cost or objective function) is an important part of the model training process @deep_learning_review. The purpose of the loss function is to act as a measure of the effectiveness of the model when acting on a particular batch of examples. In doing this, the loss function gives us a metric to evaluate the performance of the model, compare it against other models, and act as a guide during the training process. In specific cases, it can also act to regularise the model to prevent overfitting or to balance multiple objectives. 

In supervised learning, this loss function is some measure of the distance between the model's output and the ground truth labels of the examples fed into the model. These are one of the more common types of loss functions, but it should be noted that as long as it is differentiable, a great many terms can be included as part of the loss function, and indeed some of the more complex architectures have complex loss functions.

In unsupervised learning, Autoencoders @autoencoder_unsupervised are a formulation of a regression problem where the model input is equal to the model output, and thus, they follow the same principles as the typical regression problem, only the difference between their output, $uvectorn(y)$, and their input, $vectorn(x)$ in the loss, rather than an external label, $vectorn(y)$. Clustering, on the other hand, attempts to split a data distribution into groups by minimising the distance between elements in a given group while maintaining some balance with the number of groups generated --- there are a variety of different ways to do this depending on your desired outcome.

==== Binary Cross Entropy <binary_cross_entropy_sec>

The Binary Cross Entropy loss is used primarily for binary classification problems wherein each class label is independent @binary_cross_entropy. This can be the case either for single-class single-label tasks (binary classification tasks) or multi-class multi-label tasks. It is defined by @binary_cross_entropy_eq.

$ L(vectorn(y), uvectorn(y)) = - sum_(i=1)^N y_i log(accent(y, hat)_i) - (1 - y_i) log(1 - accent(y, hat)_i) $ <binary_cross_entropy_eq>

where $L(vectorn(y), uvectorn(y))$ is the loss function applied to the model output and ground truth vectors, $N$, is the number of elements in the output vector, $y_i$ is the i#super[th] element of the ground truth vector, and $accent(y, hat)_i$ is the i#super[th] element of the ground truth vector.

In the single-class single-label case where N = 1, @binary_cross_entropy_eq becomes

$ L(y, accent(y, hat)) = - y log(accent(y, hat)) - (1 - y) log(1 - accent(y, hat)). $ <binary_cross_entropy_eq_2>

Some confusion can arise in the case of binary classification problems @binary_cross_entropy, wherein the examples can either be in a class or not in that class since this is the same as the situation where there are two distinct classes. As such, these problems can be treated in two ways, either with a single output neuron and an output vector, $vectorn(y)$ of length one, (an output value i.e. $vectorn(y) = y$), where a high value indicates inclusion in the one class and a low-value exclusion, or with two output neurons where each neuron represents a "class", one being inside the class and the other being outside the class.

In the first case, we would use a sigmoid activation function and a binary cross-entropy loss, and in the second case, you would use a softmax activation function and categorical cross-entropy loss. These produce very similar outcomes, with the first method being slightly more straightforward, giving a directly interpretable output and reducing the number of parameters, whereas the second case, whilst increasing the model parameter count, can sometimes be more numerically stable.

==== Categorical Cross Entropy <cross_entropy_sec>

Categorical Cross Entropy loss is very similar to binary cross-entropy loss but is used primarily in multi-class single-label problems, such as the problem we presented in the MNIST @mnist classification task @deep_learning_review. It is a highly effective loss function, and it is often much easier to classify data into one class using this method than it would be to find multiple labels in a multi-class multi-label problem. So this kind of task is often a desirable framing of your problem. The loss is given by 

$ L(vectorn(y), uvectorn(y)) = - sum_(i=1)^N y_i log(accent(y, hat)_i), $ <categoricorical_cross_entropy_eq>

where $L(vectorn(y), uvectorn(y)) $ is the loss function applied to the model output and ground truth vectors, N is the number of elements in the output vector, $y_i$ is the i#super[th] element of the ground truth vector, and $accent(y, hat)_i$ is the i#super[th] element of the ground truth vector.

Both binary cross entropy and categorical cross entropy are loss functions that attempt the measure the difference between probability distributions. In the case of binary cross-entropy, it treats each output element as a separate probability distribution, whereas for categorical cross-entropy, the entire output vector is treated as one probability distribution. 

They are derived from the concept of entropy in information theory, which quantifies the expected amount of information from a source. Lower information states will have numbers that are closer to one or zero --- in that way minimising the function forces the output to values of one or zero, i.e., toward definite yes/no classifications.

==== Mean Square Error  <mse_sec>

For regression tasks, wherein the output vectors are not limited to boolean values, we must have more flexible activation and loss functions @mae_and_mae. In these cases, we still want to compare our desired output to our actual output, but we don't want to encourage the output to values near zero and one. There are a number of options to achieve this goal, the choice of which will depend on the specifics of your problem.

One option is mean square error loss, the sum of the squares of the error, $vectorn(y) - uvectorn(y)$, normalised by the number of elements in the output vector @mae_and_mae. It is defined by

$ L_op("MSE") (vectorn(y), uvectorn(y))  = 1/N sum_(i=1)^N (y_i - accent(y, hat)_i)^2, $ <MSE_eq>

where $L_op("MSE") (vectorn(y), uvectorn(y)) $ is the loss function applied to the model output and ground truth vectors, $N$ is the number of elements in the output vector, $y_i$ is the i#super[th] element of the ground truth vector, and $accent(y, hat)_i$ is the i#super[th] element of the ground truth vector.

Mean square error is a good choice for regression problems; it is fully differentiable, unlike mean absolute error; however, unlike mean absolute error, it heavily emphasises outliers which can be beneficial or detrimental depending on your scenario.

==== Mean Absolute Error <mae_sec>

The mean absolute error can be used in the same problems that the mean square error is used for @mae_and_mae. Again it is normalised by the total sum of the output vector. It is given by

$ L_op("MAE") (vectorn(y), uvectorn(y)) = 1/N sum_(i=1)^N |y_i - accent(y, hat)_i|, $ <mae_eq>

where $L_op("MSE") (vectorn(y), uvectorn(y)) $ is the loss function applied to the model output and ground truth vectors, N is the number of elements in the output vector, $y_i$ is the i#super[th] element of the ground truth vector, and $accent(y, hat)_i$ is the i#super[th] element of the ground truth vector.

Unlike mean squared error, it has a non-differentiable point at zero where the gradient must be artificially replaced, which is not a particularly elegant solution.
Mean absolute error punishes small errors more than mean squared error, but large errors less, which can be a desired trait in a model training procedure.

==== Huber <huber_sec>

Huber loss is an attempt to combine the benefits of both mean square error and mean absolute error and remove some of their respective disadvantages @huber_loss. It uses a combination of both methods to achieve differentiability at all points whilst removing mean squared error's large penalty to outliers. It does, however, introduce a new user-tuned hyperparameter $delta$, which, as has been discussed, is never ideal. It is defined by

$ L_delta (vectorn(y), uvectorn(y)) = 1/N sum_(i=1)^N  cases(
  0.5 (y_i - accent(y, hat)_i)^2 "if" |y_i - accent(y, hat)_i| ≤ delta,
  delta |y_i - accent(y, hat)_i| - 0.5 delta^2 "if" |y_i - accent(y, hat)_i| > delta
), $ <huber_eq>

where $L_delta (vectorn(y), uvectorn(y)) $ is the loss function applied to the model output and ground truth vectors, $N$ is the number of elements in the output vector, $delta$ is a user-tuned hyperparameter which controls how much of the loss function obeys mean squared error and how much obeys mean absolute error, $y_i$ is the i#super[th] element of the ground truth vector, and $accent(y, hat)_i$ is the i#super[th] element of the ground truth vector.

The choice of loss function for regression problems is very much problem-dependent and discoverable only through intuition about the dataset or failing that through investigation.

=== Network Design

The choice of loss function is largely down to the problem being attempted and, as such, is often correlated with an associated output layer activation function; see @loss_function_table.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [*Problem*], [*Example Label*], [*Activation Function*], [*Loss Function*],
    [Single-class \ 
    Single-label \ 
    Classification ],
    $[1]$ + " or " + $[0,1]$,
    link( label("sigmoid-sec") )[Sigmoid] + " or " 
    + link( label("softmax-sec") )[Softmax],
    link( label("binary_cross_entropy_sec") )[Binary] + " or "  +  link( label("cross_entropy_sec") )[\Categorical Cross Entropy],
    [Multi-class \ 
    Single-label \
    Classification ],
    $[0, 1, 0, 0]$,
    link( label("softmax-sec") )[Softmax] ,
    link( label("cross_entropy_sec") )[Categorical \ Cross Entropy],
    [Multi-class \
    Multi-label \
    Classification ],
    $[0, 1, 0, 1]$,
    link( label("sigmoid-sec") )[Sigmoid] ,
    link( label("binary_cross_entropy_sec") )[Binary Cross Entropy],
    "Regression including \n Autoencoders",
    "[0.12, -1.34]",
    "Often " + link( label("relu-sec") )[ReLU] + " or " + link( label("linear-sec") )[Linear],
    "Often " + link(label("mse_sec") )[MSE] + ", " + link( label("mae_sec") )[MAE] + ", or " + link( label("huber_sec") )[Huber]
  ),
  caption: [Problems often solvable by artificial neural networks and their associated activation and loss functions. This table demonstrates the most commonly used activation and loss functions for several common problem types that machine learning attempts to solve. The activation functions listed are described in @activation_functions_sec, whereas the loss functions were described in this section @loss_functions_sec. MSE is an abbreviation of Mean Squared Error, and MAE is an abbreviation of Mean Absolute Error.]
) <loss_function_table>

== The Gradients Must Flow <gradient-descent-sec>

Without a method to find useful parameters, artificial neural networks are useful for little more than hopelessly scrambling input data. As mentioned previously, this method is gradient descent @deep_learning_review @stocastic_gradient_descent_ref, using the local gradient of model parameters to find a path toward the minimum loss.

It is useful to imagine the entirety of all possible model parameters forming a surface defined by the model's loss function. Every combination of parameters is a coordinate on this highly dimensional landscape, where the corresponding loss function tells us the height of the land. We can abstract away all this high dimensionality and reduce the problem to a two-dimensional mountain range, as long as we keep in mind that, in actuality, the number of dimensions of our surface equals the number of parameters we are fitting to. 

Imagine we are lost in these Parameter Peaks. Our goal is to find the lowest point in this mountain range, for that will be the location of our helicopter rescue. In the same way that we can only calculate the gradient for one set of parameters at a time, we can only see the ground right beneath our feet; it is a very foggy day on the peaks. If we were to try to get to the lowest point, naturally, what we would do is look at the ground, examine the slope, and walk downhill; the same is true for gradient descent.

#figure(
    image("gradient_descent_examples.png", width: 90%),
    caption: [_ Left: _ An idealised gradient descent path. This gradient descent process quickly reaches the true function minimum where, in this case, the loss is close to zero. However, this is a constructed example by first finding a point near the function minimum and performing a gradient ascent operation. _ Right: _ A more realistic gradient descent path. This example shows a simple but real gradient descent function running on the cost function. As can be seen, it takes many more steps and has not yet converged on the true minimum; in fact, the process might be at risk of getting stuck in a local minimum. Both examples were generated using this notebook: https://tinyurl.com/3ufb5my3.] 
) <gradient_descent_examples>

This works perfectly if all gradients point toward the bottom, what is called a convex parameter space with one global minimum and no local minima. Parameter peaks, however, can often have some nasty tricks in store. Just like real-life mountain ranges, there can be local minima. Divits and valleys look like they might be the bottom of the mountain, but without some global information, it is impossible to tell. The parameter space is non-convex. Thus, we must explore before we settle on our final choice, moving up and down smaller hills, generally moving toward lower regions, but always searching for better outcomes --- that is, until our time runs out, model training ends, and we must make a final decision about where to await the helicopter. Although perhaps if our training lasts multiple epochs, we'll have multiple days to figure it out, taking new paths each time.

There are, perhaps unsurprisingly, a number of different algorithms to achieve this purpose beyond the naive descent suggested by @gradient_decent_step, which could leave you stuck blindly in a divot on the top of a mountain whilst wide chasms stretch unseen before you. The first misconception to correct beyond what has been previously discussed is that usually when performing gradient descent, it is not performed one example at a time but rather in batches of $N_op("batch")$ examples, the gradient of which is calculated simultaneously. This $N_op("batch")$ adds a further user-adjustable hyperparameter, batch size, $N_op("batch")$, to our growing collection of hyperparameters. It also creates a distinction between three distinct gradient descent modes. 

*Stochastic gradient descent* is the method that was illustrated in @training_arificial_neurons. This involves updating the model parameters based on the gradient of a single example at a time, looping through every example in the dataset one at a time, $N_op("batch") = 1$ @stocastic_gradient_descent_ref.

Stochastic gradient descent can converge faster than other methods as it updates the parameters as frequently as possible. This is more useful with larger datasets that cannot fit into memory, as it can make progress long before it has seen all the data, and ample data will help it converge on a correct solution. Stochastic gradient descent does introduce a lot of noise into the dataset as the smoothing effects from averaging across examples are not present. This has advantages and disadvantages. The noise can help prevent the descent from getting stuck in the local minima, but by the same process, it can struggle to settle in even the true minimum, and convergence can take a long time. It can also be slow, as gradients need to be calculated for each example sequentially.

*Mini-batch descent* is perhaps the most commonly used of the gradient descent paradigms @mini_batch_descent. In contrast to stochastic gradient descent, the gradient is calculated for multiple examples simultaneously. Unlike batch descent, however, it does not calculate the gradient for the entire dataset at once. The only restraints, therefore, is that the batch size, $N_op("batch")$, must be larger than one and smaller than the number of elements in your training dataset, $N_op("batch") > 1 and N_op("batch") < N_op("dataset")$. This number is usually a lot smaller than the size of the entire dataset, however, with power of two values around 32 being commonplace.

This method can produce a more stable convergence than stochastic descent. Because it averages the gradient over many examples at once, there is less noise. It is a compromise between batch and stochastic descent, and its strengths and weaknesses depend largely on the batch size you select. This is also one of its largest downsides; any additional hyperparameter is one more factor that has to be tuned by some other external method.

*Batch descent* occurs when the gradient is calculated for all examples in our training dataset simultaneously, $N_op("batch") = N_op("dataset")$; it is, therefore, in some ways, the purest form of gradient descent, as the gradient has been calculated with all available data included @batch_descent. In theory, it will have the most stable and direct convergence of all the methods, although in practice this is not often the case, @batch_descent. However, whilst sometimes producing good results, this can suffer from problems getting stuck in local minima, as there is no ability for exploration. It also has the considerable downside of being very computationally expensive. For very large training datasets, this could quickly become time and computationally impossible. This method is rarely used in modern machine learning due to infeasibly large training datasets.

What follows is a brief explanation of various optimisation algorithms that can be used during the training process. The choice of optimiser can again be considered another hyperparameter that must be externally selected.

=== Momentum

In order to avoid local minima and introduce more exploration to our training process, many optimisers introduce the concept of "momentum" to the descent process @momentum_ref @gradient_descent_algorithms @gradient_descent_algorithms_2. This cannot be applied to batch gradient descent since there is only one step in the process.

Adding momentum to a descent algorithm is quite literally what it sounds like; if we consider the descent process to be a ball rolling down a hill, momentum is a property that changes more slowly than the gradient of the terrain beneath it. In that way, it acts to smooth the inherent noise generated from gradient descent by adding a proportion of the previous gradient to the determination of the next parameter space step. This can help improve convergence and prevent progress from getting stuck in a local minimum.

In order to describe this process mathematically, we introduce the concept of a parameter space velocity, $v_theta (t)$, which is recorded independently of parameter space position, i.e. the parameter values themselves, $vectorn(theta)$. The two equations that fully describe the descent are

$ bold(vectorn(v)_theta (t)) = alpha bold(vectorn(v)_theta (t - 1)) + eta vectorn(gradient) bold(L_( M vectorn(x)_t vectorn(y)_t )) $ <descent_momentum_velocity>

and

$ vectorn(theta)_bold(t + 1) = vectorn(theta)_bold(t) - bold(vectorn(v)_theta (t)), $ <descent_momentum_position>

where $t$ is the current batch index, $ bold(vectorn(v)_theta (t))$ is the parameter velocity at the current batch, $bold(vectorn(v)_theta (t - 1))$, is the parameter velocity at the previous batch (initialized to $0$ at $t - 1$), $alpha$ is the momentum parameter, $eta$ is the learning rate, $vectorn(gradient) bold(L_( M vectorn(x)_t vectorn(y)_t ))$, is the gradient of the model parameters with respect to the loss function, $vectorn(theta)_bold(t+1)$, are the updated model parameters, and $vectorn(theta)_bold(t)$ are the model parameters at the current step. As with the previous training steps, this process can be used for either stochastic or mini-batch descent and will be repeated across all training examples or batches of training examples in the training data set. The momentum parameter is a newly introduced hyperparameter that must be set before the initiation of training. The momentum value indicates what fraction of the previous parameter velocity is added to the current velocity; for any valid descent algorithm, this must be below one, $alpha < 1$, as otherwise, the velocity will grow unbounded with each step. Common choices for momentum values hover around 0.9.

Momentum can be combined with stochastic or mini-batch descent and is an important aspect of other gradient techniques, including RMSProp and Adam @gradient_descent_algorithms @gradient_descent_algorithms_2. 

=== AdaGrad (Adaptive Gradient Algorithm)

In standard gradient descent, every parameter, $theta_i$, within your parameter vector, $vectorn(theta)$, is treated equally by the descent algorithm. We can, however, imagine scenarios where treating all parameters equally is not the ideal method. A given training dataset may not contain an equal representation of all features present in that dataset. Indeed, even individual examples may have some features that are much more common than others. Often, these rarer features can be crucial to the efficient tuning of the network. However, the parameters that represent these features might see far fewer updates than other parameters, leading to long and inefficient convergence. 

To combat this problem, AdaGrad, or the adaptive gradient algorithm, was introduced @adagrad @gradient_descent_algorithms @gradient_descent_algorithms_2. This method independently modifies the learning rate for each parameter depending on how often it is updated, allowing space parameters more opportunity to train. It achieves this by keeping a record of the previous sum of gradients squared and then adjusting the learning rate independently by using the value of this record. This is equivalent to normalising the learning rate by the L2 norm of the previous gradients. This approach is defined by

$ vectorn(g)_bold(t) = vectorn(g)_bold(t - 1) + vectorn(gradient) bold(L_( M vectorn(x)_bold(t) vectorn(y)_bold(t) ))^(compose 2) $ <adagrad_sum>

and

$ vectorn(theta)_bold(t + 1) = vectorn(theta)_bold(t) - (eta/(vectorn(g)_bold(t) + epsilon)^(compose 1/2) ) dot.circle vectorn(gradient) bold(L_( M vectorn(x)_bold(t) vectorn(y)_bold(t)) ) $ <adagrad_iteration>

where $t$ is the current batch index, $vectorn(g)_bold(t)$ is a vector containing the sum of the square of all parameter gradients up to the training iteration, $t$, $vectorn(g)_bold(t-1)$ is the sum of the square of all parameter gradients except the current gradient squares, $vectorn(gradient) bold(L_( M vectorn(x)_bold(t) vectorn(y)_bold(t) ))$ is a vector containing the gradients for each parameter at the current iteration, $vectorn(theta)_bold(t + 1) $ are the parameters at the next iteration, $vectorn(theta)_bold(t) $ are the parameters at the current iteration, and $epsilon $ is a very small value to prevent division by zero errors in the calculation. The $dot.circle $ notation is known as the Hadamard product and represents the element-wise multiplication of two vectors, i.e $vectorn(a) dot.circle vectorn(b) = [a_1 b_1, ..., a_i b_i, ..., a_N b_N]$. Similarly $vectorn(a)^(compose 2)$ refers to the element-wise square of the vector $vectorn(a)$, i.e $vectorn(a)^(compose 2) = [a_1^2, ..., a_i^2, ..., a_N^2]$, and $a^(compose 1/2)$ represents the element-wise square-root of $a$, i.e $vectorn(a)^(compose 2) = [sqrt(a_1), ..., sqrt(a_i), ..., sqrt(a_N)]$.

This method has the advantage of self-tuning the learning rate for individual parameters, removing the need for manual per-parameter tuning, and it helps the model update sparse parameters more quickly by increasing the learning rate for parameters which learn more rarely seen features @adagrad @gradient_descent_algorithms @gradient_descent_algorithms_2. These small features are often very important for whatever operation is being optimised for.

AdaGrad still leaves the global learning rate, $eta$, as an open hyperparameter which must be user-defined. It can also lead to problems when training deep networks with many layers. Similarly, the vanishing gradient problem can lead to tiny parameter updates when calculating the gradient of the network through very deep networks @adagrad @gradient_descent_algorithms @gradient_descent_algorithms_2. The vanishing learning rate problem can arise when training using AdaGrad with very large training datasets. In models with large amounts of parameters, it is crucial that the parameters continue to be updated throughout the training process to ensure that all of the many parameters meet optimally tuned values. However, if the normalisation factor, $vectorn(g)$ for some parameters, grows too big over the training process, the gradient updates can become very small, and training can slow to a crawl. Root Mean Square Propagation is a proposed solution to this problem.

=== RMSProp (Root Mean Square Propagation)

RMSProp, or root mean square propagation, is an alternative method to solve the adaptive learning rate issue, which attempts to alleviate the vanishing learning rate problem by less aggressively normalising the learning rate @rmsprop. Instead of using the L2 Norm of all previous gradients to normalise each parameter learning rate, like AdaGrad, it uses a moving average of the squared gradients. This also deals with non-convex scenarios better, as it allows the gradient descent to escape without the learning rate falling to tiny values. This process is described by

$ vectorn(E)_bold(g^2) (t) = beta vectorn(E)_bold(g^2) (t-1) + (1 - beta) (vectorn(gradient) bold(L_( M vectorn(x)_t vectorn(y)_t )))^(compose 2) $ <rms_sum>

and 

$ vectorn(theta)_bold(t + 1) = vectorn(theta)_bold(t) - (eta/(vectorn(E)_bold(g^2) (t) + epsilon)^(compose 1/2)) dot.circle vectorn(gradient) bold(L_( M vectorn(x)_t vectorn(y)_t )), $ <rms_iteration>

where $t$ is the current batch index, $vectorn(e)_bold(g^2) (t)$ is the moving average of parameter gradients squared with respect to the loss function, $beta$ is the decay rate for the moving average, which controls how quickly the effect of previous gradients on the current learning rate falls off, $vectorn(gradient) bold(L_( M vectorn(x)_t vectorn(y)_t ))$ is a vector containing the gradients for each parameter at the current iteration, $vectorn(theta)_bold(t + 1) $ are the parameters at the next iteration, $vectorn(theta)_bold(t) $ are the parameters at the current iteration, and $epsilon $ is a very small value to prevent division by zero errors in the calculation.

This is a similar method to AdaGrad, so it has many of the same strengths and weaknesses but alleviates the vanishing gradient problem @gradient_descent_algorithms @gradient_descent_algorithms_2. It also introduces one new hyperparameter, the decay rate, $beta $, which must be decided, and it does not necessarily completely eradicate the vanishing gradient problem in all situations.

=== Adam (Adaptive Moment Estimation)

Adam (Adaptive Moment Estimation) combines the advantages of AdaGrad and RMSProp @adam_optimiser. Instead of normalising by the L2 loss alone, like AdaGrad, or the moving squared average alone, like RMSProp, it uses an exponential of the moving average of both the gradient, $E_g (t)$ and the squared gradient, $E_(g^2) (t)$ and uses the parameters, $beta_1$ and $beta_2$ to control the decay rates of these averages respectively. The moving average of the gradient and the moving average of the squared gradient are

$ vectorn(E)_bold(g) bold(t) = beta_1 vectorn(E)_g bold(t-1) + (1-beta_1) vectorn(gradient) bold(L_( M vectorn(x)_bold(t) vectorn(y)_bold(t) )) $ <adam_average>

and

$ vectorn(E)_bold(g^2) (t) = beta_2 vectorn(E)_bold(g^2) (t-1) + (1-beta_2) vectorn(gradient) bold(L_( M vectorn(x)_t vectorn(y)_t ))^(compose 2). $ <adam_moving_average>

As with previous methods, both moving average values are initialised to vectors of zeros at the start of the descent @adam_optimiser. This poses an issue as early steps would be weighted toward zero. In order to solve this, the algorithm introduces two new terms, $uvectorn(E)_bold(g) bold(t) $, and $uvectorn(E)_bold(g^2) bold(t) $, to correct this issue:

$ uvectorn(E)_bold(g) (t) = vectorn(E)_bold(g) (t) / (1 - (beta_1)^t) $ <adam_average_corrected>

and 

$ uvectorn(E)_bold(g^2) bold(t) = vectorn(E)_bold(g^2) bold(t) / (1 - (beta_2)^t). $ <adam_squared_average_corrected>

These terms are then collected in @adam_iteration.

$ vectorn(theta)_bold(t + 1) = vectorn(theta)_bold(t) - eta uvectorn(E)_bold(g) dot.circle (t) / ((uvectorn(E)_bold(g^2) (t) + epsilon)^(compose 1/2)) $ <adam_iteration>

where $t$ is the current batch index, $E_(g) (t)$ is the moving average of parameter gradients with respect to the loss function, $E_(g^2) (t)$ is the moving average of parameter gradients squared with respect to the loss function, $beta_1$ and $beta_2$ are the decay rate for the moving average and the moving squared averages respectively, which controls how quickly the effect of previous gradients on the current learning rate falls off,  $vectorn(gradient) bold(L_( M vectorn(x)_t vectorn(y)_bold(t) ))$ is a vector containing the gradients for each parameter at the current iteration, $vectorn(theta)_bold(t + 1) $ are the parameters at the next iteration, $vectorn(theta)_bold(t) $ are the parameters at the current iteration, and $epsilon $ is a very small value to prevent division by zero errors in the calculation.

The Adam optimiser can intuitively be thought of as combining the adaptive learning rate methods with a form of momentum @adam_optimiser. $E_g (t)$ carries the first moment, the momentum of the past gradients, which, like momentum, will keep you moving in the general direction that you have been travelling, moderated by the $beta_1$ parameter. $E_(g^2)$ carries information about the second moment, which remembers the magnitude of the gradients. This will make the algorithm move more cautiously if it has been encountering steep gradients, which can normally cause large learning rates and make the optimiser overshoot. This can act as a break to the momentum built up in the first moment. The $beta_2$ parameter moderates this aspect.

The Adam optimiser is perhaps the most widely known and widely used in modern artificial neural network training due in large part to its efficacy @gradient_descent_algorithms @gradient_descent_algorithms_2. Although there have been many adaptations and variants of the Adam optimiser which have tried to improve its operation, none have been so successful as to overthrow its position as the standard choice for gradient descent algorithms.

=== Backpropagation <backpropagate-sec>

So far, we have been using the parameter gradient vector, $vectorn(gradient) bold(L_( M vectorn(x)_t vectorn(y)_t))$, without considering how we might calculate this value. 

In the case of a single-layer perceptron, this process is not particularly difficult. As discussed before, first, we must pass an example (or batch of examples) through a randomly initiated network. This network, though untuned, will still produce an output vector, $accent(y, hat)$, albeit a useless one. We can then work backwards from the model output, $accent(y, hat)$,  and, in the case of supervised learning, compare it to our desired output, $y$, by using the loss function, $L$. We can do this by applying the chain rule for the weights @backpropogation_ref.

Let's work through an example of how we might do this for a simple single-layer perceptron, with parameters, $vectorn(theta)$, split into a weights matrix, $matrixn(W)$, and bias vector, $vectorn(b)$.

The action of the model is defined by:

$ M(vectorn(x))= f(vectorn(z)) $

where $vectorn(z) = matrixn(W) vectorn(x) + vectorn(b) $ is the raw input to the activation function, and f is the activation function. The $i^op("th")$ element of the output is given by the softmax function of the raw input, $vectorn(z)$:

$ accent(y, hat)_i = e^(z_i) / (∑_(j=1)^N e^(z_j)) $

and the loss function is given by

$ L = -sum_(i=1)^N y_i log(accent(y, hat)_i), $

where L is the loss function, N is the number of elements in the output vector and $accent(y, hat)_i$ is the $i^op("th")$ element of the output vector. We want to find the gradients of the model parameters with respect to the loss function. In this case, $(partial L) / (partial matrixn(W))$ and $(partial L) / (partial b)$. We can start by using the chain rule to compute $(partial L) / (partial z_i)$, the derivative of the loss with respect to the $i^op("th")$ component of z:

$ (partial L) / (partial z_i) = sum_(j=1) (partial L) / (partial y_j) (partial y_j) / (partial z_i) $ <chain_rule_1>

Here, $(partial L) / (partial y_j)$ is the derivative of the loss with respect to the $j^op("th")$ output, and $(partial y_j)/ (partial z_i)$ is the derivative of the $j^op("th")$ output with respect to the $i^op("th")$ input before activation. In our case, because we are using categorical cross-entropy loss:

$ (partial L) / (partial y_j) = accent(y, hat)_j / y_j $ <chain_rule_2>

And, due to the softmax activation function, in which the value of all output neurons affects the gradient of all others, 

$ (partial accent(y, hat)_j) / (partial z_i) = cases(
  accent(y, hat)_j (1 - accent(y, hat)_j) "if" i = j,
  - accent(y, hat)_j accent(y, hat)_i "if" i ≠ j ) $ <chain_rule_3>

Substitution of @chain_rule_2 and @chain_rule_3 into @chain_rule_1 gives

$ (partial L) / (partial z_i) = - y_i / accent(y, hat)_i  accent(y, hat)_i (1 - accent(y, hat)_i) + ∑_(j ≠ i) y_j / accent(y, hat)_j (-accent(y, hat)_j accent(y, hat)_i). $

Simplifying gives:

$ (partial L)/ (partial z_i) = - y_i (1 - accent(y, hat)_i) + ∑_(j ≠ i) -y_j accent(y, hat)_i. $

We can simplify this further because $sum_(j) y_j = 1$, as the input label is a one-hot vector and will always sum to one:

// TODO: FIX THIS BASTARD (or at least understand it; the answer is right)
$ (partial L)/ (partial z_i) = y_i - accent(y, hat)_i. $

This shows that the derivative of the softmax function with respect to the sum of the weighted inputs and bias values, $(partial L)/ (partial z_i)$, is equal to the difference between the ground truth label value and the model output value. This provides us with another insight into the design of the softmax function and its use of exponentials.

We can then again use the chain rule to find the gradient of the weights and biases

$ (partial L) / (partial matrixn(W)) = (partial L) / (partial vectorn(z)) (partial vectorn(z)) / (partial matrixn(W)) = (vectorn(y) - uvectorn(y)) dot.circle vectorn(x) $ <weights_chain>

and 

$ (partial L)/ (partial vectorn(b)) = (partial L)/(partial vectorn(z)) (partial vectorn(z))/ (partial vectorn(b)) = y - uvectorn(y). $ <bias_chain>

Both of the gradients, @weights_chain, and @bias_chain, are quite intuitively what you might expect from a single-layer network. There is no non-linear behaviour, and as we previously speculated, the network is just training to find pixels that are most often activated by certain classes.

We can use a similar method for artificial neural networks of all complexities and depths. For a feed-forward dense network with $N$ layers, let us denote the weighted sums of the inputs plus the biases of a layer with index $i$, as $vectorn(z)_bold(i)$, the output of the activation function, $f$ of layer $i$ as $a_i = f(z_i)$, the weights matrix and biases vector of layer $i$ as $matrixn(W)_bold(i)$ and $ vectorn(b)_bold(i)$, and the loss function again as $L$.

First, we compute the forward propagation by running an input vector, $vectorn(x)$, or batch of input vectors, through the network to produce an output vector $uvectorn(y)$. Then follow the following procedure.

+ Compute the derivative of the loss function with respect to the final output values: $ (partial L)/( partial a_N) = (partial L)/( partial accent(y, hat))$.
+ Compute $(partial L)/ (partial z_N) = (partial L) / (partial a_N) (partial a_N)/ (partial z_N) $, where $(partial a_N)/(partial z_N)$ is the derivative of the activation function in the final layer. This gives the gradient of the loss function with respect to the final raw outputs, $vectorn(z)_bold(N)$.
+ Compute $(partial L)/ (partial matrixn(W)_bold(N)) = (partial L) / (partial z_N) (partial z_N) / (partial matrixn(W)_bold(N)) $ and $(partial L) / (partial b_N) = (partial L) / (partial z_N) (partial z_N)/ (partial b_L) $. This gives the gradients with respect to the final layer's weights and biases.
+ To propagate the error back to the previous layer, compute $(partial L)/ (partial a_(N-1)) = (partial z_N)/ (partial a_(N-1)) (partial L) / (partial z_N) = W_(N)^T (partial L) / (partial z_N)$.
+ Recursively repeat steps 1 to 4 until you reach the input layer and you have gradients for all parameters.

This method is known as backpropagation because you work backward from the output of the model toward the input vector @backpropogation_ref.

== Overfitting and Regularisation

Thus far, we have been partaking in perhaps one of the most heinous sins when developing a machine learning method --- we have not made a distinction between the dataset we use to train our model, out *training* dataset, and the dataset we use to test our model, our *testing* dataset. It is vital that whenever possible we produce these datasets independently, and keep them entirely separate so that a model never has the chance to use any information present in the testing dataset to adjust its weights, ensuring that the first time a model sees any of the testing examples, is when it is being validated with them. 

This hard segregation acts as a test to see if our model has *overfit* @overfitting. If the model learns the features of each of our training examples to the point where it can remember each specific example and match that example to a label, then our model may just associate each training example individually to an example rather than learning the general features of the dataset which will allow for classification of new unseen examples. If this is the case, when presented with new examples, such as examples from our testing dataset, the classifier will fail because it does not know what class this example belongs to. Thus keeping a separate dataset for testing is crucial to accurately assess model performance. Without such a distinct dataset, we cannot make any claims about the efficacy of our model. Whether, and to what degree, it is possible for a model to overfit to a particular dataset depends on the size and complexity of the training dataset and the size and complexity of the model. The larger the training dataset the more unlikely it is for overfitting to occur; however, a larger model ensures that a model can "remember" more data, which gives it an increased possibility to overfit to its training dataset. 

Often, a further distinction is made between testing, training, and validation datasets. Although the nomenclature is often mixed up between these three datasets. The purpose of introducing a third *validation* dataset is to act as a final check of generality. Since the training procedure and model architecture are often altered throughout the development of the model, it is important to make sure that these alterations are not also accidentally tailored to our testing dataset. This third, validation dataset is set aside, ideally until the finalization of the data analysis method, to act as a final test of performance. Ideally, this dataset would have been created prior to the commencement of the project, to ensure that there is no possibility that the validation dataset is generated with the designed method in mind.

There may also be a need for more than three datasets, for example, you might want to have a test dataset that a training model is compared against every epoch, then another test dataset after a full training procedure has completed, but before the final method has been selected, which would necessitate a fourth dataset in this case. The term validation dataset will be used throughout this thesis for any dataset that is not the training dataset.

Overfitting is one of the most prominent and difficult problems in artificial neural network development, and thus there has been a large body of methods to try and ameliorate the issues it causes @overfitting. These methods are known as regularisations, the following few sections will briefly describe some of these methods.

=== Dropout

One of the most powerful ways to deal with overfitting is to employ dropout layers with your network @dropout_ref. Dropout layers can prevent overfitting by ensuring that the model does not rely on any one given neuron (or any given set of neurons), in order to produce its final output. Dropout layers do this by randomly setting a certain percentage of the previous layer's outputs to zero, ensuring that information from that neuron cannot be used to produce the model's output during this inference. The choice of neurons that are zeroed is randomised between each training instance, this teaches the model to explore different feature recognition pathways in each training batch.

The percentage of outputs dropped is a user-selected hyperparameter that must be decided before model training is initiated, can can be anywhere from 0 (equivalent to no dropout layer) to 1 (which would stop all information flowing through the network and make training impossible), typical dropout values lie between 0.1 and 0.5 @dropout_ref. Dropout layers are only active during model training, and when in use for inference proper are not employed and can be removed without affecting model function.

By randomly dropping out neurons during model training, it reduces the information that a model can rely on to produce its final output @dropout_ref @dropout_regularisation.. Because, in almost all cases, remembering the exact form of each training example will take a larger amount of information than remembering only general features, the network is incentivized to learn input features rather than memorizing specific training examples.

Dropout can sometimes slow down model convergence, and it is not a complete solution to overfitting, but it finds use very commonly across a wide range of modern neural network architectures @dropout_regularisation.

=== Batch normalisation

Another layer type that can be added to a network to act as regularisation as well as provide other benefits is the batch normalization layer @batch_normalisation_ref. As data moves through deep networks, a phenomenon known as internal covariate shift can take place. This describes the scenario wherein values flowing through the network can occupy a wide range of distributions that vary dramatically between layers, usually increasing in size as they move through the network. 

This can be a problem as the activation functions present in a network are designed to act in specific data distributions, so large values in the network can saturate their non-linearities, and remove much of the potential for nuance, which can increase model convergence time and degrade model performance.

Batch normalization layers offer a solution to this problem by normalizing the mean and standard deviation of a layer's output distribution to certain values, normally a mean of zero and a variance of one. To ensure that this normalisation does not reduce the information content of the layer output, the layer has two tunable weight parameters per input neuron, one to scale the neuron's output after normalisation, and the other to shift it after normaliation. Finally, batch normalisation ensures that the gradients stay within a reasonable range, which also increases model convergence.

Like dropout, batch normalisation is applied differently during training and when in use in production in the inference phase of the model @batch_normalisation_ref. During training the normalisation is based on the mean and variance of the current training batch, whereas during inference the normalisation uses the moving average and moving variance computed during the training phase, primarily to ensure that the model's output is deterministic, which is often a desired characteristic.

Batch normalisation serves many purposes within the network, increasing convergence and allowing for faster training times, but it also can help to prevent overfitting because it dramatically reduces the range of possible states that can occur in inference and training. 

=== Early Stopping

One simple way to prevent overfitting that is often employed is to halt the training procedure before overfitting can occur @early_stopping. Generally, if it is possible for a given model to overfit to a given training distribution, then it will overfit more the more often it has seen that training dataset, i.e. the number of epochs that have been used in the training procedure. Shuffling the dataset each epoch will reduce this problem slightly by generating unique batch combinations and altering the order that gradient descent takes through the parameter space, but at each iteration the model is still adjusting its parameters based on the training examples, potentially closing in on a fit that is too close.

This can be alleviated by halting the training early based on model post-epoch performance when validated on your validation dataset, generated as independently as possible from your training dataset. If your model begins to overfit your training dataset, then, almost by definition, validation performance will begin to degrade or at least saturate. The model training procedure can be configured to automatically detect if this is the case, relying on a user-defined hyperparameter known as patience, which determines the number of epochs with no improvement in test dataset performance to wait before halting the training.

Stopping model training early, and restoring the model parameters that achieved the best performance on the test dataset can be an effective method to stop the model from converging on a fit that is too closely tailored to the training dataset.

== Infrastructure Layers <flatten-sec>

Most GPU vector libraries, including TensorFlow @tensorflow, have strict requirements about the shapes of vectors that flow through them. Within artificial neural network models, there is often a need to change the shape and/or dimensionality of the vectors as they flow through the network -- for example if we are moving from a 2D image to a 1D vector, as we saw when feeding 2D MNIST images into the 1D perceptron architecture we must employ a *flattening layer* which takes whatever dimensionality the input vector has and reduces it to a 1D vector. We can also use reshape layers to perform more complex reshapings between vector shapes as long as the requested resultant vector contains the same number of elements as the input vector; see @flattening_diagram.

#figure(
  image("flattening.png", width: 80%),
  caption: [A flattening layer. This layer takes a 2D input matrix $X = mat(x^1_1, x^1_2; x^2_1, x^2_1)$ and converts it into a 1D vector, $ vectorn(y) = [y_1, y_2, y_3, y_4]$, without using any learned parameters or altering the values of the data. It simply rearranges the indexes and removes all but one dimension. Reshaping layers are a more general version of a flattening layer, where an input vector or matrix can be transformed into any equivalently sized output vector or matrix.],
) <flattening_diagram>

These kinds of "infrastructure" layers will typically not be discussed nor included in network diagrams if their existence is implied by the network construction. They do not have any trainable parameters and perform no transformation on the passing data other than to change the data layout. They are only noted when newly introduced or of special interest.