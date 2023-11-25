= Skywarp: An Attention-Based model for the Detection of Gravitational-Wave Compact Binary Coalescences <skywarp-sec>

== Attention! <sec-attention>

The global information provided by an individual element within a sequence is often greater than the local information contained within the isolated datum. This extra information is stored contextually within the relationship between the given element and the other elements in the sequence, both within the information stored locally by the other elements and by the relative and absolute positions of the other elements.

The set of possible combinations of elements is large, even within relatively small sequences. Therefore, in order to regularise a machine learning model to extract contextual information efficiently, a method must be implemented to determine which elements contribute the most contextual information to a given datum. This method is attention. Attention determines which elements in the sequence contribute highly to the global information of a given element. Once attention has been determined, global contextual information can be embedded within each element’s local information. Ideally, this process makes the output elements, now with contextual information embedded locally, easier for other machine-learning methods to interpret. 

A transformer model is a machine learning algorithm that implements this method to localise global information using attention. The output to a transformer block has the same dimensionality as the block’s input, as it retains the same number of elements. Ideally, each element has been transformed to contain a proportion of the global information stored within the input sequence.

The models we describe in this section are novel in that they utilise attention mechanisms (Bahdanau et al. 2014,  Luong et al. 2015), a type of differentiable memory in which a global context vector is learned over an input sequence $x_i$. 

The attention mechanism aims to embed global context locally; in order to do this, a comparison must be made between each element of the sequence and (in the case of self-attention) each other element of the same sequence. It is trivial to see that not every element in every sequence will be equally contextual to each other and that the contextual dependence will depend on the information being extracted.

This corresponds to learning the following relation:

$ y_j = alpha^i_j x_i $
    
Where $alpha^i_j in Re$ are the scalar attention scores measuring the relative correlation between $x_i$ and $x_j$. In this way, one learns intra-sequence relations; long-term dependencies are captured because the entire input sequence is used to compute a single element of the output sequence. 

In order to calculate these attention scores, we generate three vectors, namely a query $q_i$, key $k_i$, and value $v_i$ vector, for each sequence element $x_i$, forming three matrices for the sequence as a whole: $Q$, $K$, and $V$. These matrices are created by applying projection matrices: $W_Q$, $W_K$ and $W_V$ to the input sequence, the values of these weights matrices are learned via backpropagation during the model training process.

The query, key, and value matrices are used to calculate the attention scores for each element. The query value for each sequence element is matched against the key value of each other element, the alignment of the key and query determines a weighting for the value vector, a distilled representation of the information contained within that element. The weighted value vectors are then summed to produce the new, contextually embedded, sequence.

The two most commonly used attention functions are dot-product (Luong et al. 2015) and additive attention (Bahdanau et al. 2014), our models utilise the former and so we restrict our discussion to the work of (Luong et al.
2015) and extensions. In either case, the function $alpha$ maps a set of query $q$, key $k$ and value $v$ vectors to a weighted sum of the values. 

$ alpha(q, K, V) = sum_{i} a(q, k_i)v_i $
  
Where $a(., .)$ is called the alignment function and measures the similarity between the queries and keys. In the case of dot-product attention proposed by (Luong et al. 2015)

$ a(q, k) = sigma(q^T k) $
  
where $sigma$ is the Softmax function (). This calculation is performed on each element of the sequence to produce a new sequence of equal length, hopefully with some contextual context embedded. Generalising the attention function we get

$ alpha(Q, K, V) = sigma(Q K^T)V .$
  
Where again, $sigma$ is the Softmax function.

== Transformers

Since their introduction, attention mechanisms have been utilised in a number of different neural network architectures, including transformers and stable diffusion models. Transformers were first proposed by (Vaswani et al. 2017) to solve natural-language processing tasks, showing significant improvement over previous recurrent and convolutional architectures. For these reasons, we decided to investigate a fully attention-based model, inspired by a Transformer encoder.

The transformer model uses the attention mechanism described earlier in section @sec-attention within discrete blocks called multi-attention heads. Multi-attention heads have N multiples of the weights matrices used to generate query, key, and value matrices from input sequences. These multiple heads can be thought of analogously to different convolutional filters inside a CNN layer; each head can focus on extracting a different type of contextual information from the sequence. This is necessary as the useful contextual information embedded within a sequence can be more complex than it is possible to extract with a single attention head. The output sequences of all N heads are merged after the block to ensure that the output and input sizes are the same.

Often, as is the case for our models, the multi-attention heads are followed by normalisation and one or more dense layers. These blocks of layers can then be stacked to form components of a transformer.

%embedding
%positional encoding,

== Literature

Chatterjee et al offer perhaps the most relevant work, they utilize Long Short Term Memory (LSTM) networks, a form of Recurrent Neural Network, for both the problems of signal detection and reconstruction. Recurrent neural networks have an internal state determined by previous inferences, and thus, they have the ability to retain some information about all previous data. In many ways, RNNs were the predecessor to Transformer models, largely because they are able, in some way, to make inferences from global information rather than being limited by the receptive fields of convolutional filters.

There have also been some attempts to detect other proposed signal types, known as gravitational-wave bursts, with machine learning methods, most prominently core-collapse supernovae. Although there has yet to be a real detection of any such signals, these types of signals are in many ways a more interesting candidate for machine learning methods due to the much higher degree of uncertainty in waveform modeling. The physics behind burst signals is often much less understood, as well as possessing a much larger number of degrees of freedom. There have also been a few papers that have attempted to use machine learning methods as a coherence detection technique, therefore eschewing any requirement for accurate signals. Such detection problems could also benefit from the application of transformer models and will be an area of future inquiry. 

As demonstrated, there has been a considerable investigation into the use of machine learning techniques for gravitational wave detection. However, there has not been a significant investigation into the use of transformers for such a purpose, with only this paper by Zhao et al known at this time, which focuses on the problem of space-based detection. https://arxiv.org/pdf/2207.07414.pdf

== Model

We have investigated a number of attention-based models and compared their performance to convolutional models as well as other traditional analysis techniques. We have investigated a fully attention-based model, as well as combined convolutional-attention model.

For the CNN model, we adapted a model from the literature. [] Architecture is as described in figure \ref{fig:cnn}.

#figure(
    image("skywarp_pure_conv.png",  width: 100%), 
    caption: []
) <skywarp_pure_conv>

%This needs major edits.
When transformers are utilised for Natural Language Processing (NLP) tasks, the input strings of natural language are first tokenised into discrete tokens before those tokens are fed into an embedding layer to convert the discrete tokens into continuous vectors that the network can ingest. When adapting the architecture for use on time series data, there are some design decisions that must be taken. Tokenization, although still possible, is no longer required as the input data is initially in a continuous form. However, when deciding how to feed the series into the transformer, there are several options. Although it is possible to feed an attention block with the length of one vector from the input time series, it was found that this naive approach eliminated much of the transformer's potential for element-wise comparison. To resolve this, the method used by the vision transformer can be used; the input data can be segmented into N segments, and then fed into the network. In addition or in place of such a segmentation, an embedding layer can also be employed to increase the dimensionality of the segments.

In the pure attention model, we reshaped the input time series (1s at 8192Hz) into 512 segments of size 16, these segments were then encoded into larger vectors of length 128 by a single convolutional layer with a filter size of 1. This embedding was performed to allow sufficient size for the positional encoding to be added to each vector. This solution was found after trialing several variations. See figure ref{fig: transformer}, for more detailed information on the network.

#figure(
    image("skywarp_pure_attention.png", width: 100%), 
    caption: []
) <skywarp_pure_attention>

On testing, we found that the pure attention model did not perform as well as the CNN model. It was found that the transformer model could much more easily overfit the training data, even with large training datasets. In order to combat this - a combination convolutional-attention model was introduced. This model, described in figure ref{fig:conv_transformer}, feeds the output of the convolutional layers from the CNN described by figure ref{fig: transformer} into the attention blocks described in figure ref{fig:cnn}, in attempts to gain the benefits of both methods.

#figure(
    image("skywarp_conv_attention.png", width: 100%), 
    caption: []
) <skywarp_conv_attention>

== Training, Testing, and Validation Data


== Training Procedure


== Results

To profile the performance of Skywarp we compare it against three alternative techniques: the standard matched filtering approach which is the current method used to confirm the detection of CBC signals; a Convolutional Neural Network (CNN) with architecture taken from this early paper by George et Al, as this was the first type of neural network architecture applied to CBC detection; and a Recurrent Neural Network, as another neural network architecture which is commonly used for time series analysis.

begin{table}[]
begin{tabular}{|l||l|l|l|l|}
hline
SNR & Matched Filtering & CNN & RNN & Skywarp \\ \hline
4   &                   &     &     &         \\ \hline
6   &                   &     &     &         \\ \hline
8   &                   &     &     &         \\ \hline
10  &                   &     &     &         \\ \hline
end{tabular}
end{table}

#figure(
  image("skywarp_far_curve.png", width: 100%), 
  caption: []
) <skywarp_far_curve> 

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("skywarp_efficiency_0_1.png", width: 100%) ],
        [ #image("skywarp_efficiency_0_01.png", width: 100%) ],
        [ #image("skywarp_efficiency_0_001.png", width: 100%) ],
        [ #image("skywarp_efficiency_0_0001.png", width: 100%) ],
    ),
    caption: []
) <skywarp_efficiency_curves>

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("skywarp_roc_8_20.png", width: 100%) ],
        [ #image("skywarp_roc_12.png", width: 100%) ],
        [ #image("skywarp_roc_10.png", width: 100%) ],
        [ #image("skywarp_roc_8.png", width: 100%) ],
        [ #image("skywarp_roc_6.png",   width: 100%) ],
    ),
    caption: []
) <skywarp_roc_curves>

== Discussion
