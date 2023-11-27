= Skywarp: An Attention-Based model for the Detection of Gravitational-Wave Compact Binary Coalescences <skywarp-sec>

Convolutional Neural Networks (CNNs), though effective, have not been the stars of the machine-learning world for a few years now. Just as AlexNet @image_classification paved the way for the era of CNNs, and the surge to prominence of artificial neural networks as problem-solving solutions in the image domain, a similar step change occurred when the confluence of several different technologies led to the development of Large Language Models (LLMs), most notably of which in recent years has been the Generative Pretrained Transformer (GPT) of ChatGPT fame, though it should be noted that this was far from the first LLM. 

Although CNNs were extremely successful at solving previously intractable problems in the image and audio domain, there were still many challenges remaining within Natural Language Processing (NLP), the area of study relating to data analysis of the text domain. Text sequences differ from audio time series, in that rather than vectors of continuous numerical values, they consists of sequences of discrete tokens, whether you divide it all the way into individual characters or constrain the tokens further to unique individual words. There had been some work to attempt these problems with CNNs and Recurrent Neural Networks, but it was only through the application of multiple different insights including Word2Vec, attention mechanisms, and skip connections, that the start of the avalanche of progress we have since seen in this area, began. The groundbreaking paper by Vaswani _et al._, @attention_is_all_you_need first introduced the transformer architecture to the world.

Following the success of transformers in NLP, there has been much research into the application of attention-based models to other domains, including image and audio processing. Notably, AlphaFold, a project by Google Deepmind, successfully solved the protein folding problem. Transformers have proven effective, scalable solutions due to their highly parallelizable nature, and new attention-based architectures such as generative diffusion models have seen great success in text-to-image generation problems, as is seen in products like Stable Diffusion, Midjourney, and Dall-E. Gravitational-wave astronomy has also seen the application of a large number of machine-learning approaches; however, there exists a considerable delay between the advent of new techniques and their application to gravitational wave data. Since this research was originally carried out, the gap has narrowed, nonetheless, it remains a relatively unexplored area that may prove fruitful in the future. In this chapter, we introduce Skywarp, an attention-based model for the detection of gravitational waves produced by Compact Binary Coalescences (CBCs).

The distinguishing feature of the transformer architecture is the use of attention. Attention is a technique that can determine the information value of data elements in a series, crucially including the relative information derived contextually from the value and position of all other data elements within the series. As opposed to convolutional layers, which learn feature-extracting filters that are convolved with the input data, multi-attention heads learn a weighting for each data element, identifying the information interactions between elements. One key advantage of this method is that attention is computed globally, whereas convolutional layers only use a local context dictated by their receptive field. This makes attention an ideal candidate for analyzing time-series data, wherein important contextual information can be located at a significant temporal distance from any given datum. Recurrent neural networks share many of these properties; however, they make use of an internal state, which means they are much harder to parallelize and therefore scale, and their knowledge of data outside of their receptive field is uni-directional, as opposed to transformers, which are bi-directional. In addition, it should be noted that self-attention layers are more general than convolutional layers, as it is proven that a self-attention layer can generalize any convolution.

This section is organized into the following structure. First, we will give a brief technical overview of the concepts and structure of the transformer architecture in @attention-method, including a discussion of tokenization, embedding and attention, and how these elements are assembled into transformers proper. Then we will review the small amount of relevant literature that has not already been discussed in @skywarp_review. Next will provide details on the models and training procedures we used to train Skywarp in @skywarp-method. We will show the validation results from our trained models in @skywarp-results, before finally in @skywarp-discussion, we discuss the importance of these results and investigate how the method employed by the transformer models differs from that used by CNNs by comparing their respective attention and convolution maps.

== Attend Closely <attention-method>

The transformer is a deep learning model first described in Vaswani _et al._  @attention_is_all_you_need. This paper was an NLP paper, demonstrating a deep learning sequence-to-sequence model that could ingest text data and predict the token that was most likely to appear next in the sequence. By recursing inferences of the model, new sentences could be generated based on previous inputs. In some sense, our time series data is already closer to the required input of a deep learning model, however, it is easiest to explain attention by using the NLP example. Therefore whilst describing attention we will use the text domain as an example and then replace the vectors we end up with in the NLP case, with whatever sequence of vectors we wish to analyze.

=== Tokenisation and Embedding

There are several steps required in order to condition text data for consumption by deep learning models. Artificial neural networks work solely with numerical data therefore text must somehow be converted into numbers before it is ingested by the machine. The most obvious way to do this would be to use a preexisting character format such as ASCII or Unicode. However, if we were to do this, the numbers would relate very little even to the characters they represented, let alone the words. This would make these inputs very difficult for the model to analyze. Therefore, in order to make the task easier, we must have some method of embedding the words into a numerically defined space in a manner that maintains some of their meaning in their representation. Typically this is achieved in two or three steps, tokenization, vectorisation, and embedding; see @tokenisation_diagram.

#figure(
    image("tokenisation_diagram.png",  width: 100%), 
    caption: [The process of conditioning text data for input into a deep learning model. Text data is not intrinsically digestible by artificial neural network models, as artificial neurons can only process numerical inputs. Therefore, in order to apply deep learning models to text data, we must have some method of converting the data into a numerical format. Transformers expect a sequence of same-length vectors, this diagram shows the process of conversion. Typically, this conversion is completed in three steps, tokenization, vectorisation, and embedding. However, often, and in the case of the first described transformer model vectorisation and embedding occur simultaneously as is depicted in the diagram, and are often labeled simply embedding. In the diagram, we see the sentence "The quick brown fox jumped over the lazy dog." as it is prepared for ingestion by an NLP model. *Tokenisation* is the process of splitting one contiguous sequence of characters into a number of unique discrete tokens, N. This can be done at multiple levels but is usually done at the scale of words. Sometimes, especially with longer words, words can be split into multiple tokens, as is seen in this example in the word "jumped", which is split into "jump" and "ed". There are numerous algorithms to achieve this, which will not be discussed in detail. Every word, or word subset, within the training dataset, should have a unique token id. Before running inference on new text data, that data must be tokenized, and each word in the new data is mapped onto an existing token ID that was generated during the initial tokenization process. Often some information-low words, known as "stop words", and punctuation are removed during the tokenisation process. In the example shown, the words "The", and full stops are removed from the input string. During *vectorisation*, each token is assigned a numerical vector, and *embedding* ensures that this vector is transformed into a meaningful vector space to allow for easier interpretation by the model. There are a number of methods to achieve both of these steps, some of which are simultaneous. In the example shown, each token ID is associated with a vector of tunable weights, as was the case in the first transformer paper. These vectors start training randomised, but as the model training process continues, they become tuned to values that represent the value contained by the words. In this manner, the vectorisation and embedding steps occur simultaneously after the model has been trained.]
) <tokenisation_diagram>

First, the input sequence must be tokenized. Tokenization involves splitting the input text into N unique discrete tokens. Often the text is processed first, sometimes removing low information elements known as stop-words and punctuation. Then, using one of a variety of algorithms that will not be discussed here, the text is consolidated into N tokens, often these could be whole words, but sometimes, depending on the tokenization algorithm and the size of the training data, word fragments could also be tokens. One will note that often words can have multiple meanings, which is a problem when trying to describe each token in a way that somehow presents the value it represents in a sentence, this can be a problem, and methods have been developed that can split identical sequences of letters into multiple tokens contextually. The very existence of this problem is an example of the contextual information that is provided by surrounding tokens in the sentence, this is the information that attention layers attempt to extract and distill.

After tokenization, we must convert each token to a unique vector. This can also be done through a variety of methods. In Vaswani _et al._ each token has an associated vector in a look-up table, initially these vectors are randomly generated; however, the values in each of these vectors act as tunable parameters inside the model, so that whenever a certain token is present in a particular training example, the weights of its vector are tuned through the usual gradient-descent methods. In this way, as the training progresses the vectors become, at least to the model, meaningful numerical representations of the value contained within the word. 

Gravitational-wave data is intrinsically vectorized so the embedding layer would not be much of a problem, however, it is not intrinsically discretised. Since transform are sequence-to-sequence models, they ingest a series of N vectors. It is unclear a priori, how best to split the gravitational wave data into smaller vectors. We could simply cut along equally separated lines, "chunking" our data into smaller timesteps, or we could embed the data using some learned weights, for example with one or more dense or convolutional layers, in the latter case, feeding the transformer with feature slices at different timesteps.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("skywarp_chunking.png",  width: 100%)],
        [ #image("skywarp_convolutional_embedding.png",  width: 100%)],
        [ #image("skywarp_dense_embedding.png",  width: 100%)],
    ),
    caption: ["Different embedding possibilities to discretise and embed gravitational-wave time-series data."]
) <gw_embedding>

// Gravitational-wave embedding

We have managed to transform our input text from a list of symbols into discrete tokens and finally into vectors that contain some aspect of the value represented by that token, and we have some ideas about how we might do the same with gravitational-wave data. However, unlike RNNs transformers treat each token equally, and intrinsically have no information about the location of the word in the sentence. We must use feature engineering to add to each vector, some information about the position of the token in the input sequence. 

=== Positional Encoding

Much information is embedded in the relative and absolute positions of tokens with text data. The same can said to be true of gravitational-wave data --- we would always expect the merger to come after the inspiral for example. Whilst there is some possibility within the dense layers of a traditional CNN for the model to use this ordinal information in its classification, it might be a challenging process. We can use attention layers

=== Attention! <sec-attention>

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

=== Multi-Head Attention

=== Attention Blocks

=== Transformers

Since their introduction, attention mechanisms have been utilised in a number of different neural network architectures, including transformers and stable diffusion models. Transformers were first proposed by (Vaswani et al. 2017) to solve natural-language processing tasks, showing significant improvement over previous recurrent and convolutional architectures. For these reasons, we decided to investigate a fully attention-based model, inspired by a Transformer encoder.

The transformer model uses the attention mechanism described earlier in section @sec-attention within discrete blocks called multi-attention heads. Multi-attention heads have N multiples of the weights matrices used to generate query, key, and value matrices from input sequences. These multiple heads can be thought of analogously to different convolutional filters inside a CNN layer; each head can focus on extracting a different type of contextual information from the sequence. This is necessary as the useful contextual information embedded within a sequence can be more complex than it is possible to extract with a single attention head. The output sequences of all N heads are merged after the block to ensure that the output and input sizes are the same.

Often, as is the case for our models, the multi-attention heads are followed by normalisation and one or more dense layers. These blocks of layers can then be stacked to form components of a transformer.

%embedding
%positional encoding,

== Literature <skywarp_review>

Chatterjee et al offer perhaps the most relevant work, they utilize Long Short Term Memory (LSTM) networks, a form of Recurrent Neural Network, for both the problems of signal detection and reconstruction. Recurrent neural networks have an internal state determined by previous inferences, and thus, they have the ability to retain some information about all previous data. In many ways, RNNs were the predecessor to Transformer models, largely because they are able, in some way, to make inferences from global information rather than being limited by the receptive fields of convolutional filters.

There have also been some attempts to detect other proposed signal types, known as gravitational-wave bursts, with machine learning methods, most prominently core-collapse supernovae. Although there has yet to be a real detection of any such signals, these types of signals are in many ways a more interesting candidate for machine learning methods due to the much higher degree of uncertainty in waveform modeling. The physics behind burst signals is often much less understood, as well as possessing a much larger number of degrees of freedom. There have also been a few papers that have attempted to use machine learning methods as a coherence detection technique, therefore eschewing any requirement for accurate signals. Such detection problems could also benefit from the application of transformer models and will be an area of future inquiry. 

As demonstrated, there has been a considerable investigation into the use of machine learning techniques for gravitational wave detection. However, there has not been a significant investigation into the use of transformers for such a purpose, with only this paper by Zhao et al known at this time, which focuses on the problem of space-based detection. https://arxiv.org/pdf/2207.07414.pdf
 
== Skywarp Method <skywarp-method>

=== Skywarp <skywarp-model>

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

== Training, Testing, and Validation Data <skywarp-data>


== Training Procedure <skywarp-training>


== Results <skywarp-results>

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

== Discussion <skywarp-discussion>
