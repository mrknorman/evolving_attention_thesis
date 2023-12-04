= Skywarp: An Attention-Based model for the Detection of Gravitational-Wave Compact Binary Coalescences <skywarp-sec>

#set math.equation(numbering: "6.1")
\
Convolutional Neural Networks (CNNs), though effective, have not been the stars of the machine-learning world for a few years now. Just as AlexNet @image_classification paved the way for the era of CNNs, and the surge to prominence of artificial neural networks as problem-solving solutions in the image domain, a similar step change occurred when the confluence of several different technologies led to the development of Large Language Models (LLMs), most notably of which in recent years has been the Generative Pretrained Transformer (GPT) of ChatGPT fame, though it should be noted that this was far from the first LLM. 

Although CNNs were extremely successful at solving previously intractable problems in the image and audio domain, there were still many challenges remaining within Natural Language Processing (NLP), the area of study relating to analysis of the text domain. Text sequences differ from audio time series, in that rather than vectors of continuous numerical values, they consist of sequences of discrete tokens that encode externally defined values. There had been some work to attempt these problems with CNNs and Recurrent Neural Networks (RNNs), but it was only through the application of multiple different insights including positional encoding, attention mechanisms, and skip connections, that the start of the avalanche of progress we have since seen in this area, began. The groundbreaking paper by Vaswani _et al._, @attention_is_all_you_need first introduced the transformer architecture to the world.

Following the success of transformers in NLP, there has been much research into the application of attention-based models to other domains, including image and audio processing. Notably, AlphaFold, a project by Google Deepmind, successfully solved the protein folding problem. Transformers have proven effective, scalable solutions due to their highly parallelizable nature, and new attention-based architectures such as generative diffusion models have seen great success in text-to-image generation problems, as is seen in products like Stable Diffusion, Midjourney, and Dall-E. Gravitational-wave astronomy has also seen the application of a large number of machine-learning approaches; however, there exists a considerable delay between the advent of new techniques and their application to gravitational wave data. Since this research was originally carried out, the gap has narrowed, nonetheless, it remains a relatively unexplored area that may prove fruitful in the future. In this chapter, we introduce Skywarp, an attention-based model for the detection of gravitational waves produced by Compact Binary Coalescences (CBCs).

The distinguishing feature of the transformer architecture is the use of attention. Attention is a technique that can determine the information value of data elements in a series, crucially including the relative information derived contextually from the value and position of all other data elements within that series. As opposed to convolutional layers, which learn feature-extracting filters that are convolved with the input data, attention layers learn a weighting for each data element, identifying the information interactions between elements. One key advantage of this method is that attention is computed globally, whereas convolutional layers only use a local context dictated by their receptive field. This makes attention an ideal candidate for analyzing time-series data, wherein important contextual information can be located at a significant temporal distance from any given datum. RNNs share many of these properties; however, they make use of an internal state, which means they are much harder to parallelize and therefore scale, and their knowledge of data outside of their receptive field is uni-directional, as opposed to transformers, which are bi-directional. In addition, it should be noted that self-attention layers are more general than convolutional layers, as it is proven that a self-attention layer can generalize any convolution.

This section is organized into the following structure. First, we will give a brief technical overview of the concepts and structure of the transformer architecture in @attention-method, including a discussion of tokenization, embedding, and attention, and how these elements are assembled into transformers proper. Then we will review the small amount of relevant literature that has not already been discussed in @skywarp_review. Next, we will provide details on the models and training procedures we used to train Skywarp in @skywarp-method. We will show the validation results from our trained models in @skywarp-results, before finally in @skywarp-discussion, we discuss the importance of these results and investigate how the method employed by the transformer models differs from that used by CNNs by comparing their respective attention and convolution maps.

== Attend Closely <attention-method>

The transformer is a deep learning model first described in Vaswani _et al._  @attention_is_all_you_need. This paper was an NLP paper, demonstrating a deep learning sequence-to-sequence model that could ingest text data and predict the token that was most likely to appear next in the sequence. By recursing inferences of the model, new sentences could be generated based on previous inputs. In some sense, our time series data is already closer to the required input of a deep learning model, however, it is easiest to explain attention by using NLP as an example. Therefore whilst describing attention we will use the text domain as example data before replacing the relevant vectors with gravitational-wave equivalents.

=== Tokenisation and Embedding

There are several steps required in order to condition text data for consumption by deep learning models. Artificial neural networks work solely with numerical data therefore text must somehow be converted into numbers before it is ingested by the model. The most obvious way to do this would be to use a preexisting character format such as ASCII or Unicode. However, if we were to do this, the numbers would relate very little even to the characters they represented, let alone the words. This would make these inputs very difficult for the model to analyze. Therefore, in order to make the task easier, we can use a method to embed the words into a numerically defined N-dimensional space in a manner that maintains some of their meaning in their new vectorised representation. Typically this is achieved in two or three steps, tokenization, vectorisation, and embedding; see @tokenisation_diagram.

#figure(
    image("tokenisation_diagram.png",  width: 100%), 
    caption: [The process of conditioning text data for input into a deep learning model. Text data is not intrinsically digestible by artificial neural network models, as artificial neurons can only process numerical inputs. Therefore, in order to apply deep learning models to text data, we must have some method of converting the data into a numerical format. Transformers expect a sequence of same-length vectors forming an input matrix, $X$. This diagram shows the process of converting text data into an input matrix. Typically, this conversion is completed in three steps, tokenization, vectorisation, and embedding. However, often, and in the case of the first described transformer model, vectorisation and embedding occur simultaneously,  and are often labeled simply embedding. This is the method depicted in the diagram. In the example, we see the sentence "The quick brown fox jumped over the lazy dog." as it is prepared for ingestion by an NLP model. *Tokenisation* is the process of splitting one contiguous sequence of characters into a number of unique discrete tokens, $N$. This can be done at multiple levels but is usually done at the scale of words. Sometimes, especially with longer words, words can be split into multiple tokens, as is seen in this example where the word "jumped" is split into "jump" and "ed". There are numerous algorithms to achieve this, which will not be discussed in detail. Every word, or word subset, within the training dataset, should have a unique token ID. Before running inference on new text data, that data must be tokenized, and each word in the new data will be mapped onto an existing token ID that was generated during the initial tokenisation process. Often some information-low words, known as "stop words", and punctuation are removed during the tokenisation process. In the example shown, the words "The", and full stops are removed from the input string. During *vectorisation*, each token is assigned a numerical vector, and *embedding* ensures that this vector is transformed into a meaningful vector space to allow for easier interpretation by the model. There are a number of methods to achieve both of these steps, some of which are simultaneous. In the example shown, each token ID is associated with a vector of tunable weights, as was the case in the first transformer paper. These vectors are randomised at the start of training, but as the process continues, they become tuned to values that represent the information contained by the tokens. In this manner, the vectorisation and embedding steps occur at the same time.]
) <tokenisation_diagram>

First, the input sequence must be tokenized. Tokenization involves splitting the input text into $N$ unique discrete tokens. Often the text is processed first, sometimes removing low information elements known as stop-words and punctuation. Then, using one of a variety of algorithms that will not be discussed here, the text is consolidated into $N$ tokens, often these could be whole words, but sometimes, depending on the tokenization algorithm and the size of the training data, word fragments could also be tokens. One will note that often words can have multiple meanings, which is a problem when trying to describe each token in a way that somehow presents the value it represents in a sentence, this can be a problem, and methods have been developed that can split identical sequences of characters into multiple tokens contextually. The very existence of this problem is an example of the contextual information that is provided by surrounding tokens in the sentence, this is the information that attention layers attempt to extract and distill.

After tokenization, we must convert each token into a unique vector. This can also be done through a variety of methods. In Vaswani _et al._ each token has an associated vector in a look-up table, initially these vectors are randomly generated; however, the values in each of these vectors act as tunable parameters inside the model, so that whenever a certain token is present in a particular training example, the weights of its vector are tuned through the usual gradient-descent methods. In this way, as the training progresses, the vectors become, at least to the model, meaningful numerical representations of the value contained within their associated tokens. 

Gravitational-wave data is intrinsically vectorized so the embedding layer should not be much of a problem, however, it is not intrinsically discretised. Since transformers are sequence-to-sequence models, they ingest a series of N vectors forming an input matrix, whereas gravitational-wave time series data is a single vector, at least when dealing with one detector. It is unclear how best to split the gravitational wave data into smaller vectors. We could simply cut along equally separated lines, "chunking" our data into smaller timesteps, or we could embed the data using some learned weights, for example with one or more dense or convolutional layers, in the latter case, feeding the transformer with feature slices at different timesteps; see @gw_embedding. Using different detectors as this extra dimension will only give us two to four features per timestep, which would be very small vectors for the transformer to work with.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("skywarp_chunking.png",  width: 100%) ],
        [ #image("skywarp_dense_embedding.png",  width: 100%) ],
        [ #image("skywarp_convolutional_embedding.png",  width: 100%) ]
    ),
    caption: [Different embedding possibilities to discretise and embed gravitational-wave time-series data. _Upper:_ "Chunking" method of discretisation, where the input time-series is split into $N$ equal-length segments which can be fed into an attention-based model. This method would seem to have the disadvantage that it could split the waveform at any point, leading to chunks with very different waveform content depending on the waveform offset; it also assumes that the innate interferometer output vector is a good embedding for the attention mechanism, which is not necessarily true. _Middle:_ Embedding with dense layers, this setup is similar to the chunking method, but it applies one or more dense layers to each chunk so that the model can learn an embedding that will be better adapted to the attention mechanism in subsequent layers. Since the parameters of the dense layers are repeated for each chunk, this method is equivalent to a convolutional layer with $N$ filters and no overlap, where $N$ is the size of your embedded vector output. _Lower:_ Embedding with convolutional layers. This type of embedding involves creating feature maps of the input vector using a combination of convolutional and/or pooling layers. It is the equivalent of attaching a CNN head at the front of your model. The output of a 1D CNN would be a 2D matrix where one dimension, the depth, is different features, and the other is time. This can then be split into discrete vectors by splitting it along the time dimension to create vectors of features with length equivalent to the number of features.]
) <gw_embedding>

We have managed to transform our input text from a list of symbols into discrete tokens and finally into vectors that contain some aspect of the value represented by that token, and we have some ideas about how we might do the same with gravitational-wave data. However, unlike RNNs and CNNs, transformers treat each token equally and intrinsically have no information about the location of the word in the sentence. We must use feature engineering to add to each vector, some information about the position of the token in the input sequence. 

=== Positional Encoding

Much information is embedded in the relative and absolute positions of tokens within text data. The same can said to be true of gravitational-wave data --- we would always expect the merger to come after the inspiral, for example. Whilst there is some possibility within the dense layers of a traditional CNN for the model to use this ordinal information in its classification, it might be a challenging process. We can use attention layers to look at the global information in a sequence, but since, unlike CNNs, there is no structure inherent to the architecture that maintains information about the position of the inputs, if we feed the word sequence in as is, we end up with a "bag of words"; see @bag_of_words. Whilst some models can do quite well with just a bag of words approach, able to infer context simply from the numbers of each word present, it is clear that some information is lost when discarding order.

#figure(
    image("bag_of_words.png",  width: 60%), 
    caption: [A "Bag of words". Without ordinality, the meaning represented by this sentence becomes significantly harder, if not impossible, to parse. If we had not already seen this sentence then we would not know if the fox was lazy or quick, or rather if it were the dog that was lazy or quick, and just who is jumping over whom? There are NLP models that are designed to use bag of words as inputs, but it is easy to see that much information is lost when word order is discarded, thus we can infer that the order and position of the words contain a significant amount of information. The same can be true for time series, a CBC signal that contains a merger, an inspiral, and a ringdown, in that order, can probably be discounted as a glitch, but if we feed it in as a bag of words model, there could be no distinction between this and the expected arrangement.]
) <bag_of_words>

We solve this problem by adding extra information to our input embeddings with positional encoding. To do this, we create a matrix that is the same size as our attention input matrix: [num_time_steps, num_feature_channels]. Each column in our matrix must have certain properties: it must be unique so that no two feature embeddings are given the same encoding, and it must convey information about the absolute and relative position of a given feature vector in the input sequence. We create this matrix using

$
    op("PE") (t,i) = cases(
        sin( t /(log(10000)^(i/d_"model") )) "if" i "is" "even",
        cos( t /(log(10000)^(i/d_"model") )) "if" i "is" "odd"
    )
$ <positional_encoding_eq>

where $op("PE") (t, i)$ is the positional encoding matrix, $t$ is the time index, $i$ is the feature index, and $d_"model"$ is the dimension of our model, the relevance of which will become clear later. The periodicity of sine and cosine functions enables a unique identifier for each vector whilst maintaining a consistent pattern that evolves across the time dimension. This uniqueness ensures that absolute position is encoded; all feature vectors get a unique encoding, which will be the same independent of the vector's contents so the model can learn which encodings map to which positions. The logarithmic term, $log(10000)$ ensures that the variation in frequency between steps is large enough to be detectable by the model, whereas the scaling by $d_"model"$ ensures that the positional encoding values do not become too large and overshadow the feature vectors, or become so small they are undetectable. The relative position between any two vectors in the sequence can be estimated due to the linear superposition properity of the sin and cos functions; the sum of the positional encodings will approximate $t_1 + t_2$, and the difference will approximate the difference, $t_1 - t_2$. Therefore, when the model adds or subtracts positional encodings (as it might do implicitly during training), the resulting encoding still carries meaningful positional information. This matrix is added to our sequence of input vectors by simple element-wise addition, therefore inherently encoding positional information into each feature vector. 

Fortunately, this embedding process is just as appropriate to use on gravitational wave data as it is on text data. In early testing, we found that including positional encoding improved model performance significantly.

By adding positional encoding to our input vectors, we have ensured that even if we (or a model) look at the vector in isolation we will still be able to know where in the vector it originated. So we have stored extra information within the vector, however, if we look at this new vector in isolation, there is still much contextual information provided by the rest of the sequence that we cannot access alone. If we look at the word "dog" in isolation for example, even if we knew it was the ninth word in the sequence, we would have no idea that it was lazy, or that a fox was jumping over it. To embed this kind of information, we must turn to attention layers.

=== Attention! <sec-attention>

The global information provided by an individual element within a sequence is often greater than the local information contained within the isolated element. This extra information is stored contextually within the relationship between the given element and the other elements in the sequence, both within the information stored locally by the other elements and by the relative and absolute positions of the other elements.

The set of possible combinations of elements is large, even within relatively small sequences. Therefore, in order to enable a machine learning model to extract contextual information efficiently, a method must be implemented to determine which elements contribute the most contextual information to a each element. This method is attention. Attention determines which elements in the sequence contribute highly to the global information of a given element. Once attention has been determined, global contextual information can be embedded within each element’s local information. Ideally, this process makes the output elements, now with contextual information embedded locally, easier for other machine-learning methods to interpret. 

A transformer model is a machine learning algorithm that implements this method to localise global information using attention. The output of a transformer block has the same dimensionality as the block’s input, as it retains the same number of elements. Ideally, each element has been transformed to contain a proportion of the relevent global information stored within the input sequence.

The models we describe in this section are novel in that they utilize attention mechanisms @attention_1 @attention_2, a type of differentiable memory in which a global context vector is learned over an input sequence, $X = [accent(x, arrow)_1 ... accent(x, arrow)_i ... accent(x, arrow)_n]$. The attention mechanism aims to embed global context locally; in order to do this, a comparison must be made between each element of the sequence and (in the case of self-attention) each other element of the same sequence. It is trivial to see that not every element in every sequence will be equally relevent and that this contextual dependence will depend on the information being extracted. In this way, one learns intra-sequence relations; long-term dependencies are captured because the entire input sequence is used to compute a single element of the output sequence.

The question becomes, how can we calculate the attention? We can use an analogous problem to demonstrate the principle. In search and retrieval tasks, such as a search engine query, the user, in this case, a human, must generate a *query* phrase that can be used to find relevant information. This query phrase will not contain the entire information content of whatever document we are attempting to discover, if it did then we would not need to perform the search. Instead, it is generated using words and phrases that are associated with the information we are searching for. The search engine then has the unenviable task of searching through its entire library to find documents that might have information relevant to the query. 

The first instinct might be to look through every document and check to see if there are words and phrases in that document that match the content of the query. Immediately, we can tell that this will quickly become infeasible if the library is large, and/or contains large documents --- the process of searching would rapidly become very expensive. Instead, the search engine could have preprocessed these files, and in a similar manner to how the query was generated, it could pick out the key information content of each document in a distilled form that contains the information that it is most likely to match with a query. It generates a *key* or keys for that document, which can be checked against queries much more efficiently than searching the entire content.

 Finally, the *value* of the information that the end user extracts from whatever document is returned, will not necessarily equate to the entire information content of the document. Depending on what information the user was originally searching for, and hence what query they entered into the search bar, they might only read a particular chapter of a book, or, even more specifically than that, they might only retain certain parts of information from that chapter that are relevant to their needs. During a search session, a user might enter a single query that matches well with multiple keys that return documents which the user then reads and summarises parts of the information in each document to gain new knowledge on whatever the original subject of their query was.

This analogy introduces the three key information concepts of the query, key, and value. We can use these concepts to build a deep learning layer that can, for every element of our input sequence, search through each element in the sequence and extract relevant contextual information that can then be embedded into that element, in a similar manner to how we can embed information about the elements position using positional encoding. In attention layers,  query, $accent(q, arrow)_i$, key, $accent(k, arrow)_i$, and value $accent(v, arrow)_i$ vectors are generated for each sequence element $accent(x, arrow)_i$, forming three matrices for the sequence as a whole: $Q$, $K$, and $V$. We create these matrices by multiplying three projection matrices with the input matrix, X, the query, $W_q$, key, $W_k$, and value, $W_v$, matrices. $Q$, $K$, and $V$ are generated with

$ Q = W_q X, $ <weight-1-eq>
$ K = W_k X, $ <weight-2-eq>

and

$ V = W_v X. $ <weight-3-eq>

The elements inside these weights matrices are the only tunable parameters that are learned during the model training process. During model training, the weights will adapt so that they can generate effective query, key, and value vectors that allow for proficienct model function. Since this is a neural network and these are learned weights, multiplication by these weights matricies is equivilent to application of a dense layer with no bias values.

The nature of attention layers makes it more difficult to draw artificial neuron connection diagrams as we have previously with perceptrons and CNNs, since the information flow is more complex. However, we can attempt to visualize the interaction between the various vectors as interacting functional elements, like machines in a factory, organelles in a cell, or gears in a clock; see @weights_matricies.

#figure(
    image("weights_matricies.png",  width: 70%), 
    caption: [Generation of query, key, and value vectors for each element in the input sequence of length, $N$. Before attention scores are calculated, each input vector, $accent(x, arrow)_i$ is dotted with the learned query, $W_q$, key, $W_k$, and value, $W_v$, weights projection matrices to produce a query, $accent(q, arrow)_i$, key, $accent(k, arrow)_i$, and value $accent(v, arrow)_i$ vector for the input element, $accent(x, arrow)_i$. This operation is equivalent to the multiplication of the projection matrices and the input matrix, X, to produce the query, $Q$, key $K$, and value $V$ matrices. The key takeaway is that the only tunable parameters are contained in the weights matrices, which act as projection functions to convert the input vector into functional vectors.]
) <weights_matricies>

#figure(
    image("q_k_v.png",  width: 40%), 
    caption: [Illustration of example query, key, and value vectors generated for the sentence "The quick brown fox jumped over the lazy dog.". After tokenisation and embedding, each vector in the embedded input sequence generates its own query, key, and value vector. Which together form query, key, and value matricies.]
) <qkv_diagram>

The query, key, and value matrices are used to calculate attention; see @qkv_diagram for an illustrative example of the projection matrices applied to the example sentence. The attention method aims to collect relevant information about a given sequence element within that element, extracting the information content from the position and meaning of the surrounding elements. Understandably, language does not have words for every possible concept, instead, it relies on combinations of words to provide many more concepts than single words could alone. For example, language could have developed a single word for "lazy-dog" and "quick-fox"; but you would soon end up with an extraordinarily large vocabulary (assuming that new words were invented rather than just cheating and compounding words with a hypen). If we wanted to include more complex concepts like "quick-brown-fox-that-jumped-over-the-lazy-dog" and "lazy-dog-that-has-had-quick-brown-fox-jump-over-it", the number of potential concepts becomes vast. Within the vector space, however, we are not limited by discretized tokens, and such concepts can all exist in a highly multi-dimensional space, since, in effect, we can add vectors together to sum their meanings. Attention layers essentially attempt to assemble these complex words.

In order to assemble these new vectors with embedded contextual meaning, we must work out the magnitude to which each other element affects the meaning of that element. This score is the "attention" for which the process is named. In the example sentence, "The quick brown fox jumped over the lazy dog," we can see that almost all of the concepts are somehow interacting with each other in a significant manner. If we were to extend the string however say to, "The quick brown fox jumped over the lazy dog. Incy wincy spider climbed up the water spout.", we can see that tokens in the second sentence have very little effect on the concepts in the first sentence, so we might expect the attention scores between tokens in different sentences to be much lower than in the same sentence. Now in very advanced LLMs there could be some cross-sentence attention as the model tries to determine why those two sentences in particular are next to each other, a question which could certainly hold some information, but this would be at a much higher level of abstraction than the simpler cases we have been discussing.

The query value for each sequence element is matched against the key value of each other element; see @alignment-diagram. The alignment of the key and query determines a weighting for the value vector, a distilled representation of the relevant information contained within that element; see @scaling-diagram. The weighted value vectors are then summed to produce the new, contextually embedded, element. The two most common attention methods are dot-product @attention_2 and additive attention @attention_1, our models utilise the former and so we restrict our discussion to the work of Luong _et al._ and extensions. In either case, the function $alpha$ maps a set of query $q$, key $k$, and value $v$ vectors to a weighted sum of the values. This is given by

$ alpha(q_i, K, V) = sum_(j=1)^N a(q_i, k_j)v_j $ <attention-eq>
  
where $a(., .)$ is called the alignment function and measures the similarity between the queries and keys. In the case of dot-product attention

$ a(q, k) = sigma(q^T k) $ <alignment-eq>
  
where $sigma$ is the Softmax function; see @softmax-sec.

#figure(
    image("alignment_function_diagram.png",  width: 100%), 
    caption: [Illustration of the operation of how the alignment function utilizes the query and key vectors to produce alignment scores for each sequence element. In dot-product attention, this is acheived using @alignment-eq. Note that the numbers used here are for illustratory purposes only and not extracted from a real model.]
) <alignment-diagram>

#figure(
    image("scale_and_sum.png",  width: 80%), 
    caption: [Illustation of how the alignment scores are used to scale the respective value vectors for each sequence element, and are then summed to produce a new vector which contains global information embedded contextually. Each value vector is multiplied by the respective score, and then these scaled elements are summed together to produce the new vector.]
) <scaling-diagram>

This calculation is performed on each element of the sequence to produce a new sequence of equal length, hopefully with some contextual context embedded. Generalizing @attention-eq for the entire input matrix, $X$, we get

$ alpha(Q, K, V) = sigma(Q K^T)V . $ <attention-eq-general>
  
Where again, $sigma$ is the Softmax function. Comining @attention-eq-general and with @weight-1-eq, @weight-2-eq and @weight-3-eq gives a mapping between the attention input matrix, $X$, and the attention output matrix, $Y$

$ Y = sigma((X W_q) (X W_k)^T) (X W_v). $ 

The convience that this complex proceedure can be perfomed with a few matrix multiplications is one of the reasons for its great successs. See @attention-diagram and @attention-network-diagram for illustrative diagrams.

#figure(
    image("attention_mechanism.png",  width: 100%), 
    caption: [Illustration of the operation of a single attention head. Here a very small three element sequence is examined. Each element of the original input sequence is couloured differently, in red, green, and blue. All vectors and scalors associated with a input element are coloured similarly. The output sequence vectors are coloured with a mix of the input colours to show their new information content which consists of distilled global information. More detailed descriptions of the processes shown can be found in @weights_matricies, @alignment-diagram, and @scaling-diagram.]
) <attention-diagram>

#figure(
    image("attention_network_diagram.png",  width: 80%), 
    caption: [_Upper:_ Alternate method of visualising attention mechanism as a network diagram. Although this is more similar to how networks have been displayed elsewhere in the thesis, it might obfuscate some aspects of the reasoning behind the attention layer operation. As in @attention-diagram, this illustrates the operation of the attention mechanism on a sequence of length three, with each input vector coloured differently, in red, green, and blue. In this representation, the projection matricies, $W_q$, $W_k$, and $W_v$, are represented as dense layers, which are applied to each of the column vectors that comprise the input matrix in turn. It should be noted that although the dense layers are coloured differently as they applied to each input element, this is just to show the different data flows, the weights are maintained by each application of each dense layer. The key, query, and value dense layers however, have different weights, and notably, no activation function, as they are just supplying a linear mapping rather than any more complex behaviour. _Lower:_ Abstraction of a single attention head layer, that will be used in future diagrams of model which contain attention layers, in order to limit model complexity.]
) <attention-network-diagram>

=== Multi-Head Attention

Thus far, the process we have described is the operation performed in a single attention head. We have worked under the assumption that all contectual infomation can be embedded locally with one pass from one head. In reality this is not true, except for in trivially simple sequences it would not be posissible to emebed all global information in one pass. In a similar manner to convolutional filters, wherein each filter looks at a particular feature of the input data, an attention layer typically has multiple-heads each which focus on a particular information feature. One could look at colour for example, whilst another focuses on punctuation (if not removed in tokenisation), or sentence structure. 

In multi-head attention layers the number of heads is a user specified hyperparameter, N, just like the number of filters in a convolutional layer. Each head has independent weights for query, $W_q$, key, $W_k$, and value, $W_v$, projection matrices, which are each tuned to find specific features in the data. After these heads have been appplied the output is concatanated along the feature dimension, and then multiplied by a further weights matrix, used to mix the outputs of different heads together and to reshape the output vector to a desired size, which does not neccisarily have to be the same size as the input vector, though this is a common choice; see @multi-head-diagram for a representation of a multi-attention head.

It should be noted that in practice, the entire query key and value matricies for each head are calulated at once with the same large weights matricies which produce output matrices that can then be split along the feature vectors and fed into the individual heads for the alignment scores calculation and vector summation. This is done to reduce the number of matrix multiplications required for the layer as a whole.

#figure(
    image("multi_attention_head.png",  width: 80%), 
    caption: [_Upper:_ Network diagram of multi-attention head. Similar to how multiple convolutional kernels work in tandem in convolutional layers, multiple attention heads work together in multi-attention heads to focus on different information aspects of the input vector. These are then concatanated along the feature axis before finally being multiplied by a further weights matrix, here shown as a dense layer, which serves to mix the output of the different heads and to reshape the output to a desired size. _Lower:_ Abstraction of a multi-head attention layer, that will be used in future diagrams of model which contain attention layers.]
) <multi-head-diagram>

=== Attention Blocks

Within transformers and other similar architectures, multi-head attention layers are often paired with a number of complementary layers within a residual block. The input and output matricies of this block are usually have identical shape so that the block can be repeated, $N$ times without having any intermediate reshaping layers. Attention blocks typically feature a number of dense layers with activation functions in order to perform non-linear computation, regularisation methods such as dropout and batch normalisation, and a residual skip connection wherein the block input is added to the block output, in order to reduce the vanishing gradient problem that can occour in very deep networks; see @attention-block-diagram.

#figure(
    image("attention_block.png",  width: 100%), 
    caption: [Typical atention block comprising multiple layers. Residual attention blocks vary in design between architecures but usually maintain the consistent elements shown. The skip connection is here represented by the encirciling arrow, which shows that the input of the block is fed to the output before it is returned. There are also several regularisation methods present, batch normalisation, and dropout. Finally, the addition of dense layers and activation functions ensures that non linear computation can be performed. Sometimes, if a reduction in total model parameter count and inference time is required, convolutional layers can be used in place of dense layers. ]
) <attention-block-diagram>

=== Transformers

Since their introduction, attention mechanisms have been utilised in a number of different neural network architectures, including transformers and stable diffusion models. Transformers were first proposed by (Vaswani et al. 2017) to solve natural-language processing tasks, showing significant improvement over previous recurrent and convolutional architectures. For these reasons, we decided to investigate a fully attention-based model, inspired by a Transformer encoder.

The transformer model uses the attention mechanism described earlier in section @sec-attention within discrete blocks called multi-attention heads. Multi-attention heads have N multiples of the weights matrices used to generate query, key, and value matrices from input sequences. These multiple heads can be thought of analogously to different convolutional filters inside a CNN layer; each head can focus on extracting a different type of contextual information from the sequence. This is necessary as the useful contextual information embedded within a sequence can be more complex than it is possible to extract with a single attention head. The output sequences of all N heads are merged after the block to ensure that the output and input sizes are the same.

Often, as is the case for our models, the multi-attention heads are followed by normalisation and one or more dense layers. These blocks of layers can then be stacked to form components of a transformer.

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
