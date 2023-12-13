= CrossWave: Cross-network Attention for the Detection and Parameterisation of Overlapping Compact Binary Coalescence Signals <crosswave-sec>

Thus far, we have focused our attention on perhaps one of the simpler problems in gravitational-wave data analysis, transient detection; the fact remains, that many, more complex, tasks are yet to be satisfactorily solved. One of the largest and most intriguing of these is Parameter Estimation (PE). Whilst detection merely identifies the presence of a signal, and, in a modeled search, tells us the type of signal we have detected, there is invariably other scientifically valuable information that can be extracted from a signal. During PE, we attempt to predict, with error margins, several parameters about a gravitational-wave-producing system. Typically this is a CBC system, although PE could also be performed on burst events if they were to be detected. Fortunately, CBCs can be described quite well in as few as 14 parameters that contain information both about the state of a CBC system, known as intrinsic parameters, and its relation to us as observers, known as extrinsic parameters. Care should be taken to distinguish between the parameters being extracted by PE, and the parameters of a neural network model, as they are unrelated.

Without further analysis, detection alone is useful for little more than rudimentary population analysis; PE is a crucial part of gravitation wave science. Extrinsic parameters, like the source distance and sky location, aid in population studies and multimessenger analysis, and intrinsic parameters such as the companion mass and spin properties can help unveil information about the sources themselves.

This section does not focus on a general PE method for either CBC or burst signals. Those have both been well investigated and although there is arguably a greater room for improvement and a larger need for innovation on these fronts than in detection alone it was not within the scope of this work. In this section, we present an analysis of a much smaller subproblem within PE; the detection and isolation, of overlapping waveforms. Because of the somewhat limited nature of the problem, it has not been studied as thoroughly as any of the other problems we have yet examined, which, in some ways, gives us more space for exploration, and an increased potential for novel scientific output.

== Frequency of Overlapping Compact Binary Coalescences (CBCs)

Significant improvements to our gravitational wave detection capability are anticipated within the next decade, with improvements to existing detectors such as LIGO-Voyager @LIGO_Voyager, as well as future 3#super("rd") and 4#super("th") generation space and ground-based detectors such as the Einstein Telescope (ET) @einstein_telescope, Cosmic Explorer (CE) @cosmic_explorer, and the Laser Interferometer Space Antenna (LISA) @LISA. Whilst the current detection rate ($1~2 space "week"^(-1)$ [BBHs]) and average detectable duration ($~7 s$ [BBHs]) of Compact Binary Coalescence (CBC) detection is too low for any real concern about the possibility of overlapping detections @bias_study_one, estimated detection rates ($50~300 space "week"^(-1)$ [BBHs]) and durations ($40~20000 s$ [BBHs]) for future networks will render such events a significant percentage of detections @bias_study_one, see @overlaping-event-rate for a more detailed breakdown of overlapping event estimates. Contemporary detection and parameter pipelines do not currently have any capabilities to deal with overlapping signals --- and although, in many cases, detection would still be achieved @bias_study_one @bias_study_two, PE would likely be compromised by the presence of the overlap, especially if more detailed information about higher modes and spins @bias_study_one are science goals.


#figure(
  table(
    columns: (auto, auto, auto, auto, 80pt, 70pt, auto),
    inset: 10pt,
    align: horizon,
    [*Configuration*],  [*Range (MPc)*], [*Cut Off (Hz)*], [*Mean Visible Duration (s)*], [*P(Overlap) ($"year"^(-1)$)*], [*$N_"events"$ ($"year"^(-1)$)*], [*$N_"overlaps"$ \ ($"year"^(-1)$)*],
    [aLIGO: O3], [611.0], [20], [6.735], [$3.9_(-1.3)^(+1.9) times 10^(-6)$], [$42.0_(-13.0)^(+21.0)$], [$0.0_(-0.0)^(+0.0)$],
    [aLIGO: O4], [842.5], [20], [6.735], [$1.0_(-0.3)^(+0.5) times 10^(-5)$], [$100.0_(-29.0)^(+56.0)$], [$0.0_(-0.0)^(+0.0)$],
    [aLIGO: Design], [882.9], [20], [6.735], [$1.2_(-0.4)^(+0.6) times 10^(-5)$], [$120.0_(-38.0)^(+60.0)$], [$0.0_(-0.0)^(+0.0)$],
    [LIGO-Voyager], [2684.0], [10], [43.11], [$2.3_(-0.8)^(+1.2) times 10^(-3)$], [$2700.0_(-38.0)^(+60.0)$], [$6.3_(-3.4)^(+7.7)$],
    [Einstein Telescope], [4961.0], [1], [19830.0], [$1.0_(-0.0)^(+0.0)$], [$15000.0_(-5000.0)^(+7100.0)$], [$1.0_(-0.0)^(+0.0)$],
  ),
  caption: [Estimated overlap rates of BBH signals in current and future detectors, sourced from Relton @phil_thesis. Presented error values are 90% credible intervals. Note that these are estimates of rates, including for past observing runs rather than real values, and are meant only as an illustration of the difference in overlap rates between current and future detector configurations. The number of overlapping signals, $N_"overlap"$, anticipated within one year is determined by the number of detections, $N_"events"$, and the visible duration of those detections, which are, in turn, affected by the detection range and lower frequency cut off of that detector configuration. We can see that although with the current and previous detector configurations an overlapping event is extremely unlikely, it will increase with LIGO-Voyager to the point where we would expect $6.3_(-3.4)^(+7.7)$ per year, and further increase with the Einstein telescope to the point where we would not expect any event to be detected without components of other signals also present in the detector. ]
) <overlaping-event-rate>

== Detection and Parameter Estimation (PE) of Overlapping Compact Binary Coalescences (CBCs)

Two studies examined the rate at which overlaps were likely to occur with different detector configurations and the effect of overlapping signals on PE. Samajdar _et al._ @bias_study_one, determined that during an observing period of the future Einstein telescope, the typical BNS signal will have tens of overlapping BBH signals and that there will be tens of thousands of signals that have merger times within a few seconds of each other. They found that for the most part, this had little effect on parameter recovery except in cases where a short BBH or quiet BNS overlapped with a louder BNS signal. Relton and Raymond @bias_study_two performed a similar study and produced the overlap estimates seen in @overlaping-event-rate. They found that PE bias was minimal for the larger of the two signals, when the merger time separation was greater than #box($0.1$ + h(1.5pt) + "s") and when the SNR of the louder signal was more than three times that of the quieter signal. This bias was also smaller when the two signals occupied different frequency regions, and when the louder of the two signals appeared first in the detector stream. Despite this, they found evidence of PE bias even when the smaller signal was below the detectable SNR threshold. They found that overlapping signals can mimic the effects of procession, it will be important to be able to distinguish the two when detailed procession analysis becomes possible.

Much of the work in this area focuses on performing PE with overlapping signals, and there has not been as much attention to distinguishing pairs of mergers from single mergers. Relton _et al._ @overlapping_search measured the detection performance of both a modeled (PyCBC) @pycbc and unmodeled (coherent WaveBurst [cWB]) @cWB search pipeline when searching for overlapping signals. They determined that both pipelines were able to recover signals with minimal efficiency losses ($<1%$) although they noted that the clustering algorithm used in both pipelines was inadequate to separate the two events, noting that cWB could detect both events. They concluded that adjustments to clustering could be made to both pipelines in order to return both events given a sufficient merger time separation. Using these altered pipelines it would then be possible to separate the data into two regions, which could be used for independent PE.

Once an overlapping single has been identified, the next step is to deal with PE. Although in many cases, existing PE techniques may provide results with little bias @bias_study_one @bias_study_two, there are some situations in which this may not be the case. If the PE method can be improved in order to reduce that bias, it is useful in any case, so long as it does not result in a reduction of PE accuracy that is greater than the bias introduced by the overlapping signal.

There are four types of methods we can apply to alleviate the issues with PE @phil_thesis. 

+ *Global-fit* methods attempt to fit both signals simultaneously. There have been several studies investigating this method by Antonelli _et al._ @global_fit, which attempts to apply it to both Einstein Telescope and LISA data, @hieherachical_overlapping_pe_2 which compares this method to hierarchical subtraction, and several studies focusing solely on LISA data @lisa_global_1 @lisa_global_2 @lisa_global_3. This has the advantage of being somewhat a natural extension of existing methods, with no special implementation simply an increased parameter count, but that can also be its greatest disadvantage. The total number of parameters can quickly become large when an overlap is considered, especially if multiple overlaps are present which will be expected to occur in ET and LISA data.

+ *Local-fit* methods attempt to fit each signal independently and correct for the differences. The original proposal by Antonelli _et al._  @global_fit suggests using local fits to supplement a global-fit approach. Evidently, this will reduce the number of parameters that you require your method to fit, but its efficacy is highly dependent on the efficacy of your correction method.

+ *Hierarchical Subtraction* methods suggest first fitting to the most evident signal, then subtracting the signal inferred from your original data and repeating the process @hiherachical_subtration_overlapping_pe @hieherachical_overlapping_pe_2. This method would be effective at subtracting multiple sets of parameters for overlapping signals, assuming that the overlap does not cause bias in the initial fit, which the previously mentioned studies have shown is not always a correct assumption @bias_study_one @bias_study_two. This method can also be augmented with machine learning methods

+ Finally, and most relevantly, *machine learning* methods can be employed as a global fit technique to try and extract parameters from overlapping signals. They come with all the usual advantages, (inference speed, flexibility, computational backloading) and disadvantages (lack of interpretability, unpredictable failure modes). Langendorff _et al._ @machine_learning_overlapping_pe attempt to use Normalising Flows to output estimations for parameters.

Most of the methods mentioned benefit from having prior knowledge about each of the pairs of signals, especially the merger times of each signal. As well as acting as a method to distinguish between overlapping and lone signals, CrossWave was envisioned as a method to extract the merger times of each of the binaries in order to assist further PE techniques. Crosswave was able to achieve this and also demonstrated some more general, but limited PE abilities.

== CrossWave Method

We introduce CrossWave, two attention-based neural network models for the identification and PE of overlapping CBC signals. CrossWave consists of two complementary models, one for the separation of the overlapping case from the non-overlapping case and the second as a PE follow-up to extract the merger times of the overlapping signals in order to allow other PE methods to be performed. CrossWave can differentiate between overlapping signals and lone signals with efficiencies matching that of more conventional matched filtering techniques but with considerably lower inference times and computational costs. We present a second PE model, which can extract the merger times of the two overlapping CBCs. We suggest CrossWave or a similar architecture may be used to augment existing CBC detection and PE infrastructure, either as a complementary confirmation of the presence of overlap or to extract the merger times of each signal in order to use other PE techniques on the separated parts of the signals.

Since the CrossWave project was an exploratory investigation rather than an attempt to improve the results of a preexisting machine learning method, it has a different structure to the Skywarp project. Initially, we applied architecture from the literature, again taking Gabbard _et al._ @gabbard_messenger_cnn, with architecture illustrated here @gabbard_diagram. This worked effectively for the differentiation of overlapping and lone signals. We named this simpler method was called OverlapNet. However, when attempting to extract the signal merger times from the data, we found this method to be inadequate, therefore, we utilized the attention methods described in @skywarp-sec, along with insights gained over the course of other projects to construct a more complex deep network for the task, seen in @crosswave-large-diagram. We name this network CrossWave, as it utilises cross attention between a pair of detectors. It is hoped that this architecture can go on to be used in other problems, as nothing in its architecture, other than its output features, has been purpose-designed for the overlapping waveform case.

=== Crosswave Training, Testing, and Validation Data <crosswave-data>

The dataset utilized in this section differs from previous sections, in that it was not generated using the GWFlow data pipeline. Since this was part of a PE investigation, the exact morphology of the waveforms injected into the signal is crucial to validating performance. The cuPhenom IMPhenomD waveform generator that was developed for rapid waveform generation on the GPU has a relatively high degree of mismatch (~5%) with IMPhenomD signals generated with LALSimulation in some areas of parameter space. This is thought primarily to be down to reduced precision operation (32-bit in most areas rather than 64-bit) and the lack of implementation of some post-Fourier conditioning steps. Whilst this mismatch was deemed to be mostly adequate for detection searches, especially for comparison of methods, we considered it inadequate for PE tasks. IMRPhenomD is also an older waveform aproximant, which does not take into consideration, the latest improvements to waveform aproximation and some physical phenomena, such as higher modes. Whilst there is currently no one waveform approximant that can generate waveforms that include all physical effects, we opted to use IMRPhenomTPHM, which is a Time-Domain approximant that includes the physics of precession, which allows for studies of Higher Modes.

A static dataset was created using BBH waveforms generated using LALSimulation and injected into Gaussian noise coloured by the LIGO Hanford and LIGO Livingston aLIGO design specifications using the technique described in @noise_acquisition_sec but not with the GWFlow pipeline. No BNS signals were considered. We used a #box("16" + h(1.5pt) + "s") on-source duration, to allow more space for different signal start times and to examine the effects of distant signal overlap on PE. We used a sample rate of #box($1024$ + h(1.5pt) + "Hz"), as this was considered adequate to contain the vast majority of relevant frequency content for the CBCs examined.

Unlike in the detection case, wherein our training distribution consisted of some examples with obfuscated signals and some consisting of pure noise. For this case, we assume that a detection has already been made by a detection pipeline, so our examples always contain signal content of some kind. This assumption was made to simplify the task to its minimal possible case. Our generated waveform bank consisted of $20^5$ IMRPhenomTPHM approximants. From that template bank, we constructed $20^5$ of lone signals injected into obfuscated noise and $20^5$ pairs of signals injected into obfuscated noise, totaling $40^5$ training examples. In the latter case, each waveform was unique to a single pair, generating $10^5$ pairs, but each pair was injected into two different noise realizations in order to generate identical numbers of lone and paired templates. The use of the same waveforms in both the single case and the pairs was a conscious decision that was made in order to attempt to reduce the change of the network overfitting to any particular signal morphology.

The waveforms were generated with a wide parameter range uniformly drawn from across parameter space. The primary component of each waveform was generated with masses between #box("10.0" + h(1.5pt) + $M_dot.circle$) and #box("70.0" + h(1.5pt) + $M_dot.circle$), this is notably inconsistent with our previous studies, but was reduced to reduce task complexity and because this still covers most of the range that is of interest to PE studies. This also ensured that their visual duration, between #box("20.0" + h(1.5pt) + "Hz"), which is both the whitening low-pass filter and around the limit that the detector design curve starts to make detection impossible, remained well contained within the #box("16" + h(1.5pt) + "s") on-source duration. Also unlike in our previous detection studies, the mass ratio was constrained between 0.1 and 1. Since the approximants were generated in an alternate method utalising luminosity distance as the scaling factor rather than SNR, the SNRs are not uniformly distributed, however, the Network SNR of any signal is not less than 5 or greater than 100. For each injection luminosity distance in MPc was drawn from a power law distribution with base two scaled by 145, with a minimum distance of #box("5.0" + h(1.5pt) + "MPc"), this luminosity distance range was generated by a trial and error approach to achieve the desired SNR distribution.  An overview of the parameters used to train both the CrossWave and Overlapnet models is shown in @overlap_injection_examples.

A validation dataset was also generated with independent signals and background noise, with $20^4$ singles and $20^4$ pairs generated similarly to the training data, totaling $40^4$ validation examples.

#figure(
  table(
    columns: (auto, auto),
    inset: 10pt,
    align: horizon,
    [*Hyperparameter*],  [*Value*],
    [Batch Size], [32],
    [Learning Rate], [10#super("-4")],
    [Optimiser], [ Adam ],
    [Scaling Method], [Luminosity Distance],
    [Min Luminosity Distance], [5.0],
    [Max Luminosity Distance], [N/A],
    [Luminosity Distance Distribution], [$ ("Power-Law (base 2)" times 145) + 5 "MPc"$ ],
    [Data Acquisition Batch Duration], [ N/A ],
    [Sample Rate], [ #box("1024.0" + h(1.5pt) + "Hz")],
    [On-source Duration], [ #box("16.0" + h(1.5pt) + "s")],
    [Off-source Duration], [ N/A ],
    [Scale Factor], [10#super("21") ],
    
  ),
  caption: [The training and dataset hyperparameters used in Crosswave experiments. ]
) <skywarp-training-parameters>

In the case of the pairs of injection, the two waveforms are injected so that their merger times never have a separation exceeding #box("2" + h(1.5pt) + "s"). "Signal A" is defined as the signal that arrives first, whereas "Signal B" is always defined as the signal that arrives second. This allows the model to differentiate between the two signals for the PE tasks. When only one waveform is present, that waveform is labeled "Signal A".

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("single_example.png", width: 80%) ],
        [ #image("overlap_example.png", width: 80%) ]
    ),
    caption: [Two illustrative examples of data used to train CrossWave, the upper in the single signal case, the lower in the multiple signal case. Since the real data used to train CrossWave was unwhitened, it is not easy to parse by eye. Thus, as an illustrative example, these two examples are shown in whitened data generated using cuPhenom and GWFlow. The example duration has also been cropped from #box("16" + h(1.5pt) + "s") to #box("5" + h(1.5pt) + "s"), since the merger times never have a separation greater than #box("2" + h(1.5pt) + "s") this is ample as an example. Both examples show time series from both detectors, simulating LIGO Livingstone and LIGO Hanford. _Upper:_ Single waveform injected into noise drawn from the two LIGO detectors. _Lower:_ A pair of waveforms injected into noise drawn from the two LIGO detectors. The waveforms are always injected with merger times less than #box("2" + h(1.5pt) + "s") distant.] 
) <overlap_injection_examples>


==== A note on Whitening

Interestingly, since the data was generated independently, it was not whitened prior to model injection. Since this is not a comparison to another machine learning method that uses whitening, this is not particularly an issue, but it also can't tell us about the efficacy we have lost/gained due to the lack of whitening. Since this investigation does have positive results, this could potentially be an area for future experimentation, forgoing the whitening step before ingestion by a model would streamline a lot of the problems faced by low-latency machine learning pipelines. 

This may also have some benefits in the case of overlapping signal detection and PE. Because it is expected to only become relevant in the regime of very long-lived signals, it may be difficult to get clean off-source data at a close enough time separation from the on-source data.

== Overlapnet Results

The first attempt to classify input examples generated with the method described in @crosswave-data utilized an architecture from the literature taken from Gabbard _et al._ @gabbard_messenger_cnn, the model architecture of this model can be seen at @gabbard_diagram. To distinguish this model from later models, this model was named Overlapnet. We trained a binary classifier to output a score near or equal to one if there were two signals present in the input data, and a score near or equal to zero if there was only one signal in the data.

Since data for this experiment was generated independently, validation was also performed alternately. Since we are assuming the presence of at least one signal, in either case, the problem is not hugely asymmetric as, in the CBC detection case, the penalty for incorrectly classifying a single signal as a double is much less than for classifying noise as a signal. This means we can keep our classification threshold at 0.5 and focus on optimizing our model to gain as high accuracy as possible, without needing performance in extremely low FAR regimes. This means that FAR plots are not particularly useful.

The trained model was run over the validation dataset consisting of $40^4$ examples generated independently but with the same method as the training data. The parameters for each waveform were recorded and compared with the classification results. 

This initial attempt at applying a preexisting model from the literature to the problem proved sufficient even in unwhitened noise. The model was able to correctly classify most examples where the optimal network SNR of Signal B was over 12. See @overlapnet_classification_scores.

#figure(
    image("overlapnet_classification_results.png",  width: 50%),
    caption: []
) <overlapnet_classification_scores>

#figure(
    grid(
        columns: 2,
        rows:    1,
        gutter: 1em,
        [ #image("overlapnet_signal_time_error_a.png", width: 100%) ],
        [ #image("overlapnet_signal_time_error_b.png", width: 100%) ]
    ),
    caption: []
) <overlapnet_regression_scores>

== Cross-Attention

== CrossWave

#set page(
  flipped: true
)
#set align(center)
#image("crosswave_small_diagram_expanded.png",  width: 85%)
#figure(
    image("crosswave_large_diagram_expanded.png",  width: 96%),
    caption: [],
) <crosswave-large-diagram>
#set align(left)

#set page(
  flipped: false
)

=== Arrival Time Parameter Estimation (PE):

#figure(
    grid(
        columns: 2,
        rows:    1,
        gutter: 1em,
        [ #image("error_vs_arrival_time.png", width: 100%) ],
        [ #image("binary_arrival_time_difference.png", width: 100%) ]
    ),
    caption: []
) <crosswave_regression_scores>

=== Other Parameter Estimation Results

=== Physicallised Intuition
