= CrossWave: Cross-network Attention for the Detection and Parameterisation of Overlapping Compact Binary Coalescence Signals <crosswave-sec>

Thus far, we have focused our attention on perhaps one of the simpler problems in gravitational-wave data analysis, transient detection; the fact remains, that there are many, more complex, tasks that are yet to be satisfactorily solved. One of the largest and most intriguing of these is parameter estimation. Whilst detection merely identifies the presence of a signal, and, in a modeled search, tells us the type of signal we have detected, there is invariably other scientifically valuable information that can be extracted from a signal. In fact, without further analysis, detection alone is useful for little more than rudimentary population analysis; parameter estimation is a crucial part of gravitation wave science, without which the endeavor fails to make as much sense.

This section does not focus on a general parameter estimation method for either CBC or burst signals, those have both been well investigated, and although there is arguably, more room for improvement and a need for innovation, on these fronts than in detection alone it was not within the scope of this work. In this section, we present work on a much smaller subproblem within parameter estimation; the detection and isolation, of overlapping waveforms, because of the somewhat limited nature of the problem, it is yet to be studied as thoroughly, as any of the other problems we have yet examined, which, in some ways, gives us more room for improvement.

Significant improvements to our gravitational wave detection capability are anticipated within the next decade, with improvements to existing detectors [cite], as well as future 3rd and 4th generation space and ground-based detectors such as the Einstein Telescope [cite] and Cosmic Explorer. Whilst the current rate of Compact Binary Coalescence (CBC) detection is too low (estimate) for any real concern about the possibility of overlapping detections, estimated rates for future networks (estimate) will render such events a significant percentage of detections. Contemporary detection and parameter pipelines do not currently have any capabilities to deal with overlapping signals --- and although, in many cases, detection would still be achieved [cite], parameter estimation would likely be compromised by the presence of the overlap, especially if more detailed information about higher modes and spins [cite] are science goals.

We introduce CrossWave, two attention-based neural network models∂ for the identification and parameter estimation of overlapping CBC signals. CrossWave consists of two complementary models, one for the separation of the overlapping case from the non-overlapping case and the second as a parameter estimation follow-up to extract the merger times of the overlapping signals in order to allow other parameter estimation methods to be performed. CrossWave can differentiate between overlapping signals and lone signals with efficiencies matching that of more conventional matched filtering techniques but with considerably lower inference times and computational costs. We present a second parameter estimation model, which can extract the merger times of the two overlapping CBCs. We suggest CrossWave or a similar architecture may be used to augment existing CBC detection and parameter estimation infrastructure, either as a complementary confirmation of the presence of overlap or to extract the merger times of each signal in order to use other parameter estimation techniques on the separated parts of the signals.

== CrossWave Method

Since the CrossWave project was an exploratory investigation rather than an attempt to improve the results of a preexisting machine learning method, it has a different structure to the Skywarp project. Initially, we applied architecture from the literature, again taking Gabbard _et al._ @gabbard_messenger_cnn, with architecture illustrated here @gabbard_diagram. This worked effectively for the differentiation of overlapping and lone signals. We named this simpler method was called OverlapNet. However, when attempting to extract the signal merger times from the data, we found this method to be inadequate, therefore, we utilized the attention methods described in @skywarp-sec, along with insights gained over the course of other projects to construct a more complex deep network for the task, seen in @crosswave-large-diagram. We name this network CrossWave, as it utilises cross attention between a pair of detectors. It is hoped that this architecture can go on to be used in other problems, as nothing in its architecture, other than its output features, has been purpose-designed for the overlapping waveform case.

=== Crosswave Training, Testing, and Validation Data <crosswave-data>

The dataset utilised in this section differs from previous sections, in that it was not generated using the GWFlow data pipeline. Since this was part of a parameter estimation investigation, the exact morphology of the waveforms injected into the signal are crucial to validating performance. The cuPhenom IMPhenomD waveform generator that was developed for rapid waveform generation on the GPU has a relatively high degree of mismatch (~5%) with IMPhenomD signals generated with LALSimulation in some areas of parameter space. This is thought primarily to be down to reduced precision operation (32-bit in most areas rather than 64-bit) and the lack of implementation of some post-Fourier conditioning steps. Whilst this mismatch was deemed to be mostly adequate for detection searches, especially for comparison of methods, we considered it inadequate for parameter estimation tasks. IMRPhenomD is also an older waveform aproximant, which does not take into consideration, the latest improvements to waveform aproximation and some physical phenomena, such as higher modes. Whilst there is currently no one waveform approximant that can generate waveforms that include all physical effects, we opted to use IMRPhenomTPHM, which is a Time-Domain approximant that includes the physics of precession, which allows for studies of Higher Modes.

A static dataset was created using BBH waveforms generated using LALSimulation and injected into Gaussian noise coloured by the LIGO Hanford and LIGO Livingston aLIGO design specifications using the technique described in @noise_acquisition_sec but not with the GWFlow pipeline. No BNS signals were considered. We used a #box("16" + h(1.5pt) + "s") on-source duration, to allow more space for different signal start times and to examine the effects of distant signal overlap on parameter estimation. We used a sample rate of #box($1024$ + h(1.5pt) + "Hz"), as this was considered adequate to contain the vast majority of relevant frequency content for the CBCs examined.

Unlike in the detection case, wherein our training distribution consisted of some examples with obfuscated signals and some consisting of pure noise. For this case, we assume that a detection has already been made by a detection pipeline, so our examples always contain signal content of some kind. This assumption was made to simplify the task to its minimal possible case. Our generated waveform bank consisted of $20^5$ IMRPhenomTPHM approximants. From that template bank, we constructed $20^5$ of lone signals injected into obfuscated noise and $20^5$ pairs of signals injected into obfuscated noise. In the latter case, each waveform was unique to a single pair, generating $10^5$ pairs, but each pair was injected into two different noise realizations in order to generate identical numbers of lone and paired templates. The use of the same waveforms in both the single case and the pairs was a conscious decision that was made in order to attempt to reduce the change of the network overfitting to any particular signal morphology.

The waveforms were generated with a wide parameter range uniformly drawn from across parameter space. The primary component of each waveform was generated with masses between #box("10.0" + h(1.5pt) + $M_dot.circle$) and #box("70.0" + h(1.5pt) + $M_dot.circle$), this is notably inconsistent with our previous studies, but was reduced to reduce task complexity and because this still covers most of the range that is of interest to parameter estimation studies. This also ensured that their visual duration, between #box("20.0" + h(1.5pt) + "Hz"), which is both the whitening low-pass filter and around the limit that the detector design curve starts to make detection impossible, remained well contained within the #box("16" + h(1.5pt) + "s") on-source duration. Also unlike in our previous detection studies, the mass ratio was constrained between 0.1 and 1. Since the approximants were generated in an alternate method utalising luminosity distance as the scaling factor rather than SNR, the SNRs are not uniformly distributed, however, the Network SNR of any signal is not less than 5 or greater than 100. For each injection luminosity distance in MPc was drawn from a power law distribution with base two scaled by 145, with a minimum distance of #box("5.0" + h(1.5pt) + "MPc"), this luminosity distance range was generated by a trial and error approach to achieve the desired SNR distribution. 

In the case of the pairs of injection, the two waveforms are injected so that their merger times never have a separation exceeding #box("2" + h(1.5pt) + "s"). Signal A is defined as the signal that arrives first, allowing the model to differentiate between the two signals for the parameter estimation tasks. When only one waveform is present, that waveform is labeled signal A.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("overlap_example.png", width: 100%) ],
        [ #image("single_example.png", width: 100%) ]
    ),
    caption: [Two illustrative examples of data used to train CrossWave, the upper in the single signal case, the lower in the multiple signal case. Since the real data used to train CrossWave was unwhitened, it is not easy to parse by eye. Thus, as an illustrative example, these two examples are shown in whitened data generated using cuPhenom and GWFlow. The example duration has also been cropped from #box("16" + h(1.5pt) + "s") to #box("5" + h(1.5pt) + "s"), since the merger times never have a separation greater than #box("2" + h(1.5pt) + "s") this is ample as an example. Both examples show time series from both detectors, simulating LIGO Livingstone and LIGO Hanford. _Upper:_ Single waveform injected into noise drawn from the two LIGO detectors. _Lower:_ A pair of waveforms injected into noise drawn from the two LIGO detectors. The waveforms are always injected with merger times less than #box("2" + h(1.5pt) + "s") distant.] 
) <overlap_injection_examples>

Interestingly, since the data was generated independently, it was not whitened prior to model injection. Since this is not a comparison to another machine learning method that uses whitening, this is not particularly an issue, but it also can't tell us about the efficacy we have lost/gained due to the lack of whitening. Since this investigation does have positive results, this could potentially be an area for future experimentation, forgoing the whitening step before ingestion by a model would streamline a lot of the problems faced by low-latency machine learning pipelines.

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

== Overlapnet

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

=== Arrival Time Parameter Estimation:

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

=== Other Parameter Results

=== Physicallised Intuition
