= Application

#set math.equation(numbering: "(1)")
We have demonstrated that simple artificial neural networks can be used to classify input data drawn from a restricted distribution into a number of classes, $N$, with a high ($>99.9%$) degree of accuracy. Because we didn't design the network with any particular consideration for the dataset (besides the dimensionality of its elements), we can infer that this method should be general enough to classify data drawn from other distributions that contain discrete differentiable classes. It is not clear, however, which other distributions can be classified and what magnitude of network is required to achieve a similar degree of accuracy. It is easy to imagine distributions that are considerably simpler than the MNSIT dataset @mnist and, conversely, ones that are much more complex. There may be a mathematical approach to determine the link between the distribution and required model complexity. One possible metric that touches upon this relation is the Rademacher complexity, $accent(cal(R), hat)_M$, given by 

$ accent(cal(R), hat)_M (H) = EE_in  [1/M sum_(i=1)^M epsilon_i h [accent(x, arrow)_i]] , $
 
where $M$ is the number of data points in a dataset $X = [accent(x_1, arrow), ..., accent(x_i, arrow), ..., accent(x_M, arrow)]$ where each point is a vector, $accent(x, arrow)$, in our case the input vectors of our training dataset, $accent(epsilon, arrow) = [accent(epsilon_1, arrow), ..., accent(epsilon_i, arrow), ..., accent(epsilon_M, arrow)]$ uniformly distributed in ${-1, +1}^m$, and $H$ is a real-valued function class, in our case the set of functions that can be approximated by our chosen neural network architecture. The Rademacher complexity is a measure of how well functions in $H$ can fit random noise in the data. A higher Rademacher complexity indicates that the function class can fit the noise better, which implies a higher capacity to overfit to the data. So, one approach to optimising the model would be to attempt to minimise Rademacher complexity value whilst maximising model performance. More details about this metric and its use in defining the relationship between data samples and model complexity can be found at @model_complexity. Despite the existence of this metric, however, it would appear that there has not been substantial research into the link between dataset complexity and required model size @model_complexity_2, though it is possible that such a paper has been missed.

One method that we can use to explore this question is to find out the answer empirically. As we move from the MNIST datase @mnist to distributions within gravitational-wave data science, the natural starting point is to repeat the previous experiments with gravitational-wave data, both as a comparison and as a baseline as we move forward. The following subsection will explore the possible areas for detection applications, describe the example datasets and their construction, and explore the results of repeating the previous experiments on gravitational-wave data, comparing our results to similar attempts from the literature. By the end of this chapter, we will have accumulated a large number of possible network, training, and data configurations. These form the set of hyperparameters that we must, though some approach, narrow down; we will explore how we can do this in @dragonn-sec.

== Gravitational-Wave Classifiers

The scope of gravitational-wave data problems to which we can apply artificial neural network models is large @ml_in_gw_review. However, we shall limit our investigation to perhaps the simplest type of problem --- classification, the same type of problem that we have previously demonstrated with our classification of the MNIST dataset @mnist into discrete classes. Though classification is arguably the most straightforward problem available to us, it remains one of the most crucial --- before any other type of transient signal analysis can be performed, transients must first be identified.

There are several problems in gravitational-wave data analysis which can be approached through the use of classification methods. These can broadly be separated into two classes  --- detection and differentiation. *Detection* problems are self-explanatory; these kinds of problems require the identification of the presence of features within a noisy background. Examples include Compact Binary Coalescence (CBC) @gabbard_messenger_cnn @pycbc, burst @cWB @MLy, and glitch detection @gravity_spy @idq; see @data_features for a representation of the different features present in gravitational-wave data. *Differentiation* problems, usually known simply as classification problems, involve the separation of detected features into multiple classes, although this is often done in tandem with detection. An example of this kind of problem is glitch classification, in which glitches are classified into classes of known glitch types, and the classifier must separate input data into these classes @gravity_spy.

#figure(
    image("data_features.png", width: 100%),
    caption: [ A non-exhaustive hierarchical depiction of some of the features, and proposed features, of gravitational-wave interferometer data. The first fork splits the features into two branches, representing the duration of the features. Here, *continuous* features are defined as features for which it is extremely unlikely for us to witness their start or end within the lifespan of the current gravitational-wave interferometer network and probably the current scientific community @continious_gravitational_waves. These features have durations anywhere from thousands to billions of years. *Transient* features have comparatively short durations @first_detction, from fractions of seconds in the case of stellar-mass Binary Black Hole (BBH) mergers @first_detction to years in the case of supermassive BBH mergers @supermassive_mergers. It should be noted that the detectable period of supermassive binaries could be much longer; although the mergers themselves are transient events, there is no hard cut-off between the long inspiral and the merger event. Nevertheless, the mergers are probably frequent enough that some will end within the lifetime of the proposed LISA space constellation, so in some cases, they can be considered transients @supermassive_mergers. The next fork splits features by origin. Features of *astrophysical* origin originate from beyond Earth. This distinction is practically synonymous with the distinction between gravitational waves and signals from other sources since no other astrophysical phenomena are known to have a similar effect in interferometers @first_detction. Features of *terrestrial* origin, unsurprisingly, originate from Earth. These primarily consist of detector glitches caused by seismic activity or experimental artifacts @det_char. Astrophysical transients have a further practical division into CBCs and bursts. The category of *bursts* contains all astrophysical transients that are not CBCs @bursts_O1. The primary reason for this distinction is that CBCs have been detected and have confirmed waveform morphologies @first_detction @first_bns. As of the writing of this thesis, no gravitational-wave burst events have been detected @bursts_O1 @bursts_O2 @bursts_O3. Bursts often require different detection techniques @cWB @x-pipeline; of the proposed sources, many are theorised to have waveforms with a much larger number of free parameters than CBCs, as well as being harder to simulate as the physics are less well-understood @supernovae_waveforms_2 @starquake_detection. These two facts compound to make generating large template banks for such signals extremely difficult. This means that coherence detection techniques that look for coherent patterns across multiple detectors are often used over matched filtering @x-pipeline @oLIB @cWB @BayesWave @MLy. The astrophysical leaves of the diagram represent possible and detected gravitational-wave sources; the text's colourings represent their current status. Green items have been detected using gravitational-wave interferometers, namely the merger of pairs of Binary Black Holes (BBHs) @first_detction, Binary Neutron Stars (BNSs) @first_bns, or one of each (BHNSs) @first_nsbh; see @GWTC-1 @GWTC-2 @GWTC-3 for full catalogues of detections. Yellow items have been detected via gravitational waves but using Pulsar Timing Arrays (PTAs) rather than interferometers @PTA. Blue items represent objects and systems that are theorised to generate gravitational waves and have been detected by electromagnetic observatories but not yet with any form of gravitational wave detection. This includes white dwarf binaries @white_dwarf_binary_em @white_dwarf_lisa_detection, the cosmological background @cosmological_background_em @cosmological_background_gw, starquakes @starquake_em @starquake_gw, and core-collapse supernovae CCSN @supernovae_em @supernovae_gw. This is because they are too weak and/or too uncommon for our current gravitational-wave detector network to have had a chance to detect them. Finally, red items are possible, theorised sources of gravitational waves that have not yet been detected by any means. These are, evidently, the most contentious items presented, and it is very possible that none of these items will ever be detected or exist at all. It should be noted that the number of proposed sources in this final category is extensive, and this is far from an exhaustive list. The presented proposed continuous sources are neutron star asymmetries @neutron_star_gw_review, and the presented transient sources are extraterrestrial intelligence @et_gw, cosmic string kinks and cusps @cosmic_string_cusps, accretion disk instabilities @accrection_disk_instability, domain walls @domain_walls, and nonlinear memory effects @non_linear_memory_effects. ]
) <data_features>

@data_features shows that several possible transients with terrestrial and astrophysical origins could be targeted for detection. For our baseline experiments and throughout this thesis, we will select two targets. 

Firstly, *Binary Black Holes (BBHs)*. We have the most numerous detections of BBH signals @GWTC-1 @GWTC-2 @GWTC-3, and whilst this might make them seem both less interesting and as a solved problem, they have several benefits. As test cases to compare different machine learning techniques against traditional methods, they have the most material for comparison because of their frequency; they would also see the greatest benefits from any computational and speed efficiency savings that may be wrought by the improvement of their detection methods @computational_cost. These factors may become especially relevant when the 3#super[rd] generation detectors, such as the Einstein Telescope @einstein_telescope and Cosmic Explorer @cosmic_explorer, come online. During their observing periods, they expect detection rates on the order of between $10^4$ and $10^5$ detections per year @overlapping_search, which would stretch computing power and cost if current methods remain the only options. In the shorter term, if detection speeds can be improved, faster alerts could be issued to the greater astronomical community, allowing increased opportunity for multimessenger analysis @multimessnger_review. Only one multimessenger event has thus far been detected --- a Binary Neutron Star (BNS) event @first_bns, but it is probable, due to the relative similarity in their morphologies, that methods to detect BBHs could be adapted for BNS detection.

Secondly, we will investigate the detection of unmodeled *burst* signals using a machine learning-based coherent detection technique. Bursts are exciting sources whose detection could herald immense opportunities for scientific gain @bursts_O3. Possible burst sources include core-collapse supernovae @supernovae_gw, starquakes @starquake_gw, accretion disk instabilities @accrection_disk_instability, nonlinear memory effects @non_linear_memory_effects, domain walls @domain_walls, and cosmic string cusps @cosmic_string_cusps, as well as a plethora of other proposed sources. It should be noted that whilst many bursts have unknown waveform morphologies, some, such as cosmic string cusps, are relatively easy to model and are grouped with bursts primarily due to their as-yet undetected status @cosmic_string_cusps.

Our current models of the physics of supernovae are limited both by a lack of understanding and computational intractability; detecting the gravitational-wave signal of a supernova could lead to new insights into the supranuclear matter density equation of state as well other macrophysical phenomena present in such events such as neutron transport and hydrodynamics @neutron_star_equation_of_state_1 neutron_star_equation_of_state_2 @neutron_star_equation_of_state_3. We may also detect proposed events, such as accretion disk instabilities @accrection_disk_instability, which may be missed by standard searches. We can search for the gravitational-wave signals of electromagnetic events that currently have unknown sources, such as fast radio bursts @targeted_frb_search, magnetar flares @targeted_magnetar_search, soft gamma-ray repeaters @targeted_grb_search, and long gamma-ray bursts @targeted_grb_search. Although it's possible that some of these events produce simple, modelable waveforms, it is not currently known, and a general search may one day help to reveal their existence. Some of the more hypothetical proposed sources could fundamentally alter our understanding of the universe, such as evidence for dark matter @domain_wall_dark_matter and/or cosmic strings @cosmic_string_cusps, or if we fail to find them, it could also help to draw limits on theory search space. 

It is unknown whether unmodeled burst detection is a solved problem. Currently, the LIGO-Virgo-KAGRA collaboration has a number of active burst detection pipelines, X-Pipeline @x-pipeline, oLIB @oLIB, Coherent Wave Burst (cWB) @cWB and BayesWave @BayesWave. These include both offline and online searches, including targeted searches wherein a known electromagnetic event is used to limit the search space @targeted_frb_search @targeted_magnetar_search @targeted_grb_search. It could be that the current detection software is adequate and, indeed, the search is hardware rather than software-limited. Even if this is the case, there are probably computational improvements that are possible. It seems unlikely that we have reached the limit of coherent search efficiency.

Traditional coherence techniques require the different detector channels to be aligned for successful detection; therefore, because we don't know a priori the direction of the gravitational-wave sources (unless we are performing a targeted offline search), coherent search pipelines such as X-Pipeline @x-pipeline and cWB @cWB must search over a grid covering all possible incidence directions. In the case of all-sky searches, this grid will necessarily cover the entire celestial sphere. In targeted searches, the grid can be significantly smaller and cover only the uncertainty region of the source that has already been localised by an EM detection @targeted_grb_search @targeted_frb_search. Higher resolution grids will result in a superior search sensitivity; however, they will simultaneously increase computing time. Covering the entire sky with a grid fine enough to achieve the desired sensitivity can be computationally expensive. It is possible to circumnavigate the need to search over a grid using artificial neural networks, shifting much of the computational expense to the training procedure. This has been demonstrated by the MLy pipeline @MLy --- the only fully machine-learning-based pipeline currently in review for hopeful deployment before the end of the fourth observing run (O4). Improvements in the models used for this task could be used to improve the effectiveness of the MLy pipeline. Indeed, some of the work discussed in this thesis was used at an early stage in the pipeline's development to help design the architecture of the models; see @deployment-in-mly. It is hoped that in the future, more aspects of the work shown here can find use in the pipeline's development.

We will focus on the binary detection problem rather than multi-class classification, as there is only one discrete class of BBH (unless you want to draw borders within the BBH parameter space or attempt to discern certain interesting features, such as eccentricity), and in the unmodeled burst case, coherent detection techniques are not usually tuned to particular waveforms, which, in any case, are not widely available for many types of burst. In the next subsection, we will discuss how we can create example datasets to train artificial neural networks for this task.

== Dataset Design and Preparation

In the case of CBCs, we have only a very limited number ($<200$) of example interferometer detections @GWTC-1 @GWTC-2 @GWTC-3, and in the burst case, we have no confirmed examples @bursts_O1 @bursts_O2 @bursts_O3. This means that to successfully train artificial neural network models, which typically require datasets with thousands to millions of examples @dataset_size, we must generate a large number of artificial examples. 

In order to facilitate the real-time generation of training datasets, a custom Python package named GWFlow was created @gwflow_ref. GWFlow handles the generation of fake noise and waveforms (with the use of the custom cuPhenom GPU waveform generator @cuphenom_ref), as well as the acquisition and processing of real interferometer data, and the injection, projection, and scaling of waveforms. It packages this functionality into a configurable TensorFlow dataset. Since the majority of the processing, except the acquisition of real noise, is performed on the GPU, the dataset can be adjusted and training can commence without the need to pre-generate the entire dataset. This allows for much quicker iteration through dataset hyperparameters.

The following subsections describe how GWFlow handles the creation of these examples, including the acquisition of noise, the generation and scaling of simulated waveforms, and data conditioning.

=== The Power Spectral Density (PSD) <psd-sec>

The Power Spectral Density (PSD) is an important statistical property that is used by several elements of dataset design @psd_ref. Since a custom function was written for this thesis in order to speed up the calculation of the PSD, and since it is helpful to have an understanding of the PSD in order to understand many of the processes described in subsequent sections, a brief explanation is presented.

The PSD is a time-averaged description of the distribution of a time series's power across the frequency spectrum @psd_ref. Unlike a Fourier transform, which provides a one-time snapshot, the PSD conveys an averaged view, accounting for both persistent and transient features; see @psd_eq for a mathematical description. The PSD is used during data conditioning in the whitening transform, wherein the raw interferometer data is processed so that the noise has roughly equal power across the frequency domain, see @feature-eng-sec. For some types of artificial noise generation, the PSD can be used to colour white noise in order to generate more physically active artificial noise; see @noise_acquisition_sec. The PSD is also used to calculate the optimal Signal to Noise ratio, which acts as a metric that can be used to measure the detectability of an obfuscated feature and thus can be used to scale the amplitude of the waveform to a desired detection difficulty.

Imagine a time series composed of a stationary #box("20" + h(1.5pt) + "Hz") sine wave. In the PSD, this would materialise as a distinct peak at #box("20" + h(1.5pt) + "Hz"), effectively capturing the concentrated power at this specific frequency: the frequency is constant, and the energy is localised. If at some time, $t$, we remove the original wave and introduce a new wave at a different frequency, #box("40" + h(1.5pt) + "Hz"), the original peak at #box("20" + h(1.5pt) + "Hz")would attenuate but not vanish, as its power is averaged over the entire time-series duration. Concurrently, a new peak at #box("40" + h(1.5pt) + "Hz") would appear. The power contained in each of the waves, and hence the heights of their respective peaks in the PSD, is determined by the integrated amplitude of their respective oscillations; see @psd-example for a depiction of this example. When applied to a more complicated time series, like interferometer noise, this can be used to generate an easy-to-visualise mapping of the distribution of a time series's power across frequency space.

#show figure: set block(breakable: true) 
#figure(
    image("example_psd.png", width: 100%),
    caption: [Examples of Power Spectral Density (PSD) transforms. _Left:_ Two time domain series. The red series is a #box("20" + h(1.5pt) + "Hz") wave with a duration of #box("0.7" + h(1.5pt) + "s"), and the blue series is this same time series concatenated with a #box("40" + h(1.5pt) + "Hz") wave from $t = 0.7#h(1.5pt)s$ onwards. _Right:_ The two PSDs of the time series are displayed in the left panel. The red PSD was performed across only the #box("0.7" + h(1.5pt) + "s") of the red wave's duration, whereas the blue PSD was taken over the full #box("2.0" + h(1.5pt) + "s") duration. As can be seen, the blue PSD has two peaks, representing the two frequencies of the two waves combined to make the blue time series --- each peak is lower than the red peak, as they are averaged across the full duration, and their respective heights are proportional to their durations as both waves have the same amplitude and vary only in duration.]
) <psd-example>

The PSD can be calculated using Welch's method, which uses a periodogram to calculate the average power in each frequency bin over time @welchs_method. More specifically, the following steps are enacted:

+ First, the time series is split up into $K$ segments of length $L$ samples, with some number of overlapping samples $D$; if $D = 0$, this method is equivalent to Bartlett's method. 
+ Each segment is then windowed with a user-chosen window function, $w(n)$. This is done in order to avoid spectral leakage, avoid discontinuities in the data, smoothly transition between segments, and control several other factors about the method, which allow for fine-tuning to specific requirements.
+ For each windowed segment, $i$, we then estimate the power of the segment, $I_i (f_k)$, at each frequency, $f_k$, by computing the periodogram with

$ I_i (f_k) = 1/L|X_i (k)|^2 $ <periodogram>

where $I_i (f_k)$ is the result of the periodogram, $X_i (k)$ is the FFT of the windowed segment, and $f_k$ is the frequency corresponding to the $k^op("th")$ FFT sample.

4. Finally, we average the periodograms from each segment to get the time-average PSD:

$  S(f_k) =  1/K sum_(i=1)^K I_i (f_k) $ <average_periodograms>

where where $S(f_k)$ is the PSD. Combining @periodogram and @average_periodograms gives

$ S(f_k) =  1/K sum_(i=1)^K 1/L|X_i (k)|^2 $ <psd_eq>

To compute the PSD with enough computational speed to perform rapid whitening and SNR, $rho_"opt"$, calculation during model training and inference, an existing Welch method from the SciPy scientific Python library @scipy was adapted and added to the GWFlow pipeline @gwflow_ref, converting its use of the NumPy vectorised CPU library @numpy to the TensorFlow GPU library @tensorflow; this converted code is seen in @psd_calculation.

#figure(
```py
@tf.function 
def calculate_psd(
        signal : tf.Tensor,
        nperseg : int,
        noverlap : int = None,
        sample_rate_hertz : float = 1.0,
        mode : str ="mean"
    ) -> (tf.Tensor, tf.Tensor):
    
    if noverlap is None:
        noverlap = nperseg // 2
        
    signal = detrend(signal, axis=-1, type='constant')
    
    # Step 1: Split the signal into overlapping segments
    signal_shape = tf.shape(signal)
    step = nperseg - noverlap
    frames = tf.signal.frame(signal, frame_length=nperseg, frame_step=step)
        
    # Step 2: Apply a window function to each segment
    # Hanning window is used here, but other windows can be applied as well
    window = tf.signal.hann_window(nperseg, dtype = tf.float32)
    windowed_frames = frames * window
    
    # Step 3: Compute the periodogram (scaled, absolute value of FFT) for each 
    # segment
    periodograms = \
        tf.abs(tf.signal.rfft(windowed_frames))**2 / tf.reduce_sum(window**2)
    
    # Step 4: Compute the median or mean of the periodograms based on the 
    #median_mode
    if mode == "median":
        pxx = tfp.stats.percentile(periodograms, 50.0, axis=-2)
    elif mode == "mean":
        pxx = tf.reduce_mean(periodograms, axis=-2)
    else:
        raise "Mode not supported"
    
    # Step 5: Compute the frequencies corresponding to the power spectrum values
    freqs = fftfreq(nperseg, d=1.0/sample_rate_hertz)
    
    #Create mask to multiply all but the 0 and nyquist frequency by 2
    X = pxx.shape[-1]
    mask = \
        tf.concat(
            [
                tf.constant([1.]), 
                tf.ones([X-2], dtype=tf.float32) * 2.0,
                tf.constant([1.])
            ], 
            axis=0
        )
        
    return freqs, (mask*pxx / sample_rate_hertz)


```,
caption : [_Python @python ._ TensorFlow @tensorflow graph function used by GWFlow @gwflow_ref to calculate the PSD of a signal. `signal` is the input time series as a TensorFlow tensor, `nperseg` is the number of samples per segment, $L$, and `noverlap` is the number of overlapping samples, $D$. TensorFlow has been used in order to utilise GPU parallelisation, which offers a significant performance boost over a similar function written in NumPy @numpy.]
) <psd_calculation>

A closely related property, the Amplitude Spectral Density (ASD), is given by the element-wise square root of the Power Spectral Density (PSD)

$ A(f_k) = S(f_k)^(compose 1/2). $ <asd-func>

=== Noise Generation and Acquisition <noise_acquisition_sec>

There are two possible avenues for acquiring background noise to obfuscate our injections. We can either create artificial noise or use real segments extracted from previous observing runs. As was discussed in @interferometer_noise_sec, real interferometer noise is neither Gaussian nor stationary, and many of the noise sources which compose this background are not accounted for or modelled @det_char. This means that any artificial noise will only be an approximation of the real noise --- it is not clear, intuitively, how well this approximation will be suited to training an artificial neural network. 

One perspective argues that using more approximate noise could enhance the network's generalisation capabilities because it prevents overfitting to the specific characteristics of any given noise distribution; this is the approach adopted by the MLy pipeline @MLy. Conversely, another perspective suggests that in order to properly deal with the multitude of complex features present in real noise, we should make our training examples simulate real noise as closely as possible @dataset_mismatch_problems @dataset_mismatch_problems_2 @dataset_mismatch_problems_3, even suggesting that models should be periodically retrained within the same observing run in order to deal with variations in the noise distribution. These are not discrete philosophies, and the optimal training method could lie somewhere between these two paradigms.

Evidently, in either case, we will want our validation and testing datasets to approximate the desired domain of operation as closely as possible; if they do not, we would have no evidence, other than assumption, that the model would have any practical use in real data analysis @dataset_mismatch_problems_3. The following subsection will outline the possible types of noise that could be used to create artificial training examples. Throughout the thesis, for all validation purposes, we have used real noise at GPS times, which are not used at any point during the training of models, even when the training has been done on real noise.

*White Gaussian:* The most simplistic and general approach, and therefore probably the most unlike real noise, is to use a white Gaussian background. This is as simplistic as it sounds; we generate $N$ random variables, where N is the number of samples in our noise segment. Each sample is drawn from a normal distribution with a mean of zero and some variance according to the input scaling; often, in the case of machine learning input vectors, this would be unity; see the two uppermost plots in @noise_comparison.

*Coloured Gaussian:* This noise approximation increases the authenticity of the noise distribution by colouring it with a noise spectrum; typically, we use an ASD drawn from the interferometer we are trying to imitate in order to do this; see @psd-sec. By multiplying the frequency domain transform of Gaussian white noise by a given PSD, we can colour that noise with that PSD. The procedure to do this is as follows:

+ Generate white Gaussian noise.
+ Transform the Gaussian noise into the frequency domain using a Real Fast Fourier Transform (RFFT).
+ Multiply the noise frequency spectrum by the selected ASD in order to colour it.
+ Return the newly coloured noise to the time domain by performing an Inverse RFFT (IRFFT).

There are at least two choices of PSD we could use for this process. We could use the PSD of the detector design specification. It represents the optimal PSD given perfect conditions, no unexpected noise sources, and ideal experimental function. This would give a more general, idealistic shape of the PSD across a given observing run. Alternatively, we could use the PSD of a real segment of the background recorded during an observing run; this would contain more anomalies and be a closer approximation to the specific noise during the period for which the PSD was taken. Since the PSD is time-averaged, longer segments will result in more general noise. The MLy pipeline @MLy refers to this latter kind of noise as *pseudo-real* noise; see examples of these noise realisations in the four middle plots of @noise_comparison.

*Real:* Finally, the most authentic type of noise that can be gathered is real interferometer noise. This is noise that has been sampled directly from a detector. Even assuming that you have already decided on which detector you are simulating, which is required for all but white noise generation, there are some extra parameters, shared with the pseudo-real case, that need to be decided. The detector data information, the time period from which you are sampling, and whether to veto any features that may be present in the segment --- e.g. segments which contain events, candidate events, and known glitches. 

To acquire the real data, we utilise the GWPy Python Library's @gwpy data acquisition functionality --- since there are multiple formats in which we could retrieve the data, we must specify some parameters, namely, the frame, the channel, and the state flag. Interferometer output data is stored in a custom file format called a frame file @frame-file; thus, the choice of frame determines the file to be read. Within each frame file lies multiple channels --- each of which contains data from a single output stream. These output streams can be raw data, e.g. raw data from the interferometer photodetector itself; various raw auxiliary data streams, such as from a seismometer; conditioned data, e.g., the primary interferometer output with lines removed; or the state flag channel, which contains information about the status of the detector at every time increment --- the state flag will indicate whether the detector is currently in observing mode or otherwise, so it is important to filter the data for the desired detector state. For the real noise used in this thesis, we use the frame, channel, and state flag, shown in @detector_data_table. We have excluded all events and candidate events listed in the LIGO-Virgo-Kagra (LVK) collaboration event catalogues  @GWTC-1 @GWTC-2 @GWTC-3 but included detector glitches unless otherwise stated.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [*Detector*],  [*Frame*], [*Channel*], [*State Flag*],
    [LIGO Hanford (H1)], [HOFT_C01], [H1:DCS-CALIB_STRAIN_CLEAN_C01], [DCS-ANALYSIS_READY_C0- 1:1],
    [LIGO Livingston (L1)], [HOFT_C01], [L1:DCS-CALIB_STRAIN_CLEAN_C01], [DCS-ANALYSIS_READY_C0- 1:1],
    [VIRGO (V1)], [V1Online], [V1:Hrec_hoft_16384Hz], [ITF_SCIENCE:1], 
  ),
  caption: [The frame, channel, and state flags used when obtaining data from the respective detectors during the 3#super("rd") observing run (O3). This data was used as obfuscating noise when generating artificial examples to train and validate artificial neural network models throughout this thesis. It should be noted that although the clean channels were produced offline in previous observing runs, the current observing run, O4, produces cleaned channels in its online run, so using the cleaned channels during model development ensures that the training, testing, and validation data is closer to what would be the normal operating mode for future detection methods.]
) <detector_data_table>

#figure(
    image("noise_comparison.png", width: 100%),
    caption: [One-second examples of the four possible types of simulated and real noise considered by this thesis. Where real noise is used, it is taken from the LIGO Livingston detector during the third observing run at the GPS times listed. In order, from top to bottom, these are examples of white Gaussian noise, coloured Gaussian noise, pseudo-real noise, and real noise. A description of these noise types and their generation can be found in @noise_acquisition_sec. The left column shows the unaltered values of the noise. Note that the noise has been scaled in all cases except for the pure white noise, which is generated at the correct scale initially. This scaling is used to reduce precision errors and integrate more effectively with the machine learning pipeline, as most loss and activation functions are designed around signal values near unity; see @loss_functions_sec and @activation_functions_sec. The right column shows the same noise realisations after they have been run through a whitening filter. In each case, the PSD of a #box("16.0" + h(1.5pt) + "s") off-source noise segment not displayed is used to generate a Finite Impulse Response (FIR) filter, which is then convolved with the on-source data; see @feature-eng-sec. For the simulated and pseudo-real noise cases, the off-source data is generated using the same method as the on-source data but with a longer duration. In the real noise case, the off-source data consists of real interferometer data drawn from #box("16.5" + h(1.5pt) + "s") before the start of the on-source segment to #box("0.5" + h(1.5pt) + "s") before the start of the on-source segment. This 0.5s gap is introduced because #box("0.5" + h(1.5pt) + "s") must be cropped from the data following the whitening procedure in order to remove edge effects induced via windowing, as well as acting as a buffer to reduce contamination of the off-source data with any features present in the on-source data. Note that the whitened noise plots look very similar for the three simulated noise cases --- a close examination of the data reveals that there is some small variation between the exact values. This similarity occurs because the off-source and on-source noise segments for these examples are generated with identical random seeds and thus have identical underlying noise realisations (which can be seen exactly in the unwhitened white noise plot). Since the PSDs of the on-source and off-source data are nearly identical for the simulated cases, the whitening procedure is almost perfect and reverts it nearly perfectly to its white state. If anything, this similarity boosts confidence that our custom whitening procedure is operating as expected.]
) <noise_comparison>

For our baseline training dataset used in this section, we will employ a real-noise background. An argument can be made that it is an obvious choice. It is the most like real noise, by virtue of being real noise, and thus it contains the full spectrum of noise features that might be present in a real observing run, even if it does not contain the particular peculiarities of any given future observing run in which we may wish to deploy developed models. We will experiment with different noise realisations in a future chapter @noise-type-test-sec.

In each case, we will acquire two seconds of data at a sample rate of #box("2048.0" + h(1.5pt) +"Hz"), which includes #box("0.5" + h(1.5pt) + "s") of data on either side of the time series, which will be cropped after whitening. The whitening is performed similarly in all cases in order to ensure symmetry when comparing obfuscation methods. A power-of-two value is used as it simplifies many of the mathematical operations that need to be performed during signal and injection processing, which may, in some cases, improve performance, as well as help to avoid edge cases that may arise from odd numbers. This frequency was selected as its Nyquist frequency of #box("1024.0" + h(1.5pt) + "Hz") will encompass nearly the entirety of the frequency content of BBH signals; it also covers a large portion of the search space of proposed transient burst sources. The duration of #box("1.0" + h(1.5pt) + "s") is a relatively arbitrary choice; however, it is one that is often the choice for similar examples found in the literature @gabbard_messenger_cnn @george_huerta_cnn, which makes comparison easier. It also encompasses the majority of the signal power of BBH waves @first_detction, as well as the theoretically detectable length of many burst sources @bursts_O1. For each on-source noise example gathered or generated, #box("16.0" + h(1.5pt) + "s") of off-source background noise is also acquired to use for the whitening procedure; see @feature-eng-sec.

In the case where multiple detectors are being used simultaneously during training or inference, such as coherence detection, noise is generated independently for each interferometer using the same methods, with the restriction that noise acquired from real interferometer data is sampled from each detector within a common time window of #box("2048.0" + h(1.5pt) + "s") so that the noise all originates from a consistent time and date. This is done as there are periodic non-stationary noise features that repeat in daily, weekly, and yearly cycles due to weather, environmental conditions, and human activity @det_char. When validating methods, we want to make our validation data as close as possible to reality whilst maintaining the ability to generate large datasets. As we are only ever training our method to operate in real noise conditions (which our validation data attempts to mimic), there is no need to deviate from this method of acquiring noise for our training datasets.

=== Waveform Generation <injection-gen-sec>

Once the background noise has been acquired or generated, the next step is to introduce some differentiation between our two classes, i.e. we need to add a transient signal into some of our noise examples so that our model can find purpose in its existence. When we add a transient into background noise that was not there naturally, we call this an *injection*, since we are artificially injecting a signal into the noise. This injection can be a transient of any type.

Typically, this injection is artificially simulated both due to the limited @GWTC-1 @GWTC-2 @GWTC-3 (or non-existent @bursts_O1 @bursts_O2 @bursts_O3) number of real examples in many cases and because we will only be able to obtain the real signal through the lens of an interferometer, meaning it will be masked by existing real detector noise. If we were to inject a real injection into some other noise realisation, we would either have to perform a denoising operation (which, even when possible, would add distortion to the true signal) or inject the injection plus existing real noise into the new noise, effectively doubling the present noise and making injection scaling a difficult task. Thus, we will be using simulated injections to generate our training, testing, and validation datasets.

Luckily, this is not unprecedented, as most other gravitational-wave detection and parameter estimation methods rely on simulated signals for their operation, including matched filtering @pycbc. Therefore, there is a well-developed field of research into creating artificial gravitational-wave waveform "approximants", so named because they only approximate real gravitational-wave waveforms @imrphenom_d. Depending on the complexity and accuracies of the chosen approximant and the source parameter range you are investigating, there will be some level of mismatch between any approximant and the real waveform it is attempting to simulate, even when using state-of-the-art approximants @imrphenom_future.

To simulate BBH waveforms, we will be using a version of the IMRPhenomD approximant @imrphenom_d, which has been adapted to run on GPUs using NVIDIAs CUDA GPU library. We name this adapted waveform library cuPhenom @cuphenom_ref for consistency with other CUDA libraries such as cuFFT @cufft. More details on this adaptation can be found in @software-dev-sec. PhenomD has adjustable parameters that can be altered to generate BBHs across a considerable parameter space, although it should be noted that it does not simulate eccentricity, non-aligned spins, or higher modes. It is also a relatively outdated waveform, with the paper first published in 2015 @imrphenom_d. Newer waveforms, such as those of the IMRPhenomX family, are now available @imrphenom_future. IMRPhenomD was initially chosen due to its simpler design and as a test case for the adaptation of Phenom approximants to a CUDA implementation. It would not be ideal for implementation into a parameter estimation pipeline due to its mismatch, but the accuracy requirements for a detection pipeline are significantly less stringent.

The IMRPhenomD @imrphenom_d approximant generates a waveform by simulating the Inspiral, Merger, and Ringdown regions of the waveform, hence the IMR in the approximant name. The waveform is generated in the frequency domain before being transformed back into the time domain. The inspiral is generated using post-Newtonian expressions, and the merger ringdown is generated with a phenomenological ansatz; both parts of the model were empirically tuned using a small bank of numerical relativity waveforms. Detailed investigation of aproximant generation was out of the scope of this thesis and will not be covered. See @example_injections for examples of waveforms generated using cuPhenom @cuphenom_ref

The increased performance of cuPhenom is significant and speeds up the training and iteration process of models considerably @cuphenom_ref. Because of cuPhenom's ability to generate injections on the fly during the training process without significant slowdown, it allows for very quick alteration of dataset parameters for training adjustments. It was felt that this advantage outweighed any gains that would be achieved by using newer waveform models that had not yet been adapted to the GPU, as it seems unlikely, especially in the detection case, that the newer waveform models would make for a significantly harder problem for the model to solve. This statement is, however, only an assumption, and it would be recommended that an investigation is carried out to compare the differences between approximants before any of the methods are used in a real application. A final retraining with these more accurate models would be recommended, in any case.

In the case of unmodelled burst detection, the accuracy of the signal shape is not as fundamental, as the ground truth shapes are not known and, for some proposed events, cover a very large shape space @supernovae-review. In order to cover the entire search space, GWFlow uses artificially generated White Noise Bursts (WNBs) generated on the GPU via a simple custom Python @python function utilising TensorFlow @tensorflow. The procedure for generating WNBs with randomised duration and frequency content is as follows.

+ A maximum waveform duration is decided; typically, this would be less or equal to the duration of the example noise that you are injecting the waveform into, with some room for cropping.
+ Arrays of durations, minimum frequencies, and maximum frequencies are generated, each with a number of elements, $N$, equal to the number of waveforms that we wish to generate. These arrays can be pulled from any distribution as long as they follow the following rules. Duration cannot be larger than our maximum requested duration or less than zero. The frequency bounds cannot be less than zero or greater than the Nyquist frequency.
+ It is enforced that the maximum frequency is greater than the minimum frequency for any waveform by swapping values where this is not the case.
+ Gaussian white noise is generated with as many samples, which, given the selected sample rate, will produce a time series with the same duration as our requested max waveform duration.
+ A number of samples at the end of each waveform are zeroed so that each waveform has a number of samples equivalent to the randomised duration assigned to that signal.
+ Each waveform is transformed into the frequency domain by a RFFT. 
+ Samples are zeroed at each end of each frequency-domain signal in order to perform a bandpass and limit the waveform between the assigned frequency constraints for each waveform.
+ The remaining signal is windowed using a Hann window to reduce the effects of the discontinuities generated by the bandpass operation.
+ The frequency domain signal is then returned to the time domain via a IRFFT.
+ Finally, the time-domain waveform is enveloped by a sigmoid window.
+ Assuming the plus polarisation component of the waveform strain was generated first, repeat with the same parameters but different initial noise distributions for the cross polarisation component.

Because we have used random noise across a range of frequency spaces, our distribution will, in theory, cover all possible signals within the specified parameter range. These WNBs can generate waveforms that look qualitatively similar to many proposed burst sources, including current supernovae simulations; see @supernovae_example. See @example_injections for examples of our WNBs and @wnb_calculation for the code used to generate these waveforms.

#figure(
```py 
@tf.function
def generate_white_noise_burst(
    num_waveforms: int,
    sample_rate_hertz: float,
    max_duration_seconds: float,
    duration_seconds: tf.Tensor,
    min_frequency_hertz: tf.Tensor,
    max_frequency_hertz: tf.Tensor
) -> tf.Tensor:
        
    # Casting
    min_frequency_hertz = tf.cast(min_frequency_hertz, tf.float32)
    max_frequency_hertz = tf.cast(max_frequency_hertz, tf.float32)

    # Convert duration to number of samples
    num_samples_array = tf.cast(sample_rate_hertz * duration_seconds, tf.int32)
    max_num_samples = tf.cast(max_duration_seconds * sample_rate_hertz, tf.int32)

    # Generate Gaussian noise
    gaussian_noise = tf.random.normal([num_waveforms, 2, max_num_samples])

    # Create time mask for valid duration
    mask = tf.sequence_mask(num_samples_array, max_num_samples, dtype=tf.float32)
    mask = tf.reverse(mask, axis=[-1])
    mask = tf.expand_dims(mask, axis=1)
    
    # Mask the noise
    white_noise_burst = gaussian_noise * mask

    # Window function
    window = tf.signal.hann_window(max_num_samples)
    windowed_noise = white_noise_burst * window

    # Fourier transform
    noise_freq_domain = tf.signal.rfft(windowed_noise)

    # Frequency index limits
    max_num_samples_f = tf.cast(max_num_samples, tf.float32)
    num_bins = max_num_samples_f // 2 + 1
    nyquist_freq = sample_rate_hertz / 2.0

    min_freq_idx = tf.cast(
        tf.round(min_frequency_hertz * num_bins / nyquist_freq), tf.int32)
    max_freq_idx = tf.cast(
        tf.round(max_frequency_hertz * num_bins / nyquist_freq), tf.int32)

    # Create frequency masks using vectorized operations
    total_freq_bins = max_num_samples // 2 + 1
    freq_indices = tf.range(total_freq_bins, dtype=tf.int32)
    freq_indices = tf.expand_dims(freq_indices, 0)
    min_freq_idx = tf.expand_dims(min_freq_idx, -1)
    max_freq_idx = tf.expand_dims(max_freq_idx, -1)
    lower_mask = freq_indices >= min_freq_idx
    upper_mask = freq_indices <= max_freq_idx
    combined_mask = tf.cast(lower_mask & upper_mask, dtype=tf.complex64)
    combined_mask = tf.expand_dims(combined_mask, axis=1)

    # Filter out undesired frequencies
    filtered_noise_freq = noise_freq_domain * combined_mask

    # Inverse Fourier transform
    filtered_noise = tf.signal.irfft(filtered_noise_freq)
    
    envelopes = generate_envelopes(num_samples_array, max_num_samples)
    envelopes = tf.expand_dims(envelopes, axis=1)
        
    filtered_noise = filtered_noise * envelopes

    return filtered_noise
```,
caption : [_ Python @python . _ TensorFlow @tensorflow graph function to generate the plus and cross polarisations of WNB waveforms; see @injection-gen-sec for a description of the generation method. `num_waveforms` takes an integer value of the number of WNBs we wish to generate. `sample_rate_hertz` defines the sample rate of the data we are working with. `max_duration_seconds` defines the maximum possible duration of any signals within our output data. `duration_seconds`, `min_frequency_hertz`, and `max_frequency_hertz` all accept arrays or in this case TensorFlow tensors, of values with a number of elements equal to `num_waveforms`, each duration. Both polarisations of the WNB are generated with parameters determined by the value of these three arrays at the equivalent index. This method is implemented by the GWFlow pipeline @gwflow_ref.]
) <wnb_calculation>

#figure(
    image("example_injections.png", width: 100%),
    caption: [Eight simulated waveforms that could be used for injection into noise to form an obfuscated training, testing, or validation example for an artificial neural network. Note that only the plus polarisation component of the strain, $h_plus$, has been plotted in order to increase visual clarity. The leftmost four injections are IMRPhenomD waveforms generated using cuPhenom @cuphenom_ref; see @cuphenom-sec, with parameters (shown in the adjacent grey information boxes) drawn from uniform distributions between #box("5.0" + h(1.5pt) + $M_dot.circle$) and #box("95.0" + h(1.5pt) + $M_dot.circle$) for the mass of both companions and between -0.5 and 0.5 for the dimensionless spin component. Note that during injection generation, the two companions are always reordered so that the mass of companion one is greater and that the IMRPhenomD waveform ignores the x and y spin components. They are included just for code completion. The rightmost four injections consist of WNB waveforms generated via the method described in @injection-gen-sec. Their parameters are again drawn from uniform distributions and are shown in the grey box to their right. The durations are limited between #box("0.1"+ h(1.5pt) + "s") and #box("1.0" + h(1.5pt) + "s"), and the frequencies are limited to between #box("20.0" + h(1.5pt) + "Hz") and #box("500.0" + h(1.5pt) + "Hz"), with the minimum and maximum frequencies automatically swapped.]
) <example_injections>

#figure(
    image("supernova_example.png", width: 80%),
    caption: [The plus polarisation component of the gravitational-wave strain of a simulated core-collapse supernova at a distance of #box("10" + h(1.5pt) + "kpc"), this data was taken from @supernovae_waveforms. Although some structures can clearly be observed, it is possible to imagine that a method trained to detect WNB signals, such as those presented in @example_injections, might be able to detect the presence of such a signal. ]
) <supernovae_example>

=== Waveform Projection <projection-sec>

As has been discussed, gravitational waves have two polarisation states plus, $plus$, and cross, $times$, which each have their own associated strain values $h_plus$ and $h_times$ @gravitational_waves_ref @gravitational_wave_interfereometers. Since these strain polarisation states can have different morphologies and since the polarisation angle of an incoming signal paired with a given interferometer's response will alter the proportion of each polarisation that is perceptible by the detector, our aproximant signals are also generated with two polarisation components. Before being injected into any data, the waveforms must be projected onto each detector in our network in order to simulate what that signal would look like when observed with that detector. This projection will account for the full antenna response of each detector @gravitational_wave_interfereometers. Since a given interferometer has different sensitivities depending on both the direction of the source and the polarisation angle of the incoming wave, some waves will be entirely undetectable in a given detector. 

If we want accurate data when simulating multi-interferometer examples, we must account for both the polarisation angle and direction of the source so that the relative strain amplitudes and morphologies in each detector are physically realistic @gravitational_wave_interfereometers. 

Since the detectors have a spatial separation, there will usually, depending on source direction, also be a difference in the arrival time of the waves at the different detectors @gravitational_wave_interfereometers --- this discrepancy is especially important for localising sources, as it provides the possibility for source triangulation, which, along with the antenna responses of each detector, can be used to generate a probability map displaying the probability that a wave originated from a given region of the sky. In coherence detection methods, it also allows for the exclusion of multi-interferometer detections if the detections arise with an arrival time difference greater than that which is physically possible based on the spatial separation of the detectors.

None of this is essential when dealing with single detector examples --- in those cases, we could choose to forgo projection entirely and inject one of the strain polarisation components directly into the obfuscating noise as there are no time separations to model accurately and signal proportionality between detectors is also irrelevant. 

The projection from both the antenna response parameters and the arrival time delay are dependent on the source direction @gravitational_wave_interfereometers. The plane of the wavefront and the direction of travel of the wave are dependent on the direction of the source. Since the sources are all extremely distant, the wavefront is considered a flat plane. Waves have some time duration, so both the time delay and antenna response parameters will change over the course of the incoming wave's duration as the Earth and the detectors move in space. As we are dealing with relatively short transients ($< 1.0 space s$), the change in these factors will be considered negligible and is not included in projection calculations.

Assuming that we ignore the Earth’s motion, the final waveform present in a detector is given by

$ h(t) = F_plus h_plus (t + Delta t) + F_times h_times (t + Delta t) $ <projection_equ>

where $h(t)$ is the resultant waveform present in the detector output at time $t$; $F_plus$ and $F_times$ are the detector antenna response parameters in the plus and cross polarisations for a given source direction, polarisation angle, and detector; $h_plus$ and $h_times$ are the plus and cross polarisations of the gravitational-wave strain of simulated or real gravitational waves; and $Delta t$ is the arrival time delay taken from a common reference point, often another detector or the Earth’s centre.

We can also calculate the relative times that the signals will arrive at a given detector,

$ Delta t = frac( (accent(x_0, arrow) - accent(x_d, arrow)) ,  c) dot.op accent(m, arrow) $ <time-delay_eq>

where $Delta t$ is the time difference between the wave's arrival at location $accent(x_d, arrow)$ and $accent(x_0, arrow)$, $c$ is the speed of light, $accent(x_0, arrow)$ is some reference location, often taken as the Earth’s centre, $accent(x_d, arrow)$ is the location for which you are calculating the time delay, in our case, one of our interferometers, and $accent(m, arrow)$ is the direction of the gravitational-wave source. If we work in Earth-centred coordinates and take the Earth's centre as the reference position so that $x_0 = [0.0, 0.0, 0.0]$ we can simplify @time-delay_eq to

$ Delta t = - frac(accent(x, arrow) ,  c) dot.op accent(m, arrow). $ <time-delay_eq_sim>

Finally, combining @projection_equ and @time-delay_eq_sim, we arrive at

$ h(t) = F_plus h_plus (t - frac(accent(x, arrow) ,  c) dot.c accent(m, arrow)) + F_times h_times (t - frac(accent(x, arrow) ,  c) dot.c accent(m, arrow)) . $ <final_response_equation>

In practice, for our case of discretely sampled data, we first calculate the effect of the antenna response in each detector and then perform a heterodyne shift to each projection to account for the arrival time differences. When multiple detector outputs are required for training, testing, or validation examples, GWFlow performs these calculations using a GPU-converted version of the PyCBC @pycbc project_wave function; see @projection_examples for example projections.

#figure(
    image("projection_examples.png", width: 80%),
    caption: [Example projection of two artificial gravitational-wave waveforms. The blue waveforms have been projected into the LIGO Livingston interferometer, the red waveforms have been projected into the Ligo Hanford interferometer, and the green waveforms have been projected into the VIRGO interferometer. The left column displays different projections of an IMRPhenomD waveform generated with the cuPhenom GPU library @cuphenom_ref; see @cuphenom-sec. The right column displays different projections of a WNB waveform generated with the method described in @injection-gen-sec. The projections are performed using a GPU adaptation of the PyCBC Python library's @pycbc project_wave function. Both waveforms are projected from different source locations; the projection and time displacement are different in each case. ]
) <projection_examples>

=== Waveform Scaling

Once waveforms have been projected to the correct proportionality, we must have some method to inject them into obfuscating noise with a useful scaling. If using physically scaled approximants, such as the IMRPhenomD waveform, we could forgo scale by calculating the resultant waveform that would be generated by a CBC at a specified distance from Earth, then injecting this into correctly scaled noise (or simply raw real noise). However, since we are also using non-physical waveforms such as WNBs, and because we would like a more convenient method of adjusting the detectability of our waveforms, we will use a method to scale the waveforms to a desired proportionality with the noise.

Evidently, if we injected waveforms that have been scaled to values near unity into real unscaled interferometer noise (which is typically on the order of $10^(-21)$), even a very simple model would not have much of a problem identifying the presence of a feature. Equally, if the reverse were true, no model could see any difference between interferometer data with or without an injection. Thus, we must acquire a method to scale our injections so that their amplitudes have a proportionality with the background noise that is similar to what might be expected from real interferometer data. 

Real data holds a distribution of feature amplitudes, with quieter events appearing in the noise more commonly than louder ones @gravitational_wave_population @network_snr --- this is because gravitational-wave amplitude scales inversely with distance @network_snr @gravitation, whereas the volume of searchable space, and thus matter and, generally, the number of systems which can produce gravitational waves, scale cubically with distance from Earth. 

Features with quieter amplitudes will, in general, be harder for a given detection method to identify than features with louder amplitudes. We must design a training dataset that contains a curriculum which maximises model efficacy across our desired regime, with examples that are difficult but never impossible to classify and perhaps some easier cases that can carve channels through the model parameters, which can be used to direct the training of more difficult examples.

In any given noise distribution, there will, for any desired false alarm rate, be a minimum detectable amplitude below which it becomes statistically impossible to make any meaningful detections [cite]. This minimum amplitude occurs because even white Gaussian noise will occasionally produce data which looks indistinguishable from a certain amplitude of waveform. 

We can use matched filtering statistics to prove this point, as we know that given an exactly known waveform morphology and perfect Gaussian noise, matched filtering is the optimal detection statistic. The probability that a matched filtering search of one template produces a false alarm is dependant only on the rate at which you are willing to miss true positives. We can use the $cal(F)$-statistic, $cal(F)_0$, for our probability metric to adjust this rate. Assuming that the noise is purely Gaussian, and we are only searching for one specific template, the probability of false detections of this exact waveform, i.e. $P(cal(F) > cal(F)_0)$, can be expressed as

$ P_F (cal(F)_0) = integral_(cal(F)_0)^infinity p_0(cal(F))d cal(F) = exp(-cal(F_0)) sum_(k=0)^(n/2-1) frac(cal(F)_0^k, k!) $ <false_alarm_rate_eq>

where n is the number of degrees of freedom of $chi^2$ distributions. We can see from @false_alarm_rate_eq that the False Alarm Rate (FAR) in this simple matched filtering search is only dependent on the arbitrary choice of $cal(F)_0$. However, in practice, the choice of $cal(F)_0$ will be determined by the minimum amplitude waveform you wish to detect because the probability of detection, $P_d$ given the presence of a waveform, is dependent on the optimal SNR, $rho_"opt"$ of that waveform, $rho$, which has a loose relationship to the amplitude of the waveform. The probability of detection is given by

$ P_D (rho, cal(F)_0) = integral_(cal(F)_0)^infinity frac((2 cal(F))^((n/2 - 1)/2), rho^(n/2 - 1)) I_(n/2-1) (rho sqrt(2 cal(F))) exp(-cal(F) - 1/2 rho^2) d cal(F) $

where $I_(n/2-1)$ is the modified Bessel function of the first kind and order $n/2 -1$. For more information on this, please refer to @gw_gaussian_case.

More complex types of noise, however, like real LIGO interferometer noise, could potentially produce waveform simulacra more often than artificially generated white noise @det_char. 

Louder false alarms are less likely than quieter ones, and at a certain amplitude, a given detection method will start producing a greater number of false alarms than the desired false alarm rate. If our training dataset includes waveforms with an amplitude that would trigger detections with a false alarm rate near or less than our desired rate, this could significantly reduce the performance of our network @feature_noise, so we must select a minimum amplitude that maximises our detection efficiency at a given false alarm rate. 

Our minimum possible detection amplitude is limited by the combination of the noise and the false alarm rate we desire. There is not a maximum possible signal amplitude, other than some very unuseful upper bound on the closest possible gravitational-wave-producing systems to Earth (a nearby supernova or CBC, for example), but these kinds of upper limit events are so astronomically rare as not to be worth considering. Events will, however, follow a distribution of amplitudes @network_snr @gravitational_wave_population. As is often the case, we can try to generate our training data using a distribution that is as close as possible to the observed data, with the exception of a lower amplitude cutoff @dataset_mismatch_problems_3, or we can instead use a non-realistic distribution, uniformly or perhaps Gaussianly distributed across some amplitude regime which contains the majority of real signals --- making the assumption that any detection methods we train using this dataset will generalise to higher amplitudes, or failing that, that the missed signals will be so loud that they would not benefit greatly from improved detection methods.

Thus far in this subsection, we have been talking rather nebulously about waveform "amplitude", as if that is an easy thing to define in a signal composed of many continuous frequency components. There are at least three properties we might desire from this metric. Firstly, magnitude, some measure of the energy contained by the gravitational wave as it passes through Earth --- this measure contains a lot of physical information about the gravitational wave source. Secondly, significance, given the circumstances surrounding the signal, we may want to measure how likely the signal is to have been astrophysical rather than terrestrial, and finally, closely related to the significance and perhaps most importantly when designing a dataset for artificial neural network training, the detectability, given a chosen detection method this would act as a measure of how easy it is for that method to detect the signal.

Naively, one might assume that simply using the maximum amplitude of the strain, $h_op("peak")$, would be a good measure, and indeed, this would act as a very approximate measure of the ease of detection --- but it is not a complete one. Consider, for a moment, a sine-Gaussian with an extremely short duration on the order of tens of milliseconds but a maximum amplitude that is only slightly louder than a multi-second long BNS signal @first_bns. You can imagine from this example that the BNS would be considerably easier to detect, but if you were going by $h_op("peak")$ alone, then you would have no idea.

Within gravitational-wave data science, there are nominally two methods for measuring the detectability of a signal --- the Root-Sum-Squared strain amplitude @gravitational_wave_interfereometers @hrss_ref, $h_op("rss")$, and the optimal matched filter Signal Noise Ratio, $rho_"opt"$ @snr_ref @gravitational_wave_interfereometers. What follows is a brief description of these metrics.

==== The Root-Sum-Squared strain amplitude, $h_op("rss")$ <hrss-sec>

The Root-Sum-Squared strain amplitude, $h_op("rss")$:, is a fairly simple measure of detectability @hrss_ref. Unlike $rho_"opt"$, it is exclusive to gravitational-wave science. It accounts for the power contained across the whole signal by integrating the square of the strain across its duration, essentially finding the area contained by the waveform. It is given by

$ h_op("rss") = sqrt(integral (h_plus (t)^2 + h_times (t)^2 )d t) $

or written in its discrete form, which is more relevant for digital data analysis

$ h_op("rss") = sqrt(sum_(i=1)^(N) (h_plus [t_i]^2 + h_times [t_i]^2)) $

when $h_op("rss")$ is the root-sum-squared strain amplitude, $h_plus (t)$ and $h_times (t)$ are the plus and cross polarisations of the continuous strain, $h_plus (t_i)$ and $h_times (t_i)$ are the plus and cross polarisations of the discrete strain at the i#super("th") data sample, and $N$ is the number of samples in the waveform. 

It should be noted that with any measure that utilises the strain, such as $h_op("peak")$ and $h_op("rss")$, there is some ambiguity concerning where exactly to measure strain. You could, for example, measure the raw strains $h_plus$ and $h_times$ before they have been transformed by the appropriate detector antenna response functions, or you could take the strain $h$ after it has been projected onto a given detector. The advantage of the former is that you can fairly compare the magnitude of different gravitational waves independent of information about the interferometer in which it was detected. This is the commonly accepted definition of the $h_op("rss")$.

The $h_op("rss")$ is most often used during burst analysis as a measure of the detectability, magnitude, and significance of burst transients. Within CBC detection SNR, $rho_"opt"$, is often preferred. Whilst $h_op("rss")$ is a simple and convenient measure, it ignores noise, so it cannot by itself tell us if a signal is detectable.

==== Optimal Signal-to-Noise Ratio (SNR) ($rho_"opt"$) <snr-sec>

The optimal Signal-to-Noise Ratio (SNR), $rho_"opt"$, solves both of these issues by acting as a measure of detectability, magnitude, and significance in comparison to the background noise. Consequently, because it is relative to the noise, the magnitude of a given waveform can only be compared to the optimal SNR of a waveform that was obfuscated by a similar noise distribution. If a real gravitational-wave signal were detected in a single LIGO detector, say, LIGO Hanford, for example, then its optimal SNR would be significantly larger than the same signal detected only in VIRGO, even if the signal was aligned in each case to the original from the optimally detectable sky location. This is because the sensitivity of the VIRGO detector is substantially lower than the two LIGO detectors @detector_sensitivity, so the noise is proportionally louder compared to the waveforms.

It is, however, possibly a good measure of detectability, as detection methods do not much care about the actual magnitude of the signal when they are attempting to analyse one; the only relevant factors, in that case, are the raw data output, consisting of the portion of the gravitational-wave strain perceptible given the detector's antenna response function, see @final_response_equation, and the interferometer noise at that time.

The SNR can also sometimes be an ambiguous measurement, as there are multiple different metrics that are sometimes referred to by this name, most prominently, a ratio between the expected value of the signal and the expected value of the noise, or sometimes the ratio between the root mean square of the signal and noise.  Within gravitational-wave data science, though there is sometimes confusion over the matter, the commonly used definition for SNR is the matched filter SNR, $rho_"opt"$ @snr_ref. Since matched filtering is the optimal method for detecting a known signal in stationary Gaussian noise @snr_ref, we can use the result of a matched filter of our known signal with that signal plus noise as a measure of the detectability of the signal in a given noise distribution.

The optimal SNR, $rho_"opt"$, is given by 

$ rho_"opt" = sqrt(4 integral_0^infinity (|accent(h, tilde.op)(f)|^2)/ S(f) d f) $

where $rho_"opt"$ is the optimal SNR, S(f) is the one sided PSD, and 

$ accent(h, tilde.op)(f) = integral_(-infinity)^infinity h(x) e^(-i 2 pi f t) d t  $

is the Fourier transform of h(f). The coefficient of 4 is applied since, in order to use only the one-sided transform, we assume that $S(f) = S(-f) $, which is valid because the input time series is entirely real. This applies a factor of two to the output, and since we are only integrating between 0 and $infinity$ rather than $-infinity$ to $infinity$, we apply a further factor of 2. 

Because, again, for data analysis purposes, the discrete calculation is more useful, the $rho_"opt"$ of discrete data is given by

$ rho_"opt" = sqrt(4 sum_(k=1)^(N - 1) (|accent(h, tilde.op)[f_k]|^2)/ S(f_k)) $ <snr-equation>

where $N$ is the number of samples, and, in this case, the discrete Fourier transform $accent(h, tilde)[f]$ is given by

$ accent(h, tilde)[f_k] = sum_(i=1)^(N - 1) h[t_i] e^(-(2 pi)/N k i)  $ <fourier-transform-eq>

For the work during this thesis, we have added a TensorFlow @tensorflow implementation for calculating the $rho_"opt"$ to GWFlow @gwflow_ref. This implementation is shown in @snr_calculation.

#figure(
```py 
@tf.function 
def calculate_snr(
    injection: tf.Tensor, 
    background: tf.Tensor,
    sample_rate_hertz: float, 
    fft_duration_seconds: float = 4.0, 
    overlap_duration_seconds: float = 2.0,
    lower_frequency_cutoff: float = 20.0,
    ) -> tf.Tensor:
    
    injection_num_samples      = injection.shape[-1]
    injection_duration_seconds = injection_num_samples / sample_rate_hertz
        
    # Check if input is 1D or 2D
    is_1d = len(injection.shape) == 1
    if is_1d:
        # If 1D, add an extra dimension
        injection = tf.expand_dims(injection, axis=0)
        background = tf.expand_dims(background, axis=0)
        
    overlap_num_samples = int(sample_rate_hertz*overlap_duration_seconds)
    fft_num_samples     = int(sample_rate_hertz*fft_duration_seconds)
    
    # Set the frequency integration limits
    upper_frequency_cutoff = int(sample_rate_hertz / 2.0)

    # Calculate and normalize the Fourier transform of the signal
    inj_fft = tf.signal.rfft(injection) / sample_rate_hertz
    df = 1.0 / injection_duration_seconds
    fsamples = \
        tf.range(0, (injection_num_samples // 2 + 1), dtype=tf.float32) * df

    # Get rid of DC
    inj_fft_no_dc  = inj_fft[:,1:]
    fsamples_no_dc = fsamples[1:]

    # Calculate PSD of the background noise
    freqs, psd = \
        calculate_psd(
            background, 
            sample_rate_hertz = sample_rate_hertz, 
            nperseg           = fft_num_samples, 
            noverlap          = overlap_num_samples,
            mode="mean"
        )
            
    # Interpolate ASD to match the length of the original signal    
    freqs = tf.cast(freqs, tf.float32)
    psd_interp = \
        tfp.math.interp_regular_1d_grid(
            fsamples_no_dc, freqs[0], freqs[-1], psd, axis=-1
        )
        
    # Compute the frequency window for SNR calculation
    start_freq_num_samples = \
        find_closest(fsamples_no_dc, lower_frequency_cutoff)
    end_freq_num_samples = \
        find_closest(fsamples_no_dc, upper_frequency_cutoff)
    
    # Compute the SNR numerator in the frequency window
    inj_fft_squared = tf.abs(inj_fft_no_dc*tf.math.conj(inj_fft_no_dc))    
    
    snr_numerator = \
        inj_fft_squared[:,start_freq_num_samples:end_freq_num_samples]
    
    if len(injection.shape) == 2:
        # Use the interpolated ASD in the frequency window for SNR calculation
        snr_denominator = psd_interp[:,start_freq_num_samples:end_freq_num_samples]
    elif len(injection.shape) == 3: 
        snr_denominator = psd_interp[:, :, start_freq_num_samples:end_freq_num_samples]
        
    # Calculate the SNR
    SNR = tf.math.sqrt(
        (4.0 / injection_duration_seconds) 
        * tf.reduce_sum(snr_numerator / snr_denominator, axis = -1)
    )
    
    SNR = tf.where(tf.math.is_inf(SNR), 0.0, SNR)
    
    # If input was 1D, return 1D
    if is_1d:
        SNR = SNR[0]

    return SNR
```,
caption : [_ Python @python. _ The GWFlow TensorFlow @tensorflow graph function to calculate the optimal SNR, $rho_"opt"$, of a signal. `injection` is the input signal as a TensorFlow tensor, `background` is the noise into which the waveform is being injected, `sample_rate_hertz` is the sample rate of both the signal and the background, `fft_duration_seconds` is the duration of the FFT window used in the PSD calculation, `overlap_duration_seconds` is the duration of the overlap of the FFT window in the PSD calculation, and `lower_frequency_cutoff` is the frequency of the lowpass filter, below which the frequency elements are silenced.]
) <snr_calculation>

Once the optimal SNR or $h_op("rss")$ of an injection has been calculated, it is trivial to scale that injection to any desired optimal SNR or $h_op("rss")$ value. Since both metrics scale linearly when the same coefficient scales each sample in the injection,

$ h_op("scaled") = h_op("unscaled") M_op("desired") / M_op("current") $ <scaling-equation>

where $h_op("scaled")$ is the injection strain after scaling, $h_op("unscaling")$ is the injection strain before scaling, $M_op("desired")$ is the desired metric value, e.g. $h_op("rss")$, or $rho_"opt"$, and $M_op("current")$ is the current metric value, again either $h_op("rss")$, or $rho_"opt"$. Note that since $h_op("rss")$ and $rho_"opt"$ are calculated using different representations of the strain, $h_op("rss")$ before projection into a detector, and $rho_"opt"$ after, the order of operations will be different depending on the scaling metric of choice, ie. for $h_op("rss")$: scale $arrow$ project, and for $rho_"opt"$: project $arrow$ scale. 

#figure(
    image("scaling_comparison.png", width: 100%),
    caption: [Eight examples of artificial injections scaled to a particular scaling metric and added to a real noise background to show variance between different scaling methods. The blue line demonstrates the whitened background noise plus injection; the red line represents the injection after being run through the same whitening transform as the noise plus injection, and the green line represents the injection after scaling to the desired metric. The leftmost column contains an IMRPhenomD waveform, generated using @cuphenom_ref, injected into a selection of various background noise segments and scaled using SNR; see @snr-sec. From upper to lower, the SNR values are 4, 8, 12, and 16, respectively. The rightmost column displays a WNB injected into various noise distributions, this time scaled using $h_op("rss")$; see @hrss-sec. From upper to lower, the $h_op("rss")$ values are as follows: $8.52 times 10^(-22)$, $1.70 times 10^(-21)$, $2.55 times 10^(-21)$, and $3.41 times 10^(-21)$. As can be seen, though both sequences are increasing in linear steps with a uniform spacing of their respective metrics, they do not keep in step with each other, meaning that if we double the optimal SNR of a signal, the $h_op("rss")$ does not necessarily also double.]
) <scaling_comparison>

For the experiments performed later in this section, we will use SNR as our scaling metric drawn from a uniform distribution with a lower cutoff of 8 and an upper cutoff of 20. These values are rough estimates of a desirable distribution given the SNR values of previous CBC detections.

If we wish to utilise multiple detectors simultaneously as our model input, we can scale the injections using either the network SNR or the $h_op("rss")$ before projection into the detectors. In the case of $h_op("rss")$, the scaling method is identical, performed before detection and injection. Network SNR is computed by summing individual detector SNRs in quadrature @network_snr, as shown by

$ rho_op("network") = sqrt(sum_(i=1)^(N) rho_i^2) $ <network-snr>

where $rho_op("network")$ is the network SNR, $N$ is the total number of detectors included in the input, and $rho_i$ is the detector SNR of the i#super("th") detector given in each case by @snr-equation. To scale to the network, SNR @scaling-equation can still be used, with the network SNR of @network-snr as the scaling metric, by multiplying the resultant projected injection in each detector by the scaling coefficient.

=== Data Dimensionality and Layout <dim_sec>

Interferometer output data is reasonably different from the example MNIST data @mnist we have been using to train models thus far, the primary difference being that it is one-dimensional rather than two, being more similar to audio than image data. In fact, most of the features we are looking for within the data have a frequency that, when converted to sound, would be audible to the human ear @human_hearing, so it is often useful to think of the problem in terms of audio classification. In many ways, this reduced dimensionality is a simplification of the image case. In pure dense networks, for example, we no longer have to flatten the data before feeding it into the model; see @flatten-sec.

There are, however, multiple interferometers across the world. During an observing run, at any given time, there are anywhere between zero to five operational detectors online: LIGO Livingston (L1), LIGO Hanford (H1), Virgo (V1), Kagra (K1), and GEO600 (G1) @open_data (although as of this thesis, there has yet been a time when all five detectors were online). GEO600 is not considered sensitive enough to detect any signals other than ones that would have to be so local as to be rare enough to dismiss the probability, so it is usually not considered for such analysis @geo_sensitivity. It should also be noted that during O4, both Virgo and Kagra are currently operating with a sensitivity and up-time frequency that makes it unlikely they will be of much assistance for detection @current_status. It is hoped that the situation at these detectors will improve for future observing runs. Even with just the two LIGO interferometers, it is possible to include multiple detectors within our model input, and in fact, such a thing is necessary for coherence detection to be possible @x-pipeline @cWB.

This multiplicity brings some complications in the construction of the input examples. Currently, we have only seen models that ignore the input dimensionality; however, with other network architectures, such as Convolutional Neural Networks (CNNs), this is not always the case  @deep_learning_review. Therefore, we must consider the data layout. In the simplest cases, where we are not modifying the shape of the data before injection, we can imagine three ways to arrange the arrays; see @layout_options for a visual representation.

- *Lengthwise*: wherein the multiple detectors are concatenated end to end, increasing the length of the input array by a factor equal to the number of detectors. This would evidently still be a 1D problem, just an extended one. While perhaps this is the simplest treatment, we can imagine that this might perhaps be the hardest to interpret by the model, as we are mostly discarding the dimensionality, although no information is technically lost.
- *Depthwise*: Here, the detectors are stacked in the depth dimension, an extra dimension that is not counted toward the dimensionality of the problem, as it is a required axis for the implementation of CNNs, in which each slice represents a different feature map; see @cnn_sec. Often, this is how colour images are injected by CNNs, with the red, green, and blue channels each taking up a feature map. This would seem an appropriate arrangement for the detectors. However, there is one significant difference between the case of the three-colour image and the stacked detectors, that being the difference in signal arrival time between detectors; this means that the signal will be offset in each channel. It is not intuitively clear how this will affect model performance, so this will have to be empirically compared to the other two layouts.
- *Heightwise*: The last possible data layout that could be envisioned is to increase the problem from a 1D problem to a 2D one. By concatenating the arrays along their height dimension, the 1D array can be increased to a 2D array.

#figure(
    image("data_layout.png", width: 100%),
    caption: [Possible data layouts for multi-detector examples. Here, $d$ is the number of included detectors, and $N$ is the number of input elements per time series. There are three possible ways to align interferometer time-series data from multiple detectors. These layouts are discussed in more detail in @dim_sec. ]
) <layout_options>

For pattern-matching methods, like that which is possible in the CBC case, there are also advantages to treating each detector independently. If we do this, we can use the results from each model as independent statistics, which can then be combined to create a result with a far superior False Alarm Rate (FAR) @false_alarm_rate_ref. We could combine the score from both models and calculate a false alarm rate empirically using this combined score, or use each detector as a boolean output indicating the presence of a detector or not, and combine the FARs using @comb_far_eq.

For the first case treating the two models as one, the combined score is calculated by

$ op("S")_(op("comb")) = product_(i=1)^N op("S")_i $ 

where $op("S")_(op("comb"))$ is the combined classification score, which can be treated approximately as a probability if the output layer uses a softmax, or single sigmoid, activation function, see @softmax-sec, $op("S")_i$ is the output score of the $i^op("th")$ classifier input with the data from the $i^op("th")$ detector, and $N$ is the number of included detectors. Note that one could employ a uniquely trained and/or designed model for each detector or use the same model for each detector.

In the second case, treating each model as an independent boolean statistic and assuming that the output of the detectors is entirely independent except for any potential signal, the equation for combining FARs is

$ op("FAR")_(op("comb")) = (w - o)  product_(i=1)^N op("FAR")_i $ <comb_far_eq>

where $op("FAR")_(op("comb"))$ is the combined FAR, $N$ is the number of included detectors, $w$ is the duration of the input vector, and $o$ is the overlap between windows @false_alarm_rate_ref. This equation works in the case when a detection method tells you a feature has been detected within a certain time window, $w$, but not the specific time during that window, meaning that $t_"central" > w_"start" and t_"central" < w_"end"$, where $t_"central"$ is the signal central time, $w_"start"$ is the input vector start time and $w_"end"$ is the input vector end time. 

If a detection method can be used to ascertain a more constrained time for a feature ($w_"duration" < "light_travel_time"$), then you can use the light travel time between the two detectors to calculate a FAR @false_alarm_rate_ref. For two detectors, combing the FAR in this way can be achieved by

$ op("FAR")_(1, 2) = 2 op("FAR")_1 op("FAR")_2 w_(1,2) $

where $op("FAR")_(op("comb"))$ is the combined FAR, and $w_(1,2)$ is the light travel time between detectors 1 and 2, as this is the largest physically possible signal arrival time separation between detectors; gravitational waves travel at the speed of light, and detector arrival time difference is maximised if the direction of travel of the wave is parallel to the straight-line path between the two detectors.

/*
Generalising this kind of FAR combination to $N$ detectors becomes difficult because each pair of detectors will have a different maximum time separation, meaning that the sequence in which the detection occurs becomes important. The general formula is

$ op("FAR")_(op("comb")) = sum_op("all sequences S") op("FAR")_(S_1) times product_(i=1)^(N-1) op("FAR")_(S_(K+1)) times w_(S_(K) S_(K+1)) $

where a given sequence S is an order of detectors, i.e. $S = [1,2,3...N]$, and $"all sequences S"$ are all possible combinations of $S$, meaning that there are $N!$ possible sequences. For a more simplistic approach that will slightly overestimate the true FAR value, the maximum time separation between any pair of detectors is sometimes used instead of unique separations for each pair. Note that Equation 59 is the general case where only cases when all $N$ detectors have detections are considered detections; if less than $N$ detectors are required for detection, then we should use all combinations of sequences of length $X$ when $X$ is the number of detectors required for a confirmed detection.

*/

In the case where we are using $t_"central"$ and coincidence times to calculate our combined FAR, if we use overlapping data segments to feed our model, we must first group detections that appear in multiple inferences and find one central time for the detection. We can use an empirical method to determine how best to perform this grouping and identify if and how model sensitivity varies across the input window.

=== Feature Engineering and Data Conditioning <feature-eng-sec>

Invariably, there are data transforms that could be performed prior to ingestion by the model. If there are operations that we imagine might make the task at hand easier for the model, we can perform these transforms to improve network performance. Because we are attempting to present the data to the model in a form that makes the features easier to extract, this method of prior data conditioning is known as *feature engineering* @feature_engineering_ref. It should be noted that feature engineering does not necessarily add any extra information to the data. In fact, in many cases, it can reduce the overall information content whilst simultaneously simplifying the function that the model is required to approximate in order to operate as intended @feature_engineering_ref, see for example the whitening procedure described in @whitening-sec. As we have said before, although the dense neural network with a CAP above two, is, at its limit, a universal function approximator @universal_aproximators, there are practical limitations to finding the right architecture and parameters for a given function, so sometimes simplifying the task can be beneficial. This can reduce the model size and training time, as well as improve achievable model performance when the time available for model and training optimisation is limited @feature_engineering_performance.

==== Raw Data

When designing the package of information that will be presented to the network at each inference, the simplest approach would be to feed the raw interferometer data directly into the model. There are certainly some methodologies that consider it optimal to present a model with as much unaltered information as possible @deep_learning_review. By performing little to no data conditioning, you are allowing the network to find the optimal path to its solution; if all the information is present and an adequate model architecture is instantiated, then a model should be able to approximate the majority of possible conditioning transforms during model training, not only this, but it may be able to find more optimal solutions that you have not thought of, perhaps ones customised to the specific problem at hand, rather than the more general solutions that a human architect is likely to employ. This methodology, however, assumes that you can find this adequate model architecture and have an adequate training procedure and dataset to reach the same endpoint that could be achieved by conditioning the data. This could be a more difficult task than achieving a result that is almost as good with the use of feature engineering.

==== Whitened Data <whitening-sec>

One type of data conditioning that we will employ is time-series whitening @whitening_ref. As we have seen in @interferometer_noise_sec, as well as containing transient glitches, the interferometer background is composed of many different continuous quasi-stationary sources of noise, the frequency distributions of which compose a background that are unevenly distributed across our frequency search space @det_char. This leaves us with 1D time series that have noise frequency components with much greater power than any interesting features hidden within the data. This could potentially make detections using most methods, including artificial neural networks, much more difficult, especially when working in the time domain; see @whitnening_examples for an example of the PSD of unwhitened noise.

#figure(
    image("whitening_examples.png", width: 100%),
    caption: [An example of a segment of interferometer data before and after whitening. The two leftmost plots in blue show the PSD, _upper_, and raw data, _lower_, output from the LIGO Hanford detector before any whitening procedure was performed. The two rightmost plots show the same data after the whitening procedure described in @whitening-sec has been implemented. The data was whitened using the ASD of a #box("16.0" + h(1.5pt) + "s") off-source window from #box("16.5" + h(1.5pt) + "s") before the start of the on-source window to #box("0.5" + h(1.5pt) +"s") before. The #box("0.5" + h(1.5pt) +"s") gap is introduced as some data must be cropped after whitening due to edge effects caused by windowing. This also acts to ensure that it is less likely that any features in the on-source data contaminate the off-source data, which helps reduce the chance that we inadvertently whiten any interesting features out of the data.]
) <whitnening_examples>

Fortunately, there exists a method to flatten the noise spectrum of a given time series whilst minimising the loss of any transient features that don't exist in the noise spectrum @whitening_ref. This requires an estimate of the noise spectrum of the time series in question, which does not contain the hidden feature. In this case, this noise spectrum will take the form of an ASD; see @asd-func. 

Since the noise spectrum of the interferometer varies with time, a period of noise close to but not overlapping with the section of detector data selected for analysis must be chosen --- we call this time series the *off-source* period. The period being analysed, the *on-source* period, is not included in the off-source period so that any potential hidden features that are being searched for, e.g. a CBC signal, do not contribute significant frequency components to the ASD, which may otherwise end up dampening the signal along with the noise during the whitening procedure. It should be noted, then, that whitening via this process uses additional information from the off-source period that is not present in the on-source data. During this thesis, we have elected to use an off-source window duration of #box("16.0" + h(1.5pt) +"s"), as this was found to be an optimal duration by experiments performed as part of previous work during the development of MLy @MLy, although it should be noted that we have taken the on-source and crop regions after the off-source as opposed to the initial MLy experiments wherein it was taken at the centre of the off-source window. See @onsource_offsource_regions for a depiction of the relative locations of the on-source and off-source segments.

#figure(
    image("onsource_offsource_regions.png", width: 100%),
    caption: [Demostration of the on-source and off-source regions used to calculate the ASD used during the whitening operations throughout this thesis wherever real noise is utilised. Where artificial noise is used, the off-source and on-source segments are generated independently but with durations equivalent to what is displayed above. The blue region shows the #box("16.0" + h(1.5pt) + "s") off-source period, the green region shows the #box("1.0" + h(1.5pt) + "s") on-source period, and the two red regions represent the #box("0.5" + h(1.5pt) + "s") crop periods, which are removed after whitening. During an online search, the on-source region would advance in second-long steps, or if some overlap was implemented, less than second-long steps, meaning all data would eventually be searched. The leading #box("0.5" + h(1.5pt) + "s") crop region will introduce an extra #box("0.5" + h(1.5pt) + "s") of latency to any search pipeline. It may be possible to avoid this latency with alternate whitening methods, but that has not been discussed here. ]
) <onsource_offsource_regions>

We can whiten the data by convolving it with a suitably designed Finite Impulse Response (FIR) filter. This procedure is described by the following steps:

+ Calculate the ASD using @asd-func, this will act as the transfer function, $G(f)$, for generating the FIR filter. This transfer function is a measure of the frequency response of the noise in our system, and during the whitening process, we will essentially try to normalise the on-source by this off-source noise in order to flatten its PSD. We generate a filter with a #box("1" + h(1.5pt) + "s") duration.
+ Next, we zero out the low and high-frequency edges of the transfer function with

$ G_"trunc" (f) = cases(
  0 "if" f <= f_"corner",
  G(f) "if" f_"corner" < f < f_"Nyquist" - f_"corner",
  0 "if" >= f_"Nyquist" - f_"corner"  
). $

 This stage discards frequency components which we no longer care about both because these frequencies are outside of the band we are most interested in and because discarding them can improve function stability and performance whilst reducing artifacting.
 
3. Optionally, we can apply a Planc-taper window to smooth the discontinuities generated by step 2; we will apply this window in all cases. The Planc-taper window has a flat centre with smoothly tapering edges, thus the windowing is only applied as such to remove discontinuities whilst affecting the central region as little as possible.

$ G_"smoothed" (f) = G_"trunc" (f) dot W(f). $

4. Next we compute the inverse Fourier transform of $G_"smoothed" (f)$ to get the FIR filter, $g(t)$, with 

$ g(t) = 1 / (2 pi) integral_(-infinity)^infinity G_"smoothed" (f) e^(j f t) d f. $

This creates a time-domain representation of our noise characteristics, which can then be used as a filter to remove similar noise from another time-domain signal. In practice, we utilise an RFFT function to perform this operation on discrete data. As opposed to an FFT, this transform utilises symmetries inherent when transforming from complex to real data in order to halve the computational and memory requirements.

5. Finally, we convolve our FIR filter, $g(t)$, with the data we wish to whiten, $x(t)$,

$ x_"whitened" (t) = x(t) ast.op g(t) $ 

where $x_"whitened" (t)$ is the resultant whitened time-series, $x(t)$ is the original unwhitened data, and $g(t)$ is the FIR filter generated from the off-source ASD. This convolution effectively divides the power of the noise at each frequency by the corresponding value in $G(f)$. This flattens the PSD, making the noise uniform across frequencies; see @whitnening_examples for an example of this transform being applied to real interferometer data.

This method was adapted from the GWPy Python library @gwpy and converted from using NumPy functions @numpy to TensorFlow GPU operations @tensorflow in order to work in tandem with the rest of the GWFlow @gwflow_ref pipeline and allow for rapid whitening during the training process.

==== Pearson Correlation
A method of feature engineering that is employed prominently by the MLy pipeline @MLy involves extracting cross-detector correlation using the Pearson correlation @pearson_ref.  The Pearson correlation is given by

$ r = frac( N (sum_(i=0)^N x_i y_i) - (sum_(i=0)^N x_i) (sum_(i=0)^N y_i ) , sqrt( [N sum_(i=0)^N x_i^2 - (sum_(i=0)^N x_i)^2] times [N sum_(i=0)^N y_i^2 - (sum_(i=0)^N y_i)^2] ) ) $

where r is the Pearson correlation coefficient, N is the number of data points in each input array, and $x_i$ and $y_i$ are the i#super("th") elements of the $accent(x, arrow)$ and $accent(y, arrow)$ arrays respectively @pearson_ref.

Nominally, this produces one scalar output value given two input vectors, $accent(x, arrow)$ and $accent(y, arrow)$, of equal length, $N$. A value of $r = 1$ indicates perfect correlation between the two vectors, whereas a value of $r = -1$ indicates perfect anti-correlation. Finally, a value of $r = 0$ indicates no correlation between the vectors. Note that if one of the vectors is entirely uniform, then the result is undefined. 

This calculation assumes that the two vectors are aligned such that the value in $x_i$ corresponds to the value in $y_i$. If this is not the case, as would happen for interferometer data if there is an arrival time difference (which there will be for most sky locations), then this will be an imperfect measure of correlation, even discarding the obfuscation of the noise. Because, as was discussed previously in @projection-sec, we do not know the direction of the source a priori, MLy @MLy calculates the correlation for all possible arrival times given the light travel time between the two detectors in question. It uses minimum increments of the sample duration so that no heterodyning is necessary. This is done with the assumption that any difference in arrival time less than the sample duration will have a negligible effect on the correlation. It should be noted that this method is still hampered by the different polarisation projections dependent on the source polarization and by the obfuscating noise. See @pearson_example for examples of the rolling Pearson correlation calculated for LIGO Hanford and LIGO Livingston interferometer data.

#figure(
    image("pearson_example.png", width: 100%),
    caption: [Example whitened on-source and correlation plots of real interferometer noise from a pair of detectors, in this case, LIGO Livingston and LIGO Hanford, with either coherent, incoherent, or no injections added. The leftmost plots adjacent to the info panels are grouped into pairs. In each case, LIGO Livingston is at the top, and LIGO Hanford is underneath. Identical on-source and off-source noise segments are used for each example of the same detector, and noise for each detector was gathered with a time difference of no more than #box("2048.0" + h(1.5pt) + "s"). In the leftmost plots, the green series is the unwhitened but projected waveform to be injected into the real noise from that detector. The red series is that same injection but subject to the same whitening procedure that will also be applied to the on-source plus injections, and the blue series is the whitened on-source plus injections. The rightmost plots each correspond to a pair of detectors and display the rolling Pearson correlation values between those two whitened on-source plus injection series. Since there is approximately a max arrival time difference of #box("0.01" + h(1.5pt) + "s") between LIGO Livingston and LIGO Hanford, the number of correlation calculations performed corresponds to the rounded number of samples required to represent #box("0.02" + h(1.5pt) + "s") of data at #box("2048.0" + h(1.5pt) + "Hz"). This number is two times the maximum arrival time difference because the difference could be positive or negative. In this case, that difference comes to 40 samples. All injections have been scaled to an optimal network SNR of 30 using the method described in @snr-sec. The upper pair of detectors has no injection. As would be expected, the correlation is low regardless of the assumed arrival time difference. The second pair from the top has been injected with a coherent white noise burst (WNB), see @injection-gen-sec, which has been projected onto the two detectors using a physically realistic mechanism previously described in @projection-sec. Here, the correlation is much stronger. We can see it rise and fall as the waveforms come in and out of coherence. The third from the top, the central plot, shows an injection of two incoherent WNBs. They are processed identically to the coherent case, but the initial waveforms are generated independently, including their durations. The Pearson correlation looks very similar to the pure noise case in the uppermost plot, as might be expected. The second from the lowest pair has been injected with a coherent IMRPhenomD waveform, which again has been correctly projected. We can observe that a small correlation is observed at an arrival time difference of around #box("0.005" + h(1.5pt) + "s"), suggesting that the two waveforms arrived at the detectors #box("0.005" + h(1.5pt) + "s") apart. Finally, the lowest plot depicts two incoherent IMRPhenomD waveforms projected into the noise. Though these are generated with different parameters, the shared similarities in morphology between all CBC waveforms cause correlation to be registered. By maximum amplitude alone, it may even appear as though there is more correlation happening here than in the correlated case. This highlights one potential weakness of using the Pearson correlation, which can sometimes show some degree of correlation even if the two waveforms are not produced using the same physically simulated mechanism.]
) <pearson_example>

As with most mathematical functions, we have created a new GPU-based function for the calculation of the Pearson correlation in Python @python, using the TensorFlow GPU library @tensorflow for computational speed and easy integration with the rest of the GWFlow pipeline @gwflow_ref.

==== Fourier Transform

So far, we have looked at data conditioning, which produces results in the time domain. As we know, and as has been demonstrated by the previous discussion, many aspects of time series processing are performed in the frequency domain. Often, features that are hard to distinguish in the time domain are relatively easy to spot in the frequency domain, even with the human eye. Many have characteristic morphologies, such as distinct lines due to powerline harmonics and violin modes. If we make the assumption that if it is easier for a human, it might also be easier for a machine learning method, we should certainly examine feature engineering methods that take us into the frequency domain. The most obvious way to do this would be to use a simple Fourier transform @fourier_transform_ref, which takes us directly from a time-domain series to a frequency-domain one. The discrete form of the Fourier transform is given above in @fourier-transform-eq.

==== Power Spectral Density (PSD) and Amplitude Spectral Density (ASD)

As discussed in @psd-sec @psd_ref, the PSD is used in many calculations and transforms in gravitational wave data analysis, so it makes sense that along with the closely related property, the ASD, it may also be useful information to provide to a model. Since the PSD has already been discussed in detail in @psd-sec, we will not linger on it here.

==== Spectrograms

The final feature engineering method that we will discuss allows us to represent data in both the time and frequency domains simultaneously. Spectrograms are visualisations of the Short-Time Fourier Transform (STFT) of a time series @spectrograms_ref. The STFT is computed by dividing a time series into many smaller periods, much like in the calculation of a PSD; however, instead of being averaged, you can simply use this 2D output as an image in its own right, which displays how the frequency components of a time series fluctuate over its duration. This retains some information from the time domain. The 2D STFT of a continuous time series, $x(t)$, is given by

$ op("STFT")(x)(t, f) = integral_(-infinity)^infinity x(tau) w(t - tau) e^(-i 2 pi f tau) d tau  $

where $op("STFT")(x)(f, t)$ is the value of the STFT of $x(t)$ at a given time, $t$, and frequency, $f$, $w(t)$ is a configurable window function that helps to minimize the boundary effects, and $tau$ is a dummy integration variable used to navigate through the time domain at the expense of losing some information from the frequency domain, making the spectrogram, like whitening, a lossy transform. In its discrete form, this becomes

$  op("STFT")(x)[n, k] = sum_(m = 0)^(N-1) x[m] w[n - m] e^((-i 2 pi k m) / N)  $ <stft-eq>

where $op("STFT")(x)[n, k]$ is the value of the discrete STFT of a discrete time series, $x[m]$ at a given time index, $n$, and frequency index, $k$, $w[t]$ is a discrete window function, and N is the number of samples in our discrete time series. It should be noted that there are two time indices present, $n$ and $m$, because a reduction in dimensionality along the time axis usually occurs since the step between adjacent FFT segments is commonly greater than one.

When creating a spectrogram, the values are typically squared,

$ S[k, n] = (op("STFT")(x)[n, k])^2 $ <stft_sq>

to represent the power of the frequency components, similar to the process of calculating the PSD. Alternatively, the magnitude can be taken with

$ S[k, n] = |op("STFT")(x)[n, k]|. $

Before plotting, the data is often converted into decibels to better visualize the dynamic range,

$ op("DATA") = 10 times log (S[k, n]). $ <dec-eq>

We have created a custom Python TensorFlow function @tensorflow to perform these calculations on the GPU; see @spectrogram_examples for illustrations of this in use on real noise with injected waveform approximants. As is the case with multiple 1D time series, the question also remains of how to combine multiple spectrograms in the case of multiple detector outputs, see @dim_sec.

#figure(
    image("spectrogram_examples.png", width: 85%),
    caption: [Six example noise segments and their corresponding spectrograms. In all cases, the noise is real interferometer data acquired from the LIGO Hanford detector during the 3#super("rd") observing run. It is whitened using the procedure described in @whitening-sec. For the time series plots, the green series represents the original, unwhitened waveform before injection, the red series is the waveform with the same whitening transform applied to it as was applied to the on-source background plus injection, and the blue series is the whitened on-source background plus injection, except for the first two time series plots which contain no injection. The spectrograms are generated using the STFT described by @stft-eq, converted into power with @stft_sq, and finally transformed into a decibel logarithmic scale for plotting using @dec-eq. The two uppermost plots and their respective spectrograms have no injections. The two middle plots and their respective spectrograms have IMRPhenomD @imrphenom_d approximants created with cuPhenom injected into the noise @cuphenom_ref, and the two lower plots and their respective spectrograms, have White Noise Burst (WNB) waveforms generated using the method described in @injection-gen-sec, injected into the noise. In all cases, the injections are scaled to an optimal SNR randomly selected between 15 and 30; these are quite high values chosen to emphasize the features in the spectrograms. As can be seen, the whitened noise that contains injected features has spectrograms with highlighted frequency bins that have a magnitude much larger than the surrounding background noise; the different signal morphologies also create very different shapes in the spectrograms. This allows us to see the frequency components of the signal more easily, observe the presence of interesting features, and differentiate between the WNB and the CBC case. ]
) <spectrogram_examples>

==== Summary

There are multiple different possibilities for how to condition the data before it is fed into any potential machine learning model; see @feature-enginering-types, and we have only covered some of the possibilities. Most methods come at the cost of removing at least some information from the original data. It remains to be seen, however, if this cost is worthwhile to ensure adequate model performance and feasible training durations.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [*Possible Model Inputs*],  [*Dimensionality of Output*], [*Output Domain*],
    [Raw Onsource + Injection], [1], [Time],
    [Whitened Onsource + Injection], [1], [Time],
    [Pearsons Corrleation], [1], [Time],
    [Fourier Transform (PSD)], [1], [Frequency],
    [Power Spectral Density (PSD)], [1], [Frequnecy],
    [Spectrogram ], [2], [Time and Frequency]
  ),
  caption: [A non-exhaustive table of possible data conditioning modes. Feature engineering is often used in order to simplify a problem before it is presented to a machine learning model. There are many ways we could do this with gravitational-wave data. Presented are some of the most common. Each is described in more detail in @feature-eng-sec.]
) <feature-enginering-types>

=== Transient Glitch Simulation

As has previously been noted, as well as a quasi-stationary coloured Gaussian background, interferometer noise also contains transient detector glitches caused by a plethora of sources, both known and unknown. These glitches have a prominent effect on the upper-sensitivity bound of most types of search, so it may be important to represent features of this type in our training pipeline. Previous experiments performed during the development of the MLy pipeline @MLy had shown that networks can often have greatly increased FARs when performing inference on data segments that contain transient glitches, even when those glitches were only present in the off-source segment used to generate the PSD used for data whitening. As such, a method to add glitches to the training distribution should be considered so that methods to deal with features of this type can hopefully be incorporated into the model's learned parameters during training.

There have been multiple attempts to classify and document the many transient glitches found in real interferometer data @noise_clasification @dict_glitch_classifer @gravity_spy, both through automated and manual means @online_glitch_classification_review. During operation within a standard observing run, there are both intensive manual procedures @O2_O3_DQ to characterise the detector state and automated pipelines such as the iDQ pipeline @idq. There is also a large amount of work done offline to characterise the noise in a non-live environment @O2_O3_DQ. These methods utilize correlation with auxiliary channels, frequency of triggers, and other information about the detector state to ascertain the likelihood that a given feature is a glitch or of astrophysical origin.

One of the most prominent attempts to classify transient glitches is the Gravity Spy project @gravity_spy, which combines machine learning and citizen science to try and classify the many morphologies of transient glitches into distinct classes. Successful methods to classify glitches are highly useful since if a similar morphology appears again in the data it can be discounted as a probable glitch. Gravity Spy differentiates glitches into 19 classes plus one extra "no_glitch" class for noise segments that are proposed that do not contain a glitch. The other 19 classes are as follows: air_compressor, blip, chirp, extremely_loud, helix, koi_fish, light_modulation, low_frequency_burst, low_frequency_lines, none_of_the_above, paired_doves, power_line, repeating_blips, scattered_light, scratchy, tomte, violin_mode, wandering_line, and whistle. Some types, such as blips, are much more common than others.

There are two options we could use as example data in our dataset in order to familiarise the model with glitch cases. We could either use real glitches extracted from the interferometer data using the timestamps provided by the Gravity Spy catalog @gravity_spy or simulated glitches we generate ourselves. The forms of each would vary depending on whether it was a multi, or single-detector example and whether we are attempting to detect CBCs or bursts.

*Real Glitches:* The addition of real glitches to the training dataset is a fairly intuitive process, though there are still some parameters that have to be decided upon. By using timestamps from the Gravity Spy catalog @gravity_spy, we can extract time segments of equal length to our example segments, which contain instances of different classes of glitches. We should process these identically to our regular examples with the same whitening procedure and off-source segments. Real glitches have the distinct advantage that any model will be able to use commonalities in their morphology to exclude future instances; this is also, however, their disadvantage. If you train a model on specific morphologies, then the introduction of new glitch types in future observing runs, which may well be possible given the constant upgrades and changes to detector technology, then it may be less capable of rejecting previously unseen glitch types @gravity_spy. However, it is still possible that these glitches will help the model to reject anything other than the true type of feature it has been trained to recognise by weaning it off simple excess power detection.

*Simulated Glitches:* The other option is to use simulated glitches. The form of these glitches depends highly on the nature of the search, primarily because you wish to avoid confusion between the morphology of the feature you want the method to identify and simulated glitches. For example, in a CBC search, you could use WNBs as simulated glitches, as their morphologies are entirely distinct, and there is no possibility of confusion. However, if we are using coherent WNBs across multiple detectors to train a model to look for coherence, then we must be careful that our glitch cases do not look indistinguishable from true positive cases, as this would poison the training pool by essentially mislabeling some examples. We could, in this case, use incoherent WNBs as simulated glitches as, ideally, we want our coherent search to disregard incoherent coincidences. This is the approach taken by the MLy pipeline @MLy, as a method to train the models to reject counterexamples of coherent features.

Other than the question of whether to use simulated or real glitches or maybe even both, a few questions remain: what is the ideal ratio between examples of glitches and non-glitched noise examples? Should the glitched background also be injected with waveforms at some rate? A real search would occasionally see glitches overlapping real signals, though this would occur in a relatively low number of cases, and including these types of signal-glitch overlaps could perhaps interfere with the training process whilst not adding a great deal of improvement to the true positive rate. Should glitches form their own class so that the model instead has to classify between signal, noise, or glitch rather than just signal or noise? These questions must be answered empirically.

For the multi-detector case, and thus also the burst detection case, we must decide how to align glitches across detectors. It seems safe to assume that adding coherent glitches across multiple detectors would be a bad idea in a purely coherence-based search pipeline --- although perhaps if the model can learn to disregard certain morphologies based on prior experience, this would be a nice extension. For some simple glitch types, coincident and fairly coherent instances across detectors are not extremely unlikely. For example in the case of the most common glitch class identified by GravitySpy @gravity_spy, blips, we often see coincident glitches in multiple detectors with a physically plausible arrival time difference, and because they are only glitches, their morphologies can often be similar.

We could also include cases of incoherent glitches across detectors but of the same class, incoherent glitches across detectors but of different classes, and any combination of glitches found in less than the full complement of detectors. Perhaps it would be the case that a good mix of all of these cases would better inoculate our model against glitches.

== Perceptron Results <perceptron-results>

Now that we have finally assembled all the pieces required to generate training, testing, and validation datasets that can acquire training examples using and/or real data, we can finally repeat the experiments we performed on the MNIST data in @mnist-test-sec, with both single and multi-layer perceptrons. The model architectures are similar, though the input vectors are now the size of our simulated interferometer output examples: `(NUM_EXAMPLES_PER_BATCH, NUM_SAMPLES)` in the case of the single detector CBC search and `(NUM_EXAMPLES_PER_BATCH, NUM_DETECTORS, NUM_SAMPLES)` in the multi-detector coherent burst search. We will use 32 training examples per batch, `NUM_EXAMPLES_PER_BATCH = 32`, as this is a standard power-of-two value used commonly across artificial neural network literature, and, in the multi-detector case, we will use only LIGO Hanford and LIGO Livingston, for now, excluding the Virgo detector, `NUM_DETECTORS = 2`. We have chosen to use only the two LIGO detectors as in many ways, this is the simplest possible multi-detector network case; signals projected onto these two detectors will have a greater similarity than signals projected onto either of these two detectors and the Virgo detector, both due to sensitivity and orientation and position differences. We have chosen to use a sample rate of #box("2048.0" + h(1.5pt) + "Hz") and an on-source duration of #box("1.0" + h(1.5pt) + "s"), allowing an additional crop region #box("0.5" + h(1.5pt) + "s") either side of the onsource segment to remove edge effects created when whitening with #box("16.0" + h(1.5pt) + "s") of off-source background. The reasoning for these choices has been described previously in this chapter. This means we will have 2048 samples per detector, `NUM_SAMPLES = 2048`, after it has been passed through the whitening layer. A flattening layer, see @flatten-sec, will only be required in the multi-detector case; in the single-detector case, the input is already one-dimensional. The batch dimensions are not a dimension of the input data and simply allow for parallel processing and gradient descent; see @gradient-descent-sec. 

The obfuscating noise consists of real data taken from LIGO Hanford and LIGO Livingston @open_data for each respective detector. Locations of confirmed and candidate events are excluded from the data, but known glitch times have been included in the training, testing, and validation datasets.

For the single CBC case, cuPhenom @cuphenom_ref waveforms with masses drawn from uniform distributions between #box("5.0" + h(1.5pt) + $M_dot.circle$) and #box("95.0" + h(1.5pt) + $M_dot.circle$) for the mass of both companions and between -0.5 and 0.5 for the dimensionless aligned-spin component are injected into the noise and scaled with optimal SNR values taken from a uniform distribution of between 8.0 and 15.0 unless explicitly stated.

For the multi-detector Burst case, coherent WNBs are injected with durations between #box("0.1"+ h(1.5pt) + "s") and #box("1.0" + h(1.5pt) + "s"), and the frequencies are limited to between #box("20.0" + h(1.5pt) + "Hz") and #box("500.0" + h(1.5pt) + "Hz"). The injected bursts are projected correctly onto the detectors using a physically realistic projection. The bursts are injected using the same scaling type and distribution as the CBC case, although notably, the network SNR was used rather than a single detector SNR.

During network training, the gradients are modified by batches consisting of 32 examples at a time, chosen as an industry standard batch size, and with a learning rate of 1.0 $times$ 10#super("-4"), and using the Adam optimiser @adam_optimiser, which again is a common standard across the industry @improved_adam. During training epochs, $10^5$ examples are used before the model is evaluated against $10^4$ examples of the previously unseen test data. It should be noted that due to the nature of the generators used for the training, unlike in standard model training practices, no training examples are repeated across epochs, but the test dataset is kept the same for each epoch. After each epoch, if the validation loss for that epoch is the lowest yet recorded, the model is saved, replacing the existing lowest model. If no improvement in validation loss is seen in ten epochs (patience), the training is halted, and the best model is saved for further validation tests. @perceptron-training-parameters shows a large number of the training and dataset hyperparameters. 

#figure(
  table(
    columns: (auto, auto),
    inset: 10pt,
    align: horizon,
    [*Hyperparameter*],  [*Value*],
    [Batch Size], [32],
    [Learning Rate], [10#super("-4")],
    [Optimiser], [ Adam ],
    [Scaling Method], [SNR],
    [Minimum SNR], [8.0],
    [Maximum SNR], [15.0],
    [SNR Distribution], [Uniform],
    [Data Acquisition Batch Duration], [ #box("2048.0" + h(1.5pt) + "s") ],
    [Sample Rate], [ #box("2048.0" + h(1.5pt) + "Hz")],
    [On-source Duration], [ #box("1.0" + h(1.5pt) + "s")],
    [Off-source Duration], [ #box("16.0" + h(1.5pt) + "s")],
    [Scale Factor], [10#super("21") ],
    
  ),
  caption: [The common training and dataset hyperparameters shared by the CBC and Burst perceptron experiments. Note that the scale factor here refers to the factor used during the upscaling of the CBC waveforms and real interferometer noise from their extremely small natural dimensions to make them artificial neuron-friendly. This is done both to ensure that the input values work well with the network activation functions and learning rates, which are tuned around values near one, and to reduce precision errors in areas of the code that use 32-bit precision, employed to reduce memory overhead, computational cost and duration. Data acquisition batch duration is a parameter of the GWFlow data acquisition module @gwflow_ref. For speed, the GWFlow data acquisition system downloads data in larger segments than is required for each training batch, then randomly samples examples from this larger segment to assemble each training batch. The data acquisition batch duration determines how long this larger batch is. Smaller values will result in a more evenly mixed training data set and a lower overall GPU memory overhead but will be more time-consuming during the training process. ]
) <perceptron-training-parameters>

==== Architectures

We used architectures with four different layer counts: zero, one, two, and three hidden layers; see @perceptron-cbc-architectures. All models have a custom-implemented whitening layer, which takes in two vectors, the on-source and off-source segments, and performs a whitening operation as described in @whitening-sec. They also all have a capping dense layer with a single output value that represents either the presence of a feature or the absence of one. The capping layer uses the Sigmoid activation function; see @softmax, and the other hidden layers use ReLU activation functions, see @relu. 

Layers are built with a number of neurons selected from this list $[64, 128, 256, 512]$, though fewer combinations are tested in architectures with a greater number of model layers. Models tested have these 14 configurations of neuron numbers per layer, specified as [num_hidden_layers:num_neurons_in_layer_1 ... num_layers_in_layer_n]: ([0], [1:64], [1:128], [1:256], [1:512], [2:64,64], [2:128,64], [2:128,128], [2:256,64], [2:256,128], [2:256,256], [3:64,64,64], [3:128,128,128], [3:256,256,256]). These combinations were chosen to give a reasonable coverage of this section of the parameter space, though it is notably not an exhaustive hyperparameter search. From the performances demonstrated in this search compared to other network architectures, it was not deemed worthwhile to investigate further. 

#figure(
    image("perceptron_diagrams.png", width: 90%),
    caption: [Perceptron diagrams. The four different architectures used to test the use of purely dense models for both the single-detector CBC detection case and the multi-detector burst detection problem. The only differences are that the input vector sizes are different between the cases: `(NUM_EXAMPLES_PER_BATCH, NUM_SAMPLES)` in the case of the single detector CBC search and `(NUM_EXAMPLES_PER_BATCH, NUM_DETECTORS, NUM_SAMPLES)` in the multi-detector coherent burst search. All models take in two input vectors into a custom-designed GWFlow whitening layer, the off-source and the on-source vectors; see @whitening-sec for more information about the whitening procedure, and all models are capped with a dense layer with a single output neuron that is used to feed the binary loss function, with a sigmoid activation function. Each hidden layer has been tested with 64, 128, and 256 neurons, and one hidden layer was tested with 512 as a sample with higher neuron counts:  _Top:_ Zero-hidden layer model. _Second to top:_ Two-hidden layer model. _Second to bottom:_ Three-hidden layer model. _Bottom:_ One hidden layer model. ]
) <perceptron-cbc-architectures>

=== CBC Detection Dense Results

==== Training 

First, we can examine the results of applying dense-layer perceptrons to the CBC single-detector morphology detection problem. Even during the training process, it is clear that, at least amongst the selected hyperparameters, these models will not be useful; see @perceptron_single_accuracy and @perceptron_single_loss. None reach an accuracy of above 75% with a training patience of ten epochs. Setting a training patience of ten ensures that if no improvement in the validation loss is seen within ten epochs, the training process is halted. Examining the plots; see @perceptron_single_accuracy and @perceptron_single_loss, it seems possible that some of the perceptrons are on a very slow training trajectory and could have seen some marginal improvement if the training patience had been increased. It is also possible that other larger perceptron architectures may achieve greater success, as this was far from an exhaustive or even guided search of the perceptron hyperparameter space. However, as can be seen in @perceptron_single_accuracy, the models take a significant number of epochs to reach the values they do, which is what we would expect from entirely dense models. As will be seen in later sections, see @cnn-literature, other architectures can achieve much better results in fewer epochs. These results are here to act as an example of the difficulties of training dense networks for complex recognition tasks. For comparison with other methods, a more sophisticated analysis will be shown after the training history plots; see @single-perceptron-validation-sec. 

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("perceptron_single/perceptron_single_training_accuracy.png",   width: 100%) ],
        [ #image("perceptron_single/perceptron_single_validation_accuracy.png", width: 100%) ],
    ),
    caption: [The accuracy history of perceptron models training to detect IMRPhenomD waveforms generated using cuPhenom @cuphenom_ref that have been obfuscated by real interferometer noise sampled from the LIGO Livingston detector during the 3#super("rd") observing run. Visit #link("http://tinyurl.com/ypu3d97m") for interactive plots, whilst they're still working. The optimal SNR of waveforms injected into the training and validation sets was uniformly distributed between 8 and 15. Input was from a single detector only. A rough search was performed over a relatively arbitrary selection of model architectures, which varied the number of layers and the number of perceptrons in each layer. The architectures of each model can be seen in the figure legends as a list of numbers where each digit is the number of artificial neurons in that layer. All are trained with the same training hyperparameters, details of which can be found in @perceptron-training-parameters. Each epoch consisted of $10^5$ training examples, and it should be noted that, unlike the regular training pipelines, each training epoch consisted of newly generated waveforms injected into unseen noise segments, though the validation examples are consistent. Training of each model was halted after ten consecutive epochs with no improvement to validation loss, the values of which are shown in @perceptron_single_loss. Validation noise was drawn from a separate pool of data segments inaccessible to the training data loader. We can see that the maximum accuracy achieved by any perceptron model only approaches 75%. Although these validations are performed with a pool containing mixed waveform SNRs and at an unrestrained False Alarm Rate (FAR) (this accuracy uses a score threshold of 0.5 regardless of FAR), it is clear that this is insufficient to be useful. _Upper:_ Plot of model accuracies when measured with training data ($10^5$ epoch-unique examples). _Lower:_ Plot of model accuracies when mesured with validation data ($10^4$ epoch-consistent examples).]
) <perceptron_single_accuracy>

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("perceptron_single/perceptron_single_training_loss.png",   width: 100%) ],
        [ #image("perceptron_single/perceptron_single_validation_loss.png", width: 100%) ],
    ),
    caption: [Training loss history of perceptron models training to detect IMRPhenomD waveforms generated using cuPhenom @cuphenom_ref, obfuscated by real interferometer noise from the LIGO Livingston detector from the 3#super("rd") observing run. The loss is computed using binary cross entropy loss function and is used by the gradient descent algorithm, in this case, the Adam optimizer, as a minimization target. It also acts as the monitor by which the pipeline knows to stop the training process early. If the pipeline detects that the validation model loss has not decreased in more than 10 epochs, training is halted.
    Visit #link("http://tinyurl.com/ypu3d97m") for interactive plots. See @perceptron_single_accuracy for a more detailed description of the training data. _Upper:_ Plot of model loss when measured with training data ($10^5$ epoch-unique examples). _Lower:_ Plot of model loss when mesured with validation data ($10^4$ epoch-consistent examples).]
) <perceptron_single_loss>

==== Validation <single-perceptron-validation-sec>

Although the perceptron training performance was low, and probably sufficient to tell us that at least these configurations of perceptrons are not capable enough for CBC detection, a more complete validation was nonetheless performed on the trained models using the third as-yet-unseen validation dataset. This was both for comparison with later methods and to ensure that our initial assessment of the results was correct. Although it is easy to draw quick conclusions from the training results, it is not an accurate profile of the model performance, as the training validation results draw from a mixed pool of SNR values, do not consider the classes independently, and in the case of the accuracy result, use an uncalibrated detection threshold of 0.5. This means that if a model outputs a score over 0.5 it is considered a detection, and a score lower than 0.5 is considered noise. By tuning this threshold, we can arrive at the desired False Alarm Rate (FAR), though this will have an inverse effect on the sensitivity, (the true positive rate) of the model.

Before we can apply this tuning we must evaluate our model's performance on a dataset consisting exclusively of noise examples. The perfect classifier would output zero for all examples in such a dataset. We are not dealing with perfect classifiers, so the model will output a score value for each pure noise example. If our classifier has good performance most of these scores will be low, preferably near zero, but some will inevitably rise above whatever detection threshold we set, dependant of course on the size of our validation dataset, the larger the dataset the larger the expected value of our largest noise score. The size of the dataset required for threshold calibration will depend on the value of FAR that is desired, with smaller FARs requiring larger datasets. We will require a dataset in which the combined example durations sum to at least the duration of time wherein, given our desired FAR, we would expect one detection. However, since this is a statistical result, having only the exact duration required for our FAR would result in a great deal of error on that value. The larger the validation dataset, the more confident we can be in our calculation of the required FAR threshold. We will attempt to use a validation dataset around ten times larger than the minimum required, so we would, on average, expect ten false alarms total from running the model on the dataset with the given threshold.

Of course, there is only so far we can tune the threshold value within the precision available to us with 32-bit floats, and if the model gives scores to pure noise examples of exactly one, there is no way to differentiate them from true positive classifications. This means any model will have a maximum possible threshold, and therefore minimum FAR, beyond which it cannot distinguish positive results from negative ones.

In order to determine the score threshold of a model for a given FAR, we can run that model over a sufficiently large pure noise dataset, sort these scores from smallest to highest, and then assign each score an equivalent FAR. For example, if we sorted the scores from lowest to highest, and the first score was above the score threshold, then the FAR would be $1.0 / d_op("example") "Hz"$, where $d_op("example")$ is the length of the input example in our case #box("1"+ h(1.5pt) + "s"). If we set the threshold to be smaller than the smallest score this would mean that almost every noise example would score above the threshold, therefore the model would produce a false alarm nearly every time it ran. If the second sorted score was above the threshold but not the first, then all but one of the examples would be a false alarm, therefore we can estimate the FAR to be $(d_op("total") - d_op("example")) / d_op("total") times 1.0 / d_op("example") "Hz"$, $d_op("total")$ is the total duration of examples in the validation set. This gives a general formula for the y-axis,

$ y = (d_op("total") - i times d_op("example")) / d_op("total") times 1.0 / d_op("example") "Hz" , $ <FAR_index_calc>

where $i$ is the x-axis index. The FAR is plotted against the required model threshold to achieve that FAR in @perceptron_single_far. 

#figure(
  image("perceptron_single/perceptron_single_far_curves.png", width: 100%), 
  caption: [Perceptron False Alarm Rate (FAR) curves. This plot was created by running each of our 14 models over a pure noise validation dataset of $10^5$ noise examples. A relatively small number of noise examples are used due to the observed inaccuracy of the models during training which suggested that they would not be able to reach low FAR scores and thus would not necessitate a larger validation dataset. The output scores of the model from each inference over the pure noise validation dataset are sorted and plotted on this graph. The x-axis is the output score of the model inference on that example of noise. The y-axis is calculated by using @FAR_index_calc and provides the estimated number of false alarms that the model would output per second of pure noise data given the threshold score displayed on the x-axis. We can use this graph to calculate positive result thresholds for our classifier, at different false alarm rates. Once again, the models are listed with the number of artificial neurons in each hidden layer. Visit #link("http://tinyurl.com/2wkaarkh") to view an interactive plot. ]
) <perceptron_single_far>

Using @perceptron_single_far, we can select the score index that is closest to our desired FAR, and find the threshold that will generate a FAR of approximately this value. With a method to calculate threshold values in hand, we can create efficiency curves at specific FARs. Efficiency curves allow us to examine the sensitivity of the model to detect signals at different optimal SNR values. This time we can utilize datasets containing true results at set SNR values. We can run the models over these datasets and extract model scores for each true example. From those scores, we can calculate the sensitivity at different FARs. The sensitivity is given by

$ "sensitivity" = (|| "scores" > "score_threshold" ||) / (|| "scores" ||) $ <specificity>

where $|| "scores" > "score_threshold" ||$ is the number of scores above the score threshold, and $|| "scores" ||$ is the total number of examples tested. In @perceptron_efficiency_curves_single, we present the efficiency curves at three different values of FAR, #box("0.1"+ h(1.5pt) + "Hz"), #box("0.01"+ h(1.5pt) + "Hz"), and #box("0.001"+ h(1.5pt) + "Hz"), which are not particularly low FARs, but as can be seen from the plots, below these values we would encounter only negligible accuracies in the SNR ranges considered. As can be seen from the curves, the models do not perform well even with very generous FAR constraints.

#figure(
  grid(
    image("perceptron_single/perceptron_single_efficiency_curve_0_1.png", width: 100%),
    image("perceptron_single/perceptron_single_efficiency_curve_0_01.png", width: 100%),
    image("perceptron_single/perceptron_single_efficiency_curve_0_001.png", width: 100%),
  ), 
  caption: [Perceptron efficiency curves. For each of the 14 perceptron models trained, 31 efficiency tests are performed at evenly spaced optimal SNR values between 0 and 15. For each test, 8192 examples with signals of the relevant SNR are examined by the model, and the percentage of those that scored above the threshold was plotted, see @specificity, for three different False Alarm Rate (FAR) thresholds: #box("0.1"+ h(1.5pt) + "Hz"), #box("0.01"+ h(1.5pt) + "Hz"), and #box("0.001"+ h(1.5pt) + "Hz"). The efficiency curve for each FAR threshold is presented on a unique plot. Some models have been excluded, they are shaded grey on the legends, because they are incapable of performing any classification at the chosen FAR thresholds. Visit #link("http://tinyurl.com/2wkaarkh") to view an interactive plot. . _Upper:_ Efficiency curves at a FAR of #box("0.1" + h(1.5pt) + "Hz"). _Middle:_ Efficiency curves at a FAR of #box("0.01" + h(1.5pt) + "Hz"). _Lower:_ Efficiency curves at a FAR of #box("0.001" + h(1.5pt) + "Hz").]
) <perceptron_efficiency_curves_single>

Finally, we can examine the model performance from a different perspective by freezing the SNR of the validation dataset and plotting the True Positive Rate (TPR), i.e. the sensitivity, against the False Alarm Rate (FAR). This will give us a Reciever Operator Curve (ROC), see @perceptron_roc_curve. We can compare the area under the curve for each model to make a comparison of its relative performance; although in this case, all the models perform very similarly at the chosen optimal SNR of eight. Eight was chosen as this is often considered a good detectability threshold for CBCs, in the catalog of events from the first half of the third joint observing run, all confident detections had an SNR above nine, and candidate signals had SNRs above eight @GWTC-2.

#figure(
  image("perceptron_single/perceptron_single_roc.png", width: 100%), 
  caption: [Reciever Operator Curve (ROC) Curve at $rho_"opt" = 8$. To create this plot a validation dataset containing waveforms all of an SNR of eight was generated. The ability of the model to detect these waveforms was then measured at different FARs. All models show very similar, poor performance. Visit #link("http://tinyurl.com/2wkaarkh") to view an interactive plot. ]
) <perceptron_roc_curve> 

From these results, we can summarise that things are as anticipated from the results of the training. None of these models would have any useful application in gravitational-wave data science, as they all fall well below the performance of matched filtering, and they are unable to perform at acceptable FARs. In order to offer a competitive approach, we must turn to other network architectures.

=== Burst Detection Dense Results

==== Training

Although it may seem unlikely that we will have better results with what is arguably a more complex problem, we present the application of dense neural networks to multi-detector arbitrary waveform detection. Note that there are no incoherent or single detector counter-examples added to either the training or validation data, so in order to function a model would only have to identify the presence of excess power. The training and validation SNR ranges were also increased from 8 to 15 to 12 to 30 since initial testing at the SNR range used for CBC detection provided small accuracies across all FARs. From the training results it was clear that this was going to be a more complex problem than CBC detection; see @perceptron_multi_accuracy. Again there is the possibility that less constrained training or larger models could lead to better performance, but even if a solution was found outside the considered hyperparameter range, training time and computational requirements would soon become prohibitive. If other, less general networks can offer far superior results, they will be preferred.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("perceptron_multi/perceptron_multi_training_accuracy.png",   width: 100%) ],
        [ #image("perceptron_multi/perceptron_multi_validation_accuracy.png", width: 100%) ],
    ),
    caption: [The accuracy history of perceptron models training to detect multi-detector WNBs generated using GWFlow and obfuscated by real interferometer noise sampled from the LIGO Livingston and LIGO Hanford detectors during the 3#super("rd") observing run. Visit ADD_LINK for interactive plots. The optimal SNR of waveforms injected into the training and validation sets was uniformly distributed between 12 and 30. The input was generated using real noise from LIGO Hanford and LIGO Livingston. The training procedure was identical to the single detector case, except for the SNR range increase and the multiple detector data supply. We can see in these training plots, that despite the increased SNR range, training and validation accuracy barely creep above 50% (which can be achieved by random selection). This indicates that dense networks are even less suited for the more complex coherence detection problem. Further validation will be performed for completion. Visit #link("http://tinyurl.com/4jj3t5fj") to view an interactive plot. _Upper:_ Plot of model accuracies when measured with training data ($10^5$ epoch-unique examples). _Lower:_ Plot of model accuracies when tested with validation data ($10^4$ epoch-consistent examples).]
) <perceptron_multi_accuracy>

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("perceptron_multi/perceptron_multi_training_loss.png",   width: 100%) ],
        [ #image("perceptron_multi/perceptron_multi_validation_loss.png", width: 100%) ],
    ),
    caption: [The loss history of perceptron models training to detect multi-detector WNBs generated using GWFlow and obfuscated by real interferometer noise sampled from the LIGO Livingston and LIGO Hanford detectors during the 3#super("rd") observing run. Visit ADD_LINK for interactive plots. The optimal SNR of waveforms injected into the training and validation sets was uniformly distributed between 12 and 30. The input was generated using real noise from LIGO Hanford and LIGO Livingston. The losses show a similar picture to the accuracy plots, and although we see a gradual decline it is very shallow and triggers the patience early stopping before it has had any chance to gain significant performance, assuming that is even possible. Patience could be increased, but as we will see in later architectures, this is not competitive. _Upper:_ Plot of model losses when measured with training data ($10^5$ epoch-unique examples). _Lower:_ Plot of model losses when tested with validation data ($10^4$ epoch-consistent examples). Visit #link("http://tinyurl.com/4jj3t5fj") to view an interactive plot.]
) <perceptron_multi_loss>

==== Validation

As with the CBC case, we first present the FAR curve that will be used to determine model FAR thresholds in @perceptron_multi_far. Then we show the efficiency curves at two FARs, #box("0.1" + h(1.5pt) + "Hz"), and #box("0.01" + h(1.5pt) + "Hz"); see @perceptron_efficiency_curves_multi. Only two FAR thresholds are presented here as lower FARs resulted in negligible accuracies. Finally, we show the ROC curves for these models, which are unsurprisingly also poor; see @perceptron_roc_curve_multi.

#figure(
  image("perceptron_multi/perceptron_multi_far_curves.png", width: 100%), 
  caption: [Perceptron False Alarm Rate (FAR) curves. This plot was created by running each of our 14 models over a pure noise validation dataset of $10^5$ noise examples. Performance is low across the board demonstrating that dense layer perceptrons are unsuitable for this kind of WNB detection, at least within the hyperparameter range tested. Visit #link("http://tinyurl.com/bdz9axpf") to view an interactive plot.]
) <perceptron_multi_far>

#figure(
  grid(
    image("perceptron_multi/perceptron_multi_efficiency_curve_0_1.png", width: 100%),
    image("perceptron_multi/perceptron_multi_efficiency_curve_0_01.png", width: 100%),
  ), 
  caption: [Perceptron efficiency curves for the multi-detector WNB detection model. For each of the 14 perceptron models trained, 31 efficiency tests are performed at evenly spaced optimal SNR values between 0 and 30. For each test, 8192 examples with signals of the relevant SNR are examined by the model. The percentage of those that scored above the threshold is plotted, see @specificity, for two different False Alarm Rate (FAR) thresholds: #box("0.1"+ h(1.5pt) + "Hz") and #box("0.01"+ h(1.5pt) + "Hz"), lower FARs are excluded due to small accuracies. _Upper:_ Efficiency curves at a FAR of #box("0.1" + h(1.5pt) + "Hz"). _Lower:_ Efficiency curves at a FAR of #box("0.01" + h(1.5pt) + "Hz"). Visit #link("http://tinyurl.com/bdz9axpf") to view an interactive plot.]
) <perceptron_efficiency_curves_multi>

#figure(
  image("perceptron_multi/perceptron_multi_roc.png", width: 100%), 
  caption: [Reciever Operator Curve (ROC) Curve at an optimal SNR of eight. To create this plot a validation dataset containing waveforms all of an SNR of eight was generated. The ability of the model to detect these waveforms was then measured at different FARs. Again, all models show very similar, poor performance. Visit #link("http://tinyurl.com/bdz9axpf") to view an interactive plot.]
) <perceptron_roc_curve_multi> 

From these validation results, we can determine that dense layer networks alone are unsuitable for the task of coherence detection. Once again these results are not surprising and presented as a reference. In the next section, we will describe another deep-learning architecture that has seen much more promising results in the literature.

== Introducing Convolutional Neural Networks (CNNs) <cnn-sec>
\
As we have seen, simple dense-layer perceptrons can not adequately perform detection tasks on gravitational-wave data. This was anticipated, given the complexity of the distribution. Perceptrons have not been at the forefront of artificial neural network science for some time. We must turn toward other architectures. Although, in some ways, specialising the network will limit the capacity of our model to act as a universal function approximator @universal_aproximators, in practice, this is not a concern, as we have at least some idea of the process that will be involved in completing the task at hand, in this case, image, or more correctly time-series recognition. 

The Convolutional Neural Network (CNN) is currently one of the most commonly used model archetypes @deep_learning_review @conv_review. In many ways, the development of this architecture was what kickstarted the current era of artificial neural network development. On 30#super("th") December 2012, the AlexNet CNN @image_classification achieved performance in the ImageNet multi-class image recognition competition, far superior to any of its competitors. This success showed the world the enormous potential of artificial neural networks for achieving success in previously difficult domains.

CNNs are named for their similarity in operation to the mathematical convolution @deep_learning_review @conv_review, although it is more closely analogous to a discrete cross-correlation wherein two series are compared to each other by taking the dot product at different displacements. Unless you are intuitively familiar with mathematical correlations, it is not a useful point of reference for understanding CNNs. So, we will not continue to refer to convolutions in the mathematical sense.

CNNs are primarily employed for the task of image and time-series recognition @deep_learning_review @conv_review Their fundamental structure is similar to dense-layer networks on a small scale @cnn_review. They are comprised of artificial neurons that take in several inputs and output a singular output value after processing their inputs in conjunction with that neuron's learned parameters; see @artificial_neuron_sec. Typical CNNs ingest an input vector, have a single output layer that returns the network results, and contain a variable number of hidden layers. However, the structure and inter-neural connections inside and between the layers of a CNN are fundamentally different.

Unlike perceptrons, layers inside CNNs are, by definition, not all dense, fully-connected layers @deep_learning_review @conv_review. CNNs introduce the concept of different types of sparsely-connected computational layers. The classical CNN comprises a variable number, $C$, of convolutional layers stacked upon the input vector, followed by a tail of $D$ dense layers, which output the result of the network. This gives a total of $N = C + D$ layers, omitting any infrastructure layers that may also be present, such as a flattening layer (which is often employed between the last convolutional layer and the first dense layer; convolutional layers inherently have multidimensional outputs and dense layers do not). Purely convolutional networks, which consist only of convolutional layers, are possible @gebhard_conv_only_cnn, but these are a more unusual configuration, especially for classification tasks. Purely convolutional networks appear more often as autoencoders @autoencoder_ref and in situations where you want to lessen the black-box effects of dense layers. Convolutional layers are often more interpretable than pure dense layers as they produce feature maps that retain the input vector's dimensionality @cnn_interpretability.

Convolutional layers can and often do appear as layers in more complex model architectures, which are not necessarily always feed-forward models @deep_learning_review @conv_review. They can appear in autoencoders @autoencoder_ref, Generative Adversarial Networks (GANs) @gan_ref, Recurrent Neural Networks (RNNs) @conv_rnn, and as part of attention-based architectures such as transformers @conv_transformer and generative diffusion models @conv_diffusion. We will, for now, consider only the classical design: several convolutional layers capped by several dense ones.

As discussed, CNNs have a more specialised architecture than dense layers @deep_learning_review @conv_review. This architecture is designed to help the network perform in a specific domain of tasks by adding _a priori_ information defining information flow inside the network. This can help reduce overfitting in some cases, as it means a smaller network with fewer parameters can achieve the same task as a more extensive dense network. Fewer parameters mean less total information can be stored in the network, so it is less likely that a model can memorise specific information about the noise present in training examples. A CNN encodes information about the dimensionality of the input image; the location of features within the input image is conserved as it moves through layers. It also utilises the fact that within some forms of data, the same feature is likely to appear at different locations within the input vector; therefore, parameters trained to recognise features can be reused across neurons. For example, if detecting images of cats, cats' ears are not always going to be in the same location within the image. However, the same pattern of parameters would be equally helpful for detecting ears wherever it is in the network.

The following subsections describe different aspects of CNNs, including a description of pooling layers, which are companion layers often employed within convolutional networks.

=== Convolutional Layers

CNNs take inspiration from the biological visual cortex @bio_inspired_conv. In animal vision systems, each cortical neuron is not connected to every photoreceptor in the eye; instead, they are connected to a subset of receptors clustered near each other on the 2D surface of the retin @receptive_field_bio. This connection area is known as the *receptive field*, a piece of terminology often borrowed when discussing CNNs @bio_inspired_conv.

*Convolutional Layers* behave similarly. Instead of each neuron in every layer being connected to every neuron in the previous layer, they are only connected to a subset, and the parameters of each neuron are repeated across the image, significantly reducing the number of model parameters and allowing for translation equivariant feature detection @deep_learning_review @conv_review. It is a common misnomer that convolutional layers are translation invariant @lack_of_invariance; this is untrue, as features can and usually do move by values that are not whole pixel widths, meaning that even if the filters are the same, the pixel values can be different and give different results. One common problem with CNNs is that very small changes in input pixel values can lead to wildly different results, so this effect should be mitigated if possible. If they do not involve subsampling, however, CNNs are sometimes equivariant. This means that independent of starting location, ignoring edge effects, if you shift the feature by the same value, the output map will be the same --- this can be true for some configurations of CNN but is also broken by most common architectures.

This input element subset is nominally clustered spacially, usually into squares of input pixels @deep_learning_review @conv_review. This means that unlike with dense input layers, wherein 2D and greater images must first be flattened before being ingested, the dimensionality of the input is inherently present in the layer output. In a dense layer, each input is equally important to each neuron. There is no distinguishing between inputs far away from that neuron and inputs closer to that neuron (other than distinctions that the network may learn during the training process). This is not the case inside convolutional layers, as a neuron on a subsequent layer only sees inputs inside its receptive field. 

As the proximity of inputs to a neuron can be described in multiple dimensions equal to that of the input dimensionality, the network, therefore, has inherent dimensionality baked into its architecture --- which is one example of how the CNN is specialised for image recognition @deep_learning_review @conv_review. In the case of a 2D image classification problem, we now treat the input vector as 2D, with the receptive field of each neuron occupying some shape, most simply a square or other rectangle, on the 2D vector's surface.

The term receptive field is usually reserved to describe how much of the input image can influence the output of a particular neuron in the network @deep_learning_review @conv_review. The set of tunable parameters that define the computation of a neuron in a convolutional layer when fed with a subset of neuron outputs or input vector values from the previous layer is called a *kernel*. Each kernel looks at a subset of the previous layers' output and produces an output value dependent on the learned kernel parameters. A kernel with parameters tuned by model training is sometimes called a *filter*, as, in theory, it filters the input for a specific translation-invariant feature (although, as we have said, this is only partially true). The filter produces a strong output if it detects that feature and a weak output in its absence. Identical copies of this kernel will be tiled across the previous layer to create a new image with the same dimensionality as the input vector, i.e. kernels in a time-series classifier will each produce their own 1D time-series feature map, and kernels fed a 2D image will each produce a 2D image feature map. In this way, each kernel produces its own feature map where highly scoring pixels indicate the presence of whatever feature they have been trained to identify, and low-scoring ones indicate a lack thereof. Because the network only needs to learn parameters for this single kernel, which can be much smaller than the whole image and only the size of the feature it recognises, the number of trainable parameters required can be significantly reduced, decreasing training time, memory consumption, and overfitting risk. For a single kernel with no stride or dilation, see @stride-sec, applied to an input vector with no depth dimension, the number of trainable parameters is given by

$ op("len")(theta_"kernel") = (product_i^N S_i) + 1 $

where $op("len")(theta_"kernel")$ is the number of trainable parameters in the kernel, N is the number of dimensions in the input vector, and $S_i$ is the configurable hyperparameter, kernel size in the i#super("th") dimension. The extra plus one results from the bias of the convolutional kernel.
 
For example, a 1D kernel of size 3, would have $3 + 1 = 4$ total parameters, independent of the size of the input vector, and a 2D kernel of size $3 times 3$ would have $3 times 3 + 1 = 10$ total parameters, again independent of the size of the 2D input vector in either dimension.  See @kernel_example for an illustration of the structure of a convolutional kernel.

#figure(
    image("convolutional_kernel.png", width: 40%),
    caption: [Diagram of a single kernel, $""_1^1k$, in a single convolutional layer. In this example, a 1D vector is being input; therefore, the single kernel's output is also 1D. This kernel has a kernel size of three, meaning that each neuron receives three input values from the layer's input vector, $accent(x, arrow)$, which in this case is length five. This means there is room for three repeats of the kernel. Its parameters are identical for each iteration of $""_1^1k$ at a different position. This means that if a pattern of inputs recognised by the kernel at position 1, $""_1^1k_1$ is translated two elements down the input vector, it will be recognised similarly by the kernel at $""_1^1k_3$. Although this translational invariance is only strict if the translation is a whole pixel multiple and no subsampling (pooling, stride, or dilation) is used in the network, this pseudo-translational invariance can be useful, as often, in images and time series data, similar features can appear at different spatial or temporal locations within the data. For example, in a speech classification model, a word said at the start of the time series can be recognised just as easily by the same pattern of parameters if that word is said at the end of the time series (supposing it lies on the sample pixel multiple). Thus, the same kernel parameters and the same filter can be repeated across the time series, reducing the number of parameters needed to train the model. This particular kernel would have $3 + 1 = 4$ total parameters, as it applied to a 1D input vector, and has a kernel size of three, with an additional parameter for the neuron bias. With only a single kernel, only one feature can be learned, which would not be useful in all but the most simple cases. Thus, multiple kernels are often used, each of which can learn its own filter. ]
) <kernel_example>

When first reading about convolutional layers, it can be confusing to understand how they each "choose" which features to recognise. What should be understood is that this is not a manual process; there is no user input on which kernels filter which features; instead, this is all tuned by the chosen optimiser during the training process @deep_learning_review @conv_review. Even the idea that each kernel will cleanly learn one feature type is an idealised simplification of what can happen during training. Gradient descent has no elegant ideas of how it should and should not use the architectures presented to it and will invariably follow the path of least resistance, which can sometimes result in strange and unorthodox uses of neural structures. The more complex and non-linear the recognition task, the more often this will occur. 

Although we do not specify exactly which features each kernel should learn, there are several hyperparameters that we must fix for each convolutional layer before the start of training @deep_learning_review @conv_review. We must set a kernel (or filter) size for each dimension of the input vector. For a 1D input vector, we will set one kernel size per kernel; for a 2D input vector, we must set two, and so on. These kernel dimensions dictate the number of input values read by each kernel in the layer and are nominally consistent across all kernels in that layer; see @kernel-size for an illustration of how different kernel sizes tile across a 2D input.

#figure(
    image("kernel_sizes.png", width: 50%),
    caption: [Illustration of how different values of kernel size would be laid out on a $4 times 4$ input image. In each case, unused input image values are shown as empty black squares on the grid, and input values read by the kernel are filled in red. The grids show the input combinations that a single kernel would ingest if it has a given size, assuming a stride value of one and zero dilation. The kernel sizes are as follows: _Upper left:_ $2 times 2$. _Upper right:_ $3 times 2$. _Lower left:_  $2 times 3$. _Lower right:_  $3 times 3$. One pixel in the output map is produced for each kernel position. As can be seen, the size of the output map produced by the kernel depends both on the input size and the kernel size; smaller kernels produce a larger output vector.]
) <kernel-size>

The other hyperparameters that must be set are the number of different kernels and the choice of activation function used by the kernel's neurons @deep_learning_review @conv_review. These hyperparameters can sometimes be manually tuned using information about the dataset, i.e. the average size of the features for kernel size and the number of features for the number of kernels, but these can also be optimised by hyperparameter optimisation methods, which might be preferable as it is often difficult to gauge which values will work optimally for a particular problem @cnn_hyperparameters.

Multiple kernels can exist up to an arbitrary amount inside a single convolutional layer @deep_learning_review @conv_review. The intuition behind this multitude is simply that input data can contain multiple different types of features, which can each need a different filter to recognise; each kernel produces its own feature map as it is tiled across its input, and these feature maps are concatenated along an extra *depth* dimension on top of the dimensionality of the input vector. A 1D input vector will have 2D convolutional layer outputs, and a 2D input vector will result in 3D convolutional outputs. The original dimensions of the input vector remain intact, whilst the extra discrete depth dimension represents different features of the image; see @multi_kernel_example. 

In the case of a colour picture, this depth dimension could be the red, green, and blue channels, meaning this dimension is already present in the input vector. The number of trainable parameters of a single convolutional layer is given by

$ op("len")(theta_"conv_layer") = K times ((D times product_i^N S_i) + 1) $ <conv-layer-size>

where $op("len")(theta_"conv_layer")$ is the total number of parameters in a convolutional layer, $K$ is the number of convolutional kernels in that layer, a tunable hyperparameter, and $D$ is the additional feature depth dimension of the layer input vector, which is determined either by the number of pre-existing feature channels in the input vector, i.e. the colour channels in a full-colour image or, if the layer input is a previous convolutional layer, the number of feature maps output by that previous layer, which is equivalent to the number of kernels in the previous layer. For example, a 1D convolutional layer with three kernels, each with size three, ingesting a 1D input with only a singleton depth dimension would have $3 times ((1 times (3)) + 1) = 12$ total trainable parameters, whereas a 2D convolutional layer with three kernels of size $3 times 3$ looking at a colour RGB input image would have $3 times (3 times ( 3 times 3 ) + 1) = 84$ total trainable parameters.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("multiple_convolutional_kernel.png",   width: 45%) ],
        [ #image("single_conv_abstraction.png", width: 100%) ],
    ),
    caption: [_Upper:_ Diagram of three convolutional kernels, $[""_1^1k, ""_2^1k, ""_3^1k]$, in a single convolutional layer. Each kernel is coloured differently, in red, green, and blue. Artificial neurons of the same colour will share the same learned parameters. Again, a 1D vector is being input; therefore, the output of each of the kernels is 1D, and the output of the kernels stack to form a 2D output vector, with one spatial dimension retained from the input vector and an extra discrete depth dimension representing the different features learned by each of the kernels. Again, each kernel has a kernel size of three. Multiple kernels allow the layer to learn multiple features, each of which can be translated across the input vector, as with the single kernel. Using @conv-layer-size, this layer would have $3 times ((1 times 3) + 1) = 12$ trainable parameters. It should be noted that this is a very small example simplified for visual clarity; real convolutional networks can have inputs many hundreds or thousands of elements long and thus will have many more iterations of each kernel, as well as many more kernels sometimes of a much larger size. _Lower:_ Abstracted diagram of the same layer with included hyperparameter information. ]
) <multi_kernel_example>

As with dense layers, multiple convolutional layers can be stacked to increase the possible range of computation available @deep_learning_review @conv_review; see @multi_cnn_layer_example. The first convolutional layer in a network will ingest the input vector, but subsequent layers can ingest the output of previous convolutional layers, with kernels slicing through and ingesting the entirety of the depth dimension. In theory, this stacking allows the convolutional layers to combine multiple more straightforward features in order to recognise more complex, higher-level features of the input data --- although, as usual, things are not always quite so straightforward in practice. When calculating the number of trainable parameters in multiple convolutional layers, we can use @conv-layer-size for each layer and sum the result.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("two_convolutional_layers.png", width: 60%) ],
        [ #image("multi_conv_abstraction.png", width: 100%) ],
    ),
    caption: [_Upper:_ Diagram of two convolutional layers, each with independent kernels. The first layer has three kernels, each with a size of three. The second layer has two kernels, both with a size of two. Again, this is a much-simplified example that would probably not have much practical use. Different kernels are coloured differently, in red, green, and blue. Although it should be noted that similar colours across layers should not be taken as any relationship between kernels in different layers, they are each tuned independently and subject to the whims of the gradient descent process. This example shows how the kernels in the second layer take inputs across the entire depth of the first layer but behave similarly along the original dimension of the input vector. In theory, the deeper layer can learn to recognise composite features made from combinations of features previously recognised by the layers below and visible in the output feature maps of the different kernels. This multi-layer network slice would have $(3 times ((1 times 3) + 1)) + (2 times ((3 times 2) + 1)) = 26$ total trainable parameters. This was calculated by applying @conv-layer-size to each layer. _Lower:_ Abstracted diagram of the same layers with included hyperparameter information. ]
) <multi_cnn_layer_example>

The result of using one or more convolutional layers on an input vector is an output vector with an extra discrete depth dimension, with each layer in the stack representing feature maps @deep_learning_review @conv_review. Whilst often considerably more interpretable than maps of the parameters in dense layers, these maps are often not very useful alone @cnn_interpretability. However, a flattened version of this vector is now, hopefully, much easier for dense layers to classify than the original image; see @flatten-sec. As such, CNNs used for classification are almost always capped by one or more dense layers in order to produce the final classification result; see @cnn_diagram for a toy example of a CNN used for binary classification.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("cnn_diagram.png",   width: 100%) ],
        [ #image("cnn_abstracted.png", width: 100%) ],
    ),
    caption: [_Upper:_ Diagram of a very simple convolutional neural network binary classifier consisting of four layers with tunable parameters plus one infrastructure layer without parameters. Two consecutive convolutional layers ingest the five-element input vector, $accent(x, arrow)$. The 2D output of the latter of the two layers is flattened into a 1D vector by a flattening layer. This flattened vector is then ingested by two dense layers, the latter of which outputs the final classification score. The first convolutional layer has three convolutional kernels, each with a size of three, and the second convolutional layer has two kernels, both with a size of two. The first dense layer has three artificial neurons, and the final output dense layer has a number of neurons dictated by the required size of the output vector. In the case of binary classification, this is either one or two. Different kernels within a layer are differentiated by colour, in this case, red, green, or blue, but a similar colour between layers does not indicate any relationship. Dimensionless neurons are shown in black; it should be noted that after flattening, dimensional information is no longer necessarily maintained by the network structure. Of course, no information is necessarily lost either, as the neuron index itself contains information about where it originated, so, during training, this information can still be used by the dense layers; it is just not necessarily maintained as it is in convolutional layers. This network will have in total $26 + (3 times 4 + 4) + (2 times 3 + 2) = 50$ trainable parameters. This network is very simple and would probably not have much practical use in real-world problems other than straightforward tasks that would probably not necessitate using neural networks. _Lower:_ Abstracted diagram of the same model with included hyperparameter information. 
 ]
) <cnn_diagram>

=== Stride, Dilation, and Padding <stride-sec>

* Stride * is a user-defined hyperparameter of convolutional layers that must be defined before training @deep_learning_review @conv_review @cnn_hyperparameters. Like kernel size, it is a multidimensional parameter with a value for each input vector dimension. A convolutional layer's stride describes the distance the kernel moves between instances. A stride of one is the most commonly used choice. For example, if the stride is one, then a kernel is tiled with a separation of one input value from its last location. Stride, $S$, is always greater than zero, $S > 0$. The kernels will overlap in the i#super("th") dimension if $S_i < k_i$. If $S_i = k_i$, there will be no overlap and no missed input vector values. If $S_i > k_i$, some input vector values will be skipped; this is not usually used. Along with kernel size, stride determines the output size of the layer. A larger stride will result in fewer kernels and, thus, a smaller output size; see @stride below for an illustration of different kernels strides.

#figure(
    image("stride.png", width: 70%),
    caption: [Illustration of how different values of kernel stride would be laid out on a $4 times 4$ input image. In each case, unused input image values are shown as empty black squares on the grid, and input values read by the kernel are filled in red. Similar to kernel size, different values of stride result in a different output vector size. The strides shown are as follows: _Upper left:_ $1, 1$. _Upper right:_ $2, 1$. _Lower left:_  $1, 2$. _Lower right:_  $2, 2$.]
) <stride>

Introducing kernel stride primarily serves to reduce the overall size of the network by reducing the output vector without adding additional parameters; in fact, the number of parameters is independent of stride @deep_learning_review @conv_review. Reducing the size of the network might be a desirable outcome as it can help reduce computational time and memory overhead. It can also help to increase the receptive field of neurons in subsequent layers as it condenses the distance between spatially separated points, so when adjusting the resolution of feature maps in a model to balance the identification of smaller and larger scale features, it could potentially be a useful dial to tune. In most cases, however, it's left at its default value of one, with the job of reducing the network size falling to pooling layers; see @pool-sec. 

One interesting and potentially unwanted effect of introducing stride into our network is that it removes the complete translation equivariance of the layer by subsampling; instead, translations are only equivariant if they match the stride size, i.e. if a kernel has a stride of two features are invariant if they move exactly two pixels, which is not a common occurrence.

*Dilation* is a further hyperparameter that can be adjusted prior to network training @deep_learning_review @conv_review @cnn_hyperparameters. Dilation introduces a spacing inside the kernel so that each input value examined by the kernel is no longer directly adjacent to another kernel input value, but instead, there is a gap wherein the kernel ignores that element; see @dilation. By default, this value would be set to zero, and no dilation would be present.  This directly increases the receptive field of that kernel without introducing additional parameters, which can be used to help the filters take more global features into account. It can also be used in the network to try and combat scale differences in features; if multiple kernels with different dilations are used in parallel on different model branches, the model can learn to recognise features at the same scale but with different dilations.

#figure(
    image("dilation.png", width: 40%),
    caption: [Diagram illustrating how different values of kernel dilation affect the arrangement of the kernel input pixels. In this example, the receptive field of a single $3 times 3$ kernel at three different dilation levels is displayed; differing colours represent the input elements at each dilation level. The shaded red kernel illustrates dilation level zero; the shaded blue region is a kernel with dilation of one, and the green kernel has a kernel dilation of two.]
) <dilation>

Particular stride, dilation, and size combinations will sometimes produce kernel positions that push them off the edge of the boundaries of the input vector. These kernel positions can be ignored, or the input vector can be padded with zeros or repeats of the nearest input value; see @padding.

#figure(
    image("padding.png", width: 40%),
    caption: [Diagram illustrating how padding can be added to the edge of an input vector in order to allow for otherwise impossible combinations of kernel, stride, size, and dilation. In each case, unused input image values are shown as empty black squares on the grid, input values read by the kernel are shaded red, and empty blue squares are unused values added to the original input vector, containing either zeros or repeats of the closest data values. In this example, the kernel size is $3 times 3$, and the kernel stride is $2, 2$.]
) <padding>

=== Pooling <pool-sec>

Pooling layers, or simply pooling, is a method used to restrict the number of data channels flowing through the network @deep_learning_review @conv_review. They see widespread application across the literature and have multiple valuable properties. They can reduce the size of the network and thus the computation and memory overhead, and they can also make the network more robust to small translational, scale, and rotational differences in your input features. Convolutional layers record the position of the feature they recognise but can sometimes be overly sensitive to tiny shifts in the values of input pixels. Small changes in a feature's scale, rotation, or position within the input can lead to a very different output, which is evidently not often desirable behavior.

Pooling layers do not have any trainable parameters, and their operation is dictated entirely by the user-selected hyperparameters chosen before the commencement of model training. Instead, they act to group pixels via subsampling throwing away excess information by combining their values into a single output. In this way, they are similar to convolutional kernels, however. instead of operating with trained parameters, they use simple operations. The two most common types of pooling layers are *max pooling* and *average pooling*; max pooling keeps only the maximum value within each of its input bins, discarding the other values; intuitively, we can think of this as finding the strongest evidence for the presence of the feature within the pooling bin and discarding the rest. Average pooling averages the value across the elements inside each pooling bin, which has the advantage that it uses some information from all the elements.

As can be imagined, the size of CNNs can increase rapidly as more layers and large numbers of kernels are used, with each kernel producing a feature map nearly as large as its input vector. Although the number of parameters is minimised, the number of operations increases with increasing input size. Pooling layers are helpful to reduce redundant information and drastically reduce network size whilst also making the network more robust to small changes in the input values.

Along with the choice of operational mode, i.e. average or maximum, pooling layers have some of the same hyperparameters as convolutional kernels, size, and stride. Unlike convolutional layers, the pooling stride is usually set to the same value as the pooling size. Meaning that there will be no overlap between pooling bins but also no gaps. This is due to the purpose of pooling layers, which attempt to reduce redundant information; if stride were set to smaller values, there would be little reduction and little point to the layer.

Because pooling with stride is a form of subsampling, it does not maintain strict translational equivariance unless the pool stride is one, which, as stated, is uncommon. Thus, as most CNN models use pooling, most CNNs are neither strictly translationally invariant nor equivariant @lack_of_invariance.

== Results from the Literature <cnn-literature>

Both gravitational-wave astrophysics and deep learning methods have been through rapid advancement in the previous decade, so it is perhaps unsurprising that there has also developed a significant intersection between the two fields @gw_machine_learning_review. Multiple artificial network architectures, CNNs @gabbard_messenger_cnn @george_huerta_cnn, autoencoders @source_agnostic_lstm, generative adversarial networks @burst_detection_gans, recurrent neural networks @bidirectional_lstm, and attention-based networks like transformers @detection_conv_transformer have been applied to numerous gravitational-wave data analysis problems. This review will focus on efforts to apply CNN classifiers to detect features hidden within interferometer data. First, we will look at attempts to detect Compact Binary Coalescences (CBCs), followed by a look at unmodeled Bursts detection attempts. More complex network architectures will be reviewed later when we examine attention layers in closer detail; see @skywarp-sec. This is not intended to be an exhaustive catalogue, although efforts have been made to be as complete as possible.

=== Convolutional Neural Networks (CNNs) for the detection of Compact Binary Coalescences (CBCs)

The earliest attempts at CBC classification using artificial neural networks were a pair of papers by George _et al._ @george_huerta_cnn which was followed up by Gabbard _et al._ @gabbard_messenger_cnn. George _et al._ @george_huerta_cnn applied CNNs to the binary classification problem and basic parameter estimation. They used CNNs with two outputs to extract parameter estimates for the two companion masses of the binary system. They used the whitened outputs of single interferometers as inputs and utilized CNNs of a standard form consisting of convolutional, dense, and pooling layers; see @gabbard_diagram and @george_diagram. They evaluated two models, one smaller and one larger. In their first paper, they used only simulated noise, but they produced a follow-up paper showing the result of the model's application to real interferometer noise @george_huerta_followup. Gabbard _et al._ @gabbard_messenger_cnn used an alternate CNN design with a different combination of layers. They only used a single network architecture, and no attempt at parameter estimation was made. A differentiating feature of their paper was the training of individual network instances to recognize different SNRs. Both George _et al._ @george_huerta_cnn and Gabbard _et al._ @gabbard_messenger_cnn achieved efficiency curves that closely resembled that of matched filtering; of note, however, both were validated at a considerably higher FAR ($tilde 10^3$ Hz) than is useful for a gravitational-wave search, this will be a consistent theme throughout the literature and is one of the greatest blockers to using CNNs in gravitational-wave transient detection.

There have been many papers that follow up on these two initial attempts. Several papers with mixed results are difficult to compare due to a variety of different conventions used to characterise signal amplitude and assess model performance. Luo _et al._ @luo_cnn attempted to improve the model described by Gabbard _et al._ They have presented their results using a non-standard "Gaussian noise amplitude parameter". Whilst within their own comparisons, they seem to have improved network operation over the original design, at least at higher FARs, it is difficult to make a comparison against other papers because of the unorthodox presentation. Schmitt _et al._ @schmitt_cnn attempted to compare the performance of one of the models presented in George _et al._ @george_huerta_cnn with three different model architectures, Temporal Convolutional Networks (TCNs), Gated Recurrent Units (GRUs), and Long Short-Term Memory (LSTMs). They seem to show that the other model architectures can achieve higher performance than CNNs, but they have used an unfamiliar waveform scaling method, so it is hard to compare to other results. 

A more interesting follow-up by Fan _et al._ @multi_detector_fan_cnn took the smaller of the two models introduced in George _et al._ @george_huerta_cnn and extended it to use multiple detectors as inputs rather than the previously mentioned studies, which looked at only single detectors. They do this for both detection and parameter estimation and appear to show improved accuracy results over the original paper @george_huerta_cnn, although they do not address the confounding factor of having to deal with real noise. Krastev _et al._ tested the use of the other larger model introduced by George _et al._ @george_huerta_cnn. They tested its use on Binary Neuron Star (BNS) signals, as well as reaffirming its ability to detect BBH signals. They used significantly longer input windows to account for the longer detectable duration of BNS signals. They found BNS detection to be possible, although it proved a significantly harder problem. 

Using a different style of architecture, Gebhard _et al._ @gebhard_conv_only_cnn argued that convolution-only structures are more robust and less prone to error, as they remove much of the black-box effect produced by dense layers and allow for multiple independently operating (though with overlapping input regions) networks, creating an ensemble which generates a predictive score for the presence of a signal at multiple time positions. This results in a time-series output rather than a single value, which allows the model to be agnostic to signal length. Their determination of the presence of a signal can thus rely on the overall output time series rather than just a single classification score. Similarly to Fan _et al._ @multi_detector_fan_cnn, they used multiple detector inputs.

There have been at least two papers that utilise ensemble approaches to the problem. Ensembles consist of multiple independently trained models in the hopes that the strengths of another will counteract the weaknesses of one under the assumption that it is less likely for them both to be weak in the same area. A joint decision is then taken through some mechanism that takes the result of all models into consideration, often waiting for certain models' votes under certain criteria. Huerta _et al._ @huerta_fusion_cnn used an approach consisting of four independently trained models, each of which has two separate CNN branches for the LIGO Hanford and LIGO Livingston detectors, which are then merged by two further CNN layers. Still, they have efficiency results down to a lower FAR than any paper reviewed so far, at $1 times 10^(-5)$, which is impressive, although the efficiency scores at these FARs are low ($<1%$). Overall, the paper is more focused on the software infrastructure for deploying neural network models. Ma _et al._ @ma_ensemble_cnn used an ensemble network that employ one of the architectures described by Gabbard _et al._ @gabbard_messenger_cnn. They utilise two "subensembles" in an arrangement in which each detector has its own ensemble composed of networks that vote on a false/positive determination; the results of both of the two subensembles are then combined for a final output score. They do not give efficiency scores at set SNRs, so again, it is difficult to compare against other results.

There have also been some interesting studies that use feature engineering to extract features from the input data before those features are fed into the CNN models, see @feature-eng-sec. Wang _et al._ @wang_cnn use a sparse matched filter search, where template banks of only tens of features, rather than the usual hundreds of thousands or millions, were used. The output of this sparce matched filter was then ingested by a small CNN, which attempted to classify the inputs. Notably, they use real noise from the 1#super("st") LIGO observing run @GWTC-1 and multi-detector inputs. Though an interesting method, their results appear uncompetitive with other approaches. Reza @matched_filtering_combination _et al._ used a similar approach but split the input into patches before applying the matched filter. However, results are not presented in an easily comparable fashion. Bresten _et al._ @bresten_cnn_topology adapts one of the architectures from George _et al._ @george_huerta_cnn but applies a feature extraction step that uses a topological method known as persistent homology before the data is ingested by the network. It is an interesting approach, but their results are unconvincing. They limited their validation data to 1500 waveforms at only 100 specific SNR values in what they term their "hardest case". They showed poor results compared to other methods, suggesting their method is undeveloped and heavily SNR-tailored. 

There have been at least three spectrogram-based attempts to solve the CBC detection problem. Yu _et al._ @spectrogram_cnn_2 used single detector spectrograms, which are first analysed in strips using multiple 1D CNNs before being fed into a 2D CNN for final classification; they achieve middle-of-the-range efficiency results. Aveiro _et al._ @bns_object_detection_spectogram focused on BNS detection and used an out-of-the-box object detection network to try and detect patterns in spectrograms. They do not state efficiencies for SNRs less than ten. Finally, there was also a search paper @o2_search_cnn, which searched through the second observing run using spectrograms-based CNNs; they detected nothing of significance.

There has also been an attempt to use wavelet decomposition for the problem. Lin _et al._ @lin_wavelet_bns focused on the detection of BNS signals by wavelet decomposition with some very promising results shown to outperform matched filtering; a subsequent follow-up paper @li_wavelet_cnn showed that the same method could be applied to BBH signals with equal promise. They achieve an efficiency of 94% when detecting waveforms with an SNR of 2 at a FAR of $1 times 10^(-3)$, which undercuts the competition by considerable margins. Their method is certainly worth investigation but was unfortunately missed until this thesis was in the latter stages of construction, so no wavelet decomposition methods have been attempted.

There have also been a number of papers utilising CNNs for specialised detection cases, such as mass asymmetric CBCs @mass_asymetric_cbcs by Andrés-Carcasona _et al._, who employ spectrogram-based CNNs to run a search over O3, and eccentric CBCs by Wei _et al._ @wei_cnn, the latter of which also focuses on early detection along with a few other papers @early_alert_bbh @early_alert_bns @early_detection which attempt to find CBC inspirals before they are detectable by standard methods. There have also been a number of papers that discuss the use of CNNs for the analysis of data from future space-based detectors @space_detection @space_detection_2. For brevity, and as they are less relevant to our problems, these special cases will not be discussed here. 

As can be seen, it is very difficult to compare the performance of many of the architectures and methods presented in the literature. The results are presented at wildly different FARs and SNR ranges, often using different incomparable metrics and with varying levels of rigour. There is a tendency to apply new tools and ideas to the problem without careful thought about how the results can be standardised. @literature-results displays results from some of the papers that were found to have at least somewhat comparable metrics.

#set page(
  flipped: true
)

#figure(
  table(
    columns: (auto, 100pt, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [*Name*],  [*Model*], [*Real Noise?*], [*Detectors*], [*Target*], [*Feature*], [*SNR Tailored*], [*FAR*], [*Acc 8*], [*Acc 6*], [*Acc 4*],
    [George _et al._ @george_huerta_cnn], [ Novel CNN ], [No], [Single], [BBH], [No], [No], [$5 times 10^(-2)$], [0.98], [0.70], [0.16],
    [-], [-], [-], [-], [-], [-], [-], [-], [0.99], [0.80], [0.21],
    [George _et al._ @george_huerta_followup], [-], [ #text(red)[*Yes*] ], [-], [-], [-], [-], [-], [0.98], [0.77], [0.18],
    [Gabbard _et al._ @gabbard_messenger_cnn], [Novel CNN], [No], [Single], [BBH], [No], [#text(red)[*Yes*]], [$1 times 10^(-1)$], [1.0], [0.88], [0.44],
    [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-2)$], [0.99], [0.69], [0.10],
    [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-3)$], [0.98], [0.49], [0.02],
    [Fan _et al._ @multi_detector_fan_cnn], [ Based on George et al. Small], [No], [#text(red)[*Three*]], [BBH], [No], [No], [$4 times 10^(-2)$], [0.99], [0.84], [0.32],
    [Krastev _et al._ @krastev_bnn_cnn ], [Based on George et al. Large], [No], [Single], [#text(red)[*BNS*]], [No], [No], [$1 times 10^(-1)$], [0.71], [0.42], [0.20],
    [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-2)$], [0.32], [0.10], [0.02],
    [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-3)$], [0.11], [0.00], [0.00],
    [Gebhard _et al._ @gebhard_conv_only_cnn], [#text(red)[Novel Conv-Only Model]], [#text(red)[*Yes*]], [#text(red)[*Two*]], [BBH], [No], [No], [$1.25  times 10^(-3)$], [0.83], [0.35], [Not Given],
    [Wang _et al._ @wang_cnn], [-], [#text(red)[*Yes*]], [#text(red)[*Two*]], [BBH], [#text(red)[*Matched Filter*]], [No], [$1 times 10^(-1)$], [0.60], [0.24], [0.12],
    [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-2)$], [0.30], [0.05], [0.00],
    [-], [-], [-], [-], [-], [-], [-], [$1 times 10^(-3)$], [0.08], [0.00], [0.00],
    [Huerta _et al._ @huerta_fusion_cnn], [#text(red)[*Novel Ensemble*]], [Yes], [Two], [BBH], [No], [No], [#text(red)[*$5 times 10^(-4)$*]],  [0.20], [0.15], [Not Given],
    [-], [-], [-], [-], [-], [-], [-], [#text(red)[*$5 times 10^(-5)$*]], [0.01], [0.001], [Not Given],
    [Yu _et al._ @spectrogram_cnn_2], [#text(red)[*Novel Multi-Branched CNN*]], [-], [-], [Yes], [Single], [BBH], [#text(red)[*Spectrogram*]], [No], [$6 times 10^(-2)$], [0.89], [0.67], [0.20] 
  ),
  caption: [A comparison of results from the literature, red values indicate the significant feature of the study. Note: Some accuracy values are extracted from plots by eye, so substantive error will have been introduced. Some results were not included as they did not state comparable performance metrics. ]
) <literature-results>

#set page(
  flipped: false
)

=== Convolutional Neural Networks (CNNs) for the detection of Gravitational-Wave Bursts

The literature surrounding burst detections with CNNs is considerably more limited than for CBCs. In all of the previously mentioned deep-learning studies, the training of the network has relied on accurate models of CBC waveforms. As has been noted, the availability of reliable waveforms for other potential gravitational-wave sources, i.e. bursts, is considerably narrower due to unknown physical processes, large numbers of free parameters, and computational intractability, making it nearly impossible to have a sufficiently sampled template bank.

Despite this, there have been some attempts, most notably using simulated supernovae waveforms, as these are the most likely candidates for initial burst detection. There have been at least five attempts to classify supernovae with this method. Iess _et al._ @supernovae_cnn_1 used a CNN mode with two separate inputs; a 1D time series and a 2D spectrogram were fed into different input branches of the model. They used supernova signals taken from simulation catalogues along with a simple phenomenological model for two transient glitches classes in order to train the CNN to distinguish between the glitches and supernovae in the hopes that if a supernova signal were to appear in future interferometer data, it could be identified as such, rather than being ruled out as a glitch. Perhaps unsurprisingly, due to the complexity of the signal compared to CBCs, they require a significantly higher SNR in order to achieve similar accuracy results as the CBC case, although they still achieve some efficiency at lower SNRs. Chan _et al._ @supernovae_cnn_2 trained a CNN using simulated core-collapse supernovae signals drawn from several catalogues covering both magnetorotational-driven and neutrino-driven supernovae. They measured the ability to detect the signal and correctly classify which of the two types it fell into. They used moderately deep CNNs and emphasised the importance of multi-detector inputs for the task. They found it possible to detect magnetorotational-driven events at considerably greater distances than neutrino-driven supernovae.

Lopez _et al._ @supernovae_cnn_3 @supernovae_cnn_4 forgoes the use of simulated template backs in their training for a phenomenological approach in an attempt to try and avoid the problem of small template banks. They used an intricate model architecture comprised of mini-inception-resnets to detect supernova signals in time-frequency images of LIGO-Virgo data. Mini-inception resnets consist of multiple network branches of different lengths, which run in parallel before combining to produce a final classification score. Having some paths through the network that are shorter than others can be beneficial to avoid the vanishing gradient problem, wherein gradients fall off to zero within the network; having shortcuts allows the network to maintain a clearer view of the inputs even when other paths have become deep @skip_connections. Blocks of layers within networks that have *skip connections* periodically like this are known as residual blocks @res_net_intro, and allow much deeper architectures than would otherwise be possible. Networks that employ skip connections are known as *residual networks* or *resnets* @res_net_intro. Inception designs have multiple different network branches, all consisting of residual blocks, so there are many paths through the network from input vector to output. 

Sasaoka _et al._ @supernovae_spectrogram @sasaoka_resnet use gradient-weighted feature maps to train CNNs to recognise supernovae spectrograms. They utilised core-collapse supernovae waveforms from a number of catalogues. However, they only achieved good classification performances at #box("1" + h(1.5pt) + "kpc"). They attributed some of their difficulties to features lost to the lower resolution of their time-frequency maps and recommended trying a different algorithm for their generation.

There have also been a few attempts to apply CNNs to the problem of unmodelled signal detection, looking for generic signals using methods that do not require a precisely tailored training set. As has been discussed, we do not yet know how well our simulations will align with real supernovae' gravitational emissions, and it is hard to tell whether the differences between our training datasets and the real signals will significantly hinder our model's ability to detect real signals. Such difficulty could certainly be a possibility; often, deep learning modules can be very sensitive to changes in their distribution and can lose significant efficacy when applied to out-of-distribution examples. If a sensitive enough generic model could be trained, this would alleviate this problem.

Marianer _et al._ @semi-supervised  attempt a generic detection method via anomaly recognition. This is not a novel idea in the field of machine learning. However, its application to a generic burst search is intriguing. They apply their model to spectrograms of known transient noise glitches and use a mini-inception resnet to classify the input spectrograms into glitch classes. While examining the feature space of the classifier as it examines data, i.e. the feature maps of the neurons in internal convolutional layers, they utilise two anomaly detection methods to identify when a particular feature space does not look like it belongs to any of the known classes. This means they do not rely directly on the model to output a "novel" class. The latter poses a difficult problem as it is unclear how to ensure the training set is well sampled over every possible counterexample.

The other and most relevant work to this thesis on unmodeled glitch detection is MLy @MLy. MLy is a deep learning pipeline that relies on CCN models, which are trained to directly identify coherence between multiple detectors rather than using any pattern recognition or anomaly rejection techniques. This makes it somewhat unique amongst the methods presented. Rather than using a dataset consisting of particular morphologies of signal, MLy utilises distributions of generic white noise burst signals that, in their entirety, will cover all possible burst morphologies with a certain frequency range and duration. One would note that these distributions would also cover all possible glitch morphologies within that parameter space. Therefore, MLy is trained not only to notice the presence of a signal but also the coherence of that signal across detectors. In that sense, it is similar to the operation of many of the preexisting burst pipelines, though it is the only purely machine-learning pipeline to attempt to do this. 

MLy achieves this goal by utilising two independent CNN models, one of which looks simply for excess power in both detectors, the coincidence model, and one of which attempts to determine coherence between detections; the second model is fed feature-engineered data in the form of the rolling Pearson correlation between detectors with a number of factor-of-sample-interval timeshifts equivalent to the maximum arrival time difference between the two detectors in question. It does this for the two LIGO detectors and the Virgo detector. It is trained on four types of example: pure noise, noise with a simulated transient (WNB) in one detector only, noise with a simulated transient in all three detectors but with enforced incoherence, and coherent WNBs projected into the three detectors in a physically realistic manner. Using this method, the coherence network can learn to differentiate between coincident glitches and coherent signals.

As baseline models to compare with the results of our improvement attempts, in the next section, we will train five model architectures using the GWFlow data acquisition and training pipeline @gwflow_ref. Since they are the basis of many of the subsequent attempts at CBC detection, we will train both the models presented in George _et al._ @george_huerta_cnn along with the model presented in Gabbard _et al._ @gabbard_messenger_cnn and for the coherence case, the two models of the MLy pipeline @MLy.

=== CBC Detection Recreation

We present an attempt to recreate the model and training procedure presented in George _et al._ @george_huerta_cnn and Gabbard _et al._, see @george_diagram and @gabbard_diagram respectively. The model architectures themselves were recreated as closely as possible to how they are presented in the literature, except for the addition of GWFlow whitening layers as their first layer in order to replicate the data conditioning performed in both studies. These models will act as performance baselines as we move forward and try to improve their operation. Rather than trying to recreate the exact training procedure and training distribution from the literature, however, which could end up being a difficult task, we have standardized the training procedure to achieve parity with the previously conducted perceptron experiments. See @perceptron-training-parameters for details of the parameters used in the training procedure.

#figure(
  grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("george_small_diagram.png",   width: 98%) ],
        [ #image("george_large_diagram.png", width: 100%) ],
    ),
    caption: [The two CNN architectures presented in George _et al._ @george_huerta_cnn. _Upper:_ Smaller model. _Lower:_ Larger model.]
) <george_diagram>

#figure(
    image("gabbard_diagram.png",   width: 100%),
    caption: [CNN architecture from Gabbard _et al._ @gabbard_messenger_cnn.]
) <gabbard_diagram>

==== Training

Looking at the training plots, shown in @cnn_single_accuracy, it is quickly obvious that these models drastically outperform any of the perceptrons. This is perhaps unsurprising, as the CNN architecture was specifically designed for pattern recognition both in images and time series. The training parameters are identical to those used for the single detector perceptron, see @perceptron-training-parameters. The models also train more rapidly, saturating their validation loss in fewer epochs than was required for the perceptrons, which only reached a lower accuracy even with a longer training time, see @cnn_single_loss. This is also as expected; more of the function that the networks are attempting to approximate has been predefined in the CNNs architecture. Convolutional kernels can learn from features wherever they appear in the image. If dense layers used a similar technique they would have to learn equivalent "kernels" individually for every possible feature location and they would have to do so with far fewer examples for each of these unique positions. In CNNs, a kernel learning to recognize a feature can train on instances of that feature wherever they appear in the image if a similar kernel-like structure were to develop in a dense layer, each instance of that kernel-like structure would have to learn to recognize a feature from its appearance at a single location, which would presumably occur much less often than the feature as a whole. Similar to the perceptron experiments, further validation results are presented in @cnn-validation-sec.

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("cnn_single/training_accuracy_cnn.png",   width: 100%) ],
        [ #image("cnn_single/validation_accuracy_cnn.png", width: 100%) ],
    ),
    caption: [The accuracy history of attempts to retrain Convolutional Neural Networks (CNNs) with architectures adapted from the literature using the GWFlow pipeline. A custom GWFlow whitening layer has been added to the start of each model in order to reproduce the whitening data conditioning step applied in the original studies. The structure of the models is otherwise identical. Differences in the training and validation procedures, however, may lead to slightly different results than in the original studies. Rather than exactly attempting to mimic the datasets and training process used in each of these studies, it has been kept consistent with the other results throughout the thesis, in order to facilitate comparison. The models presented are the two models from George _et al._ @george_huerta_cnn, labeled "George Small", and "George Large", to differentiate them in terms of parameter count, and the single model from Gabbard _et al._ @gabbard_messenger_cnn. The network structure of these models can be seen in @george_diagram and @gabbard_diagram, respectively. The training and validation datasets were maintained from the perceptron single-detector training experiment. The dataset contains IMRPhenomD waveforms generated using cuPhenom @cuphenom_ref injected into real interferometer noise sampled from the LIGO Livingston detector during the 3#super("rd") joint observing run @O2_O3_DQ. The optimal SNR of waveforms injected into the training and validation sets was uniformly distributed between 8 and 15. Input was from a single detector only. Each epoch consisted of $10^5$ training examples, and it should be noted that, unlike regular training pipelines, each training epoch consisted of newly generated waveforms injected into unseen noise segments, though the validation examples are consistent. Training of each model was halted after ten consecutive epochs with no improvement to validation loss, the values of which are shown in @cnn_single_loss. Validation noise was drawn from a separate pool of data segments inaccessible to the training data loader. It is immediately clear that this is a huge improvement over the perceptron models, and it makes it evident why we abandon the idea of perceptrons so quickly. Both the training and validation accuracies jump to above 90% almost immediately, and in the case of the model from Gabbard _et al._, and the largest of the models from George _et al._, they plateau at approximately 98% accuracy, with only marginal improvements from there. The smaller model from George _et al._ plateaus closer to 96% accuracy. Considering approximants from both the training and validation datasets are generated with CBCs drawn uniformly between an optimal SNR of 8 and 15, this demonstrates good performance. Because two of the models plateau at statistically similar accuracies with quite different architectures, it suggests that they are approaching the detectability limit in both cases. An interesting examination will be to compare their performance with FAR-calibrated detection thresholds. _Upper:_ Plot of model accuracies when measured with training data ($10^5$ epoch-unique examples). _Lower:_ Plot of model accuracies when measured with validation data ($10^4$ epoch-consistent examples).]
) <cnn_single_accuracy>

#figure(
    grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #image("cnn_single/training_loss_cnn.png",   width: 100%) ],
        [ #image("cnn_single/validation_loss_cnn.png", width: 100%) ],
    ),
    caption: [The loss history of attempts to retrain Convolutional Neural Networks (CNNs) from the literature using the GWFlow pipeline. These loss values correspond to the accuracies displayed in @cnn_single_accuracy. The models presented are the two models from George _et al._ @george_huerta_cnn, labeled "George Small", and "George Large", to differentiate them in terms of parameter count, and the single model from Gabbard _et al._ @gabbard_messenger_cnn. The network structure of these models can be seen in @george_diagram and @gabbard_diagram, respectively. The training and validation datasets were maintained from the perceptron single-detector training experiment. The dataset contains IMRPhenomD waveforms generated using cuPhenom @cuphenom_ref and real interferometer noise sampled from the LIGO Livingston detector during the 3#super("rd") joint observing run @O2_O3_DQ. The optimal SNR of waveforms injected into the training and validation sets was uniformly distributed between 8 and 15. Input was from a single detector only. Each epoch consisted of $10^5$ training examples, and it should be noted that, unlike regular training pipelines, each training epoch consisted of newly generated waveforms injected into unseen noise segments, though the validation examples are consistent. The loss is the metric used to determine when training is halted; this is done after ten epochs have passed with no improvement. Again we can see that this is a vast improvement over the perceptron case, see @perceptron_single_loss, at least in the time frame that is monitored, with loss values quickly falling to a region with a much smaller reduction gradient and then gradually improving from there with diminishing returns. It is these diminishing returns that can have a great impact on the ability of the model to sustain high accuracies with low FAR thresholds. _Upper:_ Plot of model losses when measured with training data ($10^5$ epoch-unique examples). _Lower:_ Plot of model accuracies when measured with validation data ($10^4$ epoch-consistent examples).]
) <cnn_single_loss>

==== Validation <cnn-validation-sec>

The validation results portray a similar picture --- vastly improved performance over the perceptron results. We can see in the False Alarm Rates (FAR) curves, @cnn_far_single, that we can utilize significantly lower FARs without dramatically increasing the required score threshold. In most cases, we expect this to allow higher efficiencies at lower FARs if the model has gained adequate detection ability. The benefit of this is displayed in the efficiency plots, @cnn_efficiency_curves_single, and the ROC plots, @roc_curves_single. We can very clearly see, that these models are dramatically improved over the perceptron case, whose efficiency curves can be seen in @perceptron_efficiency_curves_single, and ROC curves can be seen in @perceptron_roc_curve.

#figure(
image("cnn_single/far_plot.png", width: 100%),
  caption: [False Alarm Rate (FAR) plotted against the score threshold required to achieve that FAR, for three recreations of models from the literature. Two models are adapted from George _et al._ @george_huerta_cnn, labeled "George Small", and "George Large", to differentiate them in terms of model parameter count, and the single model from Gabbard _et al._ was also adapted. The network structure of these models can be seen in @george_diagram and @gabbard_diagram, respectively. The presented FAR curves are significantly lower than those achieved by the perceptrons in the single detector case, see @perceptron_single_far. This means that we will be able to achieve lower FARs with lower score thresholds, which typically, though not necessarily, leads to higher efficiencies at those FARs. We explore the efficiency results in @cnn_efficiency_curves_single.]
) <cnn_far_single>

#figure(
  grid(
    image("cnn_single/efficiency_curve_0_1.png", width: 100%),
    image("cnn_single/efficiency_curve_0_01.png", width: 100%),
    image("cnn_single/efficiency_curve_0_001.png", width: 100%),
    image("cnn_single/efficiency_curve_0_0001.png", width: 100%),
  ), 
  caption: [Model efficiency curves for three models adapted from the literature. Two models are adapted from George _et al._ @george_huerta_cnn, labeled "George Small", and "George Large", to differentiate them in terms of model parameter count, and the single model from Gabbard _et al._ @gabbard_messenger_cnn was also adapted. The network structure of these models can be seen in @george_diagram and @gabbard_diagram, respectively. These models verify that CNNs can achieve much higher accuracies within the training regime utilized, even when using threshold scores that are calibrated to specific False Alarm Rates (FARs). The perceptron efficiency curves for the single detector CBC detection case can be seen in @perceptron_efficiency_curves_single. They achieve higher accuracies almost across the board at the highest FARs depicted, #box("0.1" + h(1.5pt) + "Hz") and #box("0.01" + h(1.5pt) + "Hz"), except at SNRs where detection becomes virtually impossible ($<2$) in which case they perform similarly. They are also able to achieve results at lower FARs #box("0.001" + h(1.5pt) + "Hz") and #box("0.0001" + h(1.5pt) + "Hz"); at these FARs the perceptron models had negligible performance and were not depicted, so this is a significant improvement. _First:_ Efficiency curves at a FAR of #box("0.1" + h(1.5pt) + "Hz"). _Second:_ Efficiency curves at a FAR of #box("0.01" + h(1.5pt) + "Hz"). _Third:_ Efficiency curves at a FAR of #box("0.001" + h(1.5pt) + "Hz"). _Fourth:_ Efficiency curves at a FAR of #box("0.0001" + h(1.5pt) + "Hz").]
) <cnn_efficiency_curves_single>

#figure(
    image("cnn_single/roc_8.png", width: 100%),
    caption: [Reciever Operator Characteristic (ROC) curves, for three models adapted from the literature. Two models are adapted from George _et al._ @george_huerta_cnn, labeled "George Small", and "George Large", to differentiate them in terms of model parameter count, and the single model from Gabbard _et al._ @gabbard_messenger_cnn was also adapted. The network structure of these models can be seen in @george_diagram and @gabbard_diagram, respectively. In comparison with the ROC curves achieved by the perception models, see @perceptron_roc_curve, which at an optimal SNR of 8 looks to be almost randomly guessing, this is a significant improvement. The curves shown illustrate the model operating on a pool of injected signals at an optimal SNR of 8.]
) <roc_curves_single>

It would appear from our investigation that CNNs offer a far superior solution to the single-detector CBC detection problem than perceptrons do. Several questions remain, however. Are we nearing the limit of the CNN's capacity to solve this problem, or could further hyperparameter tuning, squeeze additional performance out of the architecture, especially at low False Alarm Rates? There have been many attempts to improve on these models throughout the literature @spectrogram_cnn_2 @krastev_bnn_cnn, but there lacks a systematic algorithm to search across all possible solutions, (or at least a large number of possible solutions), to find the optimal detection architecture and training procedure. We investigate this in @dragonn-sec, where we attempt to use genetic algorithms to search the large hyperparameter space presented for more optimal solutions. There are also several more recently developed architectures, including attention-based models @attention_is_all_you_need, which offer alternate and possibly superior alternatives to convolutional layers, we explore the use of attention layers for CBC classification in @skywarp-sec. Finally, there are other, potentially more challenging problems facing gravitational-wave data science, including parameter estimation. In @crosswave-sec we tackle a special case wherein two overlapping signals are present in our input data and examine if we can separate the case of single and overlapping signals. We then test if we can extract parameters from each signal, both in aid of alternate parameter estimation methods, and potentially as a precursor to a full machine learning-based parameter estimator.