= CrossWave: Detection and Parameterisation of Overlapping Compact Binary Coalescence Signals <crosswave-sec>

We introduce CrossWave, a new attention-based neural network model for the identification and parameter estimation of overlapping CBC signals. CrossWave can with efficiencies matching that of more conventional matched filtering techniques, separate the case of overlapping mergers from lone mergers, but with considerably lower inference times and computational cost. We suggest CrossWave or a similar architecture may be used to augment existing CBC detection and parameter estimation infrastructure, either as a complementary confirmation of the presence of overlap or to extract the merger times of each signal in order to use other parameter estimation techniques on the separated parts of the signals.

Significant improvements to our gravitational wave detection capability are anticipated within the next decade, with improvements to existing detectors [cite], as well as future 3rd and 4th generation space and ground-based detectors such as the Einstein Telescope [cite] and Cosmic Explorer. Whilst the current rate of Compact Binary Coalescence (CBC) detection is too low (estimate) for any real concern about the possibility of overlapping detections, estimated rates for future networks (estimate) will render such events a significant percentage of detections.

Contemporary detection and parameter pipelines do not currently have any capabilities to deal with overlapping signals - and although, in many cases, detection would still be achieved [cite], it is likely that parameter estimation would be compromised by the presence of the overlap, especially if more detailed information about higher modes and spins [cite] are science goals.

We introduce CrossWave, two attention-based neural network models for the identification and parameter estimation of overlapping CBC signals. CrossWave consists of two complementary models, one for the separation of the overlapping case from the non-overlapping case and the second as a parameter estimation follow-up to extract the merger times of the overlapping signals in order to allow other parameter estimation methods to be performed.

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


== CrossWave

#set page(
  flipped: true
)
#set align(center)
#image("crosswave_small_diagram_expanded.png",  width: 85%)
#image("crosswave_large_diagram_expanded.png",  width: 96%)
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
