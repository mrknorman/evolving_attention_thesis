#set page(numbering: "1", number-align: center)

#set math.equation(numbering: it => {[2.#it]})
#counter(math.equation).update(0)
// #set math.mat(delim: "[")

#import "../notation.typ": vectorn, uvectorn, dvectorn, udvectorn, matrixn

= Gravitational Waves <gravitational-waves-sec>

Since time immemorial, humanity has gazed at the stars. With wonder rooted deep in their minds, they imagined strange and divine mechanisms in order to try and make sense of what they saw. Over the millennia, the vast skies have revealed much about their workings, and with ever-improving tools we have come much closer to understanding their mysteries, but there is still much to be uncovered. It is unclear how deep the truth lies. Perhaps we have but only scratched at the surface. The depths are so vast we simply do not know.

Almost all of that knowledge, all of that understanding and academia, has been built upon the observation of just a single type of particle. Until very recently, the only information we had about the world above us came from light, and although the humble photon has taught us a great deal about the scope of our universe, the discovery of new messengers promises pristine, untapped wells of science. It has only been in the last century that we have achieved the technological prowess to detect any other extraterrestrial sources of data except that which fell to us as meteors. We have brought rocks home from the moon @moon_rocks. We study data sent back from probes shot out across the solar system and even ones that have peaked just beyond the Sun's mighty sphere of influence @voyager. We have seen neutrinos, tiny, almost massless particles that pass through entire planets more easily than birds through the air @neutrino, and single particles with the energy of a Wimbledon serve @cosmic_rays. Most recently of all, we have seen the skin of space itself quiver --- gravitational waves, the newest frontier in astronomy @first_detction.

Practical gravitational-wave astronomy is still in its infancy; compared to the other fields of astrophysics, it has barely left the cradle, with the first confirmed gravitational-wave detection occurring in 2015 @first_detction. Although the field has progressed remarkably quickly since its inception, there is still a lot of work to be done --- a lot of groundwork to be performed whilst we uncover the best ways to deal with the influx of new data that we are presented with. It seems likely, assuming both funding and human civilization prevail, that work undertaken now will be but the first bricks in a great wall of discovery. New gravitational-wave observatories that are even today being planned will increase our sensitive range by orders of magnitude @einstein_telescope @cosmic_explorer @LISA. With any luck, they will open our ears further to previously undiscovered wonders.

This chapter will introduce a small part of the science of gravitational waves; it will not be an extensive review as many of the particularities are not especially relevant to the majority of the content of this thesis. Instead, this section aims to act as a brief overview to give context to the purpose behind the data-analysis methods presented throughout. We will cover the theoretical underpinning of gravitational waves, and perform a glancing tour through the experiments used to detect them.

== Gravity

Gravity is one of the four fundamental forces of nature, the other three being the electromagnetic force and the strong and weak nuclear forces @four_forces. It is, in some ways, the black sheep of the interactions, as it is the only one not explained by the standard model of particle physics, which is, by some measures, the most accurate theory of physics ever described @standard_model. Luckily, gravity has its own extremely accurate descriptive theory @testing_gr. It has a storied history, which, if you are unfamiliar, is worth skimming for context. 

=== Ye Old Times

In the beginning, men threw rocks at each other and were entirely unsurprised when they hit the floor. Over time, people became more and more confused as to why this was the case. Many philosophers proposed many reasons why one direction should be preferred over all others when it came to the unrestrained motion of an object. For a long time, there was much confusion about the relationship between mass, density, buoyancy, and the nature and position of various celestial objects. Sometime after we had decided that the Earth was not, in fact, at the centre of the universe and that objects fell at the same speed irrespective of their densities, came the time of Sir Issac Newton, and along with him arrived what many would argue was one of the most influential theories in the history of physics.

The idea of gravity as a concept had been around for many thousands of years by this point @gravity_history, but what Newton did was formalise the rules by which objects behaved under the action of gravity. Newton's universal law of gravitation states that all massive objects in the universe attract all others @principia_mathematica, acting upon each other whether surrounded by a medium or not. Gravity appeared to ignore all boundaries and was described in a simple formula that seemed to correctly predict everything from the motion of the planets (mostly) to the fall of an apple

$ F = G frac(m_1 m_2, Delta r^2). $ <newtons-law-of-universal-gravitation>

where $F$ is the scalar force along the direction between the two masses, $G$ is the gravitational constant equal to #box($6.67430(15) times 10^(−11)$ + h(1.5pt) + $m^3 "kg"^(-1) s^(-2)$) @gravitational_constant, $m_1$ is the mass of the first object, $m_2$ is the mass of the second object, and $Delta r$ is the scalar distance between the two objects. In vector form, this becomes,

$ vectorn(F) = - G frac(m_1 m_2, |dvectorn(r)|^2) udvectorn("r") = - G frac( m_1 m_2, |dvectorn(r)|^3) dvectorn(r), $ <newtons-law-of-universal-gravitation-vec>

where $vectorn(F)$ is the force vector exerted on body 2 by the gravitational effect of body 1, $dvectorn(r)$ is the displacement vector between bodies 1 and 2, and $udvectorn(r)$ is the unit direction vector between bodies 1 and 2.

Newton's law of universal gravitation describes the force every massive object in the universe experiences because of every other --- an equal and opposite attraction proportional to the product of their two masses @principia_mathematica; see @newtons_law. Though we now know this equation to be an imperfect description of reality, it still holds accurate enough for many applications to this day.

#figure(
  image("newtons_law.png", width: 30%),
  caption: [An illustration of Newton's law of universal gravitation, as described by @newtons-law-of-universal-gravitation. Two particles, here distinguished by their unique masses, $m_1$ and $m_2$, are separated by a distance, $r$. According to Newton's law, they are pulled toward each other by the force of gravity acting on each object $F$ @principia_mathematica, each being pulled directly toward the other by a force that is equal and opposite to its partner's.],
) <newtons_law>

It was also at this time that a fierce debate raged over the nature of time and space. Newton proposed a universal time that ticked whether in the presence of a clock or not, and a static, ever-present grid of space that never wavered nor wandered. Both space and time would continue to exist whether the rest of the universe was there or not @newton_vs_leibniz. Leibniz, on the other hand, argued that space and time were little more than the relations between the positions of objects and their velocities. By his reasoning, if there were no objects, there would be nothing to measure, and there would be no space. If there were no events, there would be nothing to time, and there would be no time; see @absolute_time_and_space. At the time, they did not come to a resolution, and to this day we do not have a definite answer to this question @what_is_spacetime, but as we will see, each saw some aspect of the truth.

#figure(
  grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #align(center)[#image("absoloute_space.png", width: 100%)] ],
        [ #align(center)[#image("relative_space.png", width: 100%)] ],
  ),
  caption: [An illustration of two competing historical views on the nature of space and time @newton_vs_leibniz. _Upper:_ Newton's vision of absolute universal time and absolute space, wherein time moves forward at a constant and uniform rate across the universe and space is immobile and uniform. In this model, both time and space can exist independently of objects within, even in an entirely empty universe. _Lower:_ Leibniz's view wherein time and space did not and could not exist independently of the objects used to measure them. Within this model, space is simply a measure of the relative distances between objects, and time is a measure of the relative motion of objects as their relative positions change. In this model, it makes little sense to talk of a universe without objects since time and space do not exist without objects with relative positions and velocities.],
) <absolute_time_and_space>

For a good few centuries, Newton's law of universal gravitation stood as our fundamental understanding of gravity, with its impressive descriptive and predictive power @newton_dominance. As our measurements of the solar system became more precise, however, a major discrepancy was noted, one that Newton's law failed to describe. The planet Mercury, so close to the sun and so heavily influenced by its gravity, was found to be behaving ever so slightly strangely @mercury_precession. Under Newton's laws, the orbits of the planets were described precisely --- ellipses plotted through space. The influence of other gravitational bodies, such as the other planets, would cause these ellipses to precess, their perihelion increasing with time. The vast majority of the precession of Mercury was accounted for by applying Newton's laws to the solar system as a whole. However, a small amount, only the barest fractions of a degree per century, remained a mystery. For a long time, it was thought there was an extra hidden planet in the inner solar system, but none was ever found. If this extra precession was an accurate measurement, the difference was enough to state with confidence that Newton's universal law of gravitation was not a complete description of gravity.

=== Special Relativity <special-relativity-sec>

By the start of the 20#super("th") century, two more thorns in Newton's side had been revealed. Experiments failed to detect a change in the speed of light irrespective of the Earth's motion through space @absoloute_light --- if light behaved as we might expect from ordinary matter, then its measured speed should change depending on whether we are moving toward its source, and hence in opposition to its own direction of motion, or against and in unison with its direction of motion. That is not what was observed. Light appeared to move at the same speed no matter how fast you were going when you measured it, whether you measured your velocity relative to its source or any other point in the universe. There was no explanation for this behaviour under Newtonian mechanics. The second tantalising contradiction arrived when attempting to apply Maxwell's hugely successful equations describing electromagnetism, which proved incompatible with Newtonian mechanics, again in large part because of the requirement for a constant speed of light at all reference frames @maxwells. This failing of Newtonian mechanics was noted by Hendrik Lorentz and Henri Poincaré, the former of which developed many of the ideas and mathematics later used by Einstein @lorentz_relativity.

In 1905, Einstein built upon Lorentz's work @lorentz_relativity and proposed his theory of special relativity as an extension beyond standard Newtonian mechanics in a successful attempt to rectify the previously mentioned shortcomings @special_relativity. The initial presentation of special relativity was built upon two revolutionary principles. Firstly, the speed of light was the same in all reference frames, meaning that no matter how fast you were travelling relative to another body, the speed of light would, to you (and to all observers), appear the same as it always has --- light would move away from you as it always had done, unaffected by your apparent velocity. Secondly, and closely related to the first principle, special relativity states that the laws of physics will act identically in all inertial reference frames. If you are isolated from the outside world by some impenetrable shell, there is no experiment you can perform to determine that you are moving relative to another body --- the only situations between which you could tell the difference were between different non-inertial reference frames (and between a non-inertial reference frame and an inertial one), wherein the shell surrounding you was accelerating at different rates. By introducing these postulates, Einstein explained the observations of light-speed measurements and allowed for the consistent application of Maxwell's laws.
 
What special relativity implied was that there was no one true "stationary" reference frame upon which the universe was built @special_relativity, seemingly disproving Newton's ideal of an absolute universe. All inertial frames were created equal. This seemingly innocent fact had strange consequences for our understanding of the nature of space and time. In order for the speed of light to be absolute, space and time must necessarily be relative -- were they not, then the cause-and-effect nature of the universe would break down. 

We can visualize the problem in a thought experiment, as Einstein often liked to do @light_clock. Imagine an observer standing in the carriage of a train moving at a constant velocity relative to a nearby platform. The observer watches as a light beam bounces back and forth between two mirrors, one on the ceiling, and the other on the floor. From the perspective of the observer, the time taken for light to transit this fixed vertical distance is also fixed, and determined by the speed of light and the distance between the mirrors. 

A second observer stands on a nearby platform a looks into the moving train as it passes (it has big windows) @light_clock. As they watch the light beam bounce between the two mirrors, they see that, from their reference frame, the beam must take a diagonal path between the mirrors as the train moves forward. This diagonal path is longer than the vertical path observed in the carriage's reference frame. If we take special relativity to be true, the speed of light must be constant for both observers. However, in one reference frame, the light must travel a greater distance than in the other. It cannot be the case that the time taken for the photon to travel between the mirrors is the same for the observer on the carriage and the observer on the platform --- their measurements of time must differ in order to preserve the supremacy of the speed of light. The observer on the platform would indeed see time passing on the train more slowly than time on the apparently "stationary" platform around them --- this effect is known as *time dilation*, and it has since been experimentally verified @time_dilation_ref. We can quantify this effect using

$ Delta t' = (Delta t) / sqrt(1 - v^2/c^2) = gamma(v) Delta t, $ <time-dilation-eq>

where $Delta t'$ is the measured duration of an event (the time it takes light to move between the two mirrors) in a secondary inertial reference frame (the train carriage) which has a relative velocity, $v$, compared with the inertial reference frame of the current observer (the platform), $Delta t$ is the measured duration of the same event when measured in the secondary inertial reference frame (the train carriage), $c$ is the speed of light in a vaccum, #box($299,792,458$ + h(1.5pt) + $ m s^(-1)$), and $gamma$ is the Lorentz factor given by

$ gamma(v) = 1 / sqrt(1 - v^2/c^2). $ <lorentz-factor>

An illustration of this effect can be seen in @light_clock_diagram.

#figure(
  grid(
        columns: 1,
        rows:    2,
        gutter: 1em,
        [ #align(center)[#image("train_observer.png", width: 52%)] ],
        [ #align(center)[#image("platform_observer.png", width: 100%)] ],
  ),
  caption: [An illustration of the light clock thought experiment. The light clock thought experiment is a scenario that can be imagined in order to illustrate the apparent contradiction that arises from a universally constant speed of light. In order to rectify this contradiction, the concepts of time dilation and length contraction are introduced, fundamentally changing our understanding of the nature of time and space. Two observers stand in inertial reference frames. From special relativity, we know all inertial reference frames are equal, and the laws of physics, including the speed of light, should look identical @light_clock @special_relativity. _Upper:_ The observer on the train measures the time it takes a single photon of light to bounce from a mirror at the bottom of the train, to a mirror at the top, and back again. The distance travelled by the light beam is two times the height of the train, $H$, which gives $2H$. The time it takes a particle to transit a given distance, $D$, is given by $Delta t = D / v$. Since light always travels at $c$, we know the measured photon transit time in this reference frame will be $Delta t = 2H / c$. _Lower:_ A second observer, standing on a platform, watches as the train passes at a constant velocity, $v$. Through a large window in the carriage, they observe the first observer performing their experiment. However, from the second observer's reference frame, the light now has to move on a diagonal path created by the motion of the train, we can calculate its new transit length $2D$, using Pythagoras's theorem. Each of the two transit, will, by definition take half of the total transit time measured by the platform observer, $1/2 Delta t'$, and in this time the train will have moved, $1/2 Delta t' v$, this gives us $D = sqrt(H^2 + (1/2 Delta t' v)^2)$. If we substitute this new distance into the original equation to calculate the duration of the transit $Delta t' = D / v$, we get $Delta t' = sqrt(H^2 + (1/2 Delta t' v)^2) / v$. This means that the platform observer measures a longer transit duration. Since the bouncing light beam is a type of clock, a light clock, and all functioning clocks in a given inertial reference will tick at a consistent rate, we can conclude that time is passing more slowly for the observer on the train when observed from the platform's reference frame. In reality, these effects would only become noticeable to a human if the velocities involved were significant fractions of the speed of light. In everyday life, the effects of special relativity are negligible, which was probably why it took so long for anyone to notice.]
) <light_clock_diagram>

Similarly, if we orient the mirrors horizontally, so that the light travels along the length of the carriage, a different relativistic effect becomes apparent @light_clock. The observer on the platform, observing the light's path as longer due to the train's motion, must reconcile this with the constant speed of light. This reconciliation leads to the conclusion that the train, and the distance between the mirrors, are shorter in the direction of motion from the platform observer's perspective. This phenomenon, where objects in motion are contracted in the direction of their movement, is known as *length contraction* and is described by

$ L' = L sqrt(1 - v^2/c^2) = L / gamma(v), $ <length-contraction-eq>

where $L'$ is the length of an object when measured from an inertial reference frame that has a velocity, $v$, relative to the inertial frame of the measured object, L is the "proper length" of the object when its length is measured in the object's inertial frame, $c$ is the speed of light in a vacuum, #box($299,792,458$ + h(1.5pt) + $ m s^(-1)$), and $gamma$ is the Lorentz factor given by @lorentz-factor.

Together, length contraction and time dilation shatter Newton's notions of absolute time and space @special_relativity. It should be remembered, however, that neither the carriage observer nor the platform observer can be said to be in the true stationary reference frame. The observer standing in the station is in the same inertial reference frame as the rest of the Earth, but that doesn't make it any more valid than any other. If the observer at the station had a similar setup of mirrors and light beams, and the train occupant looked out at them, the train occupant would observe the same phenomenon. To the passenger, time outside the train appears slowed, and the station shorter than it ought to be. This seems to be a paradox, often known as the twin paradox. What happens if the train later stopped and the two observers were to meet? Who would have experienced more time? It is a common misconception that acceleration must be introduced in order to reconcile the two clocks, however, even staying within the regime of special relativity, we can observe an asymmetry between the two observers @twin_paradox. In order for the two observers to meet in a shared reference frame, one of the observers, in this case, the train passenger, must change between reference frames, even if that change is instantaneous. This asymmetry allows us to solve the paradox, but the explanation is a little complex so will not be discussed here.

In order to transfer values between two coordinate frames we may use what is referred to as a Lorentz transform, the simplest of which involves moving from the coordinates of one inertial reference frame to another moving at a velocity, $v$, relative to the first. We can see that from @time-dilation-eq and @length-contraction-eq, this transform is given by

$ t' = gamma(t - frac(v x, c^2) ) , $ <lorentz_t_eq>

$ x' = gamma(x - v t), $ <lorentz_x_eq>

$ y' = y, $

and

$ z' = z . $

Noting that as expected, there are no changes in the $y$ and $z$ direction.

Though the world presented by special relativity may at first seem counter-intuitive and hard to belive, there have been inumerable experiments verifying its predictions, most famously, the Global Positioning System (GPS) network of satalites, would be unable to operate without accouting for the time-dilation induced by their relative velocity @gps_relativity, due to the exteremely precise time mesurements required. 


=== Minkowski Spacetime <minkowski-sec>

Although the notions of independent and absolute time and space were dislodged, it is still possible to describe the new universe illuminated by special relativity as an all-pervasive 4D geometry inside which the universe sits. Unlike Newton's world, however, space and time are inseparably linked into one joint four-dimensional continuum wherein motion can affect the relative measurements of time and space. We call this geometry *spacetime*. Time intervals between events are not fixed, and observers don't necessarily agree on their order. Distances must be described by a combination of temporal and spatial coordinates, and because all inertial reference frames are equal, all inertial coordinate systems (ways of assigning reference values to points in spacetime) are also equally valid.

Special relativity deals with flat spacetime. This type of spacetime is known as *Minkowski space* @gravitation; see @flat for an illustration. Although it is non-Euclidian, and its geometry can sometimes be counterintuitive to people used to travelling at pedestrian velocities, it is still isotropic and homogeneous; it looks identical, no matter where in it you are, or what velocity you are traveling at relative to any other point or object.

We can fully describe a given geometry by constructing a metric that can return the distance between any two points in that geometry. In standard 3D Euclidean geometry, which is the most instinctively familiar from everyday life, a point can be represented by a three-vector comprised of $x$, $y$, and $z$ components,

$ vectorn(r) = mat(x; y; z;) . $<euclidean_point>

The scalar distance, $Delta r$, between two points each described by @euclidean_point is given by the Euclidean distance formula --- the expansion of Pythagoras' theorem from two dimensions into three,

$ Delta r^2 = ||dvectorn(r)||^2 = Delta x^2 + Delta y^2 + Delta z^2 $ <euclidean_formula>

where $Delta r$ is the scalar distance between two points separated by $Delta x$, $Delta y$, and $Delta z$ in the $x$, $y$, and $z$ dimensions respectively, and $dvectorn(r)$ is the displacement vector between the two points. This relationship assumes a flat geometry and does not consider the role that time plays in special relativity. In the case of Euclidean geometry, the metric that we have omitted in @euclidean_formula is the $3 times 3$ Euclidean metric

$ matrixn(g) = mat(
  1, 0, 0;
  0, 1, 0;
  0, 0, 1;
). $ <identity_metric>

We can use @identity_metric and @euclidean_formula to construct a more complete expression, which can be adjusted for different geometries,

$ Delta r^2 = ||dvectorn(r)||^2 = dvectorn(r)^bold(T) matrixn(g) dvectorn(r) = mat(Delta x, Delta y, Delta z;) mat(
  1, 0, 0;
  0, 1, 0;
  0, 0, 1;
) mat(
  Delta x;
  Delta y;
  Delta z;
) = Delta x^2 + Delta y^2 + Delta z^2 . $

In this case, the inclusion of this metric does not change the calculation of the scalar distance between two points, however, as we have seen in @special-relativity-sec, in order to represent the spacetime described by special relativity, we must include the time dimension, $t$, which does not behave symmetrically with the others. The Minkowski metric allows us to explore beyond standard 3D Euclidean geometry by including a 4#super("th") dimension, time

$ matrixn(eta) = mat(
  -1, 0, 0, 0;
  0, 1, 0, 0;
  0, 0, 1, 0;
  0, 0, 0, 1;
). $ <minkowski_metric>

Using @minkowski_metric, which describes a flat spacetime, we can use this metric to compute the interval between two events in flat Minkowski space, whose positions can be described as four-positions (four-vectors), #vectorn("s"), of the following form:

$ vectorn(s) = mat(c t; vectorn(r);) = mat(c t; x; y; z;) $ <four-vector>

where $vectorn(s)$, is the four-position of an event in spacetime, $c$ is the speed of light in a vacuum, #box($299,792,458$ + h(1.5pt) + $ m s^(-1)$), $t$ is the time component of the four-position, and #vectorn("r") is a position in 3D Euclidean space. We set $s_0 = c t$ rather than just $t$ to ensure that each element of the four-position is in the same units. 

From @minkowski_metric and @four-vector, it follows that the displacement four-vector between two events in Minkowski spacetime, $dvectorn(s)$, can be computed with 

$ Delta s^2 = dvectorn(s)^bold(T) matrixn(eta) dvectorn(s)= - c^2 Delta t^2 + Delta x^2 + Delta y^2 + Delta z^2 . $ <spacetime-interval>

Even though two observers may disagree on the individual values of the elements of the vector describing the four-displacement, #dvectorn("s"), between the two events, $Delta s$, known as the spacetime interval, is invariant and has a value that all observers will agree on, independent of their reference frame. Using @spacetime-interval, we can describe the relationship of events and interactions in a flat Minkowski spacetime.

We can show that the Minkowski metric is consistent with length contraction and time dilation, described by @time-dilation-eq and @length-contraction-eq respectively, by showing that the spacetime interval, $Delta s$, is equal in two different coordinate frames that disagree on the values of $Delta t$ and $Delta x$.

In a second, boosted coordinate frame moving with a velocity $v$ (in the x-axis alone) relative to our initial frame, @spacetime-interval becomes

$ Delta s^2 = - c^2 Delta t'^2 + Delta x'^2 + Delta y'^2 + Delta z'^2 . $ <spacetime-interval-shifted>

We can substitute @lorentz_t_eq and @lorentz_x_eq into @spacetime-interval-shifted, and show that $Delta s^2$ remains the same. Substituting we get 

$ Delta s^2 = - c^2 gamma ^2 ( Delta t - frac(v Delta x, c^2))^2 + gamma^2 (Delta x - v Delta t)^2 Delta y^2 + Delta z^2 . $

We can also substitute our definition for the Lorentz factor, $gamma$, given by @lorentz-factor to get 

$ Delta s^2 = - c^2 (1 / sqrt(1 - v^2/c^2))^2 ( Delta t - frac(v Delta x, c^2))^2 + (1 / sqrt(1 - v^2/c^2))^2 (Delta x - v Delta t)^2 + Delta y^2 + Delta z^2 . $

Expanding the squares gives us

$ Delta s^2 = frac( - c^2 ( Delta t - frac(v Delta x, c^2)) ( Delta t - frac(v Delta x, c^2)) , 1 - v^2/c^2) + frac((Delta x - v Delta t)(Delta x - v Delta t), 1 - v^2/c^2) + Delta y^2 + Delta z^2 . $

We can then multiply out the brackets to get

$ Delta s^2 = frac( - c^2 Delta t^2 + 2 c^2 Delta t frac( v Delta x, c^2) - c^2 frac(v^2 Delta x^2, c^4), 1 - v^2/c^2) + frac(Delta x^2 - 2 v Delta t Delta x + v^2 Delta t^2, 1 - v^2/c^2) + Delta y^2 + Delta z^2 , $

and we can cancel this further to get

$ Delta s^2 = frac( - c^2 Delta t^2 + 2 v Delta t Delta x - frac(v^2 Delta x^2, c^2), 1 - v^2/c^2) + frac(Delta x^2 - 2 v Delta t Delta x + v^2 Delta t^2, 1 - v^2/c^2) + Delta y^2 + Delta z^2 . $

Next, we can merge the first two terms under their common denominator, $1 - v^2/c^2$, to get 

$ Delta s^2 = frac( - c^2 Delta t^2 + 2 v Delta t Delta x - frac(v^2 Delta x^2, c^2) + Delta x^2 - 2 v Delta t Delta x + v^2 Delta t^2, 1 - v^2/c^2) + Delta y^2 + Delta z^2 . $

This reduces to

$ Delta s^2 = frac( - c^2 Delta t^2 - frac(v^2 Delta x^2, c^2) + Delta x^2 + v^2 Delta t^2, 1 - v^2/c^2) + Delta y^2 + Delta z^2 . $

We can then rewrite the numerator in terms of $Delta t^2$ and $Delta x^2$, since we are aiming to reduce it to this form. This gives us

$ Delta s^2 = frac( -(c^2 + v^2) Delta t^2  + (1 - frac(v^2, c^2)) Delta x^2, 1 - v^2/c^2) + Delta y^2 + Delta z^2 . $

We can then split the common demoniator into two fractions, giving us

$ Delta s^2 = frac( -(c^2 + v^2) Delta t^2, 1 - v^2/c^2) + frac((1 - frac(v^2, c^2)) Delta x^2, 1 - v^2/c^2) + Delta y^2 + Delta z^2 . $

The coefficients in the term in $Delta x^2$ cancel to leave us with only $Delta x^2$, and we can divide the coefficients of the $Delta t^2$ term by a factor of $c^2$ to give us

$ Delta s^2 = frac( - c^2 (1+ v^2 / c^2 ) Delta t^2, 1 - v^2/c^2) + Delta x^2 + Delta y^2 + Delta z^2 . $

Which cancels and returns us to our original expression @spacetime-interval,

$ Delta s^2 = - c^2 Delta t^2 + Delta x^2 + Delta y^2 + Delta z^2 . $

This shows, that after performing a Lorentz transform by a constant velocity, $v$, in the $x$-axis, the spacetime interval, $ Delta s$, remains constant, i.e,

$ Delta s^2 = - c^2 Delta t^2 + Delta x^2 + Delta y^2 + Delta z^2 =  - c^2 Delta t'^2 + Delta x'^2 + Delta y'^2 + Delta z'^2 . $

This demonstrates that performing a Lorentz transform between two inertial reference frames is consistent with the formulation of Minkowski spacetime described by @minkowski_metric. 

As alluded to, special relativity, and Minkowski Spacetime, only deal with inertial reference frames, hence it is a "special" case of a larger, cohesive theory --- that theory, developed by Einstein in the following years, is general relativity @gravitation. 

When dealing with the gravitational effects of spacetime, we are often considering point-like particles or spherical masses; for this reason, it is very often convenient to work with spherical coordinates with the basis $t$, $r$, $theta$, and $phi$ rather than the Euclidean coordinate system we have been using so far. In spherical coordinates @spacetime-interval becomes

$ Delta s^2 = -c^2 Delta t^2 + Delta r^2 + r^2 Delta Omega^2 $ <minkowski-interval-spherical>

where

$ Delta Omega^2 = Delta theta^2 + sin^2 theta Delta phi^2 $

is the standard metric used on the surface of a two-sphere --- a 2D spherical surface embedded in a 3D space. @minkowski-interval-spherical will become a valuable reference when we move to examine curved spacetime under the influence of gravity.

=== General Relativity <general-relativity-sec>

Einstein realized that by introducing deformations to the otherwise flat Minkowski spacetime described by special relativity you could induce accelerations in particles within this spacetime without invoking any forces @gravitation. Rather than being attracted by some gravitational "force", the particles continue to behave as they always had, following their natural paths or *geodesics*. A geodesic is the shortest path between two points in a given geometry; in Euclidian geometry, all geodesics are straight lines, in other geometries however, this is not necessarily the case. Thus, depending on the shape of the spacetime they exist within, particles can accelerate with respect to each other whilst remaining within inertial frames. This is the reason that it is often stated that gravity is "not a force" --- gravitational attraction is a result of the geometry of the spacetime in which objects exist, rather than because of any fundamental attraction caused by something with the traditional properties of a force. 

It should be noted that although under general relativity gravity is not described in the same way as the other fundamental forces, it is still often useful and valid to describe it as such. We don't yet have a unifying theory of all of the forces, so they may end up being more similar than current theories describe.

After observing that deformations in spacetime would cause apparent accelerations akin to a force of gravity, the natural jump to make is that massive objects that have gravity deform spacetime @gravitation. The more massive the object, the larger the gravitational well and the more negative the gravitational potential energy of an object within that valley. The more dense the object, the steeper the gravitational well, and the stronger the gravitational attraction. See @gravitaional-potentials for an illustration. 

What we experience as the force of gravity when standing on the surface of a massive body like Earth, is an upward acceleration caused by the electromagnetic force of the bonds between atoms within the Earth. These atoms exert upward pressure on the soles of our feet. We know we are accelerating upward because we are not in freefall, which would be the case if gravity was a force that was balanced against the force of the planet below. Our bodies, and all particles, simply wish to continue on their geodesics, and in the absence of any other forces, that path would be straight down toward the centre of the Earth.

In general relativity, spacetime is described as a four-dimensional *manifold* @gravitation. A manifold is a type of space that resembles Euclidean space locally irrespective of its global geometry. This is why on the scales humans are used to dealing with, we experience space as Euclidean and never anything else. Consequentially, the flat spacetime described by Minkowski space is also a manifold. Specifically, the type of manifold that represents spacetime is known as a *Lorenzian manifold*, which has all the properties thus far described, plus some extra conditions. The Lorenzian manifold is a differentiable manifold, meaning that its differential is defined at all points without discontinuities between different regions.

Einstein formulated ten equations that describe how gravity behaves in the presence of mass and energy, known as Einstein's Field Equations (EFEs). The full complexity of EFEs is not required for this brief introduction, however, they take the general form of

$ matrixn(G_(mu v)) + Lambda matrixn(g_(mu v)) = frac(8 pi G, c^4) matrixn(T_(mu v)) $

where $matrixn(G_(mu v))$ is the Einstein tensor, describing the curvature of spacetime given the specific distribution of mass-energy described by $matrixn(T_(mu v))$, $Lambda$ is the cosmological constant, $matrixn(g_(mu v))$ is the metric tensor, describing the generic geometric structure of spacetime, and $matrixn(T_(mu v))$ is the stress-energy tensor, describing the distribution of mass and energy across a given spacetime.

#figure(
  grid(
        columns: 2,
        rows:    1,
        gutter: 1em,
        [ #align(center)[#image("flat.png", width: 100%)]],
        [ #align(center)[#image("earth.png", width: 100%)] ],
  ),
  caption: [Two depictions of Einsteins's spacetime. For illustrative purposes, since we are not 4D beings and the paper on which this will be printed very much isn't, the four dimensions of our universe have been compacted down into two. It should also be noted that these illustrations were not generated with correct physical mathematics but only to give an impression of the concepts being described. _Left:_ Minkowski space --- in the absence of any mass, spacetime will not experience any curvature @gravitation. This is the special case that Einstien's special relativity describes. If we were to place a particle into this environment, it would not experience any acceleration due to gravity. If the particle were massive, it would distort the spacetime, and the spacetime would no longer be considered Minkowski space even though, alone, the particle would not experience any acceleration. Often, when dealing with particles of low mass, their effects on the distortion of spacetime are ignored, and we can still accurately describe the scenario with special relativity @special_relativity. _Right:_ Spacetime distorted by a massive object, shown in blue. Curved space is described by Einstein's more general theory, general relativity @gravitation. In this scenario, we can see how the presence of mass imprints a distortion into the shape of spacetime. Any particles also present in the same universe as the blue object, assuming it has existed indefinitely, will experience an apparent acceleration in the direction of the blue sphere. A beam of light, for example, comprised of photons and entirely massless, would be deflected when moving past the sphere. Even though light will always travel along its geodesic through the vacuum of space, the space itself is distorted; therefore, a geodesic path will manifest itself as an apparent attraction toward the sphere. Notice that the mass of the photon is zero; therefore, using Newton's universal law of gravitation @newtons-law-of-universal-gravitation, it should not experience any gravitational attraction, and indeed, gravitational lensing of the passage of starlight, as it moved past the Sun, was one of the first confirmations of Einstein's theory of general relativity @gravitational_lensing. Even if we assume the photon has some infinitesimal mass, Newtonian mechanics predicts a deflection angle that is only half as large as General Relativity predicts, and half as large as what is observed. Were this sphere several thousand kilometres in diameter, any lifeforms living on its surface, which would appear essentially flat at small scales, would experience a pervasive and everpresent downward force. Note that the mass of the object is distributed throughout its volume, so in regions near the centre of the sphere, the spacetime can appear quite flat, as equal amounts of mass surround it from all directions.],
) <flat>

Perhaps not the first question to arise, but certainly one that would come up eventually, would be, what happens if we keep increasing the density of a massive object? Is there a physical limit to the density of an object? Would gravity keep getting steeper and steeper? The mathematical solution to this question was inadvertently answered by Karl Schwarzschild, who found the first non-flat solutions to EFEs @gravitation. The solution described the exterior of a spherical mass @gravitation. The Schwarzschild metric that describes the geometry of this manifold is 

$ matrixn(g_(mu v)) = mat(
  - (1 - frac(r_s, r)), 0, 0, 0;
  0, (1 - frac(r_s, r))^(-1), 0, 0;
  0, 0, r^2, 0;
  0, 0 , 0, r^2 sin^2 theta;
) $

and the spacetime line element for the Lorenzian manifold described by metric is given by 

$ d s = - (1 - frac(r_s, r) ) c^2 d t^2 + (1 - frac(r_s, r))^(-1) d r^2 + r ^2 d Omega^2 $

where $r_s$ is the Schwarzschild radius of the massive body inducing the spacetime curvature. The Schwarzschild radius is given by

$ r_s = frac(2 G M , c^2) . $

As can be seen from inspection, this metric introduces multiple singularities. The singularity introduced at $r = r_s$, can be shown to be a coordinate singularity alone, that can be removed via choice of coordinate system. However, the other singularity that is introduced, at the centre of the mass, often known simply as "the singularity", cannot be removed by such a trick. There was at first much confusion about the nature of the singularity, it was assumed by some that the solution was theoretical alone and such an object could not exist in nature. 

It was later discovered that there were indeed physical scenarios in which matter could become so compressed there was nothing to stop it from collapsing into what can mathematically be described as a single point @gravitation. This state occurs when a given spherical volume with a radius, $r$, contains a mass-energy content larger than $M >= frac(r c^2 , 2 G)$. No known repulsive forces exist which are strong enough to prevent this kind of gravitational collapse. Such objects would create a gravitational well so steep that light itself would not be able to escape, and since light travels at the fastest possible velocity, nothing else could either. It was from this complete state of darkness that these objects received their name --- black holes. See @gravitaional-potentials for a depiction of a black hole.

#figure(
  grid(
        columns: 2,
        rows:    1,
        gutter: 1em,
        [ #align(center)[#image("earth_moon.png", width: 100%)] ],
        [ #align(center)[#image("black_hole.png", width: 100%)] ],
  ),
  caption: [Two further depictions of spacetime. Again, these images are a 2D representation of 4D spacetime, and they were generated without correct physical descriptions but for illustrative purposes alone. _Left:_ Two objects, one in blue with a lesser mass and one in yellow with a greater mass. Objects with a larger mass distort spacetime to a greater extent. Objects close to either sphere will experience acceleration as the space curves and the objects continue to move in a straight line. In this scenario, if stationary, the yellow and blue objects will accelerate and move toward each other and, without outside interference, inevitably collide. However, if either the blue or yellow ball is given an initial velocity perpendicular to the direction of the other sphere so that its straight-line path orbits the other sphere, they can remain equidistant from each other in a stable orbit for potentially very long periods of time. As we will see, this orbit will eventually lose energy and decay, but depending on the masses of the two objects, this could take an extremely long time. _Right:_ A black hole. The three red lines represent the geodesic paths of three light beams as they move past the black hole at different distances. Thus far, we have assumed that the mass of the yellow and blue objects are evenly distributed through their volume, so the spacetime at the very centre of the object is, at its limit, entirely flat. In many scenarios, this is a physically possible arrangement of matter, as although gravity pulls on every particle within the object, pulling it toward the centre, it is a very weak pull compared to the other forces of nature, which push back out and stop the particles continuing on their naturally preferred trajectory. This prevents a complete collapse of the object. Gravity, however, has one advantage on its side, and that is that there is no negative mass, only positive, so whereas large bodies tend to be electrically neutral as positive and negative charges cancel each other out, gravity always grows stronger. If enough mass congregates in the same place, or if the forces pushing matter away from the centre stop, there's nothing to stop gravity from pulling every particle in that object right to the centre, right into a singular point of mass with infinite density known as the singularity. As this collapse occurs, the curvature of spacetime surrounding the object gets stronger and stronger, eventually reaching the point where within a region around the singularity, known as the event horizon, all straight-line paths point toward the singularity. Meaning that no matter your speed, no matter your acceleration, you cannot escape, even if you are light itself. Consequently, no information can ever leave the event horizon, and anything within is forever censored from the rest of the universe.]
) <gravitaional-potentials>

== Orbits are Not Forever

=== Orbits

In both Newtonian mechanics and general relativity it is possible to describe two objects that are in an excited state of constant motion, each object gravitationally bound to the other but never directly touching, similar to an electron caught in an atomic orbital. As expected, both theories are correctly describing existent phenomena. When in this state, the objects are said to be *orbiting* each other. If one object is significantly smaller than the other, then the smaller is usually referred to as the orbiter, and the larger the orbited, although in reality, they both exert equal force on each other and the centre of their orbit, known as the barycentre, will never quite align with the centre of the more massive object, even if it is negligibly close.

It is also quite easy to arrive at the notion of an orbit starting from everyday intuition. We can imagine that we live on the surface of a massive spherical object, such as a planet. Hopefully, this is not a particularly hard thing to imagine. We feel an apparent gravitational attraction toward the planet's centre, but the planet's crust prevents us from following our natural geodesic. If we drop something it will fall to the ground until it hits something stopping its motion, the ground. If we throw something, it will still fall, but it will also move a distance across the surface of the sphere since we have imparted some velocity onto the object. Now if we imagine this planet, for some reason, has an incredibly tall mountain and no atmosphere, and we go to the top of that mountain with a suitably sized cannon, we can throw objects (cannon balls in this case), much further. As we increase the amount of gunpowder we use to propel our cannonball, we impart more and more initial velocity onto the balls. We start to notice that as the velocity increases the ball takes longer and longer to reach the ground as the surface of the planet below curves away from it as it falls toward it. Eventually, if we increase the initial velocity enough, we reach a point where the curvature of the planet below exactly matches the rate at which the ball falls toward the centre of the planet. Assuming no external forces, and that the ball doesn't crash into the back of your head as it completes its first full orbit, this ball could circle the planet forever; see @orbits-diagram. Whilst in orbit the ball would be moving along its natural geodesic and would experience no net forces and hence no acceleration, it would be in freefall. This is the microgravity experienced by astronauts aboard the international space station, their distance from Earth's centre is not massively larger than on the surface of the planet and things would still quite happily fall down if left at that altitude with no velocity with respect to the planet's surface.

#figure(
  grid(
        columns: 2,
        rows:    1,
        gutter: 1em,
        [ #align(center)[#image("orbits.png", width: 100%)] ],
        [ #align(center)[#image("barycentre.png", width: 100%)] ],
  ),
  caption: [Two illustrations of scenarios involving simple orbital mechanics. _Left:_ In this thought experiment we imagine a cannon atop a large mountain on an unphysically small spherical planet with mass, $m$. As is described in both Newtonian mechanics and general relativity, objects are attracted toward the centre of mass of the planet. Left to their own devices they will fall until they meet some force resisting their motion, most likely, the surface of the planet. The cannon operator can control the velocity of the projected cannon balls. They note that the more velocity they impart, the longer it takes for the ball to impact the surface of the planet. The balls can travel further before impacting the ground when their velocity is greater, even if the time to impact remains the same. However, with this increased distance travelled along the surface of the sphere, the distance between the ball and the ground increases as the surface of the planet curves away from the ball. Eventually, the ball's trajectory will circularise around the planet, and, if not impeded by any other forces, the ball would remain on this circular trajectory indefinitely. _Right:_ Two identical massive objects, such as planets, in a circular orbit with a shared centre, called a barycentre (note that the objects do not have to have equal mass or be in a circular orbit, to have a shared barycentre, in fact, this will always be the case). Any massive objects can orbit each other, including black holes.]
) <orbits-diagram>

=== Gravitational Radiation

In Newtonian mechanics, assuming no other gravitational interactions, and no energy losses through tidal heating or other means (so not in reality), orbits are eternal and will never decay. This is not the case under general relativity, however, where orbiting bodies will release energy through gravitational radiation otherwise known as gravitational waves @gravitation. Two objects in orbit will continuously emit gravitational waves which will carry energy away from the system and gradually decay the orbit until eventually, the two objects merge. For most objects in the universe, the energy released through gravitational radiation will be almost negligible, and orbital decay from other factors will usually be vastly more significant. However, when we look again at the densest objects in the universe, black holes, and their slightly less dense cousins, neutron stars, their gravitational wells are so extreme that the energy lost through the emission of gravitational waves becomes significant enough for them to merge within timeframes less than the lifespan of the universe, outputting a colossal amount of energy in a frenzy of ripples in the moments before their collision. These huge amounts of energy propagate out through the universe at the speed of light, causing imperceptible distortions in the geometry of spacetime as they go. They pass through planets, stars, and galaxies with almost no interaction at all.

Like many things, the existence of gravitational waves was predicted by Einstein @einstein_grav_waves (although there had been earlier proposals based on different physical theories), as a consequence of the mathematics of general relativity. General relativity predicts that any non-axisymmetric acceleration of mass, linear or circular, will generate gravitational waves. This is because these motions induce changes in the system's quadrupole moment. A perfect rotating sphere will not produce any gravitational waves, no matter how fast it is spinning, because there is no change in the quadrupole moment. A sphere with an asymmetric lump however, like a mountain, will produce gravitational radiation @neutron_star_gw_review, as will two spheres connected by a bar spinning around their centre, or a massive alien starship accelerating forward using immense thrusters. However, as Einstein quite rightly calculated, for most densities and velocities, the energy released in such a manner is minuscule.

Under general relativity, gravitational waves travel at the speed of light @gravitation. They are not predicted by Newtonian mechanics, as in Newtonian mechanics the propagation of gravitational effects is instant. Special and general relativity, do not allow any information to travel faster than the speed of light, gravitational information is no different @special_relativity @gravitation. All current observations suggest gravitational waves appear to travel at, or very close to, the speed of light, however, this is still some limited debate on the matter @speed_of_gravity. As a perfectly spherical body rotates, the gravitational field remains constant in all directions. Due to the lack of a quadrupole moment, its rotation has no effect on the surrounding spacetime, thus no waves are created that can propagate, and no energy is lost from the spinning sphere.

/*
If a lone black hole, were for some reason, to move rapidly back and forth along a straight line, this, in isolation, would not produce gravitational waves. This would be an example of a dipole motion, where the center of mass of the object moves. In general relativity, changes in the dipole moment, such as the linear motion of black holes, do not produce gravitational waves because no oscillations would be produced in the surrounding spacetime. This is a consequence of the conservation of linear momentum @gravitation. If the oscillating mass was a perfectly symmetric black hole, but perhaps a planet with an asymmetric internal distribution of mass, or some enigmatic alien space station with lopsided corridors and internal voids, then a changing quadrupole moment again becomes a possibility.
*/

Aside from detections of the stochastic gravitational wave background @PTA, we have, thus far, only detected, gravitational waves from extremely dense binary systems consisting of pairs of black holes @first_detction, neutron stars @first_bns, and their combination @first_nsbh. These systems, known as Compact Binary Coalescences (CBCs), have a clear quadrupole moment that produces strong gravitational waves that propagate out through the universe, removing energy from the system which will eventually result in the merger of the companions into one body. See @waves for an illustration. Gravitational waves from many events of this type pass through the Earth regularly, at the moment, it is only the loudest of these that we can detect. The fact that we can detect them all, however, remains an impressive feat only possible due to the nature of gravitational waves themselves. The amplitude of gravitational waves scales inversely with distance from their source, rather than by the inverse square law as might naively be expected. If this were not the case, detection would be all but impossible. The energy contained within the waves still decreases with the inverse square law, so the conservation of energy is maintained @gravitation. 

#figure(
  image("waves.png", width: 70%),
  caption: [A depiction of the region of spacetime surrounding two inspiraling black holes. The spacetime grid visible is a 2D representation of the true 4D nature of our universe as described by general relativity @gravitation. This depiction was not produced by an accurate simulation but was constructed as a visual aid alone. Two massive objects can orbit each other if they have sufficient perpendicular velocity; this is a natural state for objects to find themselves trapped in because the chances of direct collisions between objects are low, and any objects that find themselves gravitationally bound together and do not experience a direct collision will eventuate in an orbit. The same is true for black holes; whether they form from pairs of massive stars that both evolve into black holes after the end of their main sequence lives or whether they form separately and through dynamical interaction, end up adjoined and inseparable, the occurrence of two black holes orbiting is not inconceivable @black_hole_binary_formation. Over time, small amounts of energy will leak from these binaries; ripples are sent out through the cosmos, carrying energy away from the system and gradually reducing the separation between the companions. As they get closer, the curvature of the spacetime they occupy increases, and thus, their acceleration toward each other grows. They speed up, and the amount of energy that is lost through gravitational radiation increases, further increasing the speed of their inspiral in an ever-accelerating dance. If they started just close enough, this process would be enough to merge them within the lifetime of the universe; they will inevitably collide with an incredible release of energy out through spacetime as powerful gravitational waves. It is these waves, these disturbances in the nature of length and time itself, that we can measure here on Earth using gravitational wave observatories.]
) <waves>

Gravitational waves have two polarization states, typically named plus, $+$, and cross, $times$. They are named as such due to the effect the different polarisations have on spacetime as they propagate through it. In both cases, the two polarisations cause distortions in the local geometry of spacetime along two axes at once, this is a result of their quadrupole nature. Gravitational waves are transverse waves, meaning they oscillate in a direction that is perpendicular to their direction of propagation. They alternate between stretching spacetime along one of the two axes of oscillation and squeezing along the other, to the inverse, as the wave oscillates. See @wobble for an illustration of the effect of the passage of a gravitational wave through a region of spacetime. It is this stretching and squeezing effect that we have been able to detect in gravitational wave detectors on Earth. It is worth noting that because they are quadrupole waves and oscillate in two directions simultaneously, the polarisation states are $45 degree$ apart rather than the $90 degree$ separation of states seen in electromagnetic waves. This means that any two points on a line that is at a $45 degree$ angle to the polarisation of an incoming wave, will not see any effect due to the passing wave.

#figure(
  image("wibble_wobble.png", width: 100%),
  caption: [The effect of two polarisation states of gravitational waves as they oscillate whilst passing through a region of spacetime. Each of the black dots represents freely falling particles unrestricted by any other forces. The plus and cross polarisations shown are arbitrary names, and the polarisation can be at any angle, but plus and cross are a convention to distinguish the two orthogonal states.]
) <wobble>

== Gravitational Wave Detection

Detecting gravity is quite easy, just let go of whatever you're holding. Detecting gravitational waves, however, requires the use of some of the most precise measurement instruments humanity has ever constructed. This subsection will cover the basics of how we detect gravitational waves and the challenges that our detection methods embedded into the data.

=== Interferometry

After the notion of detectable gravitational waves became more widespread, a few methods were put forward as possible avenues of investigation, the most notable alternative to current methods was the resonant bar antenna @gravitational_wave_detectors. In the end, interferometers have been proven as viable gravitational wave detectors @first_detction, along with, more recently, pulsar timing arrays @PTA. These two detection methods operate in very different frequency regimes and so can detect very distinct gravitational wave phenomena --- the former able to detect gravitational waves from individual CBC events, and the latter able to detect the pervasive stochastic gravitational wave background, generated by the overlapping and interfering signals of many supermassive black hole mergers.

We will focus our discussion on laser interferometry, as that is the most relevant to work in this thesis. As illustrated by @wobble, gravitational waves have a periodic effect on the distance between pairs of freely falling particles (assuming their displacement doesn't lie at $45degree$ to the polarisation of the wave). We can use this effect to create a detection method if we can measure a precise distance between two freely floating masses @LIGO_interferometers. In the absence of all other interactions (hence freely falling), the distance between two particles should remain constant. If there is a change in this distance we can deduce that this arises from a passing gravitational wave.

Masses suspended by a pendulum are effectively in a state of free fall in the direction perpendicular to the suspension fibers, this allows us to build test masses that are responsive to gravitational wave oscillations in one direction, provided they have significant isolation from other forces --- which is no small task; a considerable amount of engineering goes into ensuring these test masses are as isolated as possible from the outside world @LIGO_interferometers.

Once we have our test masses we must be able to measure the distance between them with incredible accuracy. The LIGO interferometers can measure a change in the length of their four-kilometer arms of only #box($10^(-18)$ + h(1.5pt) + "m"), a distance equivalent to $1/200$#super("th") of the diameter of a proton @LIGO_interferometers, a truly remarkable feat. In order to achieve this degree of accuracy they use laser interferometers. 

Interferometers use lasers to accurately measure a change in the length of two arms --- in the case of all current interferometers these arms are perpendicular to each other, but there are designs for future gravitational wave interferometers that use different angles but combine multiple overlapping interferometers @einstein_telescope. In the case of single interferometer designs, right-angled arms capture the largest possible amount of information about one polarisation state, so they are preferred.

What follows is a very rudimentary description of the optics of a gravitational wave detecting interferometer @interferometers. The real detectors have a complex setup with many additional optics that will not be discussed here. A single laser beam produced by a coherent laser source is split between the two arms by a beam-splitting optic. Each of the beams travels down the length of its respective arm before being reflected off of a mirror suspended by multiple pendulums --- the test masses. These beams are reflected back and forth along the arms thousands of times, before leaving the cavity and being recombined with the beam from the other detector and directed into a photodetector. The path lengths of the two beams are very slightly different, calibrated so that under normal operation the two beams will destructively interfere with each other, resulting in a very low photodetector output. This is the output expected from the interferometer if there are no gravitational waves within the sensitive amplitude range and frequency band passing through the detector. When a detectable gravitational wave passes through the interferometer, it will generate an effective difference in the arm lengths that will cause the distance between the freely falling mirrors to slightly oscillate. This oscillation will create a difference in the beam path lengths, and the two beams will no longer exactly cancel each other causing the photodetector to detect incoming laser light. If the detector is working correctly the amount of light detected will be proportional to the amplitude of the incoming gravitational wave at that moment. See @interferometer_diagram.

#figure(
  image("interferometer.png", width: 80%),
  caption: [A very simplified interferometer diagram. Real gravitational wave detection apparatus have considerably more optics than what is shown. The power recycling and signal recycling mirrors help maintain a high laser power within the cavities. Higher laser powers are preferable as they help reduce quantum shot noise, the limiting source of noise at high frequencies.]
) <interferometer_diagram>

A detector of this kind can only detect elements of incoming gravitational wave signals that align with its polarisation @antenna_pattern. An incoming signal that was completely antialigned to the detector arms would be entirely undetectable. Fortunately, this is a rare occurrence as most signals are at least partially aligned with a given detector. Interferometers are also sensitive to the angle between their tangent and the source direction, known as the orientation. Interferometers are most sensitive to signals that lie directly above or below the plane of the detector arms, and least sensitive to signals whose propagation direction is parallel to the plane of the arms. These two factors combine to generate the antenna pattern of the detector, which dictates which regions of the sky the detector is most sensitive to.

=== The Global Detector Network

There are currently five operational gravitational wave detectors worldwide: LIGO Livingston (L1), LIGO Hanford (H1), Virgo (V1), Kagra (K1), and GEO600 (G1) @open_data. See @global_network. Several further future detectors are planned, including LIGO India @LIGO_India, and three future next-generation detectors: the Einstein Telescope @einstein_telescope, Cosmic Explorer @cosmic_explorer, and LISA @LISA, a space-based detector constellation. 

Having multiple geographically separated detectors has multiple advantages. 

- *Verification:* Separation provides verification that detected signals are from gravitational wave sources, and are not local experimental glitches or terrestrial phenomena that appear to be signals @network_snr. Since there are no other known phenomena that can cause a similar effect in such spatially separated detectors, if we see a similar signal in multiple detectors we can say that either they were caused by gravitational wave signals or a statistical fluke. The chances for the latter to occur decrease with the number of detectors.

- *Sky Localisation:* Gravitational-Wave detectors cannot be targeted in a particular area of the sky. Other than their antenna pattern, which is fixed and moves with the Earth, they can sense detections from many directions @network_snr. This means we don't have to worry about choosing where to point our detectors, but it also means that we have very little information about the source location of an incoming signal, other than a probabilistic analysis using the antenna pattern. Because gravitational waves travel at the speed of light, they won't usually arrive in multiple detectors simultaneously. We can use the arrival time difference between detectors to localize the gravitational wave sources with a much higher constraint than using the antenna pattern alone. With two detectors we can localize to a ring in the sky, and with three we can localize further to two degenerate regions of the sky, narrowing it down to one with four detectors. Adding this triangulation method with the antenna patterns of each of the detectors in the network can provide good localization if all detectors are functioning as expected.

#figure(
  image("network.png", width: 100%),
  caption: [Location of currently operation LIGO detectors: LIGO Livingston (L1), LIGO Hanford (H1), Virgo (V1), Kagra (K1), and GEO600 (G1) @open_data. Arm angles are accurate, the arm lengths were generated with a relative scale with the real detectors: 4 km for the two LIGO detectors, 3 km for Virgo and Kagra, and 600 m for GEO600. ]
) <global_network>

=== Interferometer Noise <interferometer_noise_sec>

Perhaps the area of experimental gravitational wave science that is most relevant to gravitational wave data analysis is interferometer noise. Data scientists must examine the interferometer photodetector outputs, and determine whether a gravitational wave signal is present in the data (signal detection), then make statements about the properties of any detected signals, and how they relate to the gravitational wave source and its relation to us (parameter estimation). 

Because it is not possible to reduce noise in all areas of frequency space at once, gravitational wave interferometers are designed to be sensitive in a particular region of frequency space @ligo_o3_noise --- this region of frequency space is chosen to reveal a particular type of gravitational wave feature that is of interest to us. It makes sense then, that the current generation of detectors were designed with a sensitivity range overlapping the area in which it was correctly predicted that CBCs would lie. The design specification of LIGO Livingston can be seen in @noise_diagram, it shows the main sources of noise in the detector, which we will cover very briefly.

#figure(
  image("noise_budget.PNG", width: 80%),
  caption: [Full noise budget of the LIGO Hanford Observatory (LHO) during the 3#super("rd") joint observing run. This image was sourced from @ligo_o3_noise. ]
) <noise_diagram>

+ *Quantum* noise is the limiting noise at high frequencies @ligo_o3_noise. This is primarily in the form of quantum shot noise and quantum radiation pressure noise. It is caused by stochastic quantum fluctuations in the vacuum electric field. Quantum shot noise is the noise generated by the uncertainty in photon arrival time at the photodetector, this can be mitigated by using higher laser power. Quantum radiation pressure noise is caused by variations of the pressure on the optics caused by quantum fluctuation; this kind of noise increases with laser power, but it has a much smaller overall contribution to the noise budget so reducing shot noise is usually preferable.

+ *Thermal* noise is caused by the motion of the particles that comprise the test mass, coating, and suspensions. This is the noise caused by the random motion of particles present in all materials not cooled to absolute zero (all materials). Thermal noise dominates at lower frequencies. Reductions in thermal noise are primarily achieved through the design of the optics and suspension systems.

+ *Seismic* noise is noise generated from ground motion. One of the purposes of the test mass suspensions is to try and reduce this noise. It performs this job admirably. Seismic noise is not a dominant noise source at any frequency range.

These types of noise, plus several other accounted-for sources and small amounts of unaccounted-for noise sum to make a coloured Gaussian background. Some elements of the noise can vary depending on the time of day, year, and status of the equipment because they are sensitive to changes in the weather, local geography, and human activity. This means the noise is also non-stationary and can fluctuate quite dramatically even on an hourly basis @det_char. There are also a number of known and unknown sources of transient non-linear glitches that can cause features to appear in the noise. These are some of the most difficult noise sources to deal with and are discussed in more detail in  @glitch-sec. The non-stationary nature of the noise, in addition to the presence of non-linear glitches, makes for an intriguing backdrop in which to perform data analysis. Most of these problems already have working solutions, but there are certainly potential areas for improvement.

Gravitational wave interferometers are not perfect detectors. Their sensitivity is limited by noise present in their output. Despite best efforts, it is not, and will never be, possible to eliminate all sources of noise. When such precise measurements are taken, the number of factors that can induce noise in the output is considerable. It is remarkable that the detectors are as sensitive as they are. Nonetheless, a challenge remains to uncover the most effective techniques for extracting information from signals obfuscated by detector data. The deeper and more effectively we can peer through the noise the more information will be available to us to advance our knowledge and understanding of the universe.

This thesis focuses on a very small part of that problem. We attempt to apply the latest big thing in data science, machine learning, to both detection and parameter estimation problems in the hopes that we can make a small contribution to the ongoing effort.