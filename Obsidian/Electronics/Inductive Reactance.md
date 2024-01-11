Inductive Reactance describes the voltage induced by an inductor when the current changes. However, for the sake of simplicity during circuit analysis, it is often measured in [[Ohm|Ohms]].

Reactance is described by [[Lenz's law]] using the following formula:

$$
\varepsilon = -L\frac{di}{dt}
$$

Where:

Symbol | Meaning | Unit
--|--|--
$\varepsilon$ | Back EMF | V or $\Omega$ |
$L$ | Inductance | H |
$\frac{di}{dt}$ | Change in current | A/s |


For [[Sine and Cosine||sinusoidal]] waveforms we can take advantage of the fact that a single period of the wave is equivalent to $2\pi$ . Therefore, we can substitute ${di}/{dt}$ with $2\pi f$ as follows:
$$
\begin{align*}
\varepsilon &= -L 2\pi f\\
&\text{or}\\
\varepsilon &= -L \omega
\end{align*}
$$

When a magnetic field is induced, it stores energy that will be released when current stops and the magnetic flux collapses. The amount of energy stored is proportional to the inductance and the square of current.
$$
E = \frac{1}{2}LI^{2}
$$


