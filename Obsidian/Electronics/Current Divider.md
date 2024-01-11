For quick reference, the current divider formula is:
$$
I_{X}= I_{T}\frac{R_{T}}{R_{X}}
$$

### Deriving the formula
According to Kirchoff's current law, the amount of current flowing through a circuit is a conserved quantity:
$$
I_{T}= I_{1} + I_{2} + \dots + I_{n}
$$
This relationship is very similar to the conductance of a parallel circuit:
$$
G_{T}= G_{1} + G_{2} + \dots + G_{n}
$$

So the amount of current flowing through a given path of conductance compared to the total current will be the same ratio as the path's conductance compared to the total conductance.
$$\begin{align}
\frac{I_{X}}{I_{T}}&=\frac{G_{X}}{G_{T}} \\ 
&\text{or}\\
I_{X}&= I_{T}\times\frac{G_{X}}{G_{T}}
\end{align}
$$

Keep in mind that conductance is defined as the reciprocal of resistance:
$$
\frac{1}{R_{T}} = \frac{1}{R_{1}} + \frac{1}{R_{2}} + 
\dots + \frac{1}{R_{n}}
$$
so the previous equation can be written as:
$$
\begin{align}

\frac{I_{X}}{I_{T}}&=\frac{\frac{1}{R_{X}}}{\frac{1}{R_{T}}} \\ &\text{or} \\
\frac{I_{X}}{I_{T}}&=\frac{1}{R_{X}} \times \frac{R_{T}}{1} \\
&\text{or} \\
I_{X} &= I_{T} \times \frac{R_{T}}{R_{X}}

\end{align}
$$
