Impedance consists of resistance and [[Inductive Reactance]] and is the total opposition to sinusoidal current. Its unit is the [[Ohm]] ($\Omega$).

Because the reactance causes a phase-shift between the voltage of the inductor and the current, reactance is expressed as a [[complex number]], while resistance is expressed as a real number.

$$
Z = jX_L+R
$$
This is often expressed in polar format
$$
Z = Z\angle\theta
$$
![[Impedance vector.png]]
Where:

Symbol | Meaning | Unit
--|--|--
$Z$ | Impedance | $\Omega$ |
$j$ | The imaginary number | $\sqrt{-1}$ |
$X_L$ | Inductive Reactance | $\Omega$ |
$R$ | Resistance | $\Omega$ |

Because $X_L$ is a product of angular momentum ($2\pi f \cdot L$), the phase difference will increase as frequency increases.
![[Phase and frequency.png]]

### [[Ohm's Law]]

When applying Ohm's law to a circuit that uses impedance, use impedance in place of resistance:
$$
V = IZ
$$
This seems simple enough. However, because $Z$ is a [[complex number]], the multiplication and division required to solve the formula will also involve complex numbers.
$$
V = I\left(jX_{L}+R \right)
$$
Therefore it is often easier to convert to polar format for the operation:
$$
V = I\angle\phi \times Z\angle\theta = IZ\angle(\phi+\theta)
$$
> In the above example, $I$ would have an angle of $0^{\circ}$ , but I used $\phi$ to show that the angles are added when multiplying vectors, while the magnitudes are multiplied.

### [[Kirchoff's laws#Kirchoff's Voltage Law|Kirchoff's Voltage Law]]

Kirchoff's voltage law states that the sum of voltage drops must be equivalent to the voltage sources. However, because the voltage drop of an inductor is out of phase with the source voltage, it can appear as though the law is being violated when the numbers don't match up.

However, by representing the voltage as a [[complex number]], it actually works out so that $V_S$ is the magnitude of the real and complex voltages.
$$
V_{S}= V_{R}+jV_{L}
$$

