# Overview

Solve 1D BVP. Steady viscous Burgers equation.

$$
\begin{aligned}
-u^{\prime \prime} + \gamma u u^{\prime} = f \\
f &= 0 \\
\end{aligned}
$$

$\gamma=0.01$ results in a strong viscous case while $\gamma=100$ results in a strong nonlinear convection case.

Employ 3 time schemes to solve it: Gauss-Seidel, Multigrid using GS, Newton with a linear solver.

Test both upwind difference and central difference.

It is interesting to see that Gauss-Seidel with central difference works to give correct converged result while MG-GS and Newton-GMRES fails. Based on the textbook, the central difference is unstable for the burgers equation with a strong nonlinear convection term. Of course, all of the schemes work for the strong viscous case.

In the linear convection equation, explicit first order time scheme with second order central difference in space is unconditionally unstable while implicit first order time scheme with second order central difference in space is unconditionally stable.[^note] Of course, that is a different equation. But here, a central difference could be stable and gives converged correct result. That raises a question. Why?

[^note]: Check P297 in the book: Charles Hirsch, "Numerical Computation of Internal and External Flows, Volume 1",Elsevier LTD, Oxford , 2007.
