# Overview

Solve 1D BVP. Steady viscous Burgers equation.

Employ 3 time schemes to solve it: Gauss-Seidel, Multigrid using GS, Newton with a linear solver.

Test both upwind difference and central difference.

It is interesting to see that Gauss-Seidel with central difference works to give correct converged result while MG-GS and Newton-GMRES fails. Based on the textbook, the central difference is unstable for the burgers equation with a strong nonlinear convection term. Of course, all of the schemes work for the strong viscous case.

