#set page(
  paper: "a4",
  margin: 2.5cm,
)

#set text(
  font: "Times New Roman",
  size: 11pt,
)

#set heading(numbering: "1.")

#let code(content) = block(
  width: 100%,
  fill: rgb("#f8f9fa"),
  stroke: (left: 2.5pt + rgb("#4a7fc1"), rest: 0.5pt + rgb("#d8dde6")),
  inset: (left: 14pt, right: 14pt, top: 10pt, bottom: 10pt),
  radius: 2pt,
  [
    #set par(leading: 0.8em)
    #text(
      fill: rgb("#1c1e26"),
      font: "JetBrains Mono",
      size: 9.5pt,
      weight: "regular",
    )[#content]
  ]
)

#let frontpage(
  title,
  course: "",
  authors: (),
  date: datetime.today().display("[year]-[month]-[day]")
) = [
  #align(center)[
    #v(3cm)
    #text(size: 20pt, weight: "bold")[#title]

    #v(1.5cm)
    #text(size: 12pt)[#course]

    #v(2cm)
    #for author in authors [
      #text(size: 12pt)[#author] \
    ]

    #v(2cm)
    #text(size: 11pt)[#date]
  ]

  #pagebreak()
]

#frontpage(
  "Obligatory Assignment 1",
  course: "AI505 - Optimization",
  authors: (
    "Christian Steentoft",
    "Daniel Egedal Nissen",
  ),
)

= Introduction


= Problem Description

The trajectory consists of a sequence of points:
$
x_1, x_2, ..., x_n in RR^2
$

Where:
- $x_1$ is the fixed start point
- $x_n$ is the fixed goal point
- Intermediate points are optimization variables

The optimization objective is:
$
f(x) = f_L(x) + lambda f_S(x) + mu f_O(x)
$

This combines:
- Path length minimization
- Smoothness regularization
- Obstacle avoidance penalties

= Project Tasks

== Task 1: Problem Setup

- Define start and goal positions
- Generate straight-line initial path
- Place at least two circular obstacles
- Visualize initial configuration

== Task 2: Objective Function Implementation

=== Length

#code(```
def f_L(x):
    x = x.reshape((-1, 2))
    differences = x[1:] - x[:-1]
    return an.sum(differences ** 2)

def gradient_f_L(x):
    return grad(f_L)(x)
```)


- Implement objective function in Python
- Compute objective value
- Compute gradient analytically or automatically

== Task 3: Optimization Algorithms

Selected methods:
- Gradient Descent
- BFGS
- Momentum

For each method:
- Define stopping criteria
- Track convergence behavior
- Visualize path evolution

== Task 4: Comparison and Analysis

Compare methods based on:
- Runtime
- Number of iterations
- Final objective value
- Robustness to initialization

= Experimental Plan

Experiments will vary:
- Number of path points: 20, 50, 100
- Smoothness weight $lambda$
- Obstacle penalty weight $mu$
- Obstacle placements

= Expected Outcomes

The project aims to determine:
- Which optimization method converges fastest
- Which method is most robust
- How obstacle penalties influence convergence
- How scaling affects performance

= Conclusion

