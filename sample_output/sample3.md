**Setup**: Suppose we have a function, $f: \mathbb{R}^n \rightarrow \mathbb{R}$,
and we want to test code that computes $\nabla f$. (Note that these techniques
also apply when $f$ has multivariate output.)

## Finite-difference approximation

The main way that people test gradient computation is by comparing it against a
finite-difference (FD) approximation to the gradient:

$$
\boldsymbol{d}^\top\! \nabla f(\boldsymbol{x}) \approx \frac{1}{2 \varepsilon}(f(\boldsymbol{x} + \varepsilon \cdot \boldsymbol{d}) - f(\boldsymbol{x} - \varepsilon \cdot \boldsymbol{d}))
$$

where $\boldsymbol{d} \in \mathbb{R}^n$ is an arbitrary "direction" in parameter
space. We will look at many directions when we test. Generally, people take the
$n$ elementary vectors as the directions, but random directions are just as good
(and you can catch bugs in all dimensions with less than $n$ of them).

**Always use the two-sided difference formula**. There is a version which
doesn't add *and* subtract, just does one or the other. Do not use it ever.

**Make sure you test multiple inputs** (values of $\boldsymbol{x}$) or any thing
else the function depends on (e.g., the minibatch).

**What directions to use**: When debugging, I tend to use elementary directions
because they tell me something about which dimensions that are wrong... this
doesn't always help though. The random directions are best when you want the
test cases to run really quickly. In that case, you can switch to check a few
random directions using a
[spherical](https://github.com/timvieira/arsenal/blob/master/arsenal/math/util.py)
distribution—do *not* sample them from a multivariate uniform!

**Always test your implementation of $f$!** It's very easy to *correctly*
compute the gradient of the *wrong* function. The FD approximation is a
"self-consistency" test, it does not validate $f$ only the relationship
between $f$ and $\nabla\! f$.

Obviously, how you test $f$ depends strongly on what it's supposed to compute.

- Example: For a conditional random field (CRF), you can also test that your
  implementation of a dynamic program for computing $\log Z_\theta(x)$ is
  correctly by comparing against brute-force enumeration of $\mathcal{Y}(x)$ on
  small examples.

Similarly, you can directly test the gradient code if you know a different way
to compute it.

- Example: In a CRF, we know that the $\nabla \log Z_\theta(x)$ is a feature
  expectation, which you can also test against a brute-force enumeration on
  small examples.

### Why not just use the FD approximation as your gradient?

For low-dimensional functions, you can straight-up use the finite-difference
approximation instead of rolling code to compute the gradient. (Take $n$
axis-aligned unit vectors for $\boldsymbol{d}$.) The FD approximation is very
accurate. Of course, specialized code is probably a little more accurate, but
that's not *really* why we bother to do it! The reason why we write specialized
gradient code is *not* to improve numerical accuracy, it's to improve
*efficiency*. As I've
[ranted](http://timvieira.github.io/blog/post/2016/09/25/evaluating-fx-is-as-fast-as-fx/)
before, automatic differentiation techniques guarantee that evaluating $\nabla
f(x)$ gradient should be as efficient as computing $f(x)$ (with the caveat that
*space* complexity may increase substantially - i.e., space-time tradeoffs
exists). FD is $\mathcal{O}(n \cdot \textrm{runtime } f(x))$, where as autodiff
is $\mathcal{O}(\textrm{runtime } f(x))$.

## How to compare vectors

**Absolute difference is the devil.** You should never compare vectors in
absolute difference (this is Lecture 1 of any numerical methods course). In this
case, the problem is that gradients depend strongly on the scale of $f$. If $f$
takes tiny values then it's easy for differences to be lower than a tiny
threshold.

Most people use **relative error** $= \frac{|\textbf{want} -
\textbf{got}|}{|\textbf{want}|}$, to get a scale-free error measure, but
unfortunately relative error chokes when $\textbf{want}$ is zero.

I compute several error measures with a script that you can import from my
github
[arsenal.math.checkgrad.{fdcheck}](https://github.com/timvieira/arsenal/blob/master/arsenal/math/checkgrad.py).

I use two metrics to test gradients:

1. Relative error (skipping zeros): If relative error hits a zero, I skip
  it. I'll rely on the other measure.
2. Pearson correlation: Checks the *direction* of the gradient, but allows a
  scale and shift transformation. This measure doesn't have trouble with zeros,
  but allows scale and shift problems to pass by. *Make sure you fix those
  errors!* (e.g. In the CRF example, you might have forgotten to divide by
  $Z(x)$, which not really a constant... I've made this exact mistake a few
  times.)

I also look at some diagnostics, which help me debug stuff:

- Accuracy of predicting the sign {+,-,0} of each dimension (or dot random product).
- Absolute error (just as a diagnostic)
- Scatter plot: When debugging, I like to scatter plot the elements of FD vs. my
  implementation.

All these measurements (and the scatter plot) can be computed with
[arsenal.math.compare.{compare}](https://github.com/timvieira/arsenal/blob/master/arsenal/math/compare.py),
which I find super useful when debugging absolutely anything numerical.

## Bonus tests

**Testing modules**: You can test the different modules of your code as well
(assuming you have a composable module-based setup). E.g., I test my DP
algorithm independent of how the features and downstream loss are computed. You
can also test feature and downstream loss modules independent of one
another. Note that autodiff (implicitly) computes Jacobian-vector products
because modules are multivariate in general. We can reduce to the scalar case by
taking a dot product of the outputs with a (fixed) random vector.

Something like this:

```
r = spherical(m)  # fixed random vector |output|=|m|
h = lambda x: module.fprop(x).dot(r)   # scalar function for use in fd

module.fprop(x)  # propagate
module.outputs.adjoint = r. # set output adjoint to r, usually we set adjoint of scalar output=1
module.bprop()
ad = module.input.adjoint # grab the gradient
fd = fdgrad(h, x)
compare(fd, ad)
```

**Integration tests**: Test that running a gradient-based optimization algorithm
is successful with your gradient implementation. Use smaller versions of your
problem if possible. A related test for machine learning applications is to make
sure that your model and learning procedure can (over)fit small datasets.

**Test that batch = minibatch** (if applicable). It's very easy to get this bit
wrong. Broadcasting rules (in numpy, for example) make it easy to hide matrix
conformability mishaps. So make sure you get the same results as manual
minibatching (Of course, you should only do minibatching if are get a speed-up
from vectorization or parallelism. You should probably test that it's actually
faster.)

**Further reading**:

- I've written about gradient approximations before, you might like these
  articles:
  [Gradient-vector products](http://timvieira.github.io/blog/post/2014/02/10/gradient-vector-product/),
  [Complex-step method](http://timvieira.github.io/blog/post/2014/08/07/complex-step-derivative/).
- The foundations of backprop:
  [Backprop is not just the chain rule](http://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/).
- [I strongly recommend](http://timvieira.github.io/blog/post/2016/09/25/evaluating-fx-is-as-fast-as-fx/)
  learning how automatic differentiation works, I learned it from
  [Justin Domke's course notes](https://people.cs.umass.edu/~domke/courses/sml2011/08autodiff_nnets.pdf).
- [Justin Domke post](https://justindomke.wordpress.com/2017/04/22/you-deserve-better-than-two-sided-finite-differences/):
  Explains why we need bespoke finite-difference stencils (i.e., more than
  two-sided differences) to prevent numerical demons from destroying our results!
