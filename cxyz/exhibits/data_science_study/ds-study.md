## Article schema

{
parents: [],
children: [],
title: "",
aka: [],
body: "",
relatedLiterature: [],
relatedTools: [],
relatedTopics: []
}

// TODO place this topic

### Bagging vs. boosting

See bagging vs. boosting screenshot from reddit.

## Relationships

commutative
associative
distributive

## Types of measurement

The types of measurement define which operations are legitimate...

- nominal
- ordinal
- interval
- ratio

Other characteristics...

- discrete
- continuous

More...

- static
- dynamic

## Correlational study

Correlational research is a type of nonexperimental research in which the researcher measures two variables and assesses
the statistical relationship (i.e., the correlation) between them with little or no effort to control extraneous
variables.

There are essentially two reasons that researchers interested in statistical relationships between variables would
choose to conduct a correlational study rather than an experiment.

The first is that they do not believe that the statistical relationship is a causal one. For example, a researcher might
evaluate the validity of a brief extraversion test by administering it to a large group of participants along with a
longer extraversion test that has already been shown to be valid. This researcher might then check to see whether
participants’ scores on the brief test are strongly correlated with their scores on the longer one. Neither test score
is thought to cause the other, so there is no independent variable to manipulate. In fact, the terms independent
variable and dependent variable do not apply to this kind of research.

The other reason that researchers would choose to use a correlational study rather than an experiment is that the
statistical relationship of interest is thought to be causal, but the researcher cannot manipulate the independent
variable because it is impossible, impractical, or unethical. For example, Allen Kanner and his colleagues thought that
the number of “daily hassles” (e.g., rude salespeople, heavy traffic) that people experience affects the number of
physical and psychological symptoms they have (Kanner, Coyne, Schaefer, & Lazarus, 1981)[1]. But because they could not
manipulate the number of daily hassles their participants experienced, they had to settle for measuring the number of
daily hassles—along with the number of symptoms—using self-report questionnaires. Although the strong positive
relationship they found between these two variables is consistent with their idea that hassles cause symptoms, it is
also consistent with the idea that symptoms cause hassles or that some third variable (e.g., neuroticism) causes both.

https://opentextbc.ca/researchmethods/chapter/correlational-research/

## Ablating

Ablating, i.e. removing, part of a model and observing the impact this has on performance is a common method for
verifying that the part in question is useful. If performance doesn’t go down, then the part is useless and should be
removed. Carrying this method over to datasets, it should become common practice to perform dataset ablations, as well.

https://bheinzerling.github.io/post/clever-hans/

## Baselines

Baselines as used to orient us as we develop a system. We always aim to outperform the baseline, though we may choose
different baselines for different contexts:

- Business use-case: Choose a simple/inexpensive baseline
- Research: Choose a SOTA implementation as a baseline to outperform.

Baselines help us to:

Estimate cost/benefit: How much performance can we get with little effort (i.e. using the baseline implementation)? How
much have we invested into beating the baseline, and how much are we beating it by?

This helps to understand the complexity of the problem we are dealing with, the effectiveness of our approaches.

Classification baselines

...

Regression baselines

...

https://www.quora.com/What-does-baseline-mean-in-machine-learning

## Linear algebra

### Ax = b

Ax = b, A^-1b = x (the inverse of A and b solve x)

There are two general categories of numerical methods for solving Ax = b: Direct Methods, Iteration Methods.

#### Direct methods

These are methods with a finite number of steps; and they end with the exact solution x, provided that all arithmetic
operations are exact.

#### Iteration methods

These are used in solving all types of linear systems, but they are most commonly used with large sparse systems.

### Tensor

- Cornerstone data structure, like vectors and matrices
- A generalization of vectors and matrices: a multidimensional array
    - A vector is a 1D or first-order tensor
    - A matrix is a 2D or second-order tensor
- Tensors differ from vectors and matrices in that *they represent linear relationships between vectors (or other
  tensors) that is relative to some basis*
- Most operations performed on scalars, vectors, and matrices can be reformulated to perform on tensors
- Tensors and tensor algebra is widely used in physics and engineering
- Some operations in machine-learning can be described in terms of tensors
- Tensor operations include: tensor product, tensor dot product, tensor contraction.

### Overdetermined (overconstrained) Systems

- A system of equations with *more equations* than unknowns
- An overdetermined will *only* have solutions when some equation occurs several times in the system, or if some
  equations are linear combinations of others (a linearly dependent system)

### Hadamard product

- Element-wise matrix multiplication of two matrices with same size
- Not the typical operation when referring to matrix multiplication
- Often notated with a circle

### Matrix-matrix multiplication

- More complicated than the Hadamard prodcut
- Not all matrices can be multiplied: The number of columns in the first matrix (A) must equal the number of rows (m) in
  the second matrix (B)
- If A is of shape m x n and B is of shape n x p, then C is of shape m x p.

### What is a dot-product?

- A sum of the multiplied elements of two vectors
- The name of the dot product comes from the symbol used to denote it
- Often used to compute the weighted sum of a vector
- The intuition for the matrix multiplication is that we are calculating the dot product between each row in matrix A
  with each column in matrix B

### What is a vector norm?

- The "length" or "magnitude" of a vector; a nonnegative number that describes the extent of the vector in space
- The "length" of a vector is always a positive number, except for a vector of all zeros
- It is calculated using some measure that summarizes the distance of the vector from the origin of the vector space
- There are many different ways to calculate the length or magnitude of a vector (e.g. L1, L2, max)
- Calculating norms are part of regularization, as well as broader vector and matrix operation

#### L1 norm

- The sum of the absolute vector values
- Also known as the taxicab or Manhattan distance
- Helpful for discriminating between elements that are exactly zero and elements that are small but nonzero
- A function that grows at the same rate in all locations, but retains mathematical simplicity
- Used as a regularization method

#### L2 norm

- The square-root of the sum of squared vector values
- Also known as the Euclidean norm
- The most commonly used norm in machine-learning, by far
- Used as a regularization method

#### Max norm

- The maximum of vector values
- Notated with L^inf

### Types of matrices

#### Sparse matrices

- A matrix comprised of mostly zero values
- The sparsity of a matrix can be quanitified with a "density" score: *(count of zero elements)/total elements*
- Sparse matrices are common to many machine-learning applications
    - Data-types: Occurrence or counts
    - Encoding-schemes: One-hot encoding, TF-IDF encoding, etc.
    - Fields: NLP, recommender systems, computer-vision
- Sparsity can be exploited (i.e. encoding them with an data-structure that is different than with dense matrices),
  acheiving enormous computational savings because many large matrix problems that occur in practice are sparse
    - Dictionary of keys
    - Lists of lists
    - Coordinate list
    - Compressed sparse row (probably most common to machine-learning)
    - Compressed sparse column

#### Square matrices

- Same number of rows as columns
- Readily added and multiplied together, are the basis of many simple linear transformations (e.g. rotations)
- Graphs can be encoded as square matrices

#### Symmetric matrices

- The most important matrices the world will ever see in the theory of linear algebra and also in its applications
- Symmetric about the main diagonal axis
- Always sqauare and equal to its own transpose
- The top-right triangle is the same as the bottom-left triangle
- Undirected graphs can be represented with symmetric matrices

#### Triangular matrices

- A type of square matrix that has all values above OR below the main diagonal filled with zeros

#### Diagonal matrices

- Values outside of the main diagonal have zero values, the main diagonal has non-zero entries
- Diagonal matrices are often denoted as *D*
- Sometimes represented as a vector of values on the main diagonal

#### Identity matrices

- An identiy matrix is a square matrix that does not change a vector when multiplied
- All of the scalar values along the main diagonal (top-left to bottom-right) have the value one, while all other values
  are zero
- Is a component of important matrix operations, such as inversion
- Often denoted as *I*
- Sometimes referred to as the unit matrix

#### Orthogonal matrix

- A square matrix whose columns and rows are orthonormal unit vectors (i.e. they are perpendicular and have length or
  magnitude of 1)
- Often denoted as *Q*
- A matrix is orthogonal if:
    - Its transpose is equal to its inverse
    - The dot product of a matrix and its tranpose equals the identity matrix *I*
- Orthogonal matrices are useful tools as they are computationally cheap and stable to calculate their inverse as simply
  their transpose

#### Orthogonality

- Two vectors are orthogonal when their dot product equals zero
- The vectors are perpendicular

#### Orthonormal

1. When two vectors or orthogonal (i.e. their dot product equals zero) and...
2. Each vector has length 1 (i.e. unit vectors)

#### Hessian Matrix

...

#### Jacobian Matrix

...

### Matrix operations

Some operations can be used directly to solve key equations. Others are components of more complex operations, and
provide building blocks and shorthand notation.

#### Matrix transpose

- "Flips" the dimensions of a matrix about the main diagonal
- Denoted by the superscript *T*
- Has no effect when the matrix is symmetrical
- The most elementary interpretation I can think of is with the dot product. Consider two vectors v,w and their dot
  product, (v,w). If we have a square matrix M, then we can look at (v,Mw), transform w by M and then do the dot
  product. The transpose then satisfies (MTv,w)=(v,Mw), that is, if instead of transforming w by M, we can transform v
  by MT and we'll get the same dot product. The transpose is the unique matrix that does this for any v,w. If you have a
  symmetric matrix, then this means that you can transform v or w by the same matrix and get (Mv,w)=(v,Mw)

#### Matrix inverse

- Finds another matrix that when multiplied with the original matrix results in an identity matrix
- Used in solving systems of linear equations: where we are interested in finding vectors of unknowns
- Not typically computed directly, instead a series of operations are used involving forms of matrix decomposition
- "Primarily a theoretical tool, and should not actually be used in practice for most software applications"
- Denoted by a *-1* superscript
- Techniques:
    - Moore-Penrose method
    - Gauss-Jordan method
    - LU decomposition
    - CoFactor method

#### Pseudoinverse (aka the Moore-Penrose inverse)

- A generalization of the matrix inverse for rectangular matrices
- Calculated using the SVD
- Provides a way of solving linear regression equation when there are more rows than there are columns, which is often
  the case

#### Singular matrix

- A square matrix that is not invertible

#### Matrix decomposition

...

#### Matrix Trace

- Gives the sum of all the diagonal entries of a square matrix
- The trace of matrix *A* is denoted as *tr(A)*
- Used as a component of other operations

#### Matrix determinant

- The determinant of a square matrix is a *single* number: the product of all the eigenvalues of the matrix
- Can be thought of as a scalar summarizing the matrix
- This scalar determinant tells whether the matrix is invertible: *det(A) = 0* means *A* cannot be inverted
- Describes the relative geometry of the vectors that make up the rows of a matrix
- This scalar represents the "volume" of the matrix: the volume of a box with sides given by rows of A
- The absolute value of the determinant can be imagined as how much the matrix expands or contracts space.
- Intuition: the determinant describes the way a matrix will scale another matrix when they are multiplied together: a
  determinant of 1 preserves the space of the other matrix
- The determinant of matrix *A* is denoted as *det(A)* or *|A|*
- Used as a component of other operations

#### Matrix rank

- The estimate of the number of linearly independent rows or colums in a matrix
- The rank is not the number of dimensions of the matrix, but the number of linearly independent directions
- Intuition: consider the rank the number of dimensions spanned by all the vectors within a matrix: rank of 0 means
  vectors spand a point, a rank of 1 suggests all vectors span a line, a rank of 2 suggests all vectors span a 2D
  plane...
- Estimated numerically typically using matrix decomposition, a common approach is to use the SVD for short
- Often denoted a the function *rank(A)*

## Matrix decomposition / matrix factorization

- A matrix decomposition is a way of reducing a matrix into its constituent parts.
- Decomposition does not compress matrices, it instead breaks it into parts that make certain downstream operations
  easier to perform
- It is an approach that can simplify more complex matrix operations that can be performed on the decomposed matrix
  rather than on the original matrix itself.
- Because complex matrix operations cannot be solved efficiently or with numerical stability using the limited precision
  of computers
- Reducing matrices into constituent parts (decomposition/factorizatioin) makes it easier to calculate complex matrix
  operations
- A foundation of linear algebra in computers

### LU decomposition

- For square matrices
- Decomposes matrices into triangle matrix components: L (lower) and U (upper)
- A = L.dot(U)
- Not always possible to compute: may be impossible, or too difficult
- Often used to simplify solving systems of linear equations:
    - Finding the coefficients in a linear regression
    - Calculating the determinant and inverse of a matrix

### LUP decomposition

- A numerically more stable version of LU decomposition that involves partial pivoting
- A = P.dot(L).dot(U)
- The P matrix specifies a way to permute the result or return the result to the original order

### QR decomposition

- For square *or* rectangular matrices
- Decomposes a matrix into Q and R components: given A is *m x n*, *Q* will be *m x m* and *R* will be an upper-triangle
  matrix of size *m x n*
- A = Q.dot(R)

### Cholesky decomposition

- For square matrices with all values greater than zero (i.e. "positive definite matrices")
- A = L.dot(transpose(L))
- L is a lower triangle matrix
- Twice as efficient as LU dcomposition for symmetric matrices (remember symmetric matrices are very important!)
- The Cholesky decomposition is used for:
    - Solving linear least squares for linear regression - Simulation and optimization methods

### Eigendecomposition

- For square matrices
- One of the most widely used decomposition techniques
- Decomposes matrix into eigenvectors and eigenvalues
- Provides valuable insights into the properties of the matrix
- "Eigen" means *own* or *innate*
- Is reversable (you can get original matrix back from eigenvectors and eigenvalues)
- Not all square matrices can be decomposed into eigenvectors and eigenvalues, some can only be decomposed using complex
  numbers
- Calculated using iterative algorithm
- Used for:
    - Computing the power of a matrix
    - Principal Component Analysis (dimensionality reduction)

#### Eigenvectors

- Unit vectors (they have length/magnitude of 1)
- Column vectors (aka "right vectors)
- *v*

#### Eigenvalues

- Coefficients applied to eigenvectors that give the vectors their length or magnitude
- *lambda*

### Singular Value Decomposition (SVD)

- Probably most common decomposition method
- Allows us to discover some of the same kind of information as eigendecomposition
- More generally applicable than eigendecomposition
- *Every* matrix has an SVD: though the result may require complex number or be problematic with respect to floating
  point precision
- "The singular value decomposition (SVD) has numerous applications in statistics, machine learning, and computer
  science. Applying the SVD to a matrix is like looking inside it with X-ray vision..."
- Calculated via iterative numerical methods
- *A=U·Sigma·V^T*
    - The diagonal values of Sigma are the *singular values* of A
    - The columns of U are the *left-singular values*
    - The columns of V are the *right-singular values*
- Used for:
    - Computing inverse & pseudoinverse
    - Compressing
    - Denoising
    - Data reduction: compute SVD and select top *k* largest singular values in Sigma (the LSA or LSI in NLP)

---

## Linear regression

- A method for modeling relationship between scalar values: one or more independent variables and a dependent variable
- The model assumes that the output is a linear function (or weighted sum) of the input variable(s)
- A staple of statistics, also considered a good introductory machine-learning method
- Can be reformulated to use matrix notation and matrix operations
- No guarantee of convergence, there are several methods with some more stable than others...
- Methods:
    - Solve via Inverse
    - Solve via QR decomposition
    - Solve via SVD or pseudoinverse (The de facto standard approach. Not all matrices have an inverse or QR
      decomposition. SVD is a more stable approach.)
    - Solve via convenience function

### Multivariate regression

- Linear regression when multiple inputs (independent variables) are involved

### Components

- *x*: least-squares solution, the weights
- Residuals: Sum of residuals, the difference between the observed value of the dependent variable and the predicted
  value, the squared 2-norm for each column
- Rank: rank of input matrix (how many linearly independent rows)
- *s*: singular values of input matrix

### What are residuals? How to evaluate?

- They should be random...

---

## Statistics

### What is a statistic?

- A quantity derived from a sample
- A numerical summary of a data-set that reduces the data to one value
- A form of lossy compression

### Descriptive statistics

- Intended to be easily interpretable
- May describe a particular dimension, such as central-tendency (?) or variability

### Exploratory data anaylsis

...

### Inferential statistics / confimatory statistics

...

### Frequentist vs Bayesianist

TODO

TODO: Bayesian example from Super Forecasting

...

### Mean vs. median vs. mode

- The most well-known and frequently used summaries of a dataset (i.e. statistics):
    - Mode: the highest frequency value in the dataset
    - Median: the value at the midpoint of the frequency distribution
    - Mean: the sum of the values divided by the dataset's size

- Each arise from *minimizing a particular discrepancy* between the list of numbers and the summary statistic itself:
    - Mode: minimizes the number of the times that one of the numbers in our sumarized list does not equal the summary
      that we use (discrepancy = x|xi-1|^0 = "zero-one loss", a constant measure of discrepancy)
    - Median: minimizes the average distance between each number and our summary (discrepancy = x|xi-1|^1, increases
      linearly as s gets further and further from xi)
    - Mean: minimizes the average squared distance between each number and the summary (discrepancy = x|xi-1|^2, similar
      to the median, but discprepancy increases super-linearly as s gets further from xi)

![](http://www.johnmyleswhite.com/notebook/wp-content/uploads/2013/03/discrepancy1.png)

http://www.johnmyleswhite.com/notebook/2013/03/22/modes-medians-and-means-an-unifying-perspective/

### Pythagorean means

#### What characteristics may motivate using each of the different Pythagorean means: arithmetic mean vs. geometric mean vs. harmonic mean?

###### Maximizing vs. minimizing

For any given sample, the pythagorean means compare to eachother through this equality:

```
H < G < A
```

Thus, use the appropriate one to maximize or minimize your results if needed.

##### Considering outliers

- Use the arithmetic mean when you have a sample that varies in the same interval (no outliers)
- Use the harmonic mean when your sample contains fractions and/or extreme values (either too big or too small). It is
  more stable regarding outliers.

For example: the arithmetic mean of (1,2,3,4,5,100) is 19.2 whereas the harmonic one is 2.58

##### When fractions are involved

- Use the geometric mean or harmonic mean when your sample contains fractions
- Use the harmonic mean when your sample contains fractions and/or extreme values (either too big or too small). It is
  more stable regarding outliers. It is particularly sensitive to lower than average values.

##### When 0 is involved:

- The geometric mean with be 0: This is why we generally require all values to be positive to use the geometric mean.
- The harmonic mean can't even be applied at all because 1/0 is undefined.

##### When to use each:

A practical answer is that it depends on what your numbers are measuring. Do some unit analysis and consider the
relationship between consecutive numbers in the series you’re averaging.

If you’re measuring units that add up linearly in a sequence (such as lengths, distances, weights), then an arithmetic
mean will give you a meaningful average.

If you’re measuring units that add up as reciprocals in a sequence (such as speed or distance / time over a constant
distance, capacitance in series, resistance in parallel), then a harmonic mean will give you a meaningful average. For
example, the harmonic mean of capacitors in series represents the capacitance that a single capacitor would have if only
one capacitor was used instead of the set of capacitors in series.

If you’re measuring units that multiply in a sequence (such as growth rates or percentages), then a geometric mean will
give you a meaningful average. For example, the geometric mean of a sequence of different annual interest rates over 10
years represents an interest rate that, if applied constantly for ten years, would produce the same amount growth in
principal as the sequence of different annual interest rates over ten years did.

https://www.quora.com/When-is-it-most-appropriate-to-take-the-arithmetic-mean-vs-geometric-mean-vs-harmonic-mean

#### Can you give an example of a commonly used harmonic mean?

F-measure or F1 score

### Variance and standard-deviation

variance: denoted as lower-case *sigma^2*
std: denoted as *s*

#### What is the difference between variance and standard-deviation?

The standard deviation is just the square root of the variance, with the variance being the average of the squared
differences from the arithmetic mean. The reason we square the differences is so that larger departures from the mean
are punished more severely. The other effect is that it results in treating departures in both direction (positive
errors and negative errors) equally. The standard deviation is usually preferred since it is expressed in the same unit
as the data itself, making interpretation easier.

(From Quora answer)

### Covariance

- The measure of joint probability for two random variables
- Describes how variables change together
- Intepreting covariance:
    - The sign of the covariance can be interpreted as whether the variables increase together (positive) or decrease
      together (negative)
    - The magnitude of covariance is not easly interpreted, although a covariance of zero indicates the variables are
      completely independent
- Denoted: *cov(X,Y)*

#### Covariance matrix

- A generalization of the covariance of two variables, it captures the way in which all variables in the dataset may
  change together
- A square and symmetric matrix that describes the covariance between two or more random variables
- Useful for seaparating the structured relationships in a matrix of random variables
- Widely used in multivariate-statistics
- A key element used in Principal Component Analysis (PCA)
- The diagonal are the *variances* of each of the random variables (it is sometimes called the variance-covariance
  matrix for this reason)
- Be aware of measurement scales between variables: covariance matrix is most easily interpreted when variables are
  commensurable
- Denoted as upper-case *Sigma*

### Correlation

#### Pearson correlation coefficient

- The covariance can be normalized to a score between -1 and 1 to make the magnitude interpretable by dividing it by the
  standard deviation of X and Y
- The result is called the Pearson correlation coefficient

#### The correlation matrix

- A standardized version of the covariance matrix
- Accounts for differences in measurement scales between variables

### Hypothesis testing: Statistical significance

- A dominant approach to data analysis in many fields of science
- A key technique of inferential statistics: both frequentist inference and Bayesian inference
- Also known as confirmatory data analysis: you are confirming/denying a pre-specified hypothesis

#### Process:

1. Two statistical data sets are compared (one could be synthetic data produced by idealized model)
2. A hypothesis regarding a statistical relationship between the two dataset is proposed
3. The proposed hypothesis is compared to the null hypothesis (i.e. that no relationship exists)
4. Consider the stastical assumptions
5. Decide which test statistic is appropriate
6. Select a significance level
7. Apply the test-statistic
8. Decide to reject the null hypothesis or not

or...

1. Apply the test statistic to the observations
2. Select a signifcance level
2. Calculate the p-value
3. Reject the null hypothesis, in favor of the alternative hypothesis if and only if the p-value < the significance
   level

#### p-values

...

- May be 1-tailed or 2-tailed
  ...

#### test-statistic

- A quantity derived from a sample used in stastical hypothesis testing
- A numerical summary of a data-set that reduces the data to one value that can be used to perform a hypothesis test
- In general, a test-statistic is selected or defined in such a way as to quantify, within observed data, behaviors that
  would distinguish the null from the alternative hypothesis
- A test-statistic's sampling distribution under the null hypothesis must be calculable, either exactly or
  approximately, so that p-values can be calculated
- Commonly used test statistics are the t-statistic and the F-test
  ...

#### How to decide on a test statistic?

...

## Probability theory

### Probability distribution

A probability distribution is a mathematical function that has a sample space as its input, and gives a probability as
its output.

input: set of possible outcomes -> output: probability of each outcome

They are generally divided into discrete (probability mass functions) and continuous (probability density function).

https://en.wikipedia.org/wiki/Probability_distribution

Related:

- Bournoulli trials
- Poisson process
- Conjugate priors distribution with Bayesian inference

### Joint vs. Conditional Probability

The essential difference between the two is that in for the joint case, you are looking at all different combinations
of (H, C) and assign them probabilities, you are in the universe of all possible scenarios, i.e. looking at all people
crossing the street. The case {hit, red} is one of these. In the conditional case on the other hand, you’re only
interested in what happens under the assumption that the light is red, i.e. you are now in a universe in which the light
is always red and there remain only two cases: you get hit or you don’t. In other words, you are only observing the
people crossing the street while the light is red.

https://www.quora.com/What-is-the-difference-between-conditional-probability-and-joint-probability

### Distributions

#### Bernoulli

- A binary sequence: representing "success or failure"

#### Poisson

- Where the sample is composed of counts where each count is the number of times an event occurs in an interval.

#### t-distribution

- Avoid the "infinite ladder of inference"
- "Suppose you observe some data and want to construct an interval you think 95% of future observations will fall in.
  You could estimate the true distribution and construct the shortest interval that includes 95% of the probability
  mass. But what if you were wrong about the distribution? You should adjust your interval to account for the estimated
  degree of error in your estimate of the distribution. But what if you’re wrong about the degree of error in your
  estimate of the distribution? You can see this process, called the “ladder of inference” goes on forever, increasing
  the size of the interval at each step."
- "The t-distribution showed that for the very special case of independent draws from identical Normal distributions,
  you can compute the limit of all of these uncertainties and come up with a precise confidence interval."
- "While the special case is not particularly useful, the general idea that you can construct rigorous finite confidence
  intervals even though there are infinite levels of uncertainty is important to frequentists. Bayesians avoid the
  problem by assuming you have a prior distribution that encapsulates all levels of uncertainty."

https://www.quora.com/What-is-the-t-distribution

### Gaussian processes

- "A Gaussian process is a flexible, non-linear, prior over functions. It enables us to express our belief about a class
  of functions before observing the data and *importantly* tractably compute our updated *posterior* belief about the
  functions that are consistent with our prior assumptions and the data we observed. It does this tractably, and gives
  us a full posterior distribution over functions."

https://www.quora.com/What-makes-Gaussian-processes-so-powerful

### Autoregressive models

- A type of random process
- Applied to describe time-varying processes: nature, economics, etc.
- Specifies that the output variable depends linearly on its own previous values *and* on a stochastic term (a form of a
  stochastic difference equation)
- Related: ARMA and ARIMA models

## Principle Component Analysis

TODO move to linear algebra

- Important machine-learning method for dimensionality reduction
- Uses simple matrix operations from linear algebra and statistics to calculate a projection of the original data into
  the same number *or fewer* dimensions
- The covariance method (one of a few ways to calculate):
    - Average: Calculate the means of each column
    - Center: Subtract each column mean from the corresponding column's values
    - Compare: Calculate the covariance matrix of the "centered" matrix
    - Eigendecomposition: calculate the eigen decomposition of the covariance matrix
    - Select: Ranks eigenvectors by eigenvalues, select the top k
    - Project: *P = B^T.dot(A)*, A is the original data, B is the transpose of the selected principal components
- Interpreting:
    - Eigenvectors represent directions, eigenvalues represent magnitudes for those directions
    - The values may be more generally referred to as *"singular values"* and the vectors may be more generally reffered
      to as *"principal components" (this language may be also used when using Singular-Value-Decomposition)
    - If all eigenvalues have a similar value, then we know that the existing representation may already be reasonably
      compressed/dense: the PCA projection may offer little benefit
    - Eigenvalues close to zero represent components that may be discarded

## Information theory

### Bit

- Measure of information
- Measure of surprise

### Entropy

- Entropy is a measure of uncertainty about a random variable.
- **Entropy is maximum when all outcomes are equally likely. This may be counter-intuitive: a uniform-distribution is *
  most random***. Maximum entropy (i.e. a uniform distribution, is often the assumed prior in classification scenarios).
- Anytime you move away from equally likely outcomes or introduce predictability, the entropy must go down.
- A decrease in entropy means you can ask fewer questions to guess the outcome.
- The range of Entropy: is 0 ≤ Entropy ≤ log(n), where n is number of outcomes.
- The concept has important implications for our ability to compress information, and encode predictive logic: is lossy
  compression acceptable? how many parameters are necessary for your model to

Related:

- Surprise
- Signal-processing
- Distortion functions
- [Ashby's law of requisite variety](https://en.wikipedia.org/wiki/Variety_(cybernetics)#Law_of_Requisite_Variety)
- Gestalt Laws of Grouping
- Teslers Law of Conservation
- No Silver Bullet: accidental vs. essential complexity

### Minimum description length

The minimum description length (MDL) principle is a formalization of Occam's razor in which the best hypothesis (a model
and its parameters) for a given set of data is the one that leads to the best compression of the data.

@ChrisNF and treecut for compressing hierarchical representations.

### Mutual-information

- The amount of information (that is, reduction in uncertainty) that knowing the value of one variable provides about
  the other.

## Are neural-networks deterministic?

\ # idk if this answer is legit

not-necessarily.

they are not in training (random initialization, example ordering (thread races)). that is, the state of the model after
training may be different between different trainings.

they are likely deterministic "in production". that is, they return the same output with the same input – as in they
perform the role of deterministic automata.

there do exist stochastic neural networks, which introduce random variation into the network. this random variation can
help the model escape from local minima.

Do stochastic NNs relate to Markov chains?

---

## Sampling

Sampling is an active process of gathering observations with the intent of estimating a population variable.

Even if we have the capacity to produce estimates using all available data, that data still represents a sample of
observations from an idealized population.

https://machinelearningmastery.com/statistical-sampling-and-resampling/

### Random sampling

...

### Sample-efficiency

An algorithm is sample efficient if it can get the most out of every sample. Imagine learning trying to learn how to
play PONG for the first time. As a human, it would take you within seconds to learn how to play the game based on very
few samples. This makes you very "sample efficient"

https://ai.stackexchange.com/questions/5246/what-is-sample-efficiency-and-how-can-importance-sampling-be-used-to-achieve-it

The term in commonly used with regard to reinforcement-learning.

Related:

- One shot learning

### Bootstrapping

Samples are drawn from the dataset with replacement (allowing the same sample to appear more than once in the sample),
where those instances not drawn into the data sample may be used for the test set.

https://machinelearningmastery.com/statistical-sampling-and-resampling/

### Resampling

Once we have a data sample, it can be used to estimate the population parameter.

The problem is that we only have a single estimate of the population parameter, with little idea of the variability or
uncertainty in the estimate.

One way to address this is by estimating the population parameter multiple times from our data sample. This is called
resampling.

Statistical resampling methods are procedures that describe how to economically use available data to estimate a
population parameter. The result can be both a more accurate estimate of the parameter (such as taking the mean of the
estimates) and a quantification of the uncertainty of the estimate (such as adding a confidence interval).

### Oversampling

TODO

...

### Boosting

TODO

...

https://en.wikipedia.org/wiki/Boosting_(machine_learning)

### Stratified sampling

...

---

## Classification

### Class imbalance

Dealing with an imbalanced dataset is a common challenge when solving a classification task.

The SMOTE algorithm, as a popular choice for augmenting the dataset without biasing predictions. SMOTE uses a k-Nearest
Neighbors classifier to create synthetic datapoints as a multi-dimensional interpolation of closely related groups of
true data points.

https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03

---

## Many weak learners becoming strong learners

### Boosting

...

### Gradient-boosting machines

...

### Random-forest

...

###   

...

---

## Classification

## Using labels

### Self-supervised learning

AKA self-learning or unsupervised learning

TODO

...

### Weak supervision

TODO

AKA: semi-supervised (?), distant-supervised (?), surrogate-supervised (?)

...

### Semi-supervised

...

### Distant supervision

...

### Surrogate learning

...

### Unsupervised learning

...

### Stratified sampling, a.k.a. "Stratification"

A technique used to analyze/divide a universe of data into homogeneous groups.

Allows for sampling each subpopulation independently.

Can be used to deal with class imbalances.

https://en.wikipedia.org/wiki/Stratified_sampling

## Metrics

### Precision/Recall

TODO

...

#### Micro vs. macro

TODO

...

---

## Model selection techniques:

- Cp
- Akaike information criterion

---

### type I and type II errors

Type I error: false-positive finding (**ignored by recall and yet the type of error often introduced by recall oriented
methods**).

Type II error: false-negative finding (**ignored by precision and yet the type of error often introduced by precision
oriented methods**)

---

awesome fundamental statistics overview

- especially good for set theory and distributions

http://students.brown.edu/seeing-theory/

---

null-hypothesis testing

http://rpsychologist.com/d3/NHST/

--- 

set-identities/transitive/associative?

https://cs.brown.edu/courses/cs022/static/files/documents/sets.pdf
...

---

Independent and identically distributed (iid)

...

---

t-disribution

- Good for small sample sizes
- Used when the population variance is unknown
- Approaches the normal distribution as the sample size increases
- Has heavier tails (than normal dist. with small samples) because of the uncertainty surrounding estimating the
  variance from the sample

---

t-statistic

Number of standard deviations a parameter is away from a constant.

---

chi-squared test

Tests if categorical data shows up at a rate different than random.

For feature selection: The intuition being that if a features is independent to the target it is uninformative for
classifying observations.

---

f-test, f-statistic

To test if groups of features are jointly statistically significant.

---

order-statistics

- Very simple.
- The kth order-statistic of a statistic sample is equal to its kth smallest value. Together with rank statistics, order
  statistics are among the most fundamental tools in non-parametric statistics and inference.
- Range is a function of order statistics

---

nonparametric statistics

Do not assume a *functional* form of the relationship between the features and target. Can fit a wider range of types of
relationships, but often requires a large number of observations required to fit a model of any quality.

- Statistics not based on parameterized families of probability distributions.
- Includes both descriptive and inferential statistics

---

parametric (modeling)

Reduces the problem of estimating the relationship to the problem of estimating a few parameters (approximation).

1. Make an assumption about the functional form relationship betwen x and y.

- Fit the model by estimating the parameters.

---

## Modeling

### What type of models can be interpreted as linear models?

TODO

### Maximum likelihood estimation (MLE)

- A method of estimating parameters of a statistical model given observations, by finding the parameter values that
  maximize the likelihood of making the given observations given the parameters.
- Basically maximizes "agreement" between a selected statistical model and the observed sample data.
- MLE Provides a unified approach to estimation, which is well-defined in the case of the normal disribution and many
  other problems
- R. A. Fisher pioneered this method in the 1920s. The opening image on likelihood is from his 1922 paper “On the
  Mathematical Foundations of Theoretical Statistics” in the Philosophical Transactions of the Royal Society of London
  where he first proposed this method.

http://www.dataanalysisclassroom.com/lesson63/

## Evaluating Models

What is an AUC ROC curve? What is it good for?

- A plot of the TPR (i.e. recall) vs FPR over threshold variation
- Producing the ROC AUC score provides a statistic for evaluating the model's effectiveness

---

### Log-likelihood

Simply the logarithm of probability.

Benefits from a stats perspective...

Maximizing log-likelihood or log-probability makes computing maximum likelihood estimations easier. This is because the
log function is "well-behaved" (i.e. monotonically increasing) and turns products into sums (avoiding the product rule).

It also works well for gradient-descent since log(p(x)) has good "global optimization properties", meaning it makes it
easier to identify a good step size and get determine the global optimum in fewer steps.

https://stats.stackexchange.com/questions/174481/why-to-optimize-max-log-probability-instead-of-probability

Benefits from a computer-science perspective...

**It is more efficient.** It is faster to compute because it provides a means of using addition instead of
multiplication for combining probabilities through log properties. Addition is less CPU intensive than multiplication (
the conversion to log form is expensive, but it is typcilly incurred only once).

**It improves accuracy** through better numerical-stability, preventing "underflow", whereby products get so small they
are rounded to zero (due to computers' limitations in representing real-numbers).

Log probability is also reversable.

---

maximum a posteriori probability function (MAP)

- Closely related the method of maximum-likelihood-estimation, but employes an augmented optimization objective which
  incorporates a prior distribution function (for this reason it can be viewed as regularization of MLE).

---

central limit theorem

- Regardless of the distribution you are sampling from, as your sample size becomes larger (approaches infinity), and
  the number of samples taken becomes larger (approaches infinity), the distribution of statistics computed from those
  samples will tend toward a normal distribution (e.g. computing the distribution of the sample-mean from a large number
  of samples, taken from any distribution, will appear as a normal distribution).
- The CLT applies to many statistics for a distribution: sample-mean, sample-sum, etc.
- The more independent random variables you add, their properly normalized sum tends toward a normal distribution (even
  if the original variables themselves are not normally distributed).
- It sugggests that statistical methods that work for normal distributions can be applicable to many problems involving
  other types of distributions

---

ANOVA analysis of variance

- A special case of regression

---

## Feature Selection

Strategies:

1. Remove highly correlated variables
2. Run OLS and select significant features
3. Forward selection and backward selection, or recursive
4. Random Forest feature importance
5. Lasso (regularization)

Types:

- Variance thresholding
- ...

---

Goodness-of-fit

The goodness of fit of a statistical model describes how well it fits a set of observations.

Measures of goodness of fit typically summarize the discrepancy between observed values and the values expected under
the model in question.

---

R-squared & Adjusted R-Squared

R-squared measures the proportion of the variation in your dependent variable (Y) explained by your independent
variables (X) for a linear regression model. Adjusted R-squared adjusts the statistic based on the number of independent
variables in the model.

The reason this is important is because you can "game" R-squared by adding more and more independent variables,
irrespective of how well they are correlated to your dependent variable. Obviously, this isn't a desirable property of a
goodness-of-fit statistic. Conversely, adjusted R-squared provides an adjustment to the R-squared statistic such that an
independent variable that has a correlation to Y increases adjusted R-squared and any variable without a strong
correlation will make adjusted R-squared decrease. That is the desired property of a goodness-of-fit statistic.

Intuition behind adjusted R-Squared: Once all the correct features have been added, additional features should be
penalized.

https://www.quora.com/What-is-the-difference-between-R-squared-and-Adjusted-R-squared

---

Brier Score

The Brier score is a proper score function that measures the accuracy of probabilistic predictions. It is applicable to
tasks in which predictions must assign probabilities to a set of mutually exclusive discrete outcomes.

Shows the squared mean difference between the predicted probability of all observations with their actual outcome. The
lower score the better. Ranges between 0 and 1.

https://en.wikipedia.org/wiki/Brier_score

---

Akaike information criterion

For determining which model is better.

The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a given set of
data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other
models. Thus, AIC provides a means for model selection.

https://en.wikipedia.org/wiki/Akaike_information_criterion

---

discriminator

- A function applied to produce classifications over a probability distribution

---

mutual information (related to Kullback-Leibler divergence)

TODO combine with above

- A common feature selection method
- MI measures how much information the presence / absence of a term contributes to making the correct classification
  decision

https://en.wikipedia.org/wiki/Mutual_information#Relation_to_Kullback%E2%80%93Leibler_divergence

Kullback-Leibler divergence

- It's well suited to application. Expected difference between logs, so low risk of overflow etc. It has an easy
  derivative, and there are lots of ways to estimate it with Monte Carlo methods.
- Theoretically, minimising the KL is equivalent to doing maximum likelihood in most circumstances.
- Connects to well tested things we know work well
- It's a versatile tool for comparing two arbitrary distributions on a principled, information-theoretic basis.

https://www.reddit.com/r/MachineLearning/comments/ct2o9h/d_why_is_kl_divergence_so_popular/?utm_medium=android_app&utm_source=share

---

gradient (of a function)

The gradient of a function of a vector containing all of that function's partial derivatives at some point.

---

partial derivative

Shows how an output changes when the function has two or more variables, and only one of the variables changes.

---

## Optimization algorithms

gradient-descent

- A relatively simple optimization algorithm
- Has many virtues, but is not very fast, can settle on local-optima, will not do well with "pathological curvature"

Rule of thumb: Gradient-descent will take longer when features are not similarly scaled.

---

stochastic-gradient-descent

Motivation: computing the derivative in regular (batch) gradient descent becomes very expensive because you compute
over **all** examples in your training set.

Sequence:

- Randomly shuffle training examples.
- Repeatedly adjust the parameters, once for each example.

"What SGD is doing is actually scanning through the training examples [...] and looking only at this first example, it
takes a little gradient-descent step with respect the cost of this [current] example." Andrew Ng.
https://www.youtube.com/watch?v=UfNU3Vhv5CA

Takes a more random looking path towards minimum. It wanders around continuously in a region (hopefully of the global
minimum) – it doesn't necessary settle on **the** minima.

---

momentum

Uses previous gradients to influence the movement of the optimizer.

Often used with stochastic gradient descent.

- An alternative to gradient descent:
    - Gradient-descent: a man walking down a hill following the steepest path downwards – his progress is slow and
      steady.
    - Momentum: a heavy ball rolling down the same hill – the inertia acts both as a smoother and an accelerator,
      dampening oscillations and causing us to barrel roll through narrow valleys, small humps and local minima.

- Momentum tweaks gradient-descent algorithm to include short-term memory

- possible to acheive quadratic speedups

http://distill.pub/2017/momentum/

---

fisher information

- "A way of measuring the amount of information that an observable random variable *X* carries about an unknown
  parameter *Theta* upon which the probability of *X* depends."
- A measure of how sensitive *X* is to *Theta* (?)
- Widely used in optimal experimental design

- Used in machine-learning techniques sugh as elastic weight consolidation, which reduces catastrophic forgetting in
  artifical neural networks

---

## Developing models

talk about test vs. train vs. development sets?

1. Training set, as the name suggests, this data set is used to train your classifier. It takes major chunk of your
   original data set.
2. Development (or "validation") set, used during evaluation of your classifier with different configurations or,
   variations in the feature representation. Its called development set, since you are using it while developing your
   classifier. It can be a bit biased, that's why we need third kind of data set. **For tuning hyperparameters**.
3. Test set, Data set on which you finally check the accuracy of your classifier and get the unbiased results.

Example: Your instructor wants you to develop a classifier using Training set which should perform well on Development
set. And, most likely, your instructor will finally check the results of your classifier on the Test set, which he
didn't share with you.

https://www.quora.com/What-is-the-definition-of-development-set-in-machine-learning

---

# Kernels

## What is a kernel?

A kernel is a similarity function. It is a function that you, as the domain expert, provide to a machine learning
algorithm. It takes two inputs and spits out how similar they are.

Suppose your task is to learn to classify images. You have (image, label) pairs as training data. Consider the typical
machine learning pipeline: you take your images, you compute features, you string the features for each image into a
vector, and you feed these "feature vectors" and labels into a learning algorithm.

Data --> Features --> Learning algorithm

Kernels offer an alternative. Instead of defining a slew of features, you define a single kernel function to compute
similarity between images. You provide this kernel, together with the images and labels to the learning algorithm, and
out comes a classifier.

Of course, the standard SVM/ logistic regression/ perceptron formulation doesn't work with kernels : it works with
feature vectors. How on earth do we use kernels then? Two beautiful mathematical facts come to our rescue:

1. Under some conditions, every kernel function can be expressed as a dot product in a (possibly infinite dimensional)
   feature space ( Mercer's theorem ).
2. Many machine learning algorithms can be expressed entirely in terms of dot products.

These two facts mean that I can take my favorite machine learning algorithm, express it in terms of dot products, and
then since my kernel is also a dot product in some space, replace the dot product by my favorite kernel. Voila!

## Why use a kernel?

Why kernels, as opposed to feature vectors? One big reason is that in many cases, computing the kernel is easy, but
computing the feature vector corresponding to the kernel is really really hard. The feature vector for even simple
kernels can blow up in size, and for kernels like the RBF kernel ( k(x,y) = exp( -||x-y||^2), see Radial basis function
kernel) the corresponding feature vector is infinite dimensional. Yet, computing the kernel is almost trivial.

**Many machine learning algorithms can be written to only use dot products, and then we can replace the dot products
with kernels. By doing so, we don't have to use the feature vector at all. This means that we can work with highly
complex, efficient-to-compute, and yet high performing kernels without ever having to write down the huge and
potentially infinite dimensional feature vector**. Thus if not for the ability to use kernel functions directly, we
would be stuck with relatively low dimensional, low-performance feature vectors. This "trick" is called the kernel
trick ( Kernel trick).**

## Other thoughts on kernels

I want to clear up two confusions which seem prevalant on this page:

1. A function that transforms one feature vector into a higher dimensional feature vector is not a kernel function. Thus
   f(x) = [x, x^2] is not a kernel. It is simply a new feature vector. You do not need kernels to do this. You need
   kernels if you want to do this, or more complicated feature transformations without blowing up dimensionality.
2. A kernel is not restricted to SVMs. Any learning algorithm that only works with dot products can be written down
   using kernels. The idea of SVMs is beautiful, the kernel trick is beautiful, and convex optimization is beautiful,
   and they stand quite independent.

https://www.quora.com/What-are-kernels-in-machine-learning-and-SVM-and-why-do-we-need-them

## kernel trick vs. hashing trick

TODO

## hashing trick

Essentially, you decide upon a (large) fixed-length vector (e.g. 2^28), use a hash-function *f* that maps arbitrary
input and outputs values in that range (e.g. [0, 2^28]).

Advantages include:

- Hashes are fast.
- You don't have to use a fixed predetermined vocabulary.

Disadvantages:

- Collisions? (what are the implications?)

https://medium.com/value-stream-design/introducing-one-of-the-best-hacks-in-machine-learning-the-hashing-trick-bf6a9c8af18f

---

## The log sum exp trick

TODO merge with above

- The logarithm of the sum of the exponentials of the arguments
- A smooth approximation to the maximum function
- Mainy used by machine-learning algorithms, when usual arithmetic computations are performed in the log-domain or
  log-scale
- For numerical stability / increase accuracy : prevent overflows and underflows
- Example domains: filtering problems, multinomial distribution prameterized with softmax (e.g. logistic regression)

https://en.wikipedia.org/wiki/LogSumExp
https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/

---

## Linear Models

- Strongly related to Empirical Risk Minimization

> In general, learning a linear classifier consistes of finding a weight vector *w* and a bias *b* minimizing *
> EmpricalRisk(w, b)* for a well chosen loss function *f*.

https://thomas-tanay.github.io/post--L2-regularization/

## Picking a loss function

- Needs to be compatible with learning algorithm (e.g. must be differentiable if you are using gradient descent)
- The norm of the weight vector can be interpreted as a scaling parameter for the loss function wrt to empirical risk
  minimization.
- Binary classification has three notable loss functions
    - 0-1 indicator
    - Hinge loss
    - Softplus loss

https://thomas-tanay.github.io/post--L2-regularization/

# Optimization

## What is a Minimax problem?

About minimizing the possible loss for a worst case (maximum loss) scenario.

In zero-sum games, the minimax solution is the same as the Nash equilibrium.

TODO merge with above

## Regression

### Linear regression

...

### Polynomial regression

...

### Ridge Regression, and Collinearity

A standard linear or polynomial regression will fail in the case where there is high collinearity among the feature
variables. Collinearity is the existence of near-linear relationships among the independent variables. The presence of
hight collinearity can be determined in a few different ways:

- A regression coefficient is not significant even though, theoretically, that variable should be highly correlated with
  Y.
- When you add or delete an X feature variable, the regression coefficients change dramatically.
- Your X feature variables have high pairwise correlations (check the correlation matrix).

Ridge Regression is a remedial measure taken to alleviate collinearity amongst regression predictor variables in a
model. Collinearity is a phenomenon in which one feature variable in a multiple regression model can be linearly
predicted from the others with a substantial degree of accuracy.

https://towardsdatascience.com/5-types-of-regression-and-their-properties-c5e1fa12d55e

### Lasso regression

Lasso Regression is quite similar to Ridge Regression in that both techniques have the same premise. We are again adding
a biasing term to the regression optimization function in order to reduce the effect of collinearity and thus the model
variance. However, instead of using a squared bias like ridge regression, lasso instead using an absolute value bias.

### ElasticNet regression

ElasticNet is a hybrid of Lasso and Ridge Regression techniques. It is uses both the L1 and L2 regularization taking on
the effects of both techniques.

---

\ ## What is crossfold validation? How does it work?

TODO

--- 

LDA (Linear Discriminant Analysis)

For dimensionality reduction. Finds aces that maximize separability between classes and projects data onto them.

---

## what is the difference between linear/nonlinear vs. deterministic/nondeterministic?

NN training/using would seem to exhibit the Markov property.

Nonlinear means that there may be a ***correlation*** with the input, but it is a nonlinear one. A function is say to be
linear in some argument (the input) when the ratio result/argument is constant. In your case, it is the ratio
output/input.

Non determinism is of a different nature. The input/output relation is said to be non-deterministic when ***one of
several result may occur, without any a priori known cause***. This is usually modelelled mathematically, either by
using a relation rather than a function, or by considering a function from the input domain to the domain of subsets of
the output domain.

***For example***: if you consider inputs and outputs in the domain of integer, for each integer input, you have a set
of possible outputs that are all integers.

When this set always contains only a single element, the function is deterministic, and the set can be replaced by this
unique element.

http://cs.stackexchange.com/questions/41412/differences-between-linear-nonlinear-vs-deterministic-nondeterministic-neural-n?newreg=284184b723044f0daabc4e56d51a41ae

neural-networks,

---

## What is feature weighting? What are some feature weighting schemes in IR/NLP?

TODO merge in with feature section above

Feature weighting is a technique used to approximate the optimal degree of influence of individual features using a
training set. When successfully applied relevant features are attributed a high weight value, whereas irrelevant
features are given a weight value close to zero.

In search (IR/NLP) feature-weight schemes may be applied to different feature (or "language" models) such as
bag-of-words or n-grams, which have all kinds of issues (morphological variation, misspellings, variable use of
punctuation (e.g. '-' instead of ' ') choice of stop words, etc.). Learned feature weight schemes help supress noise and
deal with these issues. Some common feature weight schemes are binary, term-frequency,
term-frequence-inverse-document-frequency.

---

## Bootstrap Aggregating, or Bagging

TODO merge with above

https://en.wikipedia.org/wiki/Bootstrap_aggregating

---

Boosting

An ensemble learning strategy that trains a series of weak models, each on attempting to correctly predict the
observations the previous model got wrong.

https://en.wikipedia.org/wiki/Boosting_(machine_learning)

---

Bootstrapping

Simulates obtaining many new datasets by repeated sampling with replacement from the original dataset.

---

One vs Rest

An extension of logistic regression to handle multiple classes.

A separate (binary classification) model is trained for each class.

## Natural Language Processing

---

### Language models

#### bag of words

We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This
specific strategy (tokenization, counting and normalization) is called the Bag of Words or “Bag of n-grams”
representation. Documents are described by word occurrences while *completely ignoring the relative position
information* of the words in the document.

http://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation

#### continuous bag of words

You can think of the task as "predicting the word given its context".

The input to the model could be 𝑤𝑖−2,𝑤𝑖−1,𝑤𝑖+1,𝑤𝑖+2, the preceding and following words of the current word we
are at. The output of the neural network will be 𝑤𝑖. Hence you can think of the task as "predicting the word given its
context"
Note that the number of words we use depends on your setting for the window size.

Several times faster to train than the skip-gram, slightly better accuracy for the frequent words
This can get even a bit more complicated if you consider that there are two different ways how to train the models: the
normalized hierarchical softmax, and the un-normalized negative sampling. Both work quite differently.

https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures

#### skip gram words

The task here is "predicting the context given a word".

The input to the model is 𝑤𝑖, and the output could be 𝑤𝑖−1,𝑤𝑖−2,𝑤𝑖+1,𝑤𝑖+2. So the task here is "predicting the
context given a word". In addition, more distant words are given less weight by randomly sampling them. When you define
the window size parameter, you only configure the maximum window size. The actual window size is randomly chosen between
1 and max size for each training sample, resulting in words with the maximum distance being observed with a probability
of 1/c while words directly next to the given word are always(!) observed.

Works well with small amount of the training data, represents well even rare words or phrases.

https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures

---

## distributional hypothesis

The assumption that many semantically similar or related words appear in similar contexts. The basis of many popular
methods for word representation. The hypothesis supports unsupervised learning of meaningful word representations from
large corpora.

---

## stemming and lemmatization

A form of normalization in NPL.

The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms
of a word to a common base form.

Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal
correctly most of the time, and often includes the removal of derivational affixes.

Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words,
normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known
as the lemma

http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

nlp, normalization

------

Stemming

Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal
correctly most of the time, and often includes the removal of derivational affixes.

The most common algorithm for stemming English, and one that has repeatedly been shown to be empirically very effective,
is Porter's algorithm (Porter, 1980)

nlp, normalization

-------

Lemmatisation

Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words,
normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known
as the lemma

nlp, normalization

--------

chunking

Chunking is also called shallow parsing and it's basically the identification of parts of speech and short phrases (like
noun phrases). Part of speech tagging tells you whether words are nouns, verbs, adjectives, etc, but it doesn't give you
any clue about the structure of the sentence or phrases in the sentence:

> President Barack Obama criticized insurance companies and banks as he urged supporters to pressure Congress to back
> his moves to revamp the health-care system and overhaul financial regulations. (source)

Like tokenization, which omits whitespace, chunking usually selects a subset of the tokens. Also like tokenization, the
pieces produced by a chunker do not overlap in the source text.

https://www.safaribooksonline.com/library/view/natural-language-processing/9780596803346/ch07s02.html
http://stackoverflow.com/questions/1598940/in-natural-language-processing-what-is-the-purpose-of-chunking

nlp, normalization

--------

ngrams vs. shingling

N-grams are a more general concept: an n-gram is a continguous sequence of *n items* from a given sequence (of text or
speech). Shingles are a specific type of n-gram composed of contiguous *words*.

--- 

what are ngrams good for? what advantages to they have over bag of words?

A collection of unigrams (what bag of words is) cannot capture phrases and multi-word epxressions, effectively
disregrading any word order dependence. Additionally, the bag of words model doesn't account for potential misspellings
or word derivations.

You could use character ngrams to as a means of addressing misspellings and word derivations.

http://scikit-learn.org/stable/modules/feature_extraction.html#limitations-of-the-bag-of-words-representation

---

--------

wh-movement (aka: wh-fronting. wh-extraction, long-distance dependency)

Special syntax rules involving the placement of interrogative words.

linguistics,

--------

dependency parsing

...

linguistics, nlp

--------

constituency parsing (aka: phrase structure grammars)

...

linguistics, nlp

--------

discontinuity

Discontinuity occurs when a given word or phrase is separated from another word or phrase it modifies in such a manner
that a direct connection cannot be established between the two without incurring crossing lines in the tree structure.

The tree structures with which discontinuities are identified and defined is based on the principle of Projectivity.
This is significant with respect to **dependency-based** parsing.

Types of discontinuity:

1. wh-movement
2. topicalization
3. scrambling
4. extraposition

=> How is this not a matter of visualization? (i.e. examples in wikipedia could be created/destroyed by visualizing
differently)

Related to, but distinguished from: inversion and shifting

linguistics,

--------

deep structure and surface structure

https://en.wikipedia.org/wiki/Deep_structure_and_surface_structure

linguistics, nlp,

--------

grammatical mood

linguistics,

--------

modality

linguistics,

--------

textual entailment

A directional relation between text fragments. The relation holds whenever the truth of one text fragment follows from
another text.

Can be viewed as a more relaxed form of pure logical entailment.

- text (t)
- hypothesis (h)
- "t entails h" (t => h) if a human reading *t* would infer that *h* is most likely true.

It is directional because even if "t entails h", the reverse "h entails t" is much less certain.

The ambiguity of natural language is an issue: several meanings can be contained in a single text and that the same
meaning can be expressed by different texts. This variability of semantic expression can be seen as the dual problem of
language ambiguity. Together they result in a many-to-many mapping between language expressions and meanings.
Interpreting a text correctly would, in theory, require a thorough semantic interpretation into a logic-based
representation of its meanings. Practical solutions for natural language processing seek to go not that deep and use
textual entailment in a more shallow way.

nlp, text-processing

https://en.wikipedia.org/wiki/Textual_entailment

--------

## Model evaluation

TODO merge with model selection above

ROC Curves vs. Precision-Recall Curves

TODO

You may be wondering where the name "Reciever Operating Characteristic" came from. ROC analysis is part of a field
called "Signal Dectection Theory" developed during World War II for the analysis of radar images. Radar operators had to
decide whether a blip on the screen represented an enemy target, a friendly ship, or just noise. Signal detection theory
measures the ability of radar receiver operators to make these important distinctions. Their ability to do so was called
the Receiver Operating Characteristics. It was not until the 1970's that signal detection theory was recognized as
useful for interpreting medical test results.

http://gim.unmc.edu/dxtests/roc3.htm

---

What is AUC, what is used for? What is it good for? What are problems with it?

A measure of accuracy when applied to ROC curves.

The area measures discriminations, the ability to correctly classify.

http://gim.unmc.edu/dxtests/roc3.htm

---

what is the central theme of mythical man month?

"adding manpower to a late software project makes it later". This idea is known as Brooks' law, and is presented along
with the second-system effect and advocacy of prototyping

--------

docker images vs containers

- Images create containers, containers are instantiated images

--------

docker why is docker less resource intensive than virtual machines?

- Because all docker containers share the same kernel (while providing the same resource isolation and allocation
  benefits as virtual machines)

---

## When do you need to use one-hot encoding?

TODO link to types of data

Essentially a way to not imply nominal data is ordinal by using integer encoding.

Categorical data in the purest sense is nominal (no meaningful order).

Integer encoding implies order, while that data is nominal.

Many machine learning algorithms cannot operate on label data directly. They require all input variables and output
variables to be numeric.

Whether an algorithm can directly operate on categorical data is influenced by the computational complexity.

https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/

---

## What is the bias-variance tradeoff?

TODO merge with type I and II error above

Bias and variance are two types of error. We hope to minimize both, but there is often a trade-off between them.

In statistics and machine learning, the bias–variance tradeoff (or dilemma) is the problem of simultaneously minimizing
two sources of error that prevent supervised learning algorithms from generalizing beyond their training set:

### Bias

The bias is error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the
relevant relations between features and target outputs (underfitting).

### Variance

TODO link to regularization

The variance is error from sensitivity to small fluctuations in the training set. High variance can cause overfitting:
modeling the random noise in the training data, rather than the intended outputs.

Variance is the amount our predicted values would change if we had a different dataset. It is the "flexibility" of our
model, balances against bias.

https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
https://elitedatascience.com/bias-variance-tradeoff

Give an example of a high-bias/low-variance classifier?

- Niave Bayes
- Regression
- Linear algorithms
- Parametric algorithms

Give an example of a low-bias/high-variance classifier?

- KNN
- Decision Trees
- Non-linear algorithms
- Non-parametric algorithms

https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/

---

Strategies for High Variance

- Weight decay
- Drop out or bagging
- Dimensionality reduction
- Feature selection

---

What does complexity have to do with bias-variance tradeoff?

- Low-bias & high-variance : more complex (more adaptable)
- High-bias & low-variance : less complex (less adaptable)

---

Capacity

The capacity of a machine learning algorithm is its ability to learn a variety of possible functions.

More capacity means a more flexible model at the risk of overfitting.

---

Conditioning

A measure of how much a function's outputs change when its inputs change. Poorly conditioned functions are highly
sensitive to rounding errors that can happen due to the way computers process real numbers.

--- 

Saturation

When a function's output is very insensitive to inputs (e.g. "tails" of sigmoid function).

---

sources of uncertainty

TODO merge with stats above

1. Inherent randomness in universe
2. Inability to completely obeserve phenomona (even it is deterministic)
3. Inability to perfectly model a phenomona

---

Underdetermined (underconstrained) systems

TODO merge with above

- A system equations with *more unkowns* than there are equations
- Always either has *no solution* or *many solutions*

--------

Mathematical Norm

TODO merge with above

- A function that assigns a strictly positive length or size to each vector in a vector space (except for the zero
  vector, which is assigned a length of zero)
- Used in linear algebra, functional analysis, and related mathematics
- Must satisfy certain properties pertaining to scalability and additivity
- A vector space on which a norm is defined is called a *normed vector space*

- Lp vector norms:
    - L0: The total number of non-zero elements in a vector
    - L1: Absolute-value, Sum-of-Absolute-Differences
    - L2: Euclidean distance, Sum-of-Squared Differences,
    - Taxicab ofr Manhattan norm/distance
    - p-norm
    - Max-norm
    - Hamming distance

- https://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/

---

# Optimization

---

## Loss-functions

Functions representing the price paid for inaccuracy of predictions in classification problems.

It's a mathematical representation of anything you want to *minimize*.

Some common loss functions include:

- Square loss
- Hinge loss
- Logistic loss
- Cross entropy loss
- ...

statistics, machine-learning, models

---

## Cost-functions

What are cost-functions?

More general than loss functions. Typically a sum of loss functions over your training set plus some model complexity
penalty (regularization).

- Mean squared error (MSE)
- Binary crossentropy
- Categorical crossentropy
- ...

https://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing

---

## Objective functions

A very general term for any function that you optimize during training.

https://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing

---

## High-level optimization methodology

- Model selection: Choose a model with some adjustable parameters.
- Loss-function selection: Choose a goodness-of-fit (i.e. reward/cost function) of the model to some data.
- Learning algorithm: Tune the parameters in order to maximize goodness-of-fit (i.e. apply learning algorithm to
  optimize reward/cost function)

#### Is optimization computationally difficult?

- Convexity makes optimization relatively easy.
- Non-convexity makes optimization hard (multiple minima, NP hard in some cases etc.).

#### Convex

- Linear models
- Kernel machines
- ...

#### Non-convex

- Artificial neural networks
- Clustering
- ...

---

### Emprical risk minimization

Fitting your model to minimize the mean of a loss function, calculated only on the training data

machine-learning, statistics, models

http://stats.stackexchange.com/questions/4961/what-is-regularization-in-plain-english

---

## Least-squares

- A standard approach in regression analysis for finding the approximate solution of an overdetermined system (a system
  with more equations than unknowns)
- Important in data fitting where the sum-of-squared residuals (error) provided by a model is minimized
- Two categories:
    - Ordinary least-squares (often useful as a baseline)
    - Non-linear least squares

---

## Ordinary least squares

Fits a linear model that minimizes residual sum of squares. It is very simple but is often useful as a baseline.

---

## Regularization

### What is regularization?

Regularization is a "penalty for model complexity".

- Regularization is a method to keep the coefficients of the model small, and in turn, the model less complex
- Simple models are often characterized by models that have smaller coefficient values. A technique that is often used
  to encourage a model to minimize the size of coefficients while it is being fit on data is called regularization.
- Often makes use of the L1 norm (or L2)

Explain what regularization is and why it is useful? What are the benefits and drawbacks of specific methods, such as
ridge regression and LASSO?

- A modification to the cost function that creates "preference" for certain parameter values (it biases the values to
  particular values)
- The regularization term is independent of the data, so emphasizing it reduces the "draw" of data (and hence reduces
  the variability of the model but may also increase it's bias)
- A clever way of balancing error and complexity

A process of introduction additional information in order to:

1. Solve ill-posed problems:
    - Ill-posed problems are where Ax = b has no x that satisfies the equation or x is not unique
2. Prevent overfitting:
    - A technique to improve the generalization of a learned model
    - Regularization introduces a penalty for exploring certain regions of the function space used to build the model,
      which can improve generalization
    - "Choosing a preferred level of model complexity so your models are appropriately generalized"

How to use it:

- Let f(x) be the classification model
- Introduce a regularization term R(f) to the loss function
- R(f) is typically a penalty on the complexity of f

https://www.youtube.com/watch?v=sO4ZirJh9ds

---

### L1 Regularization / LASSO

- (Least absolute shrinkage and selection operator)
- Can encourage sparse parameter vectors
- A regression analysis method that performs both variable selection and regularization in order to enhance the
  prediction accuracy and interpretabliity of the statistical model it produces
- ***Alters the model fitting process to select only a subset of the provided covariates for use in the model rather
  then using all of them***
- It forces certain coefficients to be set to zero by forcing the sum of the absolute value of the regression
  coefficients to be less than a fixed value
- Originally designed for least-squares, can be extended to wide-variety of statistical models
- Produces sparse coefficients, effectively performing feature selection "automatically"

https://towardsdatascience.com/5-types-of-regression-and-their-properties-c5e1fa12d55e

---

### L2 Regularization / Ridge-Regression / Tikhonov Regularization

- Most commonly used method of regularization of ill-posed problems
- Assume Ax = b
- Spreads error throughout the vector x
- Giving preference to smaller norms
- (where as L1 is more likely to produce a sparse x, meaning that some values in x are exactly zero while others may be
  relatively large)
- Also known as Tikhonov regularization, The Tikhonov-Miller method, the Phillips-Towney method...

--------

### L1 vs L2 Regularization

- L1 usually corresponds to setting a Laplacean prior on the regression coefficients and picking a maximum a posteriori
  hypothesis.
- L2 corresponds to picking a maximum a posteriori hypothesis using a Gaussian prior

---

## Expectation-maximization algorithm (EM)

- An iterative method to find (local) maximum likelihood or maximum a posteriori (MAP) estimates of parameters in
  statistical models, *where the model depends on unobserved latent variables*.
- Each iteration alternates between performing an **expectation step**, which creates a function for the expectation of
  the log-likelihood evaluated using the current estimate for parameters, and a **maximization step**, which computes
  parameters maximizing the expected log-likelihood found in the E step. These parameter-estimates are then used to
  determine the distribution of the latent variables in the next E step.
- May be useful for uncovering latent classes and incorporating unlabeled data (Tackling the Poor Assumptions of Naive
  Bayes Text Classifiers – 2003)
- Explained and given its name in 1977 paper (Dempster, Lairs, Rubin)

---

## Mixture-models

- A probabilistic model for representing the presence of subpopulations within an overall population
- Are used to make statistical inferences about the properties of the sub-populations given only observations on the
  pooled population, without sub-population identify information
- Typically hierarchical

---

## Modeling

### Exogenous vs. Endogenous variables

Terms mostly from economics parlance.

**Exogenous**:

- Independent variables.
- Value is determined *outside* the model and is imposed on the model.

**Endogenous**:

- Dependent variables.
- Value is determined *by* the model.
- Endogenous variables change in response to exogenous change imposed on the model.

### Latent variables

- Variables that are inferred instead of observed.
- Latent variables are inferred through a mathematical model from other variables that are observed

### Examples of latent variables

- Economics: quality of life, business confidence, morale

### Methods for inferring latent variables

- Hidden markov models
- Factor analysis
- Principle component analysis
- Partial least squares regression
- Latent semantic analysis
- EM algorithms

### Latent variable models

- Relate a set of observable variables to a set of latent variables
- Related: factor analysis, mixture models

### Black-box vs white-box models

#### Black-box vs. white-box models (as a function of a apriori knowledge)

- Models can be classified along the black-box/white-box spectrum according to how much apriori information is used to
  construct the system.
- Black-box is a system which uses no a priori information, a white-box model is a system where all necessary
  information is available apriori.
- It is usually preferable to use as much apriori information as possible because it helps to acheive accuracy (e.g.
  CNNs with images or RNNs with sequences).
- **A priori information usually comes in the form of knowing the type of functions relating different variables**.
- An alternative argument here is that you cannot establish apriori knowledge faster than available compute increases,
  so (in the long run) you should favor black-box models that generalize well.

#### Black-box vs. white-box models (as a function of functional-relationships and numerical parameters)

Functional relationships

- Apriori information is often captured as a set of functions describing relationships in a system.
- Extremely general functions are used when there is little apriori information.

Numerical parameters

- Functional relationships have (potentially many) parameters to be tuned.

- **Black-box models attempt to estimate both the functional form of relations between variables and the numerical
  parameters in those functions.**
- **White-box models typically need to estimate just the numerical parameters of the functional relationships (i.e. a
  priori knowlege).**

### Statistical models vs mathematical models generally

#### Mathematical models

- Static model: not expected to change once forumlated.
- Determine how the system changes from one state to the next.
- Often built from "first-principles".
- Can take many forms:
    - Logical models
    - Dynamical systems
    - Statistical models
    - Differential equations
    - Game theoretic models
    - ...
    - Hybrids: a mix of these and other types

- Mathematical models are of great importance in the natural sciences (e.g. physics).
- In many cases, the quality of a scientific field depends on how well the mathematical models developed on the
  theoretical side agree with results of repeatable experiments.
- Lack of agreement between theoretical mathematical models and experimental measurements often leads to important
  advances as better theories are developed.
- Example: string theory provides a means of exploring microscopic physics, where ordinary lab observations would not
  do (Susskind MIT AI podcast).

#### Statistical models

- A subset of mathematical models: "stochastic models", where randomness is present, and variable states are not
  described by unique values, but rather by probability distributions.
- Most often **inductive** models, which arise from empirical findings and generalization from them.
- Use numerical data to attempt to estimate probabilistic future behavior of the system, based on past behavior.

TODO link to sources of randomness

--------

## Euclid's Elements & Immanuel Kant: Critique of Pure Reason

The terms (TODO what terms?) first known appearance is in Euclid's elements (300 BC).

Kant's book is one of the most influential works in the history of philosophy, two particular terms are well
incorporated into statistics and modeling. The comprise two kinds of knowledge/justification/argument.

## A pirori

- From the earlier.
- Knowledge or jusitification that is idependent of experience (e.g. pure logic).
- With mathematical modeling:
    - It is usually preferable to use as much a prioro information as possible to make the model more accurate.
    - A priori information usually comes in the form of **knowing the type of functions relating different variables**.

## A posteriori

- From the latter.
- Knowledge or jusitification depends on experience or empirical evidence, as with most aspects of science and personal
  knowledge.

---

## Tuning vs. training

### Training

- In any non-pure white-box model contains some parameters that can be used to fit the model of the system to the data.
- The optimization of these parameters is called training.

### Tuning

- Optimization of model hyperparameters.
- Often uses cross-validation.

---

## Numerical analysis

https://www.quora.com/What-is-numerical-analysis

---

Overfitting

When a statistical model is too closely fit to the data set it was trained on. The model is describing
noise/random-error instead of the underlying relationship. Overfitting occurs when a model is excessively complex (i.e.
too many parameters relative to the number of observations). An overfit model with have poor predictive performance and
be sensitive to even minor changes in the training set.

machine-learning, statistitcs, models

https://en.wikipedia.org/wiki/Overfitting

---------

When does it make sense to use PCA?

- Ultimately a way to summarize data by "condensing" related features into a smaller set of new features: a
  dimensionality reduction technique
- PCA finds the best possible properties (among all linear combinations of the original features)
- PCA looks for (1) linear combinations that show maximum variance and (2) minimum error. These are equivalent!

- A statistical procedure that uses orthogonal transfomation to convert a set of observations of possibly correlated
  variables into a set of values of linearly uncorrelated variables called principle components.
- The number of principle components is less-than-or-equal to the number of original variables
- The transformation is definec in such a way that the first principal component has the largest possible variance (that
  is, accounts for the as much of the variability in the data as possible), and each succeeding component in turn has
  the highest variance possible under the constraint that it is orthogonal to the preceding components.
- The resulting vectors are an uncorrelated orthogonal basis set.

- In general, dimensionality reduction loses information but PCA-based dimensionality reduction minimizes that
  information loss

What is PCA used for?

- Exploratory data analysis
- Making predictive models
- Reducing the number of features (while preserving as much information as possible)

How to evaluate PCA results?

- Usually discussed in terms of component scores (or factor scores) and loadings

How does it compare to ordinary-least-squares:

- OLS minimizes the error between the dependent variable and the model
- PCA minimizes the error orthogonal (perpendicular) to the model line

Variations:

- Robust PCA: Good corrupted observations
- Sparse PCA: PCA results are typically linear combinations of all input variables. Sparse PCA finds linear combinations
  that contain a subset of input variables.

Has a number analogous approaches or simply other names across many fields:

- Principle-axis-theorem in mechanics
- Hotelling transform
- Kosambi-Karhunen-Love transform (signal processing)
- Proper orthogonal decomposition (POD) in mechanical engineering
- Singular value decomposition (SVD) of X (Golub and Van Loan, 1983)
- Eigenvalue decomposition (EVD) of XTX in linear algebra
- Schmidt–Mirsky theorem in psychometrics
- Empirical orthogonal functions (EOF) in meteorological science
- Empirical eigenfunction decomposition (Sirovich, 1987)
- Empirical component analysis (Lorenz, 1956)
- Quasiharmonic modes (Brooks et al., 1988)
- Spectral decomposition in noise and vibration
- Empirical modal analysis in structural dynamics.

http://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
https://en.wikipedia.org/wiki/Principal_component_analysis
http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf

---------

Local optimum

The best solution to a problem with respect to a small neighborhood of possible solutions. It contrasts the global
optimum, which is the optimal solutaion when every possible solution is considered.

----------

What are specific ways of determining if you have a local optimum problem?


----------

What can be done to avoid local optima?


----------

Why are convex functions important? (how was this phrased in original quora post?)

They are especially important in the study of optimization problems where they are distinguished by a number of
convenient properties.

The essential property is a (strictly) convest function has no more than one minimum.

For instance, a (strictly) convex function on an open set has no more than one minimum. Even in infinite-dimensional
spaces, under suitable additional hypotheses, convex functions continue to satisfy such properties and as a result, they
are the most well-understood functionals in the calculus of variations.

In probability theory, a convex function applied to the expected value of a random variable is always less than or equal
to the expected value of the convex function of the random variable. This result, known as Jensen's inequality,
underlies many important inequalities (including, for instance, the arithmetic–geometric mean inequality and Hölder's
inequality).

The success of deep learning draws some skepticism with regard to the idea that convexity is necessary to guarantee
convergence.

----------

What are examples of convex optimization problems?

- Linear regression with L2 regularization
- Sparse linear regression with L1 regularization
- Support Vector Machines
- Parameter estimation on linear-Gaussian time series (Kalman filter and friends)

https://www.quora.com/Why-is-Convex-Optimization-such-a-big-deal-in-Machine-Learning

---

Fowlkes-Mallows

For evaluating clusters when ground-truth is available.

---

Kalman Filter AKA Linear Quadratic Estimator

- For estimating the internal state of a linear dynamic system from a series of noisy measurements.
- You can use it in any place where you have uncertain information about a dynamic system (continuously changing), and
  you need to make an educated guess about what the system is going to do next (even with messy reality coming along
  with it).
- It's about squeezing as much information from our uncertain measurements as we possibly can!
- Applications include guidance, navigation of control of vehicles particularly aircraft and spacecraft; time-series
  analysis used in signal processing and econometrics; robotic motion planning and control; computer vision
- The algorithm is recursive, very efficient, it can run in real time
- Underlying model is a Bayesian model similar to a hidden Markov model
- Addresses arguably one of the most fundamental problems in control theory.

http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

------------

tf-idf (term frequency-inverse document frequency)

TODO move to nlp

- Essentially a means of re-weighting the count features (whether unigrams or n-grams) into floating point values
  suitable for usage by a classifier.
- A numerical statistic intended to reflect how important a word is to a document in a collection or corpus
- The tf-idf value increases proportionally to the number of times a word appears in the document, but is offset by the
  frequency of the word in the corpus, which helps to offset the fact that some words appear more frequently in general
- Used as a central tool in scoring and ranking a document's relevance to a user's query
- Can be used for stop-words filtering
- Term frequency:
    - How often does each term occur in a each document
- Inverse-document frequency:
    - log(number-of-documents/number-of-documents-containing-item)
    - How much information the word provides, whether the term is common or rare across all documents
- tf-idf:
  term-frequency / inverse-document-frequency
- High tf-idf is achieved by a high term frequency (in a given document) and a low document frequency of the term in the
  whole collection of documents

-----------

Latent Semantic Analysis / Indexing

TODO move to nlp

- An NLP technique which produces statistically derived vectors that are indicators of semantic similarities between
  documents and queries
- Affords queries and documents to have high cosine similarity even if they do not share any terms, as long as their
  terms are semantically similar.
- Can be viewed as a similarity metric that is an alternative to word overlap measures like tf-idf.
- Attempts to produce statistically derived "concept indices" instead of individual words for retreival
- It assumes there is some underlying or latent (topic) structure in word usage that is partially obscured by
  variability in word choice
- A truncated SVD is used to estimate the structure in word usage across documents. Retreival is then performed using
  the database of singular values and vectors obtained from the truncated SVD

nlp, svd, topic-modeling,

http://www.cse.msu.edu/~cse960/Papers/LSI/LSI.pdf

This is a great intro
https://www.youtube.com/watch?v=OvzJiur55vo

------------

Latent Dirichlet Allocation

TODO move to nlp

- You tell LDA how many topics you are looking for...
- LDA starts by randomly assigning a topic to each word in the corpus
- LDA then iteratively reassigns topics to words until
- A "bag of words" model
- Developed by: David Blei, Andrew Ng, Michael Jordan (2003)
- Hyperparameters (parameters of Dirichlet distribution):
    - Alpha: The prior on per document topic distributions (high-alpha means every document is likely to contain most of
      the topics, low-alpha means each document is likely to be related to just a few topics)
    - Beta: The prior on per topic word distributions (high-beta means that each topic is likely to contain a mixture of
      most of the words, low-beta means a topic may contain a mixture of just a few words)
- High alpha will make documents appear more similar to eachother
- High beta will make topics appear more similar to eachother
- The model produces a matrix of all the topics and all the words with probabilities of a word belonging to a topic

---

TextRank

- Summarization technique

1. Split sentences
2. Create a weighted graph of sentences (weight is similarity score of your choosing – the hard part)
3. Run PageRank on your weighted graph
4. Take the top n sentences as your summary

--- 

Spectral Theory / Spectral Analysis

spectral theory is an inclusive term for theories extending the eigenvector and eigenvalue theory of a single square
matrix to a much broader theory of the structure of operators in a variety of mathematical spaces.[1] It is a result of
studies of linear algebra and the solutions of systems of linear equations and their generalizations.[2

https://en.wikipedia.org/wiki/Spectral_theory

---

Random walk

A random walk is a mathematical object, known as a stochastic or random process, that describes a path that consists of
a succession of random steps on some mathematical space such as the integers.

https://en.wikipedia.org/wiki/Random_walk

---

PageRank

https://en.wikipedia.org/wiki/PageRank

---

Stop words

TODO move to NLP

- Words which are filtered out before or after processing natural language or text in general
- Often refers to the most common words in a language, though there is no single universal list of stop words used by
  NLP tools.

What are techniques for determining stop words?

- Common lists (e.g. ones available in NLP libs)
- TF-IDF: words with extremely high document frequency

------------

Saddle point

A point that is a minima along one axis and a maxima along another axis.

---

What is Bayes theorem?

TODO merge with above

---

bayesian optimization

TODO merge with above

Sample the cost function and us them to guess the true function with something called a Gaussian Process. They use that
guessed function to determine where to evaluate next. Repeat. This is much better than random sampling.

So now the question becomes: given all this useful guessed information, what point should we check next? In answering
this, there are two things we care about:

We should evaluate points we think will yield a low output value. That is, we should evaluate on points where the solid
green line is low.
We should check areas we know less about. So in the graph above, it would be wiser to check somewhere between .65 and
.75 than to check between .15 and .25 since we have a pretty good idea as to what’s going on in the latter zone. Said
differently, we should check regions which will reduce the variance in our guess the most.
Balancing these two is the exploration-exploitation trade-off. Do you look in new spots or exploit the goldmines you've
discovered thus far? We make our preference explicit with an activation function.

### Naive Bayes

- Early description appears in Duda & Hart (1973)
- Extremely common: it is fast, easy to implement, and relatively effective
- Often used as a baseline for the reasons above
- Frequently used in text classification
- "The punching bag of classifiers" (Lewis 1998)
- Variations:
    - Gaussian
    - Multinomial McCallum & Nigam (1998)
    - Bernoulli
- Systemic limitations:
    - Does not deal with class-imbalance well
    - Strongly assumes features are independent: strong dependencies between features means larger weight magnitudes,
      can be dealt with through normalization of classification weights (Tackling the Poor Assumptions of Naive Bayes
      Text Classifiers)
    - Multinomial naive Bayes does not model text well, words tend to follow power law distribution

---

What is Naive Bayes

Naive Bayes classifier is a general term which refers to conditional independence of each of the features in the model.

---

What is Multinomial Naive Bayes

The term Multinomial Naive Bayes simply lets us know that each p(fi|c) is a multinomial distribution, rather than some
other distribution. This works well for data which can easily be turned into counts, such as word counts in text.

https://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes

1

---

What is Gaussian Naive Bayes



---

Bayes: what is a prior?

- You can also interpret the “prior” as any assumption you can make on how the data behaves.
- Strong priors can be helpful with small data

----

bayesian optimization

Bayesian optimization is different from other optimization procedures in that it constructs a probabilistic model for f(
x) and then exploits this model to make decisions abou twhere in X to next evaluate the function ("query the function")
while integrating out uncertainty. The essential philosophy is to use *all* of the information available from previous
evaluations of f(x) and not simply rely on local gradient and Hessian approximations.

Imagine an unkown objective function, the Bayesian strategy is to treat it as a random function and place a prior over
it. The prior captures our beleifs about the behavior of the function. After gathering the function evaluations, which
are treated as data, the prior is ipdated to form the posterior distribution over the objective function. The posterior
distribution, in turn, is used to construct an acquisition function (often also referred to as infill sampling criteria)
that determines what the next query point should be.

Examples of acquisition functions include probability of improvement, expected improvement, Bayesian expected losses,
upper confidence bounds, Thompson sampling, or a mixture of these.

- Prior: The probability given what we know.
- Posterior: The probability given what we knew before and what we just found out.
- Acquisition function: a function developed and employed to determine the next "probe" or "query-point" of the
  objective function (an attempt to establish what evaluation will provide data to produce the most accruate posterior).

Good for finding the minimum of difficult non-convex functions with relatively few evaluations, at the cost of
performing more computation to determine next "query point". *When evaluations of f(x) are expensive to perform – as is
the case when it requires training a machine learning algorithm – then it is easy to justify some extra computation to
make better decisions*.

There are two major choices to be made when performing Bayesian optimization.

1. Selecting a prior over functions that will express assumptions about hte function being optimized
2. Choosing an acquisition function, which is used to constrcut a utility function from the model posterior, allowing us
   to determine the next point to evaluate.

https://arxiv.org/pdf/1206.2944.pdf

---

## probabilistic programming

- KEY: Automatically transform *simulation* instructions into *inference* programs, and they manage and quanitify
  uncertainty about causal explainations. They are machines designed to interpret data.

https://www.oreilly.com/ideas/probabilistic-programming

- PPLs are a high-level language that makes it easy for a developer to define probability models and then "solve" these
  models automatically.
- These languages incorporate random events as primitives and their runtime environment handles inference.
- Affords a clean separation between modeling and inference: an abstraction layer.

https://arxiv.org/pdf/1809.10756.pdf

- PP is fundamentally about developing languages that allow the denotation of inference problems and evaluators that "
  solve" inference problems (the fundamental tool)
- How do we engineer machines that reason?
- A major division in hypothesis space regarding how to engineer reasoning machines:
    1. Bayesian / probabilistic machine-learning: Random variables and probabilistic calculation are more-or-less an
       engineering requirement. Inference is a fundamental tool.
    2. Optimization is the fundamental tool, usually gradient descent.
- Conditioning: a foundational computation that is central to the fields of probabilistic machine learning and
  artificial intelligence
- First-order probabilistic programming language:
    - Whose programs define static-computation-graph, finite-variability-cardinality models
    - Fundamental inference algorithms
- Higher-order probabilistic programming languages:
    - Functionality analagous to that of established programming languages
    - Affords dynamic computation graphs at cost of requiring inference methods tthat generate samples repeatedly
      executing the program
    -

---

## tomek links

..

---

## simpsons paradox

This is the heart of Simpson's paradox. If you pool data without regard to the underlying causality, you'll get
erroneous results.
https://twitter.com/mbeckett/status/1278750652160634880


---

## Glossaries

- https://en.wikipedia.org/wiki/Glossary_of_artificial_intelligence