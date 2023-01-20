# Thinking In SIMD

## Intro

The ubiquity and effectiveness of AVX intrinsics in the most popular programming languages has opened up a whole new paradigm for software optimization. Behind the scenes, compilers have embraced the AVX instruction sets, and will automatically apply them when appropriate ("autovectorization"). But autovectorization is generally only applied on shallow/obvious use cases. Don't just take it from me, here's Linus Torvalds:

>It's easy to wave your hands and say "just use vector units". You see people doing it here. In reality, not only has auto-vectorization not ever done so great in the first place, there aren't many places to do it at all. [1]

But if you venture into the world of manually applying intrinsic functions, they can give you power far beyond what compilers can accomplish today.

## What are SIMD intrinsics?

I'm not going to answer this. But [here's a good C++ reference](http://const.me/articles/simd/simd.pdf) and [here for C#](https://devblogs.microsoft.com/dotnet/hardware-intrinsics-in-net-core/).

## What's the problem?

The biggest problem is that software architecture and API design are mostly not organized in a ways that easily take advantage of SIMD instruction sets. Ask yourself how often your programs have contiguous arrays of `intXX`/`floatXX` values that need to be transformed? If you're in scientific computing, it's probably often. But for most people, they probably have a contiguous set of `struct` or `class` types, and even if they want to do a very SIMD friendly operation on a data member of each, the data structure will not easily accommodate it, and transforming the data structure can potentially cost more than any gains from vectorizing the computation anyway.

That last sentence I think highlights the biggest problem: no one wants to use intrinsics for the sake of using intrinsics. They want to use this new functionality to optimize a computation. To that end, they've probably already been optimizing their computations: heeding their big O's, properly minimizing their cache misses, and inlining/hoisting/unrolling/batching/threading to its fullest extent. These aren't the types of programs that are intuitively organized, and to that end, adding one more effective tool to the mess that doesn't drop in easily for the vast majority of programs is maybe more of a problem than a solution.

I would like to help maybe turn what looks like a problem into a solution.

# A case study on how to organize an AVX accelerated application.

In a recent CppCon talk, Optiver's David Gross said "if you don't design your data properly, it's just going to be slow everywhere," highlighting the idea that you cannot micro-optimize out bottleneck problems if they're built on memory-inefficient data structures. So, the fundamental question I want to answer is how we build a data processing solution that effectively integrates AVX intrinsics, and as an example, I'll draw from my field of work: financial derivatives pricing.

Financial derivatives are interesting because there are a lot of derivatives active at any one time--like tens of millions--and there are a lot of things you might need to calculate. The first step is usually to split the work up among different machines (or ignore irrelevant derivatives) to reduce it down to tens of thousands, but after that, organizing all of this data to be responsive and memory efficient is critical. 

There are several types of operations we want to be able to perform on our data: long [2] (~1us per calculation), medium [3] (~50ns per calculation) and short (a few clock cycles, e.g. add two numbers). 

## The type of AVX acceleration we're targeting

Each calculation I mentioned can be done serially. One interesting thing about all of them is that they can be written as AVX functions. That is, instead of calculating one implied volatility at a time, we can use the VCL to compose a function that calculates 16 at once. So, the goal will be to calculate the desired values in chunks of 16, rather than one at a time. [`vec_black_scholes.hpp`](vec_black_scholes.hpp) features the implementations of these functions, whereas [`black_scholes.hpp`](black_scholes.hpp) has the non-vectorized implementations.

## The data structures

We need to organize eight `floats` for each derivative. I'll try three ways.

The first I think is the "obvious" way for most programmers: put all eight in a structure for each derivative, then store an array of derivatives.

```
struct alignas(32)  bs {
    public:
    float ul, tte, strike, rate, iv, vol, px, theo;
};
```

The next would be to keep each data type in a contigious array inside a data structure:

```
struct alignas(64) bsv {
    public:
    alignas(4) float* __restrict__  ul;
    alignas(4) float* __restrict__ tte;
    alignas(4) float* __restrict__ strike;
    alignas(4) float* __restrict__ rate;
    alignas(4) float* __restrict__ iv;
    alignas(4) float* __restrict__ vol;
    alignas(4) float* __restrict__ px;
    alignas(4) float* __restrict__ theo;
};
```

Neither of these leverages any intrinsic data structure. So in the third, we will replace the `float` arrays with arrays of vector types. For this, we will use Agner Fog's superb [Vector Class Library (VCL)](https://github.com/vectorclass/version2), and specifically use the `Vec16f` type, which is a 512-bit storage for 16 contiguous floats.

```
struct alignas(64)  bsv512 {
    public:
    alignas(64) Vec16f* __restrict__ ul;
    alignas(64) Vec16f* __restrict__ tte;
    alignas(64) Vec16f* __restrict__ strike;
    alignas(64) Vec16f* __restrict__ rate;
    alignas(64) Vec16f* __restrict__ iv;
    alignas(64) Vec16f* __restrict__ vol;
    alignas(64) Vec16f* __restrict__ px;
    alignas(64) Vec16f* __restrict__ theo;
};
```

## Notes on `struct bs`

This structure is the most intuitive, but extremely AVX unfriendly. Inputs to the vectorized functions are of type `Vec16f`, so for example, they take in 16 different `ul` values, but an array of type `bsv` will not store its `ul` values contiguously. For this, we use a rather ugly hack by treating the `bsv` array as a `float` array, and then using `scatter` and `gather` operations to create the function inputs and apply the output. `scatter` and `gather` are [not particularly fast](https://www.agner.org/optimize/instruction_tables.pdf), especially on AMD processors, this should be slow. And given that many calculations do not call for the use of all the structure's variables, it will be bringing in unnecessary data to the cache page, and for operations that are memory-constrained, this can matter quite a bit.

## `bsv` vs. `bsv512`

There are advantages and disadvantages to each of these, despite how similar they are. `bsv` will be able to leverage autovectorization whenever the compiler sees an opportunity. This is ideal, but as noted earlier, it will not be able to recognize many opportunities, but when it does, you almost always want to just let it take over. Arrays of floats are much more familiar to compilers than `Vec16f` types. On the other hand, arrays of type `Vec16f` do not need to be changed into vector types in order to process them, thus saving time in the `load` and `store` stages of processing.

Although I will not be showing this here, `bsv512` has the additional complication that it is probably slower to load data into than `bsv`, but that all depends on how much you can leverage AVX up and down through your application.

## The test

I would like to see how all data structures perform in all compute paradigms. Additionally, I would like to see how it performs single threaded versus using 32 threads via OpenMP (OMP). I chose a value of `N=51200` because it's appropriately large for practical uses, and it divides all my partitions evenly. I use [ubench](https://github.com/sheredom/ubench.h) as the microbenchmarking framework. It's a header-only library that I embedded in this repo, so it should work seamlessly if you'd like to clone this and run the benchmarks yourself. Here are my systems specs:

<p style="text-align:center;">
<img src="images/specs.png" width="500">

Compiled with `g++ main.cpp -std=c++20 -O3 -lm -lstdc++ -march=native -fopenmp`.


| Syntax      | Long (Imp Vol)   | Medium (Price) | Short (Abs sum) | Short Unrolled |
| :---------- | ----------- | ----------- | ----------- | --------- |
| Naive `bsv`  | 23.083ms<br>+- 0.22% | 1.069ms<br>+- 0.71% |  3.792us<br>+- 0.53% | N/A |
| Naive `bs` | 23.312ms<br>+- 0.18% | 1.074ms<br>+- 0.3% | 20.855us<br>+- 0.44% | N/A |
| AVX `bsv` | 1.727ms<br>+- 0.32%| 85.285us<br>+- 0.12% | 3.748us<br>+- 0.63% | 2.384us<br>+- 0.04% |
| AVX `bs` | 1.854ms<br>+- 0.32% | 150.135us<br>+- 0.45% | 38.283us<br>+- 0.19%| N/A |
| AVX `bsv512` | 1.538ms<br>+- 0.29% | 84.751us<br>+- 0.14% | 4.165us<br>+- 0.92% | 3.795us<br>+- 0.59% |
| OMP `bsv` | 89.559us<br>+- 2.39% | 7.871us<br>+- 1.31% | N/A | N/A |
| OMP `bs` | 91.233us<br>+- 0.33% | 13.640us<br>+- 2.22% | N/A | N/A |
| OMP `bsv512` | 76.961us<br>+- 1.75% | 7.861us<br>+- 1.2% | N/A | N/A |

## Analysis

Results are encouraging. `bs512` is slightly more performant than `bsv` for a long task, but similar or worse for short/medium ones. Depending on your preferences, it seems likely that `bsv`, the much easier and more maintainable structure is also the best from a performance point of view. Also interesting to note that the Naive `bsv` is the fastest for the Short calculation if you go through the effort to manually unroll the loop and reorder the operations to prevent blocking to wait for a calculation to finish.

One other thing of note is how across-the-board bad `bs` is. `scatter` and `gather` combined with cache thrashing is just too much overhead. Even when you use `bs` naively, the results are bad, because the compiler cannot autovectorize it easily.

## Comments

I think the first point I want to make it just look at how much AVXifying your code can help. Single threaded Naive->AVX offers a >10x speedup. That's hard to ignore, and if the concept is hard to ignore, then you need to consider from the very beginning whether your data structures will hamper your ability to leverage it. It's easy to mistakenly think ahead of time that `bs` would be a good data structure for this purpose. It's simple, and what's more is that by putting all your instrument's data together, you'll get to leverage the idea that using one piece of data implies a high probability that you will use another, so they'll both be there on the cache line already. That sounds logical to me, but then, when you're SIMD optimizing down the road, or the compiler is trying to SIMD optimize for you, you lose big time, and probably by that point, you've built so many other applications around the `bs` data structure that you cannot go back without incurring a huge expense. 

### Footnotes

[1] https://www.realworldtech.com/forum/?threadid=209249&curpostid=209596

[2] This will be calculating the inverse of the Black-Scholes option price with respect to its volatility parameter, and [you can read about Black-Scholes here](http://www.iam.fmph.uniba.sk/institute/stehlikova/fd14en/lectures/06_black_scholes_2.pdf). Aside from the formula itself being non-trival, with calls to the ERF function, it's not analytically invertible, and it's best to use a bisection root finder to solve instead. 

[3] This will be calculating an option's value in the Black-Scholes formula.