# Thinking In SIMD

## Intro

The ubiquity and effectiveness of AVX intrinsics in the most popular programming languages has opened up a whole new paradigm for software optimization. Behind the scenes, compilers have embraced the AVX instruction sets, and will automatically apply them when appropriate ("autovectorization"). But autovectorization is generally only applied on shallow/obvious use cases. Don't just take it from me, here's Linus Torvalds:

>It's easy to wave your hands and say "just use vector units". You see people doing it here. In reality, not only has auto-vectorization not ever done so great in the first place, there aren't many places to do it at all. [1]

But if you venture into the world of manually applying intrinsic functions, they can give you power far beyond what compilers can accomplish today.

## What are SIMD intrinsics?

I'm not going to answer this. But [here's a good C++ reference](http://const.me/articles/simd/simd.pdf) and [here for C#](https://devblogs.microsoft.com/dotnet/hardware-intrinsics-in-net-core/).

## What's the problem?

The biggest problem is that software architecture and API design are mostly not organized in a ways that easily take advantage of SIMD instruction sets. Ask yourself how often your programs have contiguous arrays of `intXX`/`floatXX` values that need to be transformed? If you're in scientific computing, it's probably often. But for most people, they probably have a contiguous set of `struct` or `class` types, and even if they want to do a very SIMD friendly operation on a data member of each, the data structure will not easily accommodate it, and transforming the data structure can potentially cost more than any gains from vectorizing the computation anyway.

That last sentence I think highlights the biggest problem: no one wants to use intrinsics for the sake of using intrinsics. They want to use this new functionality to optimize a computation. To that end, they've probably already been optimizing their computations: heeding their big O's, properly minimizing their cache misses, and inlining/hugepaging/hoisting/unrolling/batching/threading to its fullest extent. These aren't the types of programs that are intuitively organized, and to that end, adding one more effective tool to the mess that doesn't drop in easily for the vast majority of programs is maybe more of a problem than a solution.

I would like to help maybe turn what looks like a problem into a solution.

# A C/C++ case study on how to organize an AVX accelerated application.

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
struct alignas(4096) bsv {
    public:
    float* __restrict__  ul;
    float* __restrict__ tte;
    float* __restrict__ strike;
    float* __restrict__ rate;
    float* __restrict__ iv;
    float* __restrict__ vol;
    float* __restrict__ px;
    float* __restrict__ theo;
};
```

Neither of these leverages any intrinsic data structure. So in the third, we will replace the `float` arrays with arrays of vector types. For this, we will use Agner Fog's superb [Vector Class Library (VCL)](https://github.com/vectorclass/version2), and specifically use the `Vec16f` type, which is a 512-bit storage for 16 contiguous floats.

```
struct alignas(4096) bsv512
{
    Vec16f *__restrict__ ul;
    Vec16f *__restrict__ tte;
    Vec16f *__restrict__ strike;
    Vec16f *__restrict__ rate;
    Vec16f *__restrict__ iv;
    Vec16f *__restrict__ vol;
    Vec16f *__restrict__ px;
    Vec16f *__restrict__ theo;
};
```

## Notes on `struct bs`

This structure is the most intuitive, but extremely AVX unfriendly. Inputs to the vectorized functions are of type `Vec16f`, so for example, they take in 16 different `ul` values, but an array of type `bsv` will not store its `ul` values contiguously. For this, we use a rather ugly hack by treating the `bsv` array as a `float` array, and then using `scatter` and `gather` operations to create the function inputs and apply the output. `scatter` and `gather` are [not particularly fast](https://www.agner.org/optimize/instruction_tables.pdf), especially on AMD processors, this should be slow. And given that many calculations do not call for the use of all the structure's variables, it will be bringing in unnecessary data to the cache page, and for operations that are memory-constrained, this can matter quite a bit.

## `bsv` vs. `bsv512`

There are advantages and disadvantages to each of these, despite how similar they are. `bsv` will be able to leverage autovectorization whenever the compiler sees an opportunity. This is ideal, but as noted earlier, it will not be able to recognize many opportunities, but when it does, you almost always want to just let it take over. Arrays of floats are much more familiar to compilers than `Vec16f` types. On the other hand, arrays of type `Vec16f` do not need to be changed into vector types in order to process them, thus saving time in the `load` and `store` stages of processing.

Although I will not be showing this here, `bsv512` has the additional complication that it is probably slower to load data into than `bsv`, but that all depends on how much you can leverage AVX up and down through your application.

## The test

I would like to see how all data structures perform in all compute paradigms. Additionally, I would like to see how it performs single threaded versus using 32 threads via OpenMP (OMP). I chose a value of `N=51200` because it's appropriately large for practical uses, and it divides all my partitions evenly. Of note is that `51200*8*4=1.6MB` for the total data operated on, and my processor has 64KB of L1 cache and 1MB of L2 per core. If we want to start exceeding cache, the usefulness of each data structure is bound to change. I use [Google Benchmark](https://github.com/google/benchmark) as the microbenchmarking framework. Here are the results:

Run on (32 X 5881 MHz CPU s)

CPU Caches:
- L1 Data 32 KiB (x16)
- L1 Instruction 32 KiB (x16)
- L2 Unified 1024 KiB (x16)
- L3 Unified 32768 KiB (x2)

Load Average: 0.81, 1.04, 0.90

|Benchmark                    |          Time|              CPU|   Iterations|
|:----------------------------|-------------:|----------------:|------------:|
|iv_naive_bsv                 | 19,345,965 ns|    19,345,938 ns|           36|
|iv_naive_bs                  | 19,284,717 ns|    19,284,882 ns|           36|
|iv_avx_bsv                   |  1,695,266 ns|     1,695,281 ns|          411|
|iv_avx_bsv_omp               |     89,908 ns|        89,909 ns|         7682|
|iv_avx_bsv512                |  1,493,387 ns|     1,493,389 ns|          453|
|iv_avx_bsv512_omp            |     78,079 ns|        78,079 ns|         8399|
|iv_avx_bs                    |  1,750,544 ns|     1,750,552 ns|          389|
|iv_avx_bs_omp                |     91,852 ns|        91,854 ns|         7531|
|pricer_naive_bsv             |    216,102 ns|       216,105 ns|         3210|
|pricer_naive_bs              |    239,462 ns|       239,464 ns|         2922|
|pricer_avx_bsv               |     76,378 ns|        76,375 ns|         9113|
|pricer_avx_bsv_omp           |      8,120 ns|         8,120 ns|        79219|
|pricer_avx_bsv512            |     95,223 ns|        95,223 ns|         7174|
|pricer_avx_bsv512_omp        |      7,982 ns|         7,982 ns|        74778|
|pricer_avx_bs                |    152,942 ns|       152,943 ns|         4498|
|pricer_avx_bs_omp            |     11,329 ns|        11,329 ns|        61423|
|vol_edge_naive_bsv512        |      3,997 ns|         3,997 ns|       174072|
|vol_edge_naive_bsv           |      3,613 ns|         3,613 ns|       193195|
|vol_edge_naive_bs            |     20,501 ns|        20,501 ns|        34116|
|vol_edge_avx_bsv             |      3,744 ns|         3,744 ns|       186476|
|vol_edge_avx_unrolled_bsv    |      3,544 ns|         3,544 ns|       197404|
|vol_edge_avx_bsv512          |      4,000 ns|         4,000 ns|       174787|
|vol_edge_avx_unrolled_bsv512 |      3,655 ns|         3,655 ns|       191567|
|vol_edge_avx_bs              |     41,404 ns|        41,404 ns|        16633|


## Analysis

`bsv512` seems to be a winner. This is interesting from the perspective of any lesson the glean, because I think it just raises more questions like "what about storing data into it?" I think it's a little unfortunate as well, because `bsv512` isn't nearly as portable or intutive as `bsv`, but if your priority is doing these calculations as fast as possible, it's hard to ignore exactly how much faster `bsv512` is.

One other thing of note is how across-the-board bad `bs` is. `scatter` and `gather` combined with cache thrashing is just too much overhead. Even when you use `bs` naively, the results are bad, because the compiler cannot autovectorize it easily.

## Comments

A big takeaway is just look at how much AVXifying your code can help. Single threaded Naive-to-AVX offers a >10x speedup. That's hard to ignore, and if the concept is hard to ignore, then you should consider from the very beginning whether your data structures will hamper your ability to leverage it. It's easy to mistakenly think ahead of time that `bs` would be a good data structure for this purpose. It's simple, and what's more is that by putting all your instrument's data together, you'll get speed ups from the idea that using one piece of data implies a high probability that you will use another, so they'll both be there on the cache line already. That sounds logical to me, but then, when you're SIMD optimizing down the road, or the compiler is trying to SIMD optimize for you, you lose big time, and probably by that point, you've built so many other applications around the `bs` data structure that you cannot go back without incurring a huge expense. 

### Footnotes

[1] https://www.realworldtech.com/forum/?threadid=209249&curpostid=209596

[2] This will be calculating the inverse of the Black-Scholes option price with respect to its volatility parameter, and [you can read about Black-Scholes here](http://www.iam.fmph.uniba.sk/institute/stehlikova/fd14en/lectures/06_black_scholes_2.pdf). Aside from the formula itself being non-trival, with calls to the ERF function, it's not analytically invertible, and it's best to use a bisection root finder to solve instead. 

[3] This will be calculating an option's value in the Black-Scholes formula.