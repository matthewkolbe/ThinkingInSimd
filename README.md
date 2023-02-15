# Thinking In SIMD

## Intro

The ubiquity and effectiveness of AVX intrinsics in the most popular programming languages has opened up a whole new paradigm for software optimization. Behind the scenes, compilers have embraced the AVX instruction sets, and will automatically apply them when appropriate ("autovectorization"). But autovectorization is generally only applied on shallow/obvious use cases. Don't just take it from me, here's Linus Torvalds:

>It's easy to wave your hands and say "just use vector units". You see people doing it here. In reality, not only has auto-vectorization not ever done so great in the first place, there aren't many places to do it at all.[^1]

But if you venture into the world of manually applying intrinsic functions, they can give you power far beyond what compilers can accomplish today.

## What are SIMD intrinsics?

I'm not going to answer this. But [here's a good C++ reference](http://const.me/articles/simd/simd.pdf) and [here for C#](https://devblogs.microsoft.com/dotnet/hardware-intrinsics-in-net-core/).

## What's the problem?

The biggest problem is that software architecture and API design are mostly not organized in a ways that easily take advantage of SIMD instruction sets. Ask yourself how often your programs have contiguous arrays of `intXX`/`floatXX` values that need to be transformed? If you're in scientific computing, it's probably often. But for most people, they probably have a contiguous set of `struct` or `class` types, and even if they want to do a very SIMD friendly operation on a data member of each, the data structure will not easily accommodate it, and transforming the data structure can potentially cost more than any gains from vectorizing the computation anyway.

That last sentence I think highlights the biggest problem: no one wants to use intrinsics for the sake of using intrinsics. They want to use this new functionality to optimize a computation. To that end, they've probably already been optimizing their computations: heeding their big O's, properly minimizing their cache misses, and inlining/hugepaging/hoisting/unrolling/batching/threading to its fullest extent. These aren't the types of programs that are intuitively organized, and to that end, adding one more effective tool to the mess that doesn't drop in easily for the vast majority of programs is maybe more of a problem than a solution.

Another problem is that computer scientists haven't classically written algorithms with SIMD in mind. I don't think 30 years ago it was commonly believed that doing the same operation on multiple data would be a useful consideration when designing algorithms. Recently, [Google came out with a SIMD-accelerated quicksort](https://opensource.googleblog.com/2022/06/Vectorized%20and%20performance%20portable%20Quicksort.html), and I think there is a lot of low hanging optimizations to be had by shoehorning in some AVX wherever it can be helpful.

I would like this essay to be a jumping off point for people who don't want to cynically dismiss AVX intrinsics by walking though examples of how thinking about SIMD can offer very significant performance improvements. 

## Project organization

This project will be a sort of interactive essay looking into the various ways to change the way you think about software optimization in the presence of AVX intrinsics. There is a folder for each "chapter" in this inquiry: [data_structures](data_structures/) and [algorithms](algorithms/) for now. In general, each section will feature a Google Benchmark to compare non-vectorized and vectorized versions of code. Fair warning in advance, this is about ways to push software optimization as far as possible, and to that end, all the AVX examples involve AVX-512 intrinsics. I know a lot of people don't have processors capable of using those instructions, so the example code won't run on those computers, but the essays can still be read and hopefully enjoyed.

### Footnotes

[^1]: https://www.realworldtech.com/forum/?threadid=209249&curpostid=209596
