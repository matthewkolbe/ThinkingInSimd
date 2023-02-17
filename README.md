# Thinking In SIMD

The ubiquity and effectiveness of AVX intrinsics in the most popular programming languages has opened up a whole new paradigm for software optimization. Behind the scenes, compilers have embraced the AVX instruction sets, and will automatically apply them when appropriate ("autovectorization"). But autovectorization is generally only applied on shallow/obvious use cases. Don't just take it from me, here's Linus Torvalds:

>It's easy to wave your hands and say "just use vector units". You see people doing it here. In reality, not only has auto-vectorization not ever done so great in the first place, there aren't many places to do it at all.[^1]

But if you venture into the world of manually applying intrinsic functions, they can give you power far beyond what compilers can accomplish today.

## What are SIMD intrinsics?

I'm not going to answer this. But [here's a good C++ reference](http://const.me/articles/simd/simd.pdf) and [here for C#](https://devblogs.microsoft.com/dotnet/hardware-intrinsics-in-net-core/).

## But SIMD is Important

Despite Linus's misgivings about autovectorization, it's becoming increasingly evident that these instruction sets are both important now, and that computing architectures in the future will rely more heavily on them to increase efficiency as well.

The purpose of this essay is to convince you that ignoring the benefits of using intrinsics and vectorized libraries to improve your applications can leave you with drastically underperforming code. There are many valid complaints you'll find, especially with the AVX512 instruction set: they need to be "warm" to actually be fast, the CPU will throttle itself because they heat the unit more, and many of the problems it solves are memory-bound anyway. None of these complaints are wrong, but there are plenty of places where it offers computational speed unattainable without them, so don't let the cynics drown them out.

Already we have interesting things like [Google coming out with a SIMD-accelerated quicksort](https://opensource.googleblog.com/2022/06/Vectorized%20and%20performance%20portable%20Quicksort.html), [simd](https://github.com/simdjson/simdjson), a JSON parser that offers unmatched performance, and several math libraries like Agner Fog's [vector class library](https://github.com/vectorclass/version2)(which I will be using in some example), the Intel MKL, and [SLEEF](https://github.com/shibatch/sleef).

The first example is [data_structures](data_structures/), because SIMD can be severely kneecapped if you do not use correct data structures as a foundation.

The second is [algorithms](algorithms/), showing how vectorization can be shoehorned into common algorithms. 

The third is [optimization](optimization/), which is TBD.

In general, each section will feature a Google Benchmark to compare non-vectorized and vectorized versions of code. Fair warning in advance, this is about ways to push software optimization as far as possible, and to that end, all the AVX examples involve AVX-512 intrinsics. I know a lot of people don't have processors capable of using those instructions, so the example code won't run on those computers, but the essays can still be read and hopefully enjoyed.


### Footnotes

[^1]: https://www.realworldtech.com/forum/?threadid=209249&curpostid=209596
