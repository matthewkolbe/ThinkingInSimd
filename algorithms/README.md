# Improving a Classic Algorithm

It's 2023, and the current state of high performance computing (whether it's on x86, ARM or GPUs) has a few overarching principles that guide performance. 

(1) Non-blocking compute is abundant.

(2) Data within a cache line is free to read.

(3) Data in the closest cache is really cheap to read.

(4) Data becomes increasingly expensive and lower bandwidth the "farther" it is from the register.

(5) Doing the same operation on a cache line data is almost free (SIMD/warps)

In college-level algorithms analysis, 1-4 get treated as more or less the same. But on modern computing systems, you can do thousands of non-blocking compute operations on 64B of cache data (i.e. 16 int32s) in the same time it takes to pull a single piece of random access data from main memory to cache. Imagine the possibilities. 

But this is a project dedicated to SIMD acceleration, so let's take a look at how squeezing in some vector operations can improve common algorithms.

## Binary Search on Integers

In the data_structures example, I showed an example of how binary search (the root finding version of the algorithm) could be vectorized. It's not a particularly clever version of vectorization, because its vectorization was dependent on there being 16 different binary searches that needed to happen at the same time. What happens if we only need one done? This is where I think being knowledgeable of hardware and the particulars of AVX intrinsics can really shine. 

What if I proposed doing a linear search of 16 integers every single iteration of the binary search? It sounds silly, right? Well, it would be silly to do as a length 16 loop comparing all the values, but AVX-512 comes equipped with an intrinsic (`_mm512_cmp_epi32_mask`) to compare 16 integers against 16 integers in one operation. It's a little slower than calling `==`, but not much slower. [src/avx_binary_search.cc](src/avx_binary_search.cc) implements this concept. One little thing to make sure of is that you only call `_mm512_cmp_epi32_mask` on the single aligned 64B chunk that contains the value you want to test, or else this operation gets much more expensive[^1].

## The test

I run a [Google Benchmark](https://github.com/google/benchmark) comparing the aforementioned algorithm to `std::lower_bound`.  I vary the input sizes from 7 to 67,108,864. 

Run on (32 X 5881 MHz CPU s)

CPU Caches:
- L1 Data 32 KiB (x16)
- L1 Instruction 32 KiB (x16)
- L2 Unified 1024 KiB (x16)
- L3 Unified 32768 KiB (x2)

|Array Size       |           AVX   |    std::lower_bound |
|:----------------|----------------:|--------:|
| 7               |        3.61 ns  |         10.6 ns | 
| 15              |        7.73 ns  |         28.6 ns |
| 83              |         123 ns  |          335 ns |
| 256             |         620 ns  |        1,523 ns |
| 503             |       1,557 ns  |        3,065 ns |
| 1,024           |       3,787 ns  |        6,513 ns |
| 4,096           |      20,124 ns  |       49,306 ns |
| 16,384          |     105,277 ns  |      366,247 ns |
| 65,536          |     572,036 ns  |    1,334,993 ns |
| 1,048,576       |       13.31 ms  |        19.99 ms |
| 16,777,216      |      346.39 ms  |       390.61 ms |
| 67,108,864      |    1,767.96 ms  |     1,766.38 ms |

AVX kicks the STL's ass until memory access constraints start to take over. 

## Footnotes

[^1] To expound on this, remember principle (2) from above? "Data within a cache line is free to read." You want the AVX-512 operation to only happen on the data in the current cache line or else the operation falls into the category of (3): "Data in the closest cache is really cheap to read." "Free" and "cheap" are not the same, and it's much better to stick with free if you can. So, we pass in 64B aligned data to the function, and therefore every 16th `int` will be on a new cache line. Of note: even after I wrote this, I still messed up and passed in unaligned data, and was getting okay results for small `n`, but increasingly bad ones until `std::lower_bound` actually started performing better than mine.
