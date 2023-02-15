# Improving a Classic Algorithm

It's 2023, and the current state of high performance computing (whether it's on x86, ARM or GPUs) has a few overarching principles that guide performance. 

(1) Non-blocking compute is abundant.

(2) Data within a cache line is free to read.

(3) Data in the closest cache is really cheap to read.

(4) Data becomes increasingly expensive and lower bandwith the "farther" it is from the register.

(5) Doing the same operation on a cache line data is almost free (SIMD/warps)

In college-level algorithms analysis, 1-4 get treated as more or less the same. But on modern computing systems, you can do thousands of non-blocking compute operations on 64B of cache data (i.e. 16 int32s) in the same time it takes to pull a single piece of random access data from main memory to cache. Imagine the possibilities. 

But this is a project dedicated to SIMD acceleration, so let's take a look at how squeezing in some vector operations can improve common algorithms.

## Binary Search on Integers

In the data_structures example, I showed an example of how binary search (the root finding version of the algorithm) could be vectorized. It's not a particularly clever version of vectorization, because its vectorization was dependent on there being 16 different binary searches that needed to happen at the same time. What happens if we only need one done? This is where I think being knowledgable of hardware and the particulars of AVX intrinsics can really shine. 

What if I proposed doing a linear search of 16 integers every single iteration of the binary search? It sounds silly, right? Well, it would be silly to do as a length 16 loop comparing all the values, but AVX-512 comes equipped with an intrinsic (`_mm512_cmp_epi32_mask`) to compare 16 integers against 16 integers in one operation. It's a little slower than calling `==`, but not much slower. [src/avx_binary_search.cc](src/avx_binary_search.cc) implements this concept. One little thing to make sure of is that you only call `_mm512_cmp_epi32_mask` on the single aligned 64B chunk that contains the value you want to test, or else this operation gets much more expensive[^1].

## The test

I run a [Google Benchmark](https://github.com/google/benchmark) comparing the aforementioned algorithm to `std::lower_bound`.  I vary the input sizes from 7 to 262144. After that perfornace starts to degrade relative to `std::lower_bound` as the size exceeds my L2 and other memory considerations start to take over.

Run on (32 X 5881 MHz CPU s)

CPU Caches:
- L1 Data 32 KiB (x16)
- L1 Instruction 32 KiB (x16)
- L2 Unified 1024 KiB (x16)
- L3 Unified 32768 KiB (x2)

|Array Size       |         AVX   |    std::lower_bound |
|:----------------|--------------:|--------:|
| 7           |      3.58 ns  |    10.3 ns |
| 15          |      7.69 ns  |    28.5 ns |
| 83          |       129 ns  |     258 ns |
| 503         |      1,646 ns  |    2,603 ns |
| 1031        |      4,298 ns  |    5,604 ns |
| 2048        |     10,137 ns  |   11,577 ns |
| 4096        |     23,589 ns  |   51,022 ns |
| 8192        |     53,560 ns  |  135,986 ns |
| 16384       |    121,045 ns  |  364,385 ns |
| 32768       |    273,431 ns  |  749,288 ns |
| 65536       |    669,157 ns  | 1,498,834 ns |
| 131072      |   1,476,971 ns  | 2,792,282 ns |
| 262144      |   3,170,675 ns  | 5,395,945 ns |


## Footnotes

[^1] To expound on this, remember principle (2) from above? "Data within a cache line is free to read." You want the AVX-512 operation to only happen on the data in the current cache line or else the operation falls into the category of (3): "Data in the closest cache is really cheap to read." "Free" and "cheap" are not the same, and it's much better to stick with free if you can. So, we pass in 64B aligned data to the function, and therefore every 16th `int` will be on a new cache line. 
