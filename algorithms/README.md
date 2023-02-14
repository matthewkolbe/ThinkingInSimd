It's 2023, and the current state of high performance computing (whether it's on x86, ARM or GPUs) has a few overarching principles that guide performance. 

(1) Non-blocking compute is abundant.

(2) Data within a cache line is free to read.

(3) Data in the closest cache is really cheap to read.

(4) Data becomes increasingly expensive and lower bandwith the "farther" it is from the register.

(5) Doing the same operation on a cache line data is almost free (SIMD/warps)

In college-level algorithms analysis, 1-4 get treated as more or less the same. But on modern computing systems, you can do thousands of non-blocking compute operations on 64B of cache data (i.e. 16 int32s) in the same time it takes to pull a single piece of random access data from main memory to cache.

## Example 1: Binary Search on Integers

Binary search is pretty well-regarded, but it does suffer from (4) when data is large enough, but when you can fit the whole data structure into L1, the seemingly random access memory jumps don't cause stalled cycles waiting for data, but the pipeline shouldn't be full, either. A binary search involves a call to view the value at an index (which will probably not be in the register already), a comparison of that value to determine the search direction, and a modification of the search size aand centering before repeating the process. A compare plus two basic arithmatic operations don't fill up a whole 10 insturction pipeline that has to wait one cycle to retreive L1 data.

So, we have some "free" instructions to execute so long as they aren't blocking. 

In my opinion, (5) looks like an interesting opportunity here. The algorithim chooses a middle test value (call it `midi`) to check whether it matches the one we're searching for, and if not, then which half of the array to then look through to try to find it. Why not test all the values on `midi`'s cache line? If we use AVX-512 intrinsics, we can test 16 for equality in one operation, which gives us 16x more chance to terminate the search early. And even if we fail, we can potentially reduce the search area by 16.

In the first iteration of say a `262144` sized array, this seems pretty insignificant. There's a 1 in 16384 chance we hit it, and we only contract the next iteration's range by 0.006%. But when we've whittled it down to a length 16 search area, it's guaranteed to be found, while ordinary binary search has potentially another 4 operations to do (albeit all on data that's already in the register).