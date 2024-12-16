# SIMD equalizer #
### The data processed ###
We have a `.wav` file, the bit depth is 16, this means that each frame ("sample") is 16 bit, and the sample rate is 44100 Hz, meaning that each second we have 44100 samples. 
This configurations of bit depth + sample rate is the CD standard, usually musicians can record with a sample rate of 48000 (also more for advanced scenarios) and a bit depth of 24, but the CD standard is sufficient to 
listen to good quality music (streaming platform are way smaller).

To have more flexibility, we use the Fast Fourier Transform to have the audio in two arrays that represent complex numbers: the first array with the real part and the second with the imaginary part. This simplifies the operations
to process audio. Each array is divided in N slices (N as the number of equalizer bands, in our case is three: low freq., mid and high), every value of the slice is multiplied by a gain that increases or decreases the volume of that specific band.

### Parallel float EQ ###
First, the pattern of the "naive" equalizer in c++ is something like this:
```

```

This is shown in the file ```ParallelEQ.cpp``` scalar function.
We can easily parallelize this problem:
```

```

At first, we had to deal with double-precision floating point numbers (64 bit each one), but this format was too big: if a single register is 128 bit, the theoretical speedup is only 2...
So, in this version we have shown, we used float (32 bit). The result, in terms of audio quality, is still high so we can't hear the difference. 

### IIR ###
IIR stands for *Infinite Impulse Response*, is an alternative method for signal processing. In our use-case is a very fast method to divide frequencies. The main advantage is that we can process the 16 bit sample directly, without the necessity to process complex numbers in floating point.
**But, the result is not precise like the one we saw before.** The gain calculus and band division is more imprecise.
Even if it's not like the equalizer with the FFT, the IIR could be implemented with more advanced techniques and used for real-time applications.

For IIR this is the approach:
1. We divide each sample in three parts (LOW, MID, HIGH) fastly with IIR filter .
2. Each sample-slice is multiplied by the gain.

So we have three IIR filter: the first selects the low frequencies, the second for the mid and the third for the high frequencies. The mathematic formula to apply the filter is the same for the three filters, only some coefficients change: 
\

![Filtro IIR](https://latex.codecogs.com/png.latex?\bg_white\color{Black}y%5Bn%5D%20%3D%20%5Cfrac%7Bb_0%20x%5Bn%5D%20%2B%20b_1%20x%5Bn-1%5D%20%2B%20b_2%20x%5Bn-2%5D%7D%7B1%20%2B%20a_1%20y%5Bn-1%5D%20%2B%20a_2%20y%5Bn-2%5D%7D)

\
Where x is the input sample, y the filtered output and a,b,z the coefficients.

In scalar mode, the code is:
```

```

And the parallel function is similar:
```

```

But, as introduced before, we have some imprecision problems:
- The formula needs the sample x[n] but also the previous samples x[n-1] and x[n-2], very sequential. We can adjust this by updating the values not every sample but every 4 samples (like a "point of sync") losing precision. The audio result is still ok.
- We can't deal directly with 16 bit sample but we need to upscale to 32 bit to handle overflow. The theoretical parallelism is 4 so is still ok.

### Performances ###
