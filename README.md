# About
A program I wrote to learn rust.
It calulates fast fourier transforms using te package [rustfft](https://crates.io/crates/rustfft) and plot the result using [gnuplot](https://crates.io/crates/gnuplot) or save the result to a file.

# Usage
Make sure [gnuplot](http://gnuplot.info/) is installed before running the program.

## Options
```    -h, --help          Print the usage menu
    -p, --datapoints POINT
                        The number of points to be generated. Value must be
                        between 1 and 10e5
    -i, --invert        Reconstruct the original function by using the invert
                        fft
    -s, --silent        Do not show the plot windows
    -k, --keeptop KEEPTOP
                        Keep n top harmonics
    -m, --keepmin KEEPMIN
                        Keep harmonics where the amplitude is larger than
    -w, --write         Write the result to the given file
        --filename NAME Filename of the result file
    -t, --wavetype WAVETYPE
                        Select a wave type: sine, block, triangle, sawtooth,
                        custom
    -d, --data DATA     Data for the wave```