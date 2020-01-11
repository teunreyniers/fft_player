extern crate args;
extern crate getopts;

use getopts::Occur;

use args::validations::{Order, OrderValidation};
use args::{Args, ArgsError};

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

use std::env;
use std::error;
use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;
use std::process::exit;

use gnuplot::{AxesCommon, Caption, Figure, Graph};

use itertools::izip;

const PROGRAM_DESC: &'static str =
    "A program to play with fast fourier transforms. Written to learn rust en rustfft";
const PROGRAM_NAME: &'static str = "FFT Player";

trait Amplidute {
    fn amplitude(&self) -> f32;
}

impl Amplidute for Complex<f32> {
    fn amplitude(&self) -> f32 {
        (self.re.powi(2) + self.im.powi(2)).sqrt()
    }
}

struct Task {
    invert: bool,
    save_to_file: bool,
    plot: bool,
    filename: String,
    number_of_points: u32,
    number_of_harmonics: u32,
    min_harmonics_amplitude: f32,
    function: Function,
}

enum Tasks {
    Help(String),
    Handle(Task),
}

enum Function {
    Sine(Vec<f32>, Vec<f32>),
    Square(Vec<f32>, Vec<f32>),
    Triangle(Vec<f32>, Vec<f32>),
    Sawtooth(Vec<f32>, Vec<f32>),
    Custom(Vec<f32>),
}

fn main() {
    match parse_arguments(&env::args().collect::<Vec<String>>()) {
        Ok(Tasks::Help(message)) => print!("{}", message),
        Ok(Tasks::Handle(task)) => match handle_task(task) {
            Ok(_) => println!("Done"),
            Err(error) => {
                println!("{}", error);
                exit(1);
            }
        },
        Err(error) => {
            println!("{}", error);
            exit(1);
        }
    };
}

fn handle_task(task: Task) -> Result<(), Box<dyn error::Error>> {
    let number_of_samples = task.number_of_points;
    let harmonics_to_keep = task.number_of_harmonics;
    let min_harmonics_amplitude = task.min_harmonics_amplitude;

    let mut input: Vec<Complex<f32>> = match task.function {
        Function::Sine(h, a) => create_function(number_of_samples, &sine, &h, &a),
        Function::Square(h, a) => create_function(number_of_samples, &square, &h, &a),
        Function::Triangle(h, a) => create_function(number_of_samples, &triangle, &h, &a),
        Function::Sawtooth(h, a) => create_function(number_of_samples, &sawtooth, &h, &a),
        Function::Custom(p) => p.iter().map(|x| Complex::new(*x, 0f32)).collect(),
    };

    let fft = run_fft(&mut input, false);

    let mut ifft: Option<Vec<Complex<f32>>> = None;
    let mut gfifft = None;
    if task.invert {
        let fifft: Vec<Complex<f32>> = combine(
            fft[0..harmonics_to_keep as usize].to_vec(),
            zeros(number_of_samples - harmonics_to_keep),
        )
        .iter()
        .map(|&y| {
            if y.amplitude() >= min_harmonics_amplitude {
                y
            } else {
                Complex::zero()
            }
        })
        .collect();
        ifft = Some(run_fft(
            &mut fifft
                .iter()
                .map(|c| c * 0.5 * number_of_samples as f32)
                .collect(),
            true,
        ));
        gfifft = Some(fifft);
    }

    if task.plot {
        match (ifft.clone(), gfifft.clone()) {
            (Some(ifft), Some(gfifft)) => {
                plot_figure_time_domain(vec![
                    (&input, "Original"),
                    (&ifft, "Reconstruction"),
                ])?;
                plot_figure_frequenty_domain(vec![
                    (&fft, "FFT"),
                    (&gfifft, "Reconstruction"),
                ])?;
            },
            _ => {plot_figure_time_domain(vec![(&input, "Original")])?;
            plot_figure_frequenty_domain(vec![(&fft, "FFT")])?;}
        }
    }

    if task.save_to_file {
        let data: Vec<Vec<f32>> = match (ifft, gfifft) {
            (Some(ifft), Some(gfifft)) => izip!(
                input.iter().map(|c| c.re),
                fft.iter().map(|c| c.re),
                fft.iter().map(|c| c.im),
                ifft.iter().map(|c| c.re),
                ifft.iter().map(|c| c.im),
                gfifft.iter().map(|c| c.re),
                gfifft.iter().map(|c| c.im),
            )
            .map(|x| vec![x.0, x.1, x.2, x.3, x.4, x.5, x.6])
            .collect(),
            _ => izip!(
                input.iter().map(|c| c.re),
                fft.iter().map(|c| c.re),
                fft.iter().map(|c| c.im),
            )
            .map(|x| vec![x.0, x.1, x.2])
            .collect(),
        };
        write_to_file(&data, &task.filename)?;
    }

    Ok(())
}

fn plot_figure_frequenty_domain(
    plots: Vec<(&Vec<Complex<f32>>, &str)>,
) -> Result<(), gnuplot::GnuplotInitError> {
    let mut fg = Figure::new();
    let ax = fg
        .axes2d()
        .set_title("Frequentie domain", &[])
        .set_legend(Graph(0.5), Graph(0.9), &[], &[])
        .set_x_label("harmonics", &[])
        .set_y_label("amplitude", &[]);
    for plot in plots {
        ax.points(
            &(0..plot.0.len())
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>(),
            &plot.0.iter().map(|c| c.amplitude()).collect::<Vec<f32>>(),
            &[Caption(plot.1)],
        );
    }
    fg.show()?;

    Ok(())
}

fn plot_figure_time_domain(
    plots: Vec<(&Vec<Complex<f32>>, &str)>,
) -> Result<(), gnuplot::GnuplotInitError> {
    let mut fg = Figure::new();
    let ax = fg
        .axes2d()
        .set_title("Time domain", &[])
        .set_legend(Graph(0.5), Graph(0.9), &[], &[])
        .set_x_label("time", &[])
        .set_y_label("amplitude", &[]);
    for plot in plots {
        ax.lines(
            &(0..plot.0.len())
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>(),
            &plot.0.iter().map(|c| c.re).collect::<Vec<f32>>(),
            &[Caption(plot.1)],
        );
    }
    fg.show()?;

    Ok(())
}

fn write_to_file(data: &Vec<Vec<f32>>, filename: &String) -> Result<String, std::io::Error> {
    let text: String = data
        .iter()
        .map(|x| {
            x.iter()
                .map(|f| f.to_string())
                .collect::<Vec<String>>()
                .join(",")
        })
        .collect::<Vec<String>>()
        .join("\n");
    let mut file = File::create(&filename)?;
    file.write_all(text.as_bytes())?;

    Ok("Data written".into())
}

fn run_fft(input: &mut Vec<Complex<f32>>, invert: bool) -> Vec<Complex<f32>> {
    let mut input = input;
    let samples = input.len();
    let mut output: Vec<Complex<f32>> = vec![Complex::zero(); samples];
    let mut planner = FFTplanner::new(invert);
    let fft = planner.plan_fft(samples);
    fft.process(&mut input, &mut output);
    if invert {
        output.iter().map(|c| c / (samples as f32)).collect()
    } else {
        output.iter().map(|c| c * 2f32 / (samples as f32)).collect()
    }
}

fn zeros(size: u32) -> Vec<Complex<f32>> {
    vec![Complex::zero(); size as usize]
}

fn combine(a: Vec<Complex<f32>>, b: Vec<Complex<f32>>) -> Vec<Complex<f32>> {
    let mut x = a.clone();
    x.extend(&b);
    x
}

fn create_function(
    number_of_samples: u32,
    function: &dyn Fn(f32, &Vec<f32>, &Vec<f32>) -> f32,
    harmonics: &Vec<f32>,
    amplitudes: &Vec<f32>,
) -> Vec<Complex<f32>> {
    (0..number_of_samples)
        .into_iter()
        .map(|x| {
            Complex::new(
                function(x as f32 / number_of_samples as f32, &harmonics, &amplitudes),
                0f32,
            )
        })
        .collect()
}

fn sine(x: f32, harmonics: &Vec<f32>, amplitudes: &Vec<f32>) -> f32 {
    harmonics
        .iter()
        .zip(amplitudes.iter())
        .map(|(&h, &a)| a * f32::sin(h * PI * 2f32 * x))
        .sum()
}

fn square(x: f32, blockes: &Vec<f32>, amplitudes: &Vec<f32>) -> f32 {
    blockes
        .iter()
        .zip(amplitudes.iter())
        .map(|(&h, &a)| {
            if x % (1f32 / h) < 1f32 / (2f32 * h) {
                a
            } else {
                -1f32 * a
            }
        })
        .sum()
}

fn triangle(x: f32, counts: &Vec<f32>, amplitudes: &Vec<f32>) -> f32 {
    counts
        .iter()
        .zip(amplitudes.iter())
        .map(|(&h, &a)| {
            if x % (1f32 / h) < 1f32 / (2f32 * h) {
                a * (x % (1f32 / h)) * 2f32 * h
            } else {
                (1f32 / (h) - (x % (1f32 / h))) * 2f32 * h * a
            }
        })
        .sum()
}

fn sawtooth(x: f32, counts: &Vec<f32>, amplitudes: &Vec<f32>) -> f32 {
    counts
        .iter()
        .zip(amplitudes.iter())
        .map(|(&h, &a)| a * (x % (1f32 / h)) * h)
        .sum()
}

fn parse_arguments(input: &Vec<String>) -> Result<Tasks, ArgsError> {
    let mut args = Args::new(PROGRAM_NAME, PROGRAM_DESC);
    args.flag("h", "help", "Print the usage menu");
    args.option(
        "p",
        "datapoints",
        "The number of points to be generated. Value must be between 1 and 10e5",
        "POINT",
        Occur::Optional,
        Some(String::from("2048")),
    );
    args.flag(
        "i",
        "invert",
        "Reconstruct the original function by using the invert fft",
    );
    args.flag("s", "silent", "Do not show the plot windows");
    args.option(
        "k",
        "keeptop",
        "Keep n top harmonics",
        "KEEPTOP",
        Occur::Optional,
        None,
    );
    args.option(
        "m",
        "keepmin",
        "Keep harmonics where the amplitude is larger than",
        "KEEPMIN",
        Occur::Optional,
        None,
    );
    args.flag("w", "write", "Write the result to the given file");
    args.option(
        "",
        "filename",
        "Filename of the result file",
        "NAME",
        Occur::Optional,
        Some(String::from("result.csv")),
    );
    args.option(
        "t",
        "wavetype",
        "Select a wave type: sine, block, triangle, sawtooth, custom",
        "WAVETYPE",
        Occur::Optional,
        Some(String::from("sine")),
    );
    args.option(
        "d",
        "data",
        "Data for the wave",
        "DATA",
        Occur::Optional,
        Some(String::from("1,2,4,8;8,4,2,1")),
    );

    (args.parse(input))?;

    let help = args.value_of("help")?;
    if help {
        return Ok(Tasks::Help(args.full_usage()));
    }

    let mut wavetype = String::from("sine");
    if args.has_value("wavetype") {
        wavetype = args.value_of("wavetype")?;
    }

    let wavedata = parse_data(args.value_of("data")?);
    let function;
    if wavetype == "square" {
        function = Function::Square(wavedata[0].clone(), wavedata[1].clone());
    } else if wavetype == "triangle" {
        function = Function::Triangle(wavedata[0].clone(), wavedata[1].clone());
    } else if wavetype == "sawtooth" {
        function = Function::Sawtooth(wavedata[0].clone(), wavedata[1].clone());
    } else if wavetype == "custom" {
        function = Function::Custom(wavedata[0].clone());
    } else {
        function = Function::Sine(wavedata[0].clone(), wavedata[1].clone())
    }

    let gt_0 = Box::new(OrderValidation::new(Order::GreaterThan, 0u32));
    let lt_10e5 = Box::new(OrderValidation::new(Order::LessThanOrEqual, 100_000u32));

    let mut datapoints = if wavetype == "custom" {
        wavedata[0].len() as u32
    } else {
        2048
    };
    if args.has_value("datapoints") {
        datapoints = args.validated_value_of("datapoints", &[gt_0, lt_10e5])?;
    }

    let invert = args.value_of("invert")?;
    let silent: bool = args.value_of("silent")?;

    let gt_0 = Box::new(OrderValidation::new(Order::GreaterThan, 0u32));
    let lt_dp = Box::new(OrderValidation::new(Order::LessThanOrEqual, datapoints));

    let mut number_of_harmonics = datapoints;
    if args.has_value("keeptop") {
        number_of_harmonics = args.validated_value_of("keeptop", &[gt_0, lt_dp])?;
    }

    let gt_0_f32 = Box::new(OrderValidation::new(Order::GreaterThan, 0f32));
    let lt_10e5_f32 = Box::new(OrderValidation::new(Order::LessThanOrEqual, 10e5f32));

    let mut min_harmonics_amplitude = 0f32;
    if args.has_value("keepmin") {
        min_harmonics_amplitude = args.validated_value_of("keepmin", &[gt_0_f32, lt_10e5_f32])?;
    }

    let mut filename = "result.csv".into();
    let save_to_file: bool = args.value_of("write")?;
    if args.has_value("filename") {
        filename = args.value_of("filename")?;
    }

    Ok(Tasks::Handle(Task {
        number_of_points: datapoints,
        invert: invert,
        plot: !silent,
        save_to_file: save_to_file,
        filename: filename,
        number_of_harmonics: number_of_harmonics,
        min_harmonics_amplitude: min_harmonics_amplitude,
        function: function,
    }))
}

fn parse_data(data: String) -> Vec<Vec<f32>> {
    data.split(";")
        .into_iter()
        .map(|x| {
            String::from(x)
                .split(",")
                .filter_map(|s| s.parse::<f32>().ok())
                .collect()
        })
        .collect()
}
