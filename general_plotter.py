
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
import argparse
from pathlib import Path
import logging

# Get the directory path of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_directory, "log.txt")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the desired log level
    format='%(asctime)s [%(levelname)s] %(message)s',  # Define log message format
    datefmt='%Y-%m-%d %H:%M:%S',  # Define date and time format
    filename=log_file,  # Specify the log file name
    filemode='w+'  # Set the file mode to overwrite the log file on each run
)

# Define the logging function
def log_message(message, level='info', print_message=False):
    log_functions = {
        'info': logging.info,
        'debug': logging.debug,
        'warning': logging.warning,
        'error': logging.error,
        'critical': logging.critical
    }
    log_func = log_functions.get(level.lower())
    if log_func:
        log_func(message)
    else:
        raise ValueError(f"Invalid log level: {level}")

    if print_message:
        print(message)


def file_exists(filename):
    """Check if a file exists.

    Args:
        filename (string or Path): Path to file

    Returns:
        bool: True if file exists, False otherwise
    """
    rval = os.path.isfile(filename)
    log_message(f"File {filename} exists: {rval}")
    return rval


def plotter(file, args, plot_counter):
    """Plot data from a file.

    Args:
        file (string or Path): Path to file
        args (argparse struct): Arguments from argparse
    """
    fig, ax = plt.subplots()
    data = np.genfromtxt(file, delimiter=args.delimiter)

    # Check columns
    if args.ycols and type(args.ycols[0]) == str:
        args.ycols = [i for i in range(0, data.shape[1]) if i != args.xcol]

    if args.xcol >= data.shape[1]:
        log_message(f"xcol must be < {data.shape[1]}", level="error", print_message=True)
        sys.exit(1)
    for y in args.ycols:
        if y >= data.shape[1]:
            log_message(f"ycol must be < {data.shape[1]}", level="error", print_message=True)
            sys.exit(1)

    ax.set_xscale("log") if args.xlog else None
    ax.set_yscale("log") if args.ylog else None
    ax.set_xlabel(args.xlabel) if args.xlabel else None
    ax.set_ylabel(args.ylabels[0]) if (args.ylabels and len(args.ylabels)==1) else None
    ax.set_title(args.title) if args.title else None
    ax.set_xlim(args.xlim) if args.xlim else None
    ax.set_ylim(args.ylim) if args.ylim else None
    for h in args.hlines:
        ax.axhline(h, color="black", linestyle="--")
    for v in args.vlines:
        ax.axvline(v, color="black", linestyle="--")
    ax.grid(args.grid, which="both", axis="both", linestyle="--", linewidth=0.5, color="black", alpha=0.5)


    # Main plot loop
    for counter, y in enumerate(args.ycols):
        label = args.ylabels[counter] if (args.ylabels and len(args.ylabels)>1) else None
        ax.plot(data[:, args.xcol], data[:, y], label=label)

    ax.legend() if (args.ylabels and len(args.ylabels)>1) else None

    if args.savenames is not None:
        plt.savefig(args.savenames[plot_counter], dpi=args.dpi)
        log_message(f"Saved figure as {args.savenames[plot_counter]}")
    else:
        plt.show()
        log_message("Showing figure")


def parser():
    """Parse arguments from command line.

    Returns:
        argparse struct: Arguments from argparse
    """
    parser = argparse.ArgumentParser(description='Plot data from a CSV file.')

    # File
    parser.add_argument('files', metavar="file(s)", type=str, nargs="+", help='CSV file path(s)')
    # Columns
    parser.add_argument('--xcol', type=int, default=0, required=False, help='X column name')
    parser.add_argument('--ycol(s)', dest="ycols", metavar=("ycol", "ycols"), type=str, nargs='+', default="all", required=False, help='Y column name(s)')
    parser.add_argument('--ycol', dest="ycols", type=str, nargs='+', default="all", required=False, help=argparse.SUPPRESS)
    parser.add_argument('--ycols', dest="ycols", type=str, nargs='+', default="all", required=False, help=argparse.SUPPRESS)
    # Labels + Title
    parser.add_argument('--xlabel', type=str, default="X", required=False, help='X axis label')
    parser.add_argument('--ylabel(s)', dest="ylabels", metavar=("ylabel", "ylabels"), type=str, default=["Y"], nargs="+", required=False, help='Y axis label(s)')
    parser.add_argument('--ylabel', dest="ylabels", type=str, default=["Y"], nargs="+", required=False, help=argparse.SUPPRESS)
    parser.add_argument('--ylabels', dest="ylabels", type=str, default=["Y"], nargs="+", required=False, help=argparse.SUPPRESS)
    parser.add_argument('--title', type=str, default=None, required=False, help='Plot title')
    # Scales
    parser.add_argument('--xlog', action='store_true', help='Use logarithmic scale for x-axis')
    parser.add_argument('--ylog', action='store_true', help='Use logarithmic scale for y-axis')
    # Limits
    parser.add_argument('--xlim', type=float, nargs=2, default=None, required=False, help='X axis limits')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, required=False, help='Y axis limits')
    # hlines + vlines
    parser.add_argument('--hline(s)', type=float, dest="hlines", metavar=("hline", "hlines"), nargs='+', default=[], required=False, help='Horizontal line(s)')
    parser.add_argument('--hline', type=float, dest="hlines", nargs='*', default=[], required=False, help=argparse.SUPPRESS)
    parser.add_argument('--hlines', type=float, dest="hlines", nargs='*', default=[], required=False, help=argparse.SUPPRESS)
    parser.add_argument('--vline(s)', type=float, dest="vlines", metavar=("vline", "vlines"), nargs='+', default=[], required=False, help='Vertical line(s)')
    parser.add_argument('--vline', type=float, dest="vlines",  nargs='*', default=[], required=False, help=argparse.SUPPRESS)
    parser.add_argument('--vlines', type=float, dest="vlines",  nargs='*', default=[], required=False, help=argparse.SUPPRESS)
    # grid
    parser.add_argument('--grid', action='store_true', help='Show grid')
    # Save
    parser.add_argument('--savename(s)', dest="savenames", metavar=("savename", "savenames"), type=str, nargs='+', default=None, required=False, help='Save figure(s) as PNG')
    parser.add_argument('--savename', dest="savenames", type=str, nargs='+', default=None, required=False, help=argparse.SUPPRESS)
    parser.add_argument('--savenames', dest="savenames", type=str, nargs='+', default=None, required=False, help=argparse.SUPPRESS)
    # Misc
    parser.add_argument('--dpi', type=int, default=500, required=False, help='DPI for PNG')
    parser.add_argument('--delimiter', type=str, default=",", required=False, help='File delimiter')

    # Parse arguments
    args = parser.parse_args()

    # Check files
    for file in args.files:
        if not file_exists(file):
            log_message(f"File {file} does not exist", level="error", print_message=True)
            sys.exit(1)

    # Check columns
    if args.xcol < 0:
        log_message(f"xcol must be >= 0", level="error", print_message=True)
        sys.exit(1)
    if args.ycols[0].lower() != "all":
        for i, y in enumerate(args.ycols):
            args.ycols[i] = int(y)
            y = args.ycols[i]
            if y < 0:
                log_message(f"ycol must be >= 0", level="error", print_message=True)
                sys.exit(1)

    # Fix savename
    if args.savenames is not None:
        if len(args.savenames) != len(args.files):
            if len(args.savenames) == 1:
                args.savenames = [f"{args.savenames[0]}_{i}" for i in range(len(args.files))]

        for i, sname in enumerate(args.savenames):
            args.savenames[i] = Path(sname).with_suffix(".png")

    # Fix ylabels
    if type(args.ycols[0]) != str and args.ylabels and len(args.ylabels) != len(args.ycols):
        if len(args.ycols) == 1:
            args.ylabels = [" ".join(args.ylabels),]
        else:
            args.ylabels = [l.strip() for l in " ".join(args.ylabels).split(",")]
            if len(args.ylabels) != len(args.ycols):
                log_message(f"Number of ylabels must match number of ycols", level="error", print_message=True)
                sys.exit(1)

    return args


def main():
    """Main function."""
    log_message("Starting plotter")

    # Parse arguments
    args = parser()

    log_message(f"Arguments: {args}", level="info")

    for counter, file in enumerate(args.files):
        plotter(file, args, counter)

    log_message("Finished plotter")


if __name__ == "__main__":
    main()
