
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


def plotter(file, args):
    """Plot data from a file.

    Args:
        file (string or Path): Path to file
        args (argparse struct): Arguments from argparse
    """
    fig, ax = plt.subplots()
    data = np.genfromtxt(file, delimiter=args.delimiter)

    # Check columns
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

    # Main plot loop
    for counter, y in enumerate(args.ycols):
        label = args.ylabels[counter] if (args.ylabels and len(args.ylabels)>1) else None
        ax.plot(data[:, args.xcol], data[:, y], label=label)

    ax.legend() if (args.ylabels and len(args.ylabels)>1) else None

    if args.savename:
        plt.savefig(args.savename, dpi=args.dpi)
        log_message(f"Saved figure as {args.savename}")
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
    parser.add_argument('file', type=str, nargs="+", help='CSV file path')
    # Columns
    parser.add_argument('--xcol', type=int, default=0, required=False, help='X column name')
    parser.add_argument('--ycols', type=int, nargs='*', default=[1], required=False, help='Y column names')
    # Labels + Title
    parser.add_argument('--xlabel', type=str, default="X", required=False, help='X axis label')
    parser.add_argument('--ylabels', type=str, default=["Y"], nargs="+", required=False, help='Y axis labels')
    parser.add_argument('--title', type=str, default=None, required=False, help='Plot title')
    # Scales
    parser.add_argument('--xlog', action='store_true', help='Use logarithmic scale for x-axis')
    parser.add_argument('--ylog', action='store_true', help='Use logarithmic scale for y-axis')
    # Limits
    parser.add_argument('--xlim', type=float, nargs=2, default=None, required=False, help='X axis limits')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, required=False, help='Y axis limits')
    # Save
    parser.add_argument('--savename', type=str, default=None, required=False, help='Save figure as PNG')
    # Misc
    parser.add_argument('--dpi', type=int, default=500, required=False, help='DPI for PNG')
    parser.add_argument('--delimiter', type=str, default=",", required=False, help='File delimiter')

    # Parse arguments
    args = parser.parse_args()

    # Check files
    for file in args.file:
        if not file_exists(file):
            log_message(f"File {file} does not exist", level="error", print_message=True)
            sys.exit(1)

    # Check columns
    if args.xcol < 0:
        log_message(f"xcol must be >= 0", level="error", print_message=True)
        sys.exit(1)
    for y in args.ycols:
        if y < 0:
            log_message(f"ycol must be >= 0", level="error", print_message=True)
            sys.exit(1)

    # Fix savename
    if args.savename:
        args.savename = Path(args.savename).with_suffix(".png")

    # Fix ylabels
    if args.ylabels and len(args.ylabels) != len(args.ycols):
        if len(args.ylabels) == 1:
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

    log_message(f"Arguments: {args}", level="debug")

    for file in args.file:
        plotter(file, args)

    log_message("Finished plotter")


if __name__ == "__main__":
    main()
