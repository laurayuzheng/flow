"""Plot results from ray-based simulations.

This method accepts as input the progress file generated by ray
(usually stored at ~/ray_results/.../progress.csv)
as well as the column(s) to be plotted.

If no column is specified, all existing columns will be printed.

Example usage
-----
::
    python plot_ray_results.py </path/to/file>.csv mean_reward max_reward
"""

import csv
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


EXAMPLE_USAGE = 'plot_ray_results.py ' + \
    '~/ray_results/experiment-tag/experiment-name/seed-id/progress.csv ' + \
    'evaluation/return-average training/return-average'


def plot_progress(filepath, columns):
    """Plot ray results from a csv file.

    Plot the values contained in the csv file at <filepath> for each column
    in the list of string columns.
    """
    data = defaultdict(list)
    print(columns)

    with open(filepath) as f:
        # if columns list is empty, print a list of all columns and return
        if not columns:
            reader = csv.reader(f)
            print('Columns are: ' + ', '.join(next(reader)))
            return

        try:
            reader = csv.DictReader(f)
            for row in reader:
                for col in columns:
                    data[col].append(float(row[col]))
        except KeyError:
            print('Error: {} was called with an unknown column name "{}".\n'
                  'Run "python {} {}" to get a list of all the existing '
                  'columns'.format(__file__, col, __file__, filepath))
            raise
        except ValueError:
            print('Error: {} was called with an invalid column name "{}".\n'
                  'This column contains values that are not convertible to '
                  'floats.'.format(__file__, col))
            raise

    plt.ion()
    for col_name, values in data.items():
        plt.plot(values, label=col_name)
    plt.legend()
    plt.savefig('progress.png')
    # plt.show()

def plot_progress_mbpo(filepath1, filepath2, columns, scenario):
    """Plot ray results from a csv file.

    Plot the values contained in the csv file at <filepath> for each column
    in the list of string columns.
    """
    data = defaultdict(list)

    with open(filepath1) as f:
        # if columns list is empty, print a list of all columns and return
        if not columns:
            reader = csv.reader(f)
            print('Columns are: ' + ', '.join(next(reader)))
            return

        try:
            reader = csv.DictReader(f)
            for row in reader:
                for col in columns:
                    if col in row:
                        data["PPO"].append(float(row[col]))
        except KeyError:
            print('Error: {} was called with an unknown column name "{}".\n'
                  'Run "python {} {}" to get a list of all the existing '
                  'columns'.format(__file__, col, __file__, filepath1))
            raise
        except ValueError:
            print('Error: {} was called with an invalid column name "{}".\n'
                  'This column contains values that are not convertible to '
                  'floats.'.format(__file__, col))
            raise

    with open(filepath2) as f:
        # if columns list is empty, print a list of all columns and return
        if not columns:
            reader = csv.reader(f)
            print('Columns are: ' + ', '.join(next(reader)))
            return

        try:
            reader = csv.DictReader(f)
            for row in reader:
                for col in columns:
                    if col in row:
                        data["MBPO"].append(float(row[col]))
        except KeyError:
            print('Error: {} was called with an unknown column name "{}".\n'
                  'Run "python {} {}" to get a list of all the existing '
                  'columns'.format(__file__, col, __file__, filepath2))
            raise
        except ValueError:
            print('Error: {} was called with an invalid column name "{}".\n'
                  'This column contains values that are not convertible to '
                  'floats.'.format(__file__, col))
            raise

    plt.ion()
    for col_name, values in data.items():
        plt.plot(values, label=col_name)
    plt.legend()
    plt.xlim([0, 50])
    plt.title("Baseline " + scenario + " Scenario")
    plt.xlabel('Training Iteration')
    plt.ylabel('Average Reward')
    plt.savefig('imgs/' + scenario.lower() + '0.png')
    # plt.show()


def create_parser():
    """Parse visualization options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Plots progress.csv file generated by ray.',
        epilog='Example usage:\n\t' + EXAMPLE_USAGE)

    parser.add_argument('file', type=str, help='Path to the csv file.')
    parser.add_argument(
        'columns', type=str, nargs='*', help='Names of the columns to plot.')

    return parser


if __name__ == '__main__':
    # parser = create_parser()
    # args = parser.parse_args()
    # plot_progress(args.file, args.columns)

    # file1 = "results/ppo_bench/merge_0/progress.csv"
    # file2 = "results/benchmarks0/merge0/progress.csv"
    file1 = "results/ppo_bench/figure_eight_0/progress.csv"
    file2 = "results/benchmarks0/figureeight0/progress.csv"
    # file1 = "results/ppo_bench/grid_0/progress.csv"
    # file2 = "results/benchmarks0/grid0/progress.csv"
    # file1 = "results/ppo_bench/bottleneck_0/progress.csv"
    # file2 = "results/benchmarks0/bottleneck0/progress.csv"

    columns = ['episode_reward_mean', 'training/return-average']
    scenario = "FigureEight"
    plot_progress_mbpo(file1, file2, columns, scenario)
