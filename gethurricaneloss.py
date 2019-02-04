# Invoke a simple model of hurricane losses in Florida and the Gulf states and
# return the expected annual economic loss.
# v1.0
# Hassan Chagani 2019-02-04

import argparse
import functools
import numpy as np
import time

def parse_arguments():
    """parse_arguments()
    Checks command line arguments for validity. Requires number of years to
    simulate, annual landfall rates in Florida and the Gulf states, and mean
    and standard deviation for LogNormal distributions that describe economic
    losses of landfalling hurricanes in Florida and the Gulf states. Optional
    parameters include help and verbose flags that print instructions and mean
    loss every 100,000 simulation years respectively.
    Returns parsed arguments."""

    description = "Estimate mean financial loss per year as a result of " +\
                  "landfalling hurricanes in Florida and the Gulf states."
    parser = argparse.ArgumentParser(description=description)

    # Required argument for number of samples to run
    num_monte_carlo_samples_help = "Number of samples (i.e. simulation years) to run"
    req_named_arg = parser.add_argument_group("required named arguments")
    req_named_arg.add_argument("-n", "--num_monte_carlo_samples",\
                               help=num_monte_carlo_samples_help,\
                               required=True, dest="n_samples", type=int)

    # Positional arguments for model parameters
    annual_rate_text = "The annual rate of landfalling hurricanes in "
    lognormal_text = " of LogNormal distribution describing economic loss " +\
                     "of a landfalling hurricane in "

    parser.add_argument("florida_landfall_rate", type=float,\
                        help=annual_rate_text+"Florida")
    parser.add_argument("florida_mean", type=float,\
                        help="Mean "+lognormal_text+" Florida")
    parser.add_argument("florida_stddev", type=float,\
                        help="Standard deviation "+lognormal_text+" Florida")
    parser.add_argument("gulf_landfall_rate", type=float,\
                        help=annual_rate_text+"the Gulf states")
    parser.add_argument("gulf_mean", type=float,\
                        help="Mean "+lognormal_text+" the Gulf states")
    parser.add_argument("gulf_stddev", type=float,\
                        help="Standard deviation "+lognormal_text+" the Gulf states")

    # Optional argument to print mean loss every 100,000 simulation years
    parser.add_argument("-v", "--verbose", action="store_true",\
                        help="Print mean loss every 100,000 simulation years")

    # Parse arguments and check whether number of samples and model parameters
    # are positive
    args = parser.parse_args()
    for arg, value in vars(args).items():

        if type(value) != bool:
        
            # If negative, print error message and exit
            if value <= 0:
                parser.error("argument {}: must be positive".format(arg))

    return args   # Return parsed arguments


def gen_poisson_samples(n, landfall_rate):
    """gen_poisson_samples(n, landfall_rate)
    Returns generator for sampling from Poisson distribution."""

    return (np.random.poisson(landfall_rate) for _ in range(n))


def gen_lognormal_samples(n, mean, stddev):
    """gen_lognormal_samples(n, mean, stddev)
    Returns generator for sampling from LogNormal distribution."""

    return (np.random.lognormal(mean, stddev) for _ in range(n))


def lognormal_to_normal(lognormal_mean, lognormal_stddev):
    """lognormal_to_normal(lognormal_mean, lognormal_stddev)
    Convert mean and standard deviation that describe LogNormal distribution to
    mean and standard deviation of underlying normal distribution.
    Returns mean and standard deviation of underlying normal distribution."""

    normal_std = np.sqrt(np.log(1 + (lognormal_stddev/lognormal_mean)**2))
    normal_mean = np.log(lognormal_mean) - normal_std**2 / 2

    return normal_mean, normal_std


def timer(fun):
    """timer(fun)
    Calculate and print run time of decorated function."""

    @functools.wraps(fun)
    
    def timer_wrapper(*args, **kwargs):

        start_time = time.perf_counter()
        value = fun(*args, **kwargs)
        end_time = time.perf_counter()
        delta_time = end_time - start_time
        print("Run time for {} = {:.4f} s".format(fun.__name__, delta_time))
        return value

    return timer_wrapper


@timer
def run_simulation(args):
    """run_simulation(args)
    Calculates financial loss per year. Landfall rates are sampled from Poisson
    distribution. Financial loss for each landfalled hurricane is sampled from
    LogNormal distribution. Financial losses for each year are summed and
    divided by number of simulated years to calculate mean loss.
    Returns mean loss."""

    mean_loss = 0

    # Obtain generators to yield number of landfalling hurricanes in Florida
    # and the gulf states
    florida_landfall_rate_model = gen_poisson_samples(args.n_samples,\
                                                      args.florida_landfall_rate)
    gulf_landfall_rate_model = gen_poisson_samples(args.n_samples,\
                                                   args.gulf_landfall_rate)

    # Convert LogNormal distributions means and standard deviations to those of
    # their underlying normal distributions
    florida_normal_mean, florida_normal_stddev = lognormal_to_normal(args.florida_mean,\
                                                                     args.florida_stddev)
    gulf_normal_mean, gulf_normal_stddev = lognormal_to_normal(args.gulf_mean,\
                                                               args.gulf_stddev)

    # Loop over simulation years
    for year in range(args.n_samples):

        simulation_loss = 0   # Loss during current year

        # Determine total loss during current year in Florida
        n_florida_events = next(florida_landfall_rate_model)
        florida_loss_model = gen_lognormal_samples(n_florida_events,\
                                                   florida_normal_mean,\
                                                   florida_normal_stddev)
        for sample in florida_loss_model:
            simulation_loss += sample

        # Determine total loss during current year in Gulf states
        n_gulf_events = next(gulf_landfall_rate_model)
        gulf_loss_model = gen_lognormal_samples(n_gulf_events,\
                                                gulf_normal_mean,\
                                                gulf_normal_stddev)
        for sample in gulf_loss_model:
            simulation_loss += sample
        
        # Determine mean loss over all years
        mean_loss += simulation_loss / args.n_samples

        # Print mean loss every 100,000 years if verbose flag given
        if args.verbose == True:

            if year % 100000 == 0 and year != 0:

                mean_loss_sample = mean_loss * args.n_samples / year
                print("Year {}: mean loss = {:.2f} per year".format(year, mean_loss_sample))

    return mean_loss   # Return mean financial loss


if __name__ == "__main__":

    # Check correct arguments given on command line
    args = parse_arguments()

    # Set random seed and calculate mean loss
    np.random.seed(None)
    mean_loss = run_simulation(args)

    # Print mean loss
    print("Mean loss = {:.2f} per year".format(mean_loss))
