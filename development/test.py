import argparse
from functions import cpu_sim, gpu_sim, batch_sim_gpu

def required_args(parser):
    required = parser.add_argument_group('required arguments')
    required.add_argument("--N", metavar="Nsites",help='number of lattice sizes', required=True, type=int, default=7)
    required.add_argument("--R", help='Subsystem size', required=True, type=int, default=4)
    required.add_argument("--L", metavar="lambda", help='coupling parameter (for odd number of Nsites)', required=True, type=float, default=2.5)
    required.add_argument("--MC", metavar="MCsteps", help='Number of Monte Carlo steps', required=True, type=int, default=1000)
    #required.add_argument("-S", metavar="S", help='Number of Monte Carlo steps saving', required=True, type=int)
    return required

def exclusive_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--in_eigen", metavar='filename', help='File with the eigenvectors computed by the same program', type=str, default=None)
    group.add_argument("--save_eigen", help='Calculate initial eigenvetors and eigenstate', action='store_true', default=False)
    group.add_argument("--resume", help='Resume s/imulation from saved states - T/F', action='store_true', default=False)

def handle_GPU_args(subparser):
    parser = subparser.add_parser("GPU", help="GPU MPI version")
    requred_options = required_args(parser)
    group = requred_options.add_mutually_exclusive_group(required=True)
    group.add_argument("--in_eigen", metavar='filename', help='File with the eigenvectors computed by the same program', type=str)
    group.add_argument("--resume", help='Resume simulation from saved states - T/F', action='store_true')
    parser.add_argument("--o", metavar="filename", help='Name of the file to write raw outputs, default Renyi_entropy_raw_N_R_L_MCsteps_M.bin', type=str, default="Renyi_entropy_raw_{}_{}_{}_{}_{}.bin")

def handle_CPU_args(subparser):
    parser = subparser.add_parser("CPU", help="CPU MPI version")
    requred_options = required_args(parser)
    exclusive_args(parser)
    parser.add_argument("--o", metavar="filename", help='Name of the file to write raw outputs, default Renyi_entropy_raw_N_R_L_MCsteps_M.bin', type=str, default="Renyi_entropy_raw_{}_{}_{}_{}_{}.bin")

def handle_batch_args(subparser):
    parser = subparser.add_parser("batch", help="Batch GPU version with array of jobs")
    requred_options = required_args(parser)
    requred_options.add_argument("-s", "--sim_per_batch", metavar="", help='Number of simulations per batch job', type=int, required=True)
    requred_options.add_argument("-b", "--batch_job_id", metavar="", help='array job id parameter', type=int, required=True )
    group = requred_options.add_mutually_exclusive_group(required=True)
    group.add_argument("--in_eigen", metavar='filename', help='File with the eigenvectors computed by the same program', type=str)
    group.add_argument("--resume", help='Resume simulation from saved states - T/F', action='store_true')
    parser.add_argument("--o", metavar="filename", help='Name of the file to write raw outputs, default Renyi_entropy_raw_N_R_L_MCsteps_M.bin', type=str, default="Renyi_entropy_raw_{}_{}_{}_{}_{}.bin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MPI version of computing the Renyi2 entropy via Monte Carlo simulations')
    subparser = parser.add_subparsers(help="Modes of execution of prog", required=True, dest="mode")
    handle_GPU_args(subparser)
    handle_CPU_args(subparser)
    handle_batch_args(subparser)
    args = parser.parse_args()

    if (args.mode == 'GPU'):
       gpu_sim(args)

    if (args.mode == 'CPU'):
       cpu_sim(args)

    if (args.mode == 'batch'):
       batch_sim_gpu(args)






