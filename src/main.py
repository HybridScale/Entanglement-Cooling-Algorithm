import argparse
from setsimulation import set_simulations, Simulation

def required_args(parser):
    required = parser.add_argument_group('required arguments')
    required.add_argument("--N", metavar="Nsites",help='number of lattice sizes', required=True, type=int, default=7)
    required.add_argument("--R", help='Subsystem size', required=True, type=int, default=4)
    required.add_argument("--L", metavar="lambda", help='coupling parameter (for odd number of Nsites)', required=True, type=float, default=2.5)
    required.add_argument("--MC", metavar="MCsteps", help='Number of Monte Carlo steps', required=True, type=int, default=1000)

def exclusive_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--in_eigen", metavar='filename', help='File with the eigenvectors computed by the same program', type=str, default=None)
    group.add_argument("--save_eigen", help='Calculate initial eigenvetors and eigenstate', action='store_true', default=False)

def optional_args(parser):
    parser.add_argument("-a", "--array_job_id", metavar="", help='array job id parameter', type=int, default=0 )
    parser.add_argument("-w", "--MC_wanted", metavar="", help='MC steps wanted', type=int, default=0 )
    parser.add_argument("--o", metavar="filename", help="Name of the file to write raw outputs, default Renyi_entropy_raw_N_R_L_MCsteps_M.bin", type=str, default="Renyi_entropy_raw_{}_{}_{}_{}_{}.bin")
    parser.add_argument("-f", metavar="filename", help="Folder default saved_states_N_R_L", type=str, default=None)



def resumesubparser(subparser):
    subsubparser = subparser.add_subparsers(title="Resume simulation or begin anew", required=True, dest="resume")

    resume = subsubparser.add_parser("resume", help="resume simulation")
    new    = subsubparser.add_parser("new", help="new simulation")

    resume.add_argument("configfolder", metavar="filename", help="saved configuration folder")
    resume.add_argument("-a", "--array_job_id", metavar="", help='array job id parameter', type=int, default=0 )


    return resume, new


def handle_GPU_args(subparser):
    parserGPU = subparser.add_parser("GPU", help="GPU MPI version")

    GPUresume, GPUnew = resumesubparser(parserGPU)
    required_args(GPUnew)
    exclusive_args(GPUnew)
    optional_args(GPUnew)

def handle_CPU_args(subparser):
    parserCPU = subparser.add_parser("CPU", help="CPU MPI version")

    CPUresume, CPUnew = resumesubparser(parserCPU)
    required_args(CPUnew)
    exclusive_args(CPUnew)
    optional_args(CPUnew)
    

def handle_batchGEMM_args(subparser):
    parserBG = subparser.add_parser("batchedGEMM", help="GPU batchedGEMM version")
    requred_options = required_args(parserBG)
    exclusive_args(parserBG)
    optional_args(parserBG)
    #requred_options.add_argument("--bs", metavar="batch_size", help='Number of Monte Carlo conncurent on single GPU', required=True, type=int, default=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MPI version of computing the Renyi2 entropy by Monte Carlo simulations')
    subparser = parser.add_subparsers(help="Modes of execution of simulation", required=True, dest="mode")

    ## functions to handle subparser and it's msg
    handle_GPU_args(subparser)
    handle_CPU_args(subparser)
    #handle_batchGEMM_args(subparser)

    args = parser.parse_args()
    print(args)
    simulation = Simulation(args)
    simulation.start()

