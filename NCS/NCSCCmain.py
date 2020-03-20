from src.optimizers import OpenAIOptimizer, CanonicalESOptimizer, CanonicalESMeanOptimizer, NCSOptimizer,NCSCCOptimizer
from src.policy import Policy
from src.logger import Logger

from argparse import ArgumentParser
from mpi4py import MPI
import numpy as np
import time
import json
import gym


# This will allow us to create optimizer based on the string value from the configuration file.
# Add you optimizers to this dictionary.
optimizer_dict = {
    'OpenAIOptimizer': OpenAIOptimizer,
    'CanonicalESOptimizer': CanonicalESOptimizer,
    'CanonicalESMeanOptimizer': CanonicalESMeanOptimizer,
    'NCSOptimizer':NCSOptimizer,
    'NCSCCOptimizer':NCSCCOptimizer
}


# Main function that executes training loop.
# Population size is derived from the number of CPUs
# and the number of episodes per CPU.
# One CPU (id: 0) is used to evaluate currently proposed
# solution in each iteration.
# run_name comes useful when the same hyperparameters
# are evaluated multiple times.
def main(ep_per_cpu, game, configuration_file, run_name):
    start_time = time.time()

    with open(configuration_file, 'r') as f:
        configuration = json.loads(f.read())

    env_name = '%sNoFrameskip-v4' % game

    # MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    # One cpu (rank 0) will evaluate results
    train_cpus = cpus - 1
    k = 10
    epoch = 5
    m = 1

    # Deduce population size
    lam = train_cpus * ep_per_cpu

    # Create environment
    env = gym.make(env_name)

    # Create policy (Deep Neural Network)
    # Internally it applies preprocessing to the environment state
    policy = Policy(env, network=configuration['network'], nonlin_name=configuration['nonlin_name'])

    # Create reference batch used for normalization
    # It will be overwritten with vb from worker with rank 0
    vb = policy.get_vb()

    # Extract vector with current parameters.
    #parameter have length of 1693380
    parameters = policy.get_parameters()
    shape = policy.parameter_shapes

    # Send parameters from worker 0 to all workers (MPI stuff)
    # to ensure that every worker starts in the same position

    #comm.Bcast([parameters, MPI.FLOAT], root=0)
    comm.Bcast([vb, MPI.FLOAT], root=0)

    # Create optimizer with user defined settings (hyperparameters)
    OptimizerClass = optimizer_dict[configuration['optimizer']]
    optimizer = OptimizerClass(train_cpus,parameters, shape,lam, rank, configuration["settings"],epoch,m)


    # Set the same virtual batch for each worker
    if rank != 0:
        policy.set_vb(vb)
        rews = [0]
        e_r = 0
        p = optimizer.get_parameters()
        policy.set_parameters(p)
        for j in range(k):
            e_rew, e_len = policy.rollout()
            e_r += e_rew
        rews[0] = e_r/k
        optimizer.rew = e_r/k
        msg = np.array(rews)
        pp = p
    # Only rank 0 worker will log information from the training
    logger = Logger(optimizer.log_path(game, configuration['network'], run_name))
    if rank == 0:
        # Initialize logger, save virtual batch and save some basic stuff at the beginning
        logger.save_vb(vb)
        logger.log('Game'.ljust(25) + '%s' % game)
        logger.log('Network'.ljust(25) + '%s' % configuration['network'])
        logger.log('Optimizer'.ljust(25) + '%s' % configuration['optimizer'])
        logger.log('Number of CPUs'.ljust(25) + '%d' % cpus)
        logger.log('Population'.ljust(25) + '%d' % lam)
        logger.log('Dimensionality'.ljust(25) + '%d' % len(parameters))

        # Log basic info from the optimizer
        # optimizer.log_basic(logger)
        msg = np.zeros(1)
        pp = np.zeros(optimizer.n)
    results = np.empty((cpus, 1))
    ppp = np.empty((cpus,optimizer.n))
    comm.Allgather([msg, MPI.FLOAT], [results, MPI.FLOAT])
    comm.Allgather([pp, MPI.FLOAT], [ppp, MPI.FLOAT])
    results = results[1:, :]
    ppp = ppp[1:,:].flatten()
    rews = results[:, :1].flatten()
    BestScore = max(rews)
    Bestid = np.argmax(rews)
    BestFound = ppp[Bestid * optimizer.n:(Bestid + 1) * optimizer.n]
    if rank == 0:
        logger.log('Best'.ljust(25) + '%f' %BestScore )

    # We will count number of steps
    # frames = 4 * steps
    steps_passed = 0
    iteration =1
    while steps_passed<=25000000:
        # Iteration start time
        iter_start_time = time.time()
        if iteration % epoch == 1:
            optimizer.sigupdatelist= np.zeros(optimizer.n)
        llambda = np.random.normal(1,0.1-0.1*steps_passed/25000000)
        # Workers that run train episodes
        optimizer.RandomGrouping()
        for ii in range(m):
            optimizer.groupnum = ii
            if rank != 0:
                # Empty arrays for each episode. We save: length, reward, noise index
                lens1 = [0]
                rews1 = [0]
                orew = [0]
                # sig1 = [0]
                # For each episode in this CPU we get new parameters,
                # update policy network and perform policy rollout
                e_r = 0
                e_l = 0
                p = optimizer.get_parameters1()
                policy.set_parameters(p)
                for j in range(k):
                    e_rew, e_len = policy.rollout()
                    e_r += e_rew
                    e_l += e_len
                lens1[0] = e_l
                rews1[0] = e_r/k
                optimizer.rew1 = e_r/k
                orew[0] = optimizer.rew
                if iteration % epoch == 1:
                    sig1 = optimizer.sigmalist
                # Aggregate information, will later send it to each worker using MPI
                msg1 = np.array(rews1 + lens1 +orew,dtype = np.float64)
                pp1 = optimizer.parameters1
                if iteration % epoch == 1:
                    sigmsg1 = sig1
            # Worker rank 0 that runs evaluation episodes
            else:
                # Empty array, evaluation results are not used for the update
                msg1 = np.zeros(3 ,dtype = np.float64)
                pp1 = optimizer.parameters
                if iteration % epoch == 1:
                    sigmsg1 = np.zeros(optimizer.n)
            # MPI stuff
            # Initialize array which will be updated with information from all workers using MPI
            results1 = np.empty((cpus, 3 ),dtype = np.float64)
            ppp1 = np.empty((cpus, optimizer.n ))
            if iteration % epoch == 1:
                sigmsgs1 = np.empty((cpus, optimizer.n ))
            comm.Allgather([msg1, MPI.FLOAT], [results1, MPI.FLOAT])
            comm.Allgather([pp1, MPI.FLOAT], [ppp1, MPI.FLOAT])
            if iteration % epoch == 1:
                comm.Allgather([sigmsg1, MPI.FLOAT], [sigmsgs1, MPI.FLOAT])
            ppp1 = ppp1[1:, :].flatten()
            if iteration % epoch == 1:
                sigmsgs1 = sigmsgs1[1:, :].flatten()
            # Skip empty evaluation results from worker with id 0
            results1 = results1[1:, :]
            # Extract IDs and rewards
            rews1 = results1[:, :1].flatten()
            lens1 = results1[:, 1:2].flatten()
            oreward = results1[:, 2:].flatten()
            newBestidx = np.argmax(rews1)

            if np.max(rews1)>BestScore:
                BestScore = rews1[newBestidx]
                BestFound = ppp1[newBestidx*optimizer.n:(newBestidx+1)*optimizer.n]
            #uodate parameters, sigmas, rews
            if rank!=0:
                optimizer.update(ppp,BestScore,sigmsgs1,llambda)
            # Steps passed = Sum of episode steps from all offsprings
            steps = np.sum(lens1)
            steps_passed += steps
        # Write some logs for this iteration
        # Using logs we are able to recover solution saved
        # after 1 hour of training or after 1 billion frames
        if rank == 0:
            eval_mean_rew = np.mean(oreward)
            eval_mean_rew1 = np.mean(rews1)
            iteration_time = (time.time() - iter_start_time)
            time_elapsed = (time.time() - start_time)/60
            logger.log('------------------------------------')
            logger.log('Iteration'.ljust(25) + '%f' % iteration)
            logger.log('EvalMeanReward'.ljust(25) + '%f' % eval_mean_rew)
            logger.log('EvalMeanReward1'.ljust(25) + '%f' % eval_mean_rew1)
            logger.log('StepsThisIter'.ljust(25) + '%f' % steps)
            logger.log('StepsSinceStart'.ljust(25)+'%f' %steps_passed)
            logger.log('IterationTime'.ljust(25) + '%f' % iteration_time)
            logger.log('TimeSinceStart'.ljust(25) + '%f' %time_elapsed)
            logger.log('Best'.ljust(25) + '%f' %BestScore)
            # Give optimizer a chance to log its own stuff
            # optimizer.log(logger)
            logger.log('------------------------------------')
            if iteration % 20 == 1:
                fin_rews = 0
                p = BestFound
                policy.set_parameters(p)
                for i in range(30):
                    e_rew, e_len = policy.rollout()
                    fin_rews+=e_rew
                fin_eval = fin_rews/30
            else:
                fin_eval = 0
            # Write stuff for training curve plot
            stat_string = "{},\t{},\t{},\t{}\n".\
                format(steps_passed, (time.time()-start_time),
                        eval_mean_rew1,  fin_eval)
            logger.write_general_stat(stat_string)
            logger.write_optimizer_stat(optimizer.stat_string())
            # Save currently proposed solution every 20 iterations
            if iteration % 20 == 1:
                logger.save_parameters(BestFound, iteration)
        else:
            if iteration%epoch ==0:
                optimizer.updatesigma()
        iteration+=1
    #test best
    if rank == 0:
        final_rews = []
        p = BestFound
        policy.set_parameters(p)
        for i in range(200):
            e_rew, e_len = policy.rollout()
            final_rews.append(e_rew)
        final_eval = np.mean(final_rews)
        logger.log('Final'.ljust(25) + '%f' % final_eval)
        logger.save_parameters(BestFound,iteration)
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-e', '--episodes_per_cpu',
                        help="Number of episode evaluations for each CPU, "
                             "population_size = episodes_per_cpu * Number of CPUs",
                        default=1, type=int)
    parser.add_argument('-g', '--game', help="Atari Game used to train an agent")
    parser.add_argument('-c', '--configuration_file', help='Path to configuration file')
    parser.add_argument('-r', '--run_name', help='Name of the run, used to create log folder name', type=str)
    args = parser.parse_args()
    return args.episodes_per_cpu, args.game, args.configuration_file, args.run_name
if __name__ == '__main__':
    ep_per_cpu, game, configuration_file, run_name = parse_arguments()
    main(ep_per_cpu, game, configuration_file, run_name)
