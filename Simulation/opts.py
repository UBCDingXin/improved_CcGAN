import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--GAN', type=str, default='CcGAN',
                        choices=['cGAN', 'CcGAN'])
    parser.add_argument('--nsim', type=int, default=3,
                        help = "How many times does this experiment need to be repeated?")
    parser.add_argument('--seed', type=int, default=2020, metavar='S',
                        help='random seed')
    parser.add_argument('--root_path', type=str, default='')

    ''' Data Generation '''
    parser.add_argument('--n_gaussians', type=int, default=120,
                        help = "Number of Gaussians (number of angles's).") #half of them will be used for training and the rest are for testing
    parser.add_argument('--n_samp_per_gaussian_train', type=int, default=10) # n_gaussians*n_rsamp_per_gaussian = ntrain
    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--sigma_gaussian', type=float, default=0.02)


    ''' GAN settings '''
    parser.add_argument('--niters_gan', type=int, default=10000,
                        help='number of iterations')
    parser.add_argument('--resume_niters_gan', type=int, default=0)
    parser.add_argument('--save_niters_freq', type=int, default=1000, help='frequency of saving checkpoints')
    parser.add_argument('--lr_gan', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--dim_gan', type=int, default=2,
                        help='Latent dimension of GAN')
    parser.add_argument('--batch_size_disc', type=int, default=512,
                        help='input batch size for training discriminator')
    parser.add_argument('--batch_size_gene', type=int, default=512,
                        help='input batch size for training generator')

    parser.add_argument('--threshold_type', type=str, default='hard',
                        choices=['soft', 'hard'])
    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--kappa', type=float, default=-1.0)
    parser.add_argument('--nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL')


    ''' Sampling and Evaluation '''
    parser.add_argument('--eval', action='store_true', default=False) #evaluation fake samples
    parser.add_argument('--n_gaussians_eval', type=int, default=360) #number of labels for evaluation
    parser.add_argument('--n_samp_per_gaussian_eval', type=int, default=100) # number of fake samples for each Gaussian
    parser.add_argument('--samp_batch_size_eval', type=int, default=100)


    ''' Visualization '''
    parser.add_argument('--n_gaussians_plot', type=int, default=12) #number of unseen labels for plotting
    parser.add_argument('--n_samp_per_gaussian_plot', type=int, default=100) # number of fake samples for each Gaussian
    parser.add_argument('--samp_batch_size_plot', type=int, default=100)


    args = parser.parse_args()

    return args
