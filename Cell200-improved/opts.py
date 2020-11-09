import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--GAN', type=str, default='CcGAN', choices=['cGAN', 'CcGAN'])
    parser.add_argument('--show_real_imgs', action='store_true', default=False) #output a grid of real images for all 300 unique cell counts
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')


    ''' Dataset '''
    parser.add_argument('--num_imgs_per_count', type=int, default=10, metavar='N',
                        help='number of images for each cell count')
    parser.add_argument('--start_count', type=int, default=1, metavar='N')
    parser.add_argument('--end_count', type=int, default=200, metavar='N')
    parser.add_argument('--stepsize_count', type=int, default=2, metavar='N')
    parser.add_argument('--num_channels', type=int, default=1, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N')


    ''' GAN settings '''
    # label embedding setting
    parser.add_argument('--net_embed', type=str, default='ResNet34_embed') #ResNetXX_emebed
    parser.add_argument('--epoch_cnn_embed', type=int, default=400) #epoch of cnn training for label embedding
    parser.add_argument('--resumeepoch_cnn_embed', type=int, default=0) #epoch of cnn training for label embedding
    parser.add_argument('--epoch_net_y2h', type=int, default=500)
    parser.add_argument('--dim_embed', type=int, default=128) #dimension of the embedding space
    parser.add_argument('--batch_size_embed', type=int, default=128, metavar='N')

    parser.add_argument('--loss_type_gan', type=str, default='vanilla')
    parser.add_argument('--niters_gan', type=int, default=10000, help='number of iterations')
    parser.add_argument('--resume_niters_gan', type=int, default=0)
    parser.add_argument('--save_niters_freq', type=int, default=2000, help='frequency of saving checkpoints')
    parser.add_argument('--lr_g_gan', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--lr_d_gan', type=float, default=1e-4, help='learning rate for discriminator')
    parser.add_argument('--dim_gan', type=int, default=128, help='Latent dimension of GAN')
    parser.add_argument('--batch_size_disc', type=int, default=64)
    parser.add_argument('--batch_size_gene', type=int, default=64)
    parser.add_argument('--transform', action='store_true', default=False,
                        help='rotate or flip images for GAN training')
    parser.add_argument('--cGAN_num_classes', type=int, default=20, metavar='N') #bin label into cGAN_num_classes

    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--kappa', type=float, default=-1)
    parser.add_argument('--nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')


    ''' Sampling and Evaluation '''
    parser.add_argument('--samp_batch_size', type=int, default=1000)
    parser.add_argument('--nfake_per_label', type=int, default=1000)
    parser.add_argument('--comp_FID', action='store_true', default=False)
    parser.add_argument('--epoch_FID_CNN', type=int, default=200)
    parser.add_argument('--FID_radius', type=int, default=5)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)

    args = parser.parse_args()

    return args
