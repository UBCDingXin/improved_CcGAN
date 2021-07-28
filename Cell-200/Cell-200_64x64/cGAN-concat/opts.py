import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='/home/xin/OneDrive/Working_directory/CcGAN/Cell200')
    parser.add_argument('--data_path', type=str, default='/home/xin/OneDrive/Working_directory/CcGAN/dataset/Cell200')
    parser.add_argument('--GAN', type=str, default='cGAN-concat')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')


    ''' Dataset '''
    parser.add_argument('--num_imgs_per_count', type=int, default=10, metavar='N',
                        help='number of images for each cell count')
    parser.add_argument('--start_count', type=int, default=1, metavar='N')
    parser.add_argument('--end_count', type=int, default=200, metavar='N')
    parser.add_argument('--stepsize_count', type=int, default=2, metavar='N')
    parser.add_argument('--num_channels', type=int, default=1, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N')
    parser.add_argument('--show_real_imgs', action='store_true', default=False)
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)


    ''' GAN settings '''
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

    ''' Sampling and Evaluation '''
    parser.add_argument('--samp_batch_size', type=int, default=1000)
    parser.add_argument('--nfake_per_label', type=int, default=1000)
    parser.add_argument('--comp_FID', action='store_true', default=False)
    parser.add_argument('--epoch_FID_CNN', type=int, default=200)
    parser.add_argument('--FID_radius', type=int, default=5)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)

    args = parser.parse_args()

    return args
