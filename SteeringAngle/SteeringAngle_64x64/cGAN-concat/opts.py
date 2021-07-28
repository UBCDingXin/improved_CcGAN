import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--eval_ckpt_path', type=str, default="")
    parser.add_argument('--GAN', type=str, default='CcGAN', choices=['cGAN', 'CcGAN'])
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Dataset '''
    parser.add_argument('--min_label', type=float, default=-80.0)
    parser.add_argument('--max_label', type=float, default=80.0)
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N')
    parser.add_argument('--max_num_img_per_label', type=int, default=9999, metavar='N')
    parser.add_argument('--max_num_img_per_label_after_replica', type=int, default=50, metavar='N')
    parser.add_argument('--show_real_imgs', action='store_true', default=False)
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)


    ''' GAN settings '''
    # label embedding setting
    parser.add_argument('--GAN_arch', type=str, default='SNGAN', choices=['SNGAN','DCGAN'])
    parser.add_argument('--niters_gan', type=int, default=20000, help='number of iterations')
    parser.add_argument('--resume_niters_gan', type=int, default=0)
    parser.add_argument('--save_niters_freq', type=int, default=2000, help='frequency of saving checkpoints')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate for discriminator')
    parser.add_argument('--dim_z', type=int, default=128, help='Latent dimension of GAN')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--visualize_freq', type=int, default=2000, help='frequency of visualization')


    ''' Sampling and Evaluation '''
    parser.add_argument('--samp_batch_size', type=int, default=500)
    parser.add_argument('--nfake_per_label', type=int, default=50)
    parser.add_argument('--comp_FID', action='store_true', default=False)
    parser.add_argument('--epoch_FID_CNN', type=int, default=200)
    parser.add_argument('--num_eval_labels', type=int, default=2000)
    parser.add_argument('--FID_radius', type=int, default=2)
    parser.add_argument('--FID_num_centers', type=int, default=1000)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)

    args = parser.parse_args()

    return args
