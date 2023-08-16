import argparse
import tasks

def modify_command_options(opts):
    if opts.dataset == 'voc':
        opts.num_classes = 21
    elif opts.dataset == 'coco':
        opts.num_classes = 80

    if not opts.visualize:
        opts.sample_num = 0

    if opts.dataset == "coco-voc":
        opts.backbone = 'wider_resnet38_a2'
        opts.output_stride = 8
        opts.crop_size = 448
        opts.crop_size_val = 512
    
    opts.use_DeeplabV3_as_seg_branch = True
    opts.branch = 'ins'
    if opts.phase == 1:
        opts.branch = 'none'
        opts.flac = True
        opts.randrop = True
    if opts.phase == 2:
        opts.freeze = True
        opts.freeze_seg = True

    opts.no_overlap = not opts.overlap
    opts.pooling = opts.crop_size // opts.output_stride

    opts.lr_head = 1. if opts.step == 0 else opts.lr_head

    return opts

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def get_argparser():
    parser = argparse.ArgumentParser()

    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help='number of workers (default: 1)')
    parser.add_argument("--device", type=int, default=None,
                        help='Device ID')

    # Datset Options
    parser.add_argument("--data_root", type=str, default='data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', help='Name of dataset')
    parser.add_argument("--weakly", default=False, action='store_true')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")

    # Train Options
    parser.add_argument("--epochs", type=int, default=30,
                        help="epoch number (default: 30)")

    parser.add_argument("--batch_size", type=int, default=24,
                        help='batch size (default: 24)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size (default: 512)")
    parser.add_argument("--crop_size_val", type=int, default=512,
                        help="crop size (default: 512)")
    parser.add_argument("--optim", type=str, default='adabelief', choices=['sgd', 'adam', 'adabelief', 'adamw'],
                        help="optimizer (defalut: adabelief)")

    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument("--lr_policy", type=str, default='poly',
                        choices=['poly', 'step', 'none', 'warmup', 'one_cycle'],
                        help="lr schedule policy (default: poly)")
    parser.add_argument("--lr_decay_step", type=int, default=5000,
                        help="decay step for stepLR (default: 5000)")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="decay factor for stepLR (default: 0.1)")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")
    parser.add_argument("--bce", default=False, action='store_true',
                        help="Whether to use BCE or not (default: no)")
    parser.add_argument("--dce", default=False, action='store_true',
                        help="Whether to use DeepLabCE or not (default: no)")

    # Validation Options
    parser.add_argument("--val_on_trainset", action='store_true', default=False,
                        help="enable validation on train set (default: False)")
    parser.add_argument("--crop_val", action='store_false', default=True,
                        help='do crop for validation (default: True)')

    # Logging Options
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="path to Log directory (default: ./logs)")
    parser.add_argument("--name", type=str, default='Experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--sample_num", type=int, default=8,
                        help='number of samples for visualization (default: 0)')
    parser.add_argument("--debug", action='store_true', default=False,
                        help="verbose option")
    parser.add_argument("--visualize", action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 1)")

    # Model Options
    parser.add_argument("--model", type=str, default='PanopticDeepLab',
                        choices=['PanopticDeepLab'],
                        help='model to use (def: PanopticDeepLab)')
    parser.add_argument("--backbone", type=str, default='resnet101',
                        choices=['resnet50', 'resnet101', 'wider_resnet38_a2'],
                        help='backbone for the body (def: resnet50)')
    parser.add_argument("--output_stride", type=int, default=16,
                        choices=[8, 16], help='stride for the backbone (def: 16)')
    parser.add_argument("--no_pretrained", action='store_true', default=False,
                        help='Wheather to use pretrained or not (def: True)')
    parser.add_argument("--norm_act", type=str, default="iabn_sync",
                        help='Which BN to use (def: abn_sync')
    parser.add_argument("--pooling", type=int, default=32,
                        help='pooling in ASPP for the validation phase (def: 32)')

    # Test and Checkpoint options
    parser.add_argument("--test", action='store_true', default=False,
                        help="Whether to train or test only (def: train and test)")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--continue_ckpt", default=False, action='store_true',
                        help="Restart from the ckpt. Named taken automatically from method name.")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="epoch interval for saving model (default: 1)")
    parser.add_argument("--seg_ckpt", default=None, type=str,
                        help="path to trained seg model. Leave it None if you want to retrain your model")

    # Parameters for Knowledge Distillation of ILTSS (https://arxiv.org/abs/1907.13372)
    parser.add_argument("--freeze", action='store_true', default=False,
                        help="Use this to freeze the feature extractor in incremental steps")
    parser.add_argument("--freeze_seg", action='store_true', default=False,
                        help="Use this to freeze the feature extractor in incremental steps")
    parser.add_argument("--loss_de", type=float, default=0.,  # Distillation on Encoder
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable distillation on Encoder (L2)")
    parser.add_argument("--loss_kd", type=float, default=0.,  # Distillation on Output
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable Knowlesge Distillation (Soft-CrossEntropy)")

    # Arguments for ICaRL (from https://arxiv.org/abs/1611.07725)
    parser.add_argument("--icarl", default=False, action='store_true',
                        help="If enable ICaRL or not (def is not)")
    parser.add_argument("--icarl_importance", type=float, default=1.,
                        help="the regularization importance in ICaRL (def is 1.)")
    parser.add_argument("--icarl_disjoint", action='store_true', default=False,
                        help="Which version of icarl is to use (def: combined)")
    parser.add_argument("--icarl_bkg", type=float, default=-1,
                        help="Background interpolation (1 is new gt)")

    # METHODS
    parser.add_argument("--init_balanced", default=False, action='store_true',
                        help="Enable Background-based initialization for new classes")
    parser.add_argument("--unkd", default=False, action='store_true',
                        help="Enable Unbiased Knowledge Distillation instead of Knowledge Distillation")
    parser.add_argument("--unce", default=False, action='store_true',
                        help="Enable Unbiased Cross Entropy instead of CrossEntropy")

    # Incremental parameters
    parser.add_argument("--task", type=str, default="19-1", choices=tasks.get_task_list(),
                        help="Task to be executed (default: 19-1)")
    parser.add_argument("--step", type=int, default=0,
                        help="The incremental step in execution (default: 0)")
    parser.add_argument("--no_mask", action='store_true', default=False,
                        help="Use this to not mask the old classes in new training set")
    parser.add_argument("--overlap", action='store_true', default=False,
                        help="Use this to not use the new classes in the old training set")
    parser.add_argument("--step_ckpt", default=None, type=str,
                        help="path to trained model at previous step. Leave it None if you want to use def path")
    parser.add_argument("--phase", type=int, default=None,
                        help="select phase for incremental steps")

    # Weakly supervised Pars
    parser.add_argument("--pseudo", default=None, type=str,
                        help="Pseudo labels for steps>0")
    parser.add_argument("--pl_ckpt", default=None, type=str,
                        help="path to pseudolabeler")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="The parameter to hard-ify the soft-labels. Def is 1.")
    parser.add_argument("--pos_w", type=float, default=1.,
                        help="Positive weight")
    parser.add_argument("--affinity", action='store_true', default=False,
                        help="Use affinity on CAM")
    parser.add_argument("--affinity_method", type=str, default='pamr',
                        choices=['pamr'],
                        help="affinity method for CAM")
    parser.add_argument("--pseudo_ep", default=5, type=int,
                        help="When to start pseudolabeling")
    parser.add_argument("--lr_pseudo", default=0.01, type=float,
                        help="learning rate pseudolabeler")
    parser.add_argument("--lr_head", default=10., type=float,
                        help="learning rate pseudolabeler")
    parser.add_argument("--cam", default="ngwp", type=str,
                        help="CAM model used")
    parser.add_argument("--ss_dist", action='store_true', default=False,
                        help="Dist on bkg prior")
    parser.add_argument("--l_seg", type=float, default=1)
    
    # panoptic deeplab
    parser.add_argument("--val_thresh", type=float, default=0.1, help='threhsold for instance-groupping in validation phase')
    parser.add_argument("--val_kernel", type=int, default=41, help='kernsl size for point extraction in validation phase')
    parser.add_argument('--val_flip', type=str2bool, default=False, help='enable flip test-time augmentation in vadliation phase')
    parser.add_argument('--val_clean', type=str2bool, default=False, help='cleaning pseudo-labels using image-level labels')
    parser.add_argument('--val_ignore', type=str2bool, default=False, help='ignore')
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help='threshold for pseudo-label generation')
    parser.add_argument("--refine_thresh", type=float, default=0.3, help='threshold for refined-label generation')
    parser.add_argument("--kernel", type=int, default=41, help='kernel size for point extraction')
    parser.add_argument("--sigma", type=int, default=6, help='sigma of 2D gaussian kernel')
    parser.add_argument("--beta", type=float, default=3.0, help='parameter for center-clustering')
    parser.add_argument('--detach_instance', action='store_true', default=False, help='detach instance')
    parser.add_argument("--run_refine", type=str2bool, default=True, help='parameter for pseudo-label refinement')
    
    # pg
    parser.add_argument("--pam_alpha", type=int, default=0.7, help='alpha for pam')
    parser.add_argument("--peak_from", type=str, default='peakgenerator', help='module to generate peak')
    
    # branch
    parser.add_argument('--branch', type=str, default='all', choices=['all', 'seg', 'ins', 'none'], help='select branch to be used in panoptic-deeplab')
    parser.add_argument('--use_DeeplabV3_as_seg_branch', action='store_true', default=False, help='use DeeplabV3 as semantic branch in panoptic-deeplab')

    # CL for WSSS
    parser.add_argument('--flac', action='store_true', default=False)
    parser.add_argument('--randrop', action='store_true', default=False)
    
    return parser
