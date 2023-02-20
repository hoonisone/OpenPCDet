import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    """
        - training 수행에 필요한 파라미터 입력을 정리한다.
        - model.yaml에도 동일한 정보가 있을 수 있다.
        - 이 경우 파라미터를 1순위로 사용하며 입력이 없으면 yaml파일의 내용을 사용
    """
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    # multi-GPU를 사용하는 경우
    # gpu는 하나의 pc에 여러개가 있을 수 있고
    # pc또한 여러대일 수 있다.
    # local rank: 하나의 pc안에서 gpu번호
    # global rank: 전체 pc에 대한 gpu번호
    # 이때 각 번호는 우선순위에 해당하는 것 같음 (뭐 할 때 우선순위가 쓰이는지는 모르겠다.)

    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    # mix precision은 모델 학습시 FP(floating point - 부동소수점)을 32가 아닌 16 사용
    # 속도도 빠르고 성능은 그대로 또는 더 좋다고..

    args = parser.parse_args()
    # 여기까지는 기본적인 argparse이용법
    # args = parser.parse_args()를 하면
    # args.속성 으로 입력받은 속성에 접근 가능

    cfg_from_yaml_file(args.cfg_file, cfg)
    # 여기서 cfg는 앞으로 config정보를 저장할 빈 EasyDict 객체
    # cfg_file은 필수 입력사항인 것 같다.
    # 라이브러리를 살펴 보면 model별로 config.yaml파일이 있다.
    # 예로 pointrcnn모델의 경우 pointrcnn.yaml이 있다.
    # cfg_file의 내용을 모두 cfg객체에 옮겨넣는다.

    cfg.TAG = Path(args.cfg_file).stem
    # Path.stem은 경로 마지막에 접미사(.extension)을 제외한 마지막 부분 => 즉 파일 이름부분?? 을 반환
    # 이때 cfg_file이 보통 모델명.yaml이기 때문에 여기서 stem 즉 cfg.TAG는 모델명에 해당한다.

    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    # 라이브러리에서는 cfg_file 경로 예시로
    # --cfg_file cfgs/kitti_models/pv_rcnn.yaml
    # 이런식으로 입력
    # 위 줄을 수행하면 kitti_models만 남는다.
    # 앞 뒤로 cfgs/ 와 /pv_rcnn.yaml 을 제거하는 것
    # 즉 group-path로 모델명?? 을 이용하려는 듯


    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)
    # mixed precision을 사용할 것인가?
    # args와 cfg 둘 다 확인하여 체크


    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
        # args를 통해 기존의 cfg의 내용을 수정할 사항이 입력되면
        # cfg를 덮어씌운다.

    return args, cfg
    # args: 사용자 입력 (para)
    # cfg: 파일에 입력 (yaml)


def main():
    # [인자 정리]
    args, cfg = parse_config()
        # 설정 파일과 설정 파라미터로부터 설정 값들을 모두 정리
        # args: 파라미터를 통해 얻은 설정들
        # cfg: 파일입력으로 얻은 설정들 (파라미터와 중복되는 경우 파라미터값으로 덮어씌워짐)


    # [분산 학습 설정]
    if args.launcher == 'none':
        # launcher 분산 학습을 결정하는 인자?
        # launcher로는 ['none', 'pytorch', 'slurm']가 있다.
        # multi gpu 사용을 제공하는 라이브러리가 pytorch와 slurm이 있는게 아닐까?

        dist_train = False # distributed training(분산 학습) 안함
        total_gpus = 1 # 전체 gpu 사용 개수?
            # 나중에 배치사이즈가 gpu개수에 비례하는지 체크할 것이기 때문에 1로 설정해준다.

    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
            # common_tuils는 파일 이름
            # 해당 파일에 여러 함수가 있고 getattr로 원하는 함수를 지정하여 호출한다.
        dist_train = True
            # 분산학습 학습 함

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
            # 배치 사이즈가 따로 명시되지 않으면 파일에 있는 gpu당 배치 사이즈로 설정함
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
            # gpu개수에 비례해야 한다.

        args.batch_size = args.batch_size // total_gpus
            # 전체 배치 사이즈를 gpu에 균등 분할
            # 입력한 값은 전체 배치를 말하는 것이고
            # 앞으로 args의 batch_size는 gpu당 batch를 말함


    # [에폭 설정]
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
        # args와 cfg를 확인하여 epoch 결정


    # [랜덤 씨드 설정]
    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)
            # seed 기능이 있는 모든 라이브러리에 대해 동일한 seed로 초기화
            # seed는 기본적으로 고정하는 듯
            # 단 gpu별로 조금 변화를 주는 것 같다.


    # [저장 경로 설정]
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
        # 모델의 최종 저장 및 check point 저장 경로
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
        # 출력 및 체크포인트 경로 생성
        # parents: 부모가 없으면 만들고 계속 진행할 것인가?
        # exist_ok: 경로가 이미 있으면 에러 없이 종료할 것인가?


    # 로그 설정
    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
        # 로거 경로 계산 및 로거 생성


    # 로깅 - 환경 정보
    logger.info('**********************Start logging**********************')
    cuda_key = 'CUDA_VISIBLE_DEVICES'
    gpu_list = os.environ[cuda_key] if cuda_key in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)


    # 로깅 - 분산 학습 여부
    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')


    # 로깅 - 파리미터
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
         # 파라미터 로깅


    # 로깅 - 로그 설정 정보
    log_config_to_file(cfg, logger=logger)


    # 경로 이동
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    # SummaryWriter 설정
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
        # 학습과정을 요약해서 보여주는 객체
        # pytorch.tensorboard에서 제공되는

    logger.info("----------- Create dataloader & network & optimizer -----------")

    # data loader 설정
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )
    # 파일과 파라미터에 설정된대로 데이터 로더를 생성 및 세팅
    # dataset, data_loader, data_sampler가 한 번에 반환됨
    # data_loader에는 사실 dataset과 data_sampler가 모두 인자로 등록되어있음

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    # 모델 정보와 데이터 셋 정보에 맞는 모델을 생성하여 반환한다.
    # 모델은 데이터 셋에 종속이라고 이해??..

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
              
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
