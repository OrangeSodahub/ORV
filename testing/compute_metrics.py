import os
import fnmatch
import concurrent
import numpy as np
import cv2
import torch
import random
from tqdm import tqdm
from scipy.linalg import sqrtm
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.transforms import transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_fid.inception import InceptionV3
from concurrent.futures import ThreadPoolExecutor
from decord import VideoReader, cpu

from training.utils import CONSOLE


def read_video_decord(path):
    vr = VideoReader(path, ctx=cpu(0))
    frames = vr.get_batch(range(len(vr)))  # (N, H, W, C)
    frames = frames.asnumpy().astype(np.float32) / 255.  # -> [0, 1]

    # get the specified view!!!
    # ----------------------------------------------------------------
    # NOTE: we hard code here and assume the total view is 3
    if args.view >= 0 and frames.shape[2] > 480:
        _, _, W, _ = frames.shape
        w = W // 3
        frames = frames[:, :, w * args.view : w * (args.view + 1)]
    # ----------------------------------------------------------------

    return frames


def _compute_psnr_ssim(pair, disable_tqdm = False):
    gt_path, pred_path = pair
    if args.view >= 0:
        gt_id = os.path.basename(gt_path).removesuffix(f'_{args.view}.mp4')
    else:
        gt_id = os.path.basename(gt_path).removesuffix('.mp4')
    pred_id = os.path.basename(pred_path).removesuffix('.mp4').replace('eval_', '', 1)
    # pred_id = os.path.basename(pred_path).removesuffix('.mp4').replace('_eval', '', 1)

    try:
        if gt_id != pred_id:
            raise RuntimeError(f'Mismatched {gt_id=} and {pred_id=}')

        gt_video = read_video_decord(gt_path)
        pred_video = read_video_decord(pred_path)

        # we may drop last several frames in generations.
        min_frames = min(gt_video.shape[0], pred_video.shape[0]) - 1
        gt_video = gt_video[:min_frames]
        pred_video = pred_video[:min_frames]

        psnr_list = []
        ssim_list = []
        for i in tqdm(range(min_frames), desc='traverse frames ...', disable=disable_tqdm, leave=False):
            pred_ = pred_video[i]
            gt_ = gt_video[i]
            pred_ = cv2.resize(pred_, (320, 256), interpolation=cv2.INTER_LINEAR)
            gt_ = cv2.resize(gt_, (320, 256), interpolation=cv2.INTER_LINEAR)
            psnr = peak_signal_noise_ratio(pred_, gt_, data_range=1.)
            ssim = structural_similarity(pred_, gt_, channel_axis=-1, data_range=1.)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        psnr = np.stack(psnr_list).mean()
        ssim = np.stack(ssim_list).mean()

        tqdm.write(f'results of {gt_id}: psnr={psnr:.4f} ssim={ssim:.4f}')
        return gt_id, pred_id, min_frames, psnr, ssim

    except Exception as e:
        CONSOLE.log(f'Skipped [bold red]{pair=}[/] due to {e}!')
        if int(os.getenv('DEBUG', 0)):
            raise
        return gt_id, pred_id, -1, -1, -1


def pair_videos(gt_dir, pred_dir):

    # fetch all videos
    gt_videos = list(sorted(fnmatch.filter(os.listdir(gt_dir), '*.mp4')))
    pred_videos = list(sorted(fnmatch.filter(os.listdir(pred_dir), 'eval*.mp4')))
    CONSOLE.log(f'Found {len(gt_videos)} gt videos and {len(pred_videos)} pred videos.')

    if args.view >= 0:
        gt_videos = list(sorted(fnmatch.filter(gt_videos, f'*_{args.view}.mp4')))
        CONSOLE.log(f'Found {len(gt_videos)} gt videos for view={args.view}')

    # strictly match
    if args.view >= 0:
        gt_videos = list(sorted(filter(lambda gt_video: f"eval_{gt_video.replace(f'_{args.view}.mp4', '.mp4')}" in pred_videos, gt_videos)))
        pred_videos = list(sorted(filter(lambda pred_video: f"{pred_video.removeprefix('eval_').replace('.mp4', f'_{args.view}.mp4')}" in gt_videos, pred_videos)))
    else:
        gt_videos = list(sorted(filter(lambda gt_video: f'eval_{gt_video}' in pred_videos, gt_videos)))
    if len(gt_videos) != len(pred_videos):
        raise RuntimeError(f'Mismatched ground truth and predictions! {gt_videos[:20]=} and {pred_videos[:20]=}.')
    CONSOLE.log(f'Successfully matched {len(gt_videos)} pairs!')

    return pred_videos, gt_videos


def pnsr_ssim(gt_dir, pred_dir, use_single=False, num_workers=8):

    pred_videos, gt_videos = pair_videos(gt_dir, pred_dir)

    # format pairs
    gt_files = list(map(lambda fn: os.path.join(gt_dir, fn), gt_videos))
    pred_files = list(map(lambda fn: os.path.join(pred_dir, fn), pred_videos))
    video_pairs = list(zip(gt_files, pred_files))

    results = []

    if use_single:
        CONSOLE.log('Running in single-process mode...')
        for pair in tqdm(video_pairs, desc='Computing results ...'):
            results.append(_compute_psnr_ssim(pair))
    else:
        CONSOLE.log(f'Running in multi-process mode ...')
        with ThreadPoolExecutor(max_workers=64) as executor:
            future_to_file = {executor.submit(_compute_psnr_ssim, pair, True): pair for pair in video_pairs}
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(video_pairs), desc='Computing resutls ...'):
                result = future.result()
                results.append(result)

    CONSOLE.log(f'Calculation done, now writing files ...')

    # print out average results
    valid_psnr = [r[3] for r in results if r[2] >= 0]
    valid_ssim = [r[4] for r in results if r[3] >= 0]

    avg_psnr = sum(valid_psnr) / len(valid_psnr) if valid_psnr else 0
    avg_ssim = sum(valid_ssim) / len(valid_ssim) if valid_ssim else 0

    CONSOLE.log(f"📊 Average PSNR: {avg_psnr:.4f}")
    CONSOLE.log(f"📊 Average SSIM: {avg_ssim:.4f}")

    results = list(sorted(results, key=lambda x: (x[-2], x[-1]), reverse=True))
    metrics_name = os.path.basename(pred_dir.rstrip('/'))
    with open(f'{metrics_name}.csv', 'w') as f:
        f.write('gt_path,pred_path,psnr,ssim\n')
        for gt_id, pred_id, n_frame, psnr, ssim in tqdm(results, desc='Writing results ...'):
            f.write(f'{gt_id},{pred_id},{psnr:.4f},{ssim:.4f},{n_frame}\n')
    CONSOLE.log(f'✅ Done! Results saved to {metrics_name}.csv')


class VideoPathDataset(torch.utils.data.Dataset):

    def __init__(self, files, num_frames, transforms=None):
        self.files = files
        self.num_frames = num_frames
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        frames = read_video_decord(path)[:self.num_frames]  # (N, H, W, C)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        if self.transforms is not None:
            frames = torch.stack([self.transforms(frame) for frame in frames])
        return frames



def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def _compute_fid(paths, device, batch_size, dims, num_frames):

    def _compute_fid_stats_of_path(files, model, batch_size, dims, device, num_frames, cache_path=None):

        # compute activations
        model.eval()    

        transforms = None
        # transforms = T.Compose([
        #     T.Resize((256,)),
        #     T.CenterCrop((256, 320)),
        # ])

        dataset = VideoPathDataset(files, num_frames=num_frames, transforms=transforms)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=8,
                    )

        activations = np.empty((len(files) * num_frames, dims))

        start_idx = 0

        for batch in tqdm(dataloader):

            b, n, c, h, w = batch.shape
            batch = batch.reshape(b * n, c, h, w).to(device)

            with torch.no_grad():
                pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            activations[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

        # compute statistics
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)

        # save caches
        if cache_path is not None and not os.path.exists(cache_path):
            # np.savez(cache_path, mu=mu, sigma=sigma)
            pass

        return mu, sigma

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    gt_path, pred_path = paths
    fid_cache_path = 'none'
    if os.path.exists(fid_cache_path):
        CONSOLE.log(f'Use cached GT fid stats from {fid_cache_path}')
        gt_stats = np.load(fid_cache_path)
        m1, s1 = gt_stats['mu'][:], gt_stats['sigma'][:]
    else:
        CONSOLE.log(f'Not found fid_cache for GT, will calculate for GT first ...')
        m1, s1 = _compute_fid_stats_of_path(gt_path, model, batch_size, dims, device, num_frames, cache_path=fid_cache_path)
    m2, s2 = _compute_fid_stats_of_path(pred_path, model, batch_size, dims, device, num_frames)

    fid_value = _calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def _frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:

    compute_stats = lambda feats: (feats.mean(axis=0), np.cov(feats, rowvar=False))  # mu, sigma

    mu1, sigma1 = compute_stats(feats_fake)
    mu2, sigma2 = compute_stats(feats_real)

    m = np.square(mu1 - mu2).sum()

    if feats_fake.shape[0] > 1:
        s, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)
        fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    else:
        fid = np.real(m)

    return float(fid)


# https://github.com/universome/fvd-comparison
def load_i3d_pretrained(device=torch.device('cpu')):
    # i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    filepath = "/share/project/cwm/xiuyu.yang/work/dev6/DiffusionAsShader/data/i3d_torchscript.pt"
    i3d = torch.jit.load(filepath).eval().to(device)
    i3d = torch.nn.DataParallel(i3d)
    CONSOLE.log(f'Loaded i3d model from {filepath}.')
    return i3d


def _compute_fvd(paths, device, batch_size, num_frames, num_videos: int = -1):

    # preprocess all tensors
    def _preprocess(files):

        transforms = None
        transforms = T.Compose([
            T.Resize((224,)),
            T.CenterCrop((224, 224)),
        ])

        if num_videos > 0:
            random.seed(42)
            selected_indices = random.sample(range(len(files)), num_videos)
            files = [files[i] for i in selected_indices]

        dataset = VideoPathDataset(files, num_frames=num_frames, transforms=transforms)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=32,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=8,
                    )  # will enhance the loading efficiency greatly!

        all_tensors = torch.cat([frames for frames in tqdm(dataloader, desc='Loaded video data ...')])

        return all_tensors  # [N, T, C, H, W]

    def _get_fvd_feats(videos, model, device):

        process_single = lambda video: (video - .5) * 2  # -> [-1, 1]
        model_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

        feats = np.empty((0, 400))
        with torch.no_grad():
            for i in tqdm(range(0, len(videos), batch_size), desc='Get fvd feats ...', leave=False):
                feats = np.vstack(
                    [
                        feats,
                        model(
                            torch.stack([
                                process_single(video)
                                for video in videos[i : i + batch_size]
                            ]).to(device),
                            **model_kwargs,
                        ).cpu().numpy()
                    ]
                )

        return feats

    gt_path, pred_path = paths
    gt_videos = _preprocess(gt_path).permute(0, 2, 1, 3, 4)  # [N, C, T, H, W]
    pred_videos = _preprocess(pred_path).permute(0, 2, 1, 3, 4)

    # calculate fvd
    i3d = load_i3d_pretrained(device=device)

    fvd_scores = {}
    for n_frame in tqdm(range(16, num_frames + 1), desc='Test different video lengths ...'):

        gt_ = gt_videos[:, :, :n_frame]
        pred_ = pred_videos[:, :, :n_frame]

        feats1 = _get_fvd_feats(gt_, model=i3d, device=device)
        feats2 = _get_fvd_feats(pred_, model=i3d, device=device)

        fvd_scores[n_frame] = _frechet_distance(feats1, feats2)
        CONSOLE.log(f'FVD Score for {n_frame=}: {fvd_scores[n_frame]:.4f}')

    return fvd_scores


def fid_fvd(gt_dir, pred_dir, fid: bool = True, fvd: bool = True):

    # fetch all videos
    gt_videos = list(sorted(fnmatch.filter(os.listdir(gt_dir), '*.mp4')))
    pred_videos = list(sorted(fnmatch.filter(os.listdir(pred_dir), '*eval*.mp4')))
    CONSOLE.log(f'Found {len(gt_videos)} gt videos and {len(pred_videos)} pred videos.')

    # strictly match
    # gt_videos = list(sorted(filter(lambda gt_video: f'eval_{gt_video}' in pred_videos, gt_videos)))
    gt_videos = list(sorted(filter(lambda gt_video: f'{gt_video.rstrip(".mp4")}_eval.mp4' in pred_videos, gt_videos)))
    # pred_videos = list(sorted(filter(lambda pred_video: pred_video.lstrip('eval_') in gt_videos, pred_videos)))
    pred_videos = list(sorted(filter(lambda pred_video: f'{pred_video.rstrip("_eval.mp4")}.mp4' in gt_videos, pred_videos)))
    if len(gt_videos) != len(pred_videos):
        raise RuntimeError(f'Mismatched ground truth and predictions! {gt_videos[:20]=} and {pred_videos[:20]=}.')
    CONSOLE.log(f'Successfully matched {len(gt_videos)} pairs!')

    # format pairs
    gt_files = list(map(lambda fn: os.path.join(gt_dir, fn), gt_videos))
    pred_files = list(map(lambda fn: os.path.join(pred_dir, fn), pred_videos))
    video_pairs = (gt_files, pred_files)

    device = torch.device('cuda')

    if fid:
        dims = 2048
        batch_size = 8  # 128

        fid_score = _compute_fid(video_pairs, device, batch_size, dims, num_frames=16)
        CONSOLE.log(f"📊 FID Score: {fid_score:.4f}")

    if fvd:
        batch_size = 128

        fvd_scores = _compute_fvd(video_pairs, device, batch_size, num_frames=16, num_videos=-1)
        CONSOLE.log(f"📊 FVD Score:")
        for k, v in fvd_scores.items():
            CONSOLE.log(f"{k}: {v:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--view', type=int, default=-1)
    parser.add_argument('--psnr_ssim', action='store_true')
    parser.add_argument('--fid', action='store_true')
    parser.add_argument('--fvd', action='store_true')
    args = parser.parse_args()

    if args.psnr_ssim:
        pnsr_ssim(args.gt_dir, args.pred_dir, use_single=args.single, num_workers=args.workers)

    if args.fid or args.fvd:
        fid_fvd(args.gt_dir, args.pred_dir, fid=args.fid, fvd=args.fvd)