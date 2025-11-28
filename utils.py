import skimage.metrics
import numpy as np
from scipy.linalg import sqrtm

class ImageMetrics:
    """
    图像质量评估工具类，支持 MSE, PSNR, SSIM, LPIPS, FID 等指标。
    支持单张图片评估和批量图片评估。
    """
    def __init__(self, device=None):
        self.device = device
        self._init_torch()

    def _init_torch(self):
        """初始化 PyTorch 相关依赖，用于计算 LPIPS 和 FID"""
        try:
            import torch
            import torchvision
            import lpips
            import torch.nn.functional as F
            self.torch = torch
            self.torchvision = torchvision
            self.lpips_module = lpips
            self.F = F
            self.has_torch = True
            
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"ImageMetrics initialized on {self.device}")
        except ImportError:
            print("Warning: torch/torchvision/lpips not found. Deep learning metrics (FID, LPIPS) disabled.")
            self.has_torch = False
            self.device = 'cpu'

        self._inception = None
        self._lpips_net = None

    @property
    def inception(self):
        """懒加载 InceptionV3 模型"""
        if not self.has_torch: return None
        if self._inception is None:
            print("Loading InceptionV3 model for FID...")
            weights = self.torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            model = self.torchvision.models.inception_v3(weights=weights, transform_input=False).to(self.device)
            model.eval()
            self._inception = model
        return self._inception

    @property
    def lpips_net(self):
        """懒加载 LPIPS 模型"""
        if not self.has_torch: return None
        if self._lpips_net is None:
            print("Loading LPIPS model...")
            # net='alex' 是最常用的配置
            self._lpips_net = self.lpips_module.LPIPS(net='alex').to(self.device)
        return self._lpips_net

    def compute_mse(self, img1, img2):
        """
        计算均方误差 (MSE)。
        MSE 计算两幅图像对应像素差值的平方和的均值。
        MSE 值越小，表示两幅图像越相似。
        """
        return skimage.metrics.mean_squared_error(img1, img2)

    def compute_psnr(self, img1, img2, max_val=1.0):
        """
        计算峰值信噪比 (PSNR)。
        PSNR用于评估一幅图像与原始图像之间的相似度，尤其是在图像压缩和重建领域。
        PSNR的值越高，表示两幅图像之间的相似度越高，质量越好。
        """
        mse = self.compute_mse(img1, img2)
        if mse == 0: return float('inf')
        return 20 * np.log10(max_val) - 10 * np.log10(mse)

    def compute_ssim(self, img1, img2, max_val=1.0):
        """
        计算结构相似性 (SSIM)。
        SSIM（structural similarity）结构相似性，也是一种全参考的图像质量评价指标，它分别从亮度、对比度、结构三方面度量图像相似性。
        SSIM值在-1到1之间，值越大表示两幅图像越相似，通常SSIM值大于0.9表示图像质量较好（try-on一般不大于0.9）。
        """
        return skimage.metrics.structural_similarity(
            img1, img2, 
            win_size=7, 
            channel_axis=-1, 
            data_range=max_val, 
            gaussian_weights=True
        )

    def compute_lpips(self, img1, img2):
        """
        计算 LPIPS 距离。
        LPIPS: 学习感知图像块相似度，用于度量两张图像之间的差别，LPIPS 测量风格化图像和相应内容图像之间的内容保真度。
        它通过比较图像块的深层特征来工作，这些特征能够捕捉到人类视觉系统中对图像质量的感知。
        LPIPS的值越低表示两张图像越相似，反之，则差异越大。
        """
        if not self.has_torch: return None
        # LPIPS 通常使用原始尺寸，但需要归一化到 [-1, 1]
        t1 = self._preprocess(img1, size=None) 
        t2 = self._preprocess(img2, size=None)
        with self.torch.no_grad():
            return self.lpips_net(t1, t2).item()

    def compute_fid(self, real_imgs, gen_imgs):
        """
        计算 FID 分数。
        FID是基于Inception-V3模型（预训练好的图像分类模型）的feature vectors来计算真实图片与生成图片之间的距离，用高斯分布来表示，
        FID就是计算两个分布之间的Wasserstein-2距离。将真实图片和预测图片分别经过Inception模型中，得到2048维度（特征的维度）的embedding vector。
        把生成和真实的图片同时放入Inception-V3中，然后将feature vectors取出来用于比较。
        FID值越低，表示生成的图片与真实图片越接近，质量越高。
        注意：FID计算需要大量的样本来估计分布，因此在实际应用中，通常会使用数百甚至数千张图片来计算FID，以获得更稳定和可靠的结果。
        """
        if not self.has_torch: return None
        print("Extracting features for FID...")
        real_feats = self._extract_features(real_imgs)
        gen_feats = self._extract_features(gen_imgs)
        return self._calculate_fid_from_features(real_feats, gen_feats)

    def _preprocess(self, img, size=None):
        """预处理图像：转 Tensor，调整维度，归一化，可选 Resize"""
        # 如果是 numpy 数组，转为 tensor
        if isinstance(img, np.ndarray):
            img = self.torch.from_numpy(img).float()
        
        # 增加 batch 维度 (H, W, C) -> (1, H, W, C)
        if img.ndim == 3: 
            img = img.unsqueeze(0)
            
        # 调整维度顺序 (N, H, W, C) -> (N, C, H, W)
        if img.shape[-1] == 3: 
            img = img.permute(0, 3, 1, 2)
        
        # Resize (如果指定)
        if size:
            img = self.F.interpolate(img, size=size, mode='bilinear', align_corners=False)
            
        # 归一化: 假设输入是 [0, 1] 或 [0, 255]，统一转为 [-1, 1]
        if img.max() > 1.0: 
            img = img / 255.0
        img = (img - 0.5) * 2.0
        
        return img.to(self.device)

    def _extract_features(self, images):
        """使用 InceptionV3 提取特征"""
        features = []
        batch_size = 32
        
        # 确保输入是列表或数组
        if not isinstance(images, (list, np.ndarray)):
            images = [images]
            
        total = len(images)
        for i in range(0, total, batch_size):
            batch = images[i:i+batch_size]
            # Inception 需要 299x299
            batch_t = self._preprocess(np.array(batch), size=(299, 299))
            
            with self.torch.no_grad():
                # 手动执行 InceptionV3 的前向传播直到 mixed_7c
                x = batch_t
                model = self.inception
                
                x = model.Conv2d_1a_3x3(x)
                x = model.Conv2d_2a_3x3(x)
                x = model.Conv2d_2b_3x3(x)
                x = self.F.max_pool2d(x, kernel_size=3, stride=2)
                x = model.Conv2d_3b_1x1(x)
                x = model.Conv2d_4a_3x3(x)
                x = self.F.max_pool2d(x, kernel_size=3, stride=2)
                x = model.Mixed_5b(x)
                x = model.Mixed_5c(x)
                x = model.Mixed_5d(x)
                x = model.Mixed_6a(x)
                x = model.Mixed_6b(x)
                x = model.Mixed_6c(x)
                x = model.Mixed_6d(x)
                x = model.Mixed_6e(x)
                x = model.Mixed_7a(x)
                x = model.Mixed_7b(x)
                x = model.Mixed_7c(x)
                x = self.F.adaptive_avg_pool2d(x, (1, 1))
                x = self.torch.flatten(x, 1)
                
                features.append(x.cpu().numpy())
                
        return np.concatenate(features, axis=0)

    def _calculate_fid_from_features(self, real_feats, gen_feats):
        """根据特征向量计算 FID"""
        mu1 = np.mean(real_feats, axis=0)
        mu2 = np.mean(gen_feats, axis=0)
        sigma1 = np.cov(real_feats, rowvar=False)
        sigma2 = np.cov(gen_feats, rowvar=False)

        diff = mu1 - mu2
        diff_squared = np.dot(diff, diff)

        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff_squared + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def evaluate_batch(self, real_imgs, gen_imgs):
        """
        评估一批图像，返回平均指标和 FID。
        real_imgs, gen_imgs: list of numpy arrays or numpy array (N, H, W, C)
        """
        results = {'mse': [], 'psnr': [], 'ssim': [], 'lpips': []}
        
        print(f"Evaluating batch of {len(real_imgs)} images...")
        
        # 逐张计算 Pixel-wise metrics 和 LPIPS
        for i, (r, g) in enumerate(zip(real_imgs, gen_imgs)):
            results['mse'].append(self.compute_mse(r, g))
            results['psnr'].append(self.compute_psnr(r, g))
            results['ssim'].append(self.compute_ssim(r, g))
            
            lpips_val = self.compute_lpips(r, g)
            if lpips_val is not None:
                results['lpips'].append(lpips_val)

        # 汇总平均值
        summary = {k: np.mean(v) for k, v in results.items() if v}
        
        # 计算 FID (需要整个 batch)
        fid_val = self.compute_fid(real_imgs, gen_imgs)
        if fid_val is not None:
            summary['fid'] = fid_val
            
        return summary
