from utils.utils import ImageMetrics
import numpy as np
import os




class PairedImageDataset:
    """
    数据集类：负责管理文件索引和单样本加载
    """
    def __init__(self, real_image_folder, generated_images_folder):
        self.real_folder = real_image_folder
        self.gen_folder = generated_images_folder
        self.image_pairs = self._find_pairs()

    def _find_pairs(self):
        # 获取两个文件夹中的文件名
        exts = ('.png', '.jpg', '.jpeg')
        real_imgs = {f for f in os.listdir(self.real_folder) if f.lower().endswith(exts)}
        gen_imgs = {f for f in os.listdir(self.gen_folder) if f.lower().endswith(exts)}
        
        # 取交集并排序，确保一一对应且顺序固定
        common_names = sorted(list(real_imgs.intersection(gen_imgs)))
        return common_names

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        filename = self.image_pairs[idx]
        real_path = os.path.join(self.real_folder, filename)
        gen_path = os.path.join(self.gen_folder, filename)
        
        return self._load_image(real_path), self._load_image(gen_path)

    @staticmethod
    def _load_image(path):
        from PIL import Image
        # 使用上下文管理器确保文件关闭
        with Image.open(path) as img:
            img = img.convert('RGB')
            return np.array(img).astype(np.float32) / 255.0


class DataLoader:
    """
    数据加载器：负责批处理、打乱和迭代 (类似 torch.utils.data.DataLoader)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        batch_real = []
        batch_gen = []
        
        for idx in self.indices:
            real_img, gen_img = self.dataset[idx]
            batch_real.append(real_img)
            batch_gen.append(gen_img)
            
            if len(batch_real) == self.batch_size:
                yield np.array(batch_real), np.array(batch_gen)
                batch_real, batch_gen = [], []
        
        # 处理剩余不足一个 batch 的数据
        if len(batch_real) > 0:
            yield np.array(batch_real), np.array(batch_gen)

    def load_all(self):
        """兼容旧接口：一次性加载所有数据（慎用，仅适用于小数据集）"""
        all_real, all_gen = [], []
        for i in range(len(self.dataset)):
            r, g = self.dataset[i]
            all_real.append(r)
            all_gen.append(g)
        return np.array(all_real), np.array(all_gen)



def main():
    # 初始化评估器
    evaluator = ImageMetrics()

    dataset = PairedImageDataset(
        real_image_folder="D:/datasets/viton_test/image-backup", 
        generated_images_folder="D:/datasets/viton_test/35372-1024"
    )
    
    # 2. 创建加载器 (Batch Size = 8)
    # 使用较小的 batch_size 防止内存溢出
    batch_size = 1016
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 用于存储所有指标的累积总和 (MSE, PSNR, SSIM, LPIPS)
    metric_sums = {}
    total_samples = 0
    
    # 用于存储 FID 计算所需的特征向量
    all_real_feats = []
    all_gen_feats = []

    print(f"开始评估，共 {len(dataset)} 张图片，Batch Size = {batch_size}...")

    for i, (real_batch, gen_batch) in enumerate(loader):
        current_batch_size = real_batch.shape[0]
        
        # 1. 计算常规指标 (不计算 FID)
        # 注意：evaluate_batch 返回的是当前 batch 的平均值
        batch_results = evaluator.evaluate_batch(real_batch, gen_batch, compute_fid=False)
        
        for metric, value in batch_results.items():
            if metric not in metric_sums:
                metric_sums[metric] = 0.0
            metric_sums[metric] += value * current_batch_size
            
        # 2. 提取 FID 特征并保存 (不保存图片，只保存特征向量，节省内存)
        if evaluator.has_torch:
            real_feats = evaluator.extract_features(real_batch)
            gen_feats = evaluator.extract_features(gen_batch)
            all_real_feats.append(real_feats)
            all_gen_feats.append(gen_feats)
            
        total_samples += current_batch_size
        print(f"已处理批次 {i+1}, 当前累计样本: {total_samples}")


    # --- 汇总结果 ---
    final_results = {}
    
    # 1. 计算常规指标的全局平均值
    for metric, total in metric_sums.items():
        final_results[metric] = total / total_samples

    # 2. 计算全局 FID
    if evaluator.has_torch and all_real_feats:
        print("正在计算全局 FID...")
        # 将所有批次的特征拼接成大矩阵
        real_feats_total = np.concatenate(all_real_feats, axis=0)
        gen_feats_total = np.concatenate(all_gen_feats, axis=0)
        
        fid_score = evaluator.calculate_fid_from_features(real_feats_total, gen_feats_total)
        final_results['fid'] = fid_score

    # 将结果保存到txt中
    output_file = "35372-3.txt"
    with open(output_file, 'w') as f:
        for metric, value in final_results.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
    
    print("\n评估结果汇总:")
    for metric, value in final_results.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()