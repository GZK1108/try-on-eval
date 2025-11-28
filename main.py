from utils import ImageMetrics
import numpy as np

def main():
    # 初始化评估器
    evaluator = ImageMetrics()

    print("--- 1. 单张图片评估 ---")
    # 模拟单张图片 (H, W, C)
    img1 = np.random.rand(256, 256, 3).astype(np.float32)
    img2 = np.random.rand(256, 256, 3).astype(np.float32)

    print(f"MSE: {evaluator.compute_mse(img1, img2):.4f}")
    print(f"PSNR: {evaluator.compute_psnr(img1, img2):.4f}")
    print(f"SSIM: {evaluator.compute_ssim(img1, img2):.4f}")
    
    lpips_val = evaluator.compute_lpips(img1, img2)
    if lpips_val is not None:
        print(f"LPIPS: {lpips_val:.4f}")
    else:
        print("LPIPS: Skipped (missing dependencies)")

    print("\n--- 2. 批量图片评估 (Batch Evaluation) ---")
    # 模拟一批图片 (N, H, W, C)
    N_SAMPLES = 5
    print(f"生成 {N_SAMPLES} 对随机图像进行测试...")
    real_batch = np.random.rand(N_SAMPLES, 299, 299, 3).astype(np.float32)
    gen_batch = np.random.rand(N_SAMPLES, 299, 299, 3).astype(np.float32)

    # 一次性计算所有指标
    results = evaluator.evaluate_batch(real_batch, gen_batch)
    
    print("\n评估结果汇总:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()