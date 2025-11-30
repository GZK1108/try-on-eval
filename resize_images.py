# 图片分辨率调整工具
# 将输入目录中的所有图片调整为指定分辨率
import os
import sys
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
from utils.resize_image import resize_image


def resize_directory(input_dir, output_dir, height_size, width_size, resize_mode='crop'):
    """
    批量处理目录中的所有图片

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        height_size: 目标高度
        width_size: 目标宽度
        resize_mode: 调整模式
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"输入目录不存在: {input_dir}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    file_list = [p for p in input_path.rglob('*') if p.is_file() and p.suffix.lower() in supported_formats]

    total_files = len(file_list)
    success_count = 0
    error_count = 0

    for file_path in tqdm(file_list, desc="处理图片"):
        relative_path = file_path.relative_to(input_path)
        output_file_path = output_path / relative_path.with_suffix('.jpg')

        if resize_image(str(file_path), str(output_file_path), height_size, width_size, resize_mode):
            success_count += 1
        else:
            error_count += 1

    print(f"\n处理完成:")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {error_count}")
    print(f"输出目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='将图片调整为指定分辨率')
    parser.add_argument('input_dir', help='输入图片目录')
    parser.add_argument('output_dir', help='输出图片目录')
    parser.add_argument('--height', type=int, default=1024, help='目标高度（默认1024）')
    parser.add_argument('--width', type=int, default=1024, help='目标宽度（默认1024）')
    parser.add_argument('--no-aspect-ratio', action='store_true',
                       help='不保持宽高比（与 --mode stretch 等效）')
    parser.add_argument('--mode', choices=['crop', 'fill', 'stretch'], default='crop',
                       help='调整模式：crop(居中裁剪)、fill(白边填充)、stretch(直接拉伸)')

    args = parser.parse_args()

    resize_mode = 'stretch' if args.no_aspect_ratio else args.mode

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标尺寸: {args.width}x{args.height}")
    print(f"调整模式: {resize_mode}")
    print("-" * 50)

    resize_directory(args.input_dir, args.output_dir, args.height, args.width, resize_mode)

def batch_more(id_list, resize_mode='crop'):
    for id in id_list:
        source_directory = f"/Volumes/huanying/datasets/lora-{id}/more"
        output_directory = f"/Volumes/huanying/datasets/lora-{id}/more"
        default_height = 1024
        default_width = 1024
        resize_directory(source_directory, output_directory, default_height, default_width, resize_mode=resize_mode)

if __name__ == "__main__":
    print('hello world')
    input_directory = "D:/datasets/viton_test/35372-1024"
    output_directory = "D:/datasets/viton_test/35372-1024"
    default_height = 1024
    default_width = 768
    default_mode = 'crop'

    print("使用默认参数进行测试:")
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory}")
    print(f"目标尺寸: {default_width}x{default_height}")
    print(f"调整模式: {default_mode}")
    print("-" * 50)

    resize_directory(input_directory, output_directory, default_height, default_width, resize_mode=default_mode)
