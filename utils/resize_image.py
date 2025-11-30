import os
from PIL import Image


def resize_image(input_path, 
                 output_path, 
                 height_size, 
                 width_size, 
                 resize_mode='crop',
                 quality=100):
    """
    将单张图片调整为指定分辨率

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        height_size: 目标高度
        width_size: 目标宽度
        resize_mode: 调整模式，可选 crop(居中裁剪)、fill(白边填充)、stretch(直接拉伸)
    """
    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            mode = resize_mode.lower()

            if mode == 'crop':
                scale = max(width_size / img.width, height_size / img.height)
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                left = (new_width - width_size) // 2
                top = (new_height - height_size) // 2
                right = left + width_size
                bottom = top + height_size
                img = img.crop((left, top, right, bottom))
            elif mode == 'fill':
                scale = min(width_size / img.width, height_size / img.height)
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                background = Image.new('RGB', (width_size, height_size), color=(255, 255, 255))
                offset_x = (width_size - new_width) // 2
                offset_y = (height_size - new_height) // 2
                background.paste(resized_img, (offset_x, offset_y))
                img = background
            elif mode == 'stretch':
                img = img.resize((width_size, height_size), Image.Resampling.LANCZOS)
            else:
                raise ValueError(f"不支持的调整模式: {resize_mode}")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, 'JPEG', quality=quality)
            return True

    except Exception as e:
        print(f"处理图片 {input_path} 时出错: {e}")
        return False

if __name__ == "__main__":  
    # 测试代码
    input_image_path = "4043.jpg"
    output_image_path = "re/4043_pingpu.jpg"
    resize_image(input_image_path, output_image_path, 2048, 2048, resize_mode='crop', quality=90)