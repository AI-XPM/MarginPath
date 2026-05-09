import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils.vision_util import process_vision_info
import argparse
import traceback
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='批量测试Qwen2VL模型')
    parser.add_argument('--model_dir', type=str, default="Qwen2-VL-2B-Instruct/",
                        help='模型目录路径')
    parser.add_argument('--test_dir', type=str, default="test_data/",
                        help='测试图像目录路径')
    parser.add_argument('--output_file', type=str, default="test_results.json",
                        help='输出结果文件路径')
    parser.add_argument('--prompt', type=str,
                        default="Please describe this multiphoton microscopy image of breast cancer.",
                        help='用于所有图像的提示文本')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批处理大小')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='生成的最大新token数')
    parser.add_argument('--use_flash_attention', action='store_true',
                        help='是否使用flash attention')
    return parser.parse_args()
# python test_batch.py --model_dir /root/autodl-tmp/data/train_output/20250320105411 --test_dir /autodl-fs/data/report_test

def load_model_and_processor(args):
    print(f"正在加载模型: {args.model_dir}")

    if args.use_flash_attention:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_dir,
            torch_dtype="auto",
            device_map="auto"
        )

    processor = AutoProcessor.from_pretrained(args.model_dir, padding_side="left")
    return model, processor


def get_image_files(test_dir):
    """获取测试目录中的所有图像文件"""
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff']
    image_files = []

    for root, _, files in os.walk(test_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_formats):
                image_files.append(os.path.join(root, file))

    print(f"找到 {len(image_files)} 个图像文件用于测试")
    return image_files


def process_single_image(img_path, prompt, model, processor, max_new_tokens):
    """处理单个图像并生成描述"""
    try:
        # 创建消息格式（与原始示例代码匹配）
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 应用聊天模板（与原始代码匹配）
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        # 处理图像输入
        image_inputs, video_inputs = process_vision_info(message)

        # 如果image_inputs为空，可能是因为图像路径问题，尝试手动加载
        if not image_inputs:
            try:
                img = Image.open(img_path).convert('RGB')
                image_inputs = [img]
            except Exception as img_e:
                return False, f"无法加载图像: {str(img_e)}"

        # 创建模型输入
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # 生成文本
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        # 解码输出
        generated_ids_trimmed = generated_ids[0, len(inputs.input_ids[0]):]
        output_text = processor.decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return True, output_text

    except Exception as e:
        error_msg = f"处理图像 {img_path} 时出错: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return False, error_msg


def main():
    args = parse_args()

    # 加载模型和处理器
    model, processor = load_model_and_processor(args)

    # 获取图像文件
    image_files = get_image_files(args.test_dir)

    if not image_files:
        print("没有找到图像文件，退出程序。")
        return

    # 批量处理图像
    print("开始批量处理图像...")
    results = []

    for i, img_path in enumerate(tqdm(image_files, desc="处理图像")):
        success, output = process_single_image(
            img_path,
            args.prompt,
            model,
            processor,
            args.max_new_tokens
        )

        results.append({
            "image_path": img_path,
            "prompt": args.prompt,
            "success": success,
            "response": output
        })

        # 定期保存中间结果
        if (i + 1) % 5 == 0 or i == len(image_files) - 1:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存最终结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印处理统计信息
    success_count = sum(1 for r in results if r["success"])
    print(f"测试完成! 成功: {success_count}/{len(results)}")
    print(f"结果已保存到 {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main()