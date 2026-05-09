import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import re
from Classifier.vit import build_model
class ImageClassifier:
    def __init__(self, model_path, json_path, device):
        # 确保类别索引文件存在
        assert os.path.exists(json_path), f"File '{json_path}' does not exist."
        with open(json_path, "r") as json_file:
            self.class_indict = json.load(json_file)

        self.device = device

        # 加载模型EMO
        # self.model = torch.load(model_path, map_location=device)
        # self.model.eval()
        # self.model.to(device)

        # vit
        self.num_classes = 10
        self.model = build_model(num_classes=self.num_classes).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        # 确定模型输出的类别数
        # try:
        #     self.num_classes = self.model.fc.out_features
        # except AttributeError:
        #     try:
        #         self.num_classes = self.model.classifier.out_features
        #     except AttributeError:
        #         # 对于没有 'fc' 或 'classifier' 的模型，尝试获取最后一层
        #         last_layer = list(self.model.children())[-1]
        #         if isinstance(last_layer, torch.nn.Linear):
        #             self.num_classes = last_layer.out_features
        #         else:
        #             raise AttributeError("Unable to determine the number of output classes from the model.")

        # if self.num_classes != len(self.class_indict):
        #     raise ValueError("Mismatch between number of classes in the model and class_indices.json.")

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict_window(self, image):
        """对图像小块进行预处理并预测"""
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(image)
            out = torch.softmax(out, dim=1)  # 获取每个类别的 softmax 概率
        return out.squeeze()

    def predict_large_image(self, image):
        """
        对较大图像进行切块预测，步距重叠，然后将预测概率加权平均。
        可根据需求缩放 window_size/step_size。
        """
        window_size = (512, 512)
        step_size = (128, 128)

        softmax_output = torch.zeros(self.num_classes, dtype=torch.float).to(self.device)

        total_windows = 0
        width, height = image.size
        x_steps = range(0, width - window_size[0] + 1, step_size[0])
        y_steps = range(0, height - window_size[1] + 1, step_size[1])

        for idx, y in enumerate(y_steps):
            # 这里是个简单的例子：偶数行从左到右，奇数行从右到左
            if idx % 2 == 0:
                x_range = x_steps
            else:
                x_range = reversed(x_steps)

            for x in x_range:
                window = image.crop((x, y, x + window_size[0], y + window_size[1]))
                pred_probs = self.predict_window(window)
                softmax_output += pred_probs
                total_windows += 1

        if total_windows > 0:
            softmax_output /= total_windows

        softmax_output = softmax_output.cpu().numpy()

        # 进行再归一化
        renorm_dist = self.prediction(softmax_output)

        # 获取最可能类别
        class_idx = np.argmax(renorm_dist)
        class_name = self.class_indict[str(class_idx)]

        return class_name, renorm_dist

    def prediction(self, softmax_output):
        """
        将 softmax 输出再做一次归一化，比如合并正常类别、去除某些类别等。
        """
        renorm_dist = softmax_output / softmax_output.sum()

        # 定义正常组织类与肿瘤相关类
        normal_classes = [0, 4, 5, 6, 7,9]
        tumor_classes =  [1, 2, 3, 8]

        # 如果正常组织概率之和>0.9，则把肿瘤类清零
        normal_prob = renorm_dist[normal_classes].sum()
        if normal_prob > 0.9:
            renorm_dist[tumor_classes] = 0
            renorm_dist = renorm_dist / renorm_dist.sum()
        else:
            # 否则清零完全正常类别(0,1,2)，仅保留6,7
            renorm_dist[[0, 1, 2]] = 0
            renorm_dist = renorm_dist / renorm_dist.sum()

        return renorm_dist


class PathologyReportGenerator:
    def __init__(self, class_indices_path, examples_path):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=,  # 替换为自己的 API Key
            base_url=
        )

        # 加载类别索引
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        # 加载示例报告
        with open(examples_path, 'r') as f:
            self.examples = json.load(f)

        # TACS4-8 说明
        self.tacs_description = ( "TACS4 is defined as a network distribution of collagen fibers adjacent to continuous tumor cells, "
            "with a clear tumor boundary. TACS5 is defined as directionally distributed collagen fibers that "
            "allow tumor cells to migrate unidirectionally without a clear tumor boundary. TACS6 is defined as "
            "disordered collagen fibers that enable tumor cells to migrate multidirectionally without a clear "
            "tumor boundary. TACS7 is characterized by densely distributed collagen fibers at the tumor invasive "
            "front, with virtually no tumor cells in the leading-edge stroma. TACS8 refers to sparsely distributed "
            "collagen fibers at the tumor invasive front, also with virtually no tumor cells in the leading-edge stroma. "
            "TACS4-8 clearly reflect different tumor-stroma interactions during the invasive stages of tumor development "
            "after the DCIS-to-IBC transition."
           
        )

    def probabilities_to_natural_language(self, probabilities, image_index=None, mode="complex"):
        """
        将分类后的概率分布转换为自然语言描述，重点说明 TACS 及是否为肿瘤。
        """
        normal_classes = [0, 4, 5, 6, 7,9]
        tacs_indices = [1, 2, 3, 4, 5]  # TACS4-8
        tumor_classes = [1, 2, 3, 8]
        only_normal_classes=[0, 6, 7, 9]
        # highest_class_idx = int(np.argmax(probabilities))
        # highest_class_name = self.class_indices[str(highest_class_idx)]
        # highest_class_prob = probabilities[highest_class_idx]
        normal_prob_sum = probabilities[normal_classes].sum()

        def qualitative_desc(value):
            if value > 0.5:
                return "significant"
            elif value > 0.2:
                return "moderate"
            elif value > 0.05:
                return "mild"
            else:
                return "trace"

        def prob_str(value):
            return f"(~{value * 100:.2f}%)"

        description_parts = []
        if image_index is not None:
            description_parts.append(f"Image {image_index} Analysis:")

        if normal_prob_sum > 0.9:
            highest_class_idx = int(np.argmax(probabilities))
            highest_class_name = self.class_indices[str(highest_class_idx)]
            highest_class_prob = probabilities[highest_class_idx]
            
            description_parts.append(
                "This image predominantly shows normal tissue characteristics, indicating a histologically benign environment."
            )
            main_quality = qualitative_desc(highest_class_prob)
            description_parts.append(
                f"The most notable feature is {highest_class_name} {prob_str(highest_class_prob)}, "
                f"which is {main_quality} in presence, reinforcing the normal-like profile."
            )
        else:
            pros = probabilities
            pros[only_normal_classes] = 0.0
            s = pros.sum()
            if s > 0:
                pros = pros / s
            highest_class_idx = int(np.argmax(pros))
            highest_class_name = self.class_indices[str(highest_class_idx)]
            highest_class_prob = pros[highest_class_idx]

            description_parts.append(
                "The tissue composition suggests a more complex microenvironment, not purely benign."
            )
            main_quality = qualitative_desc(highest_class_prob)


            if highest_class_idx in tacs_indices or highest_class_idx in tumor_classes:
                description_parts.append(
                    f"The presence of {highest_class_name} {prob_str(highest_class_prob)} is {main_quality}."
                )


            # 汇总 TACS
            tacs_prob_sum = probabilities[tacs_indices].sum()
            if tacs_prob_sum > 0.05:
                tacs_quality = qualitative_desc(tacs_prob_sum)
                description_parts.append(
                    f"TACS4-8 related patterns hold a {tacs_quality} presence {prob_str(tacs_prob_sum)}."
                )

            # 对每个 TACS 单独描述
            for t_idx in tacs_indices:
                t_prob = probabilities[t_idx]
                if t_prob > 0.05:
                    t_name = self.class_indices[str(t_idx)]
                    t_quality = qualitative_desc(t_prob)
                    description_parts.append(
                        f"{t_name} is {t_quality} {prob_str(t_prob)} in the image."
                    )

            # 肿瘤组织
            tumour_prob = probabilities[8]
            if tumour_prob > 0.05:
                tumour_quality = qualitative_desc(tumour_prob)
                description_parts.append(
                    f"Direct tumor tissue shows a {tumour_quality} presence {prob_str(tumour_prob)}."
                )

        return " ".join(description_parts)

    def generate_prompt(self, classification_descriptions, multiphoton_description):
        """
        生成给大模型的提示词（Prompt）。
        核心要求：
        1. 以小模型(qwen2vl)生成的多光子描述为主；
        2. TACS 分类结果用于补充和丰富小模型描述；
        3. 若两者有不一致处，保持小模型版本的核心叙述，并在可能的地方加入 TACS 所提供的附加信息；
        4. 输出必须为 JSON，且只包含 "Tumor microenvironment characteristics" 和 "Description" 两个字段。
        """

        prompt = (
            "You are a medical expert in breast cancer pathology. You have two sources of information about the same multiphoton image:\n\n"

            "1) A primary, detailed multiphoton description generated by a smaller vision-language model (qwen2vl). "
            "   This description should serve as your main reference.\n\n"

            "2) Additional TACS classification results from high-resolution, patch-based predictions, "
            "   which provide extra insights into tumor-associated collagen signatures (TACS 4-8) and overall tumor microenvironment.\n\n"

            "Your goal is to produce a refined pathology report **rooted in the primary description** (the smaller model's text), "
            "while selectively incorporating or expanding upon it using the TACS classification data. "
            "If there is any conflict or discrepancy, preserve the essence of the smaller model's description, "
            "but feel free to integrate TACS results that do not contradict its core statements. "
            "Avoid inventing or hallucinating details not supported by either source. The final report should be as concise as possible, minimizing redundant descriptions.\n\n"

            "Your final report must strictly be returned as a single valid JSON object with the following keys only:\n"
            '  - "Tumor microenvironment characteristics"\n'
            '  - "Description"\n\n'
            "No extra keys are allowed, and no text should appear outside of the JSON.\n\n"

            "Below is reference information on TACS (tumor-associated collagen signatures):\n"
            f"{self.tacs_description}\n\n"

            "Below is an example template showing how these two JSON fields could be structured:\n"
        )

        # 将示例模板（report_template.json）添加到 Prompt
        for example in self.examples:
            prompt += json.dumps(example, indent=2, ensure_ascii=False) + "\n\n"

        # 小模型(qwen2vl)描述 —— 这是主要的信息来源
        prompt += (
            "=== Primary Low-Resolution Multiphoton Description (qwen2vl) ===\n"
            f"{multiphoton_description.strip()}\n\n"
        )

        # TACS 分类结果 —— 用来“补充、丰富”而非“替代”
        prompt += (
            "=== Supplementary TACS Classification Results ===\n"
            f"{classification_descriptions}\n\n"
        )

        # 最终要求
        prompt += (
            "Please refine the primary description by integrating TACS insights where appropriate, "
            "ensuring consistency with the smaller model's text. Your final pathology report must be valid JSON "
            "with exactly two keys: 'Tumor microenvironment characteristics' and 'Description'.\n\n"
            "Generate your final JSON report now."
        )

        return prompt

    def generate_pathology_report(self, classification_descriptions, multiphoton_description, model='gpt-4o'):
        """
        调用大模型 API，生成病理报告，并返回 (prompt, JSON报告)。
        """
        prompt = self.generate_prompt(classification_descriptions, multiphoton_description)
        print("\n============ Generated Prompt ============\n")
        print(prompt)
        print("\n==========================================\n")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=2048,
            )

            if not response.choices:
                print("No choices returned by LLM.")
                return prompt, None

            report_content = response.choices[0].message.content.strip()
            print("\n============ Raw LLM Response ============\n")
            print(report_content)
            print("\n==========================================\n")

            # 如果包含 ```，尝试提取其中的 JSON
            if '```' in report_content:
                match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', report_content, re.IGNORECASE)
                if match:
                    report_content = match.group(1).strip()
                else:
                    report_content = report_content.replace('```json', '').replace('```', '').strip()

            # 尝试直接解析 JSON
            if not report_content.startswith("{"):
                print("The model did not return a valid JSON object after cleaning.")
                return prompt, None

            try:
                report = json.loads(report_content)
                return prompt, report
            except json.JSONDecodeError as e:
                print("Failed to parse JSON output:", report_content)
                print("JSONDecodeError:", e)
                return prompt, None

        except ValueError as e:
            print(f"OpenAI API error: {e}")
            return prompt, None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return prompt, None


def process_images_individually(image_dir,
                                qwen2vl_json_path,
                                model_path,
                                class_indices_path,
                                examples_path,
                                output_report_path,
                                prompt_output_path):
    """
    对目录下所有图像逐张处理：
      1. 读取小模型(qwen2vl)的多光子描述
      2. 预测当前大图的TACS等分类结果
      3. 调用LLM整合生成病理报告
      4. 输出到JSON文件
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ImageClassifier(model_path, class_indices_path, device)
    report_generator = PathologyReportGenerator(class_indices_path, examples_path)

    # 读取 qwen2vl 产生的描述
    with open(qwen2vl_json_path, 'r', encoding='utf-8') as f:
        qwen2vl_data = json.load(f)

    # 构建映射：{文件名: 小模型多光子描述}
    multiphoton_desc_map = {}
    for item in qwen2vl_data:
        filename = os.path.basename(item["image_path"])
        multiphoton_desc_map[filename] = item.get("response", "No multiphoton description found.")

    # 收集 image_dir 下所有图像文件路径
    image_paths = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(('.tif', '.png', '.jpg'))
    ]

    all_reports = {}
    all_prompts = {}

    with tqdm(total=len(image_paths), desc="Processing images", unit="image") as pbar:
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            print(f"\n---- Processing {filename} ----")

            # 获取小模型描述（若没有则用默认）
            mp_desc = multiphoton_desc_map.get(filename, "No multiphoton description found.")

            # 大图切块预测
            image = Image.open(img_path)
            _, renorm_dist = classifier.predict_large_image(image)
            classification_text = report_generator.probabilities_to_natural_language(renorm_dist, mode="complex")

            # 调用LLM生成JSON报告
            prompt, report = report_generator.generate_pathology_report(
                classification_descriptions=classification_text,
                multiphoton_description=mp_desc,
                model='gpt-4o'  # 可换成你能用的模型名字
            )
            structured_prompt = f"""
You are an expert in pathology and tumor microenvironment analysis.

Your task is to rewrite the following multiphoton microscopy image description into a standardized and professionally structured format, using spatial regions that appear in the original description.

### Instructions:
1. Identify which spatial orientation is present (left/right, upper/lower, etc.).
2. Only include regions that are mentioned or implied.
3. Organize the rewritten description using any of the following section headers (use only those that apply):
   - Left Region:
   - Right Region:
   - Upper Region:
   - Lower Region:
   - Transition Zone / Tumor–Stroma Interface:
4. Summarize collagen features, tumor characteristics, stromal organization, adipose/necrosis features.
5. Use academic pathology terminology.
6. Avoid redundancy.

### Output Format:
Left Region:
- …

Right Region:
- …

Upper Region:
- …

Lower Region:
- …

Tumor Boundary Region:
- …

### Original Description:
{mp_desc}

Rewrite it into the structured format above.
"""
            structured_desc = report_generator.client.chat.completions.create(model="gpt-4o",messages=[{"role": "user", "content": structured_prompt}] )
            
            report["Structured Multiphoton Description"] = structured_desc.choices[0].message.content
            if report:
                all_reports[filename] = report
            else:
                all_reports[filename] = {"error": "No valid JSON report generated."}

            # 保存 prompt、描述等信息
            all_prompts[filename] = {
                "prompt": prompt,
                "multiphoton_description": mp_desc,
                "classification_results": classification_text,
                "report": report if report else None
            }

            pbar.update(1)

    # 将所有报告写入单个JSON文件
    with open(output_report_path, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\nAll single-image pathology reports saved to {output_report_path}")

    # 将所有 prompt、分类结果和报告写入另一个 JSON 文件，便于调试或训练
    with open(prompt_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    print(f"All prompts and related data saved to {prompt_output_path}")


if __name__ == '__main__':
    # 根据自身需求修改路径与文件名
    image_dir = "/root/autodl-fs/report_test_2compressed"                   # 存放待测试的大尺寸多光子图像的目录
    qwen2vl_json_path = "/root/report/report_test2/test_results.json"  # qwen2vl生成的多光子描述文件
    model_path = "/root/report/vit/best_acc.pth"     # 分类模型
    class_indices_path = "/root/report/vit/class_indices_vit.json"    # 类别索引
    examples_path = "report_template.json"       # 示例报告模板

    # 结果输出文件
    output_report_path = "report_test2/single_image_reports_ours2.json"
    prompt_output_path = "report_test2/single_image_prompts_and_reports_ours2.json"

    process_images_individually(
        image_dir=image_dir,
        qwen2vl_json_path=qwen2vl_json_path,
        model_path=model_path,
        class_indices_path=class_indices_path,
        examples_path=examples_path,
        output_report_path=output_report_path,
        prompt_output_path=prompt_output_path
    )
