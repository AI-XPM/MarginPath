import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import tifffile
import scipy.io as sio
from vit import build_model

from PIL import Image
import numpy as np
from torchvision import transforms
import gc
from torchvision.transforms import functional as TF

import matplotlib.pyplot as plt
from collections import Counter

import matplotlib.cm as cm
from scipy import ndimage
from scipy.stats import mode
from skimage import measure, morphology
from scipy.ndimage import label, distance_transform_edt
import cv2
# ===== 配置参数 =====
input_folder = '/autodl-fs/data/datasets/fig4_test'   # 输入大图文件夹
output_npy_folder = '/root/margin/vit/output/fig4_test/mat_results'   # 输出.mat文件保存位置
model_weights = '/root/margin/vit/best_acc.pth'
output_image_folder="/root/margin/vit/output/fig4_test"                            # 输出的可视化图存放文件夹 

    # 选择需要输出的图
output = {'heatmap':True,
              'overlay':True,
              'label':True,
              'classmap':True
              }


batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10
patch_size = 512
required_predictions = 16

os.makedirs(output_npy_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# ===== 加载模型 =====
model = build_model(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_weights, map_location=device))
model.eval()

def map_category(label):
    '''标签映射'''
    # mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1} #ResNet18
    mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0}
    # mapping = {0: 0, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 2, 9: 0}
    return mapping.get(label, 0)

# ===== 滑动窗口预测函数 =====
def sliding_window_predict(image, model, patch_size, required_predictions, batch_size):
    H, W, C = image.shape
    stride = patch_size // int(np.sqrt(required_predictions))
    k_size = int(np.sqrt(required_predictions))
    # Padding
    pad_h = (patch_size - H % stride) % stride
    pad_w = (patch_size - W % stride) % stride
    padded_img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    padded_H, padded_W = padded_img.shape[:2]

    # 初始结果存储
    pred_stack = np.full((required_predictions, padded_H//stride, padded_W//stride), -1,dtype=np.uint8)
    pred_index = np.zeros((stride, stride), dtype=int)

    coords = []
    patches = []


    # 切 patch + transform
    for y in range(0, padded_H - patch_size + 1, stride):
        for x in range(0, padded_W - patch_size + 1, stride):
            patch = padded_img[y:y+patch_size, x:x+patch_size, :]
            # patch_pil = Image.fromarray(patch.astype(np.uint8))
            # patch_tensor = transform(patch_pil)
            patches.append(patch)
            coords.append((y, x))

    patches_tensor = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).float() / 255.0  # [N, C, H, W]
    patches_tensor = TF.resize(patches_tensor, [224, 224])  # 使用 functional resize

    # Normalize (可以直接广播操作)
    mean = torch.tensor([0.485, 0.456, 0.406], device=patches_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=patches_tensor.device).view(1, 3, 1, 1)
    patches = (patches_tensor - mean) / std
    # 分 batch 预测
    all_preds = []

    for i in tqdm(range(0, len(patches), batch_size), desc="Predicting patches"):
        # batch = torch.stack(patches[i:i+batch_size]).to(device)
        batch = patches[i:i+batch_size].to(device)
        with torch.no_grad():
            logits = model(batch)
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(pred_labels)

  
    
    for idx, (y, x) in enumerate(coords):
        y1 = y //stride
        x1 = x //stride
        patch_pred = all_preds[idx]  # shape: (patch_size, patch_size)

        current_pred_index = pred_index[y1:y1+k_size, x1:x1+k_size]
        mask = current_pred_index < required_predictions

        # 获取 mask 区域的索引坐标
        row_inds, col_inds = np.nonzero(mask)  # shape: [N]

        # 获取该区域每个位置要写入的 pred_stack 层
        write_layer = current_pred_index[mask]  # shape: [N]

        # 将 patch_pred 中对应位置写入 pred_stack 的对应层
        pred_stack[write_layer, y1 + row_inds, x1 + col_inds] = patch_pred

        # 更新 pred_index
        pred_index[y1:y1+k_size, x1:x1+k_size][mask] += 1

    # 去掉 padding
    pred_stack = pred_stack[:, 0:H, 0:W]
    return pred_stack  # shape: [16, H, W]

def fill_zero_region_with_nearest_category(label_image):
    filled_image = label_image.copy()

    # 获取非零mask
    mask = (label_image != 0)

    # 使用距离变换，获取每个0像素最近的非零像素的坐标
    distance, indices = distance_transform_edt(~mask, return_indices=True)

    # 用非零区域最近的像素位置对应的标签来填充0区域
    filled_image[~mask] = label_image[indices[0][~mask], indices[1][~mask]]

    return filled_image

def remove_small_objects(pred, min_size=64, connectivity=1):
    '''移除面积小于min_size的连通区域'''
    labeled_array, num_features = ndimage.label(pred)
    sizes = ndimage.sum(pred, labeled_array, range(1, num_features + 1))
    mask = np.zeros_like(pred, dtype=bool)
    for i, size in enumerate(sizes):
        if size >= min_size:
            mask |= (labeled_array == (i + 1))
    cleaned_image = mask.astype(np.uint8)
    return cleaned_image

def remove_small_objects2(label_image, min_size=50):
    # 获取每个连通区域的标记，使用 8 连通性
    labeled_image = measure.label(label_image, connectivity=1)
    
    # 获取每个连通区域的属性
    properties = measure.regionprops(labeled_image)
    
    # 创建一个新的标签图，去除小目标
    cleaned_image = np.zeros_like(labeled_image)
    
    # 仅保留大于 min_size 的区域
    for region in properties:
        if region.area >= min_size:
            # 保留大目标的标签到 cleaned_image
            cleaned_image[labeled_image == region.label] = region.label
    
    # 使用 cleaned_image 更新原始类别标签图，保留去除小目标后的标签
    final_image = np.zeros_like(label_image)
    
    # 保留每个区域的类别标签
    for region in properties:
        if region.area >= min_size:
            # 只保留大目标的类别标签
            final_image[labeled_image == region.label] = label_image[labeled_image == region.label]
    
    return final_image

def visualize_heatmap(prob_map, save_path, cmap='Spectral_r'):
    '''可视化热力图'''
    norm = plt.Normalize(vmin=0, vmax=1)
    plt.figure()
    plt.imshow(prob_map, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.savefig(save_path, dpi=300,bbox_inches='tight',pad_inches=0)
    plt.close()

def overlay_heatmap_on_image(prob_map, original_image, save_path=None, alpha=0.5, colormap='jet'):
    '''热力图叠加到原图上'''
    cmap = cm.get_cmap(colormap)
    heatmap_rgb = cmap(prob_map)[:, :, :3]  # RGBA -> 只要RGB
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)  # 转成uint8图像

    heatmap_im = Image.fromarray(heatmap_rgb).convert("RGB")

    if isinstance(original_image, np.ndarray):
        original_im = Image.fromarray(original_image).convert("RGB")
    else:
        original_im = original_image.convert("RGB")

    overlay = Image.blend(original_im, heatmap_im, alpha=alpha)

    if save_path:
        overlay.save(save_path)
    return overlay


def process_npy(pred_stack,img,img_name, output_folder, output):

    img_H_size, img_W_size,c = img.shape

    N, H, W = pred_stack.shape

    # 将类别结果映射为切缘结果
    mapped_stack = np.vectorize(map_category)(pred_stack)

    valid_mask = pred_stack != 255

    # 计算多次预测的平均值
    prob_map = np.sum(mapped_stack, axis=0)
    count_valid = np.sum(valid_mask, axis=0)
    prob_map_mean = prob_map/count_valid

    # 插值到原图大小
    
    # label_map
    if(output['label']):
        save_path = os.path.join(output_folder, 'label')
        save_path = os.path.join(save_path, img_name.replace(".tif", "_probmap.png"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 得到标签二值数组
        # TODO 最佳阈值 
        threshold = 0.4
        label_map = np.where(prob_map_mean > threshold, 1, 0)
        # 去除小区域
        # TODO 最佳min_size 100>80
        min_size = 100
        label_map = remove_small_objects(label_map, min_size, connectivity=2)
        label_map = np.where(label_map == 1, 0, 1)
        label_map = remove_small_objects(label_map, min_size, connectivity=2)
        label_map = np.where(label_map == 1, 0, 1)

        binary_image = (label_map * 255)
        binary_image = Image.fromarray(binary_image.astype(np.uint8))
        binary_image = binary_image.resize((img_W_size, img_H_size), Image.NEAREST)

        binary_image.save(save_path)


    prob_map_im = Image.fromarray(prob_map_mean)
    prob_map_im = prob_map_im.resize((img_W_size, img_H_size), Image.LANCZOS)
    prob_map_im = np.array(prob_map_im)
    # heatmap
    if(output['heatmap']):
        save_path = os.path.join(output_folder, 'heatmap')
        save_path = os.path.join(save_path, img_name.replace(".tif", "_heatmap.png"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_heatmap(prob_map_im, save_path)
    
    # overlay
    if(output['overlay']):
        save_path=os.path.join(output_folder,'overlay')
        save_path = os.path.join(save_path, img_name.replace(".tif","_overlay.png"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        overlay_heatmap_on_image(prob_map_im,img,save_path,alpha=0.5,colormap='Spectral_r')
    
    # classmap
    if(output['classmap']):
        def Hex_to_RGB(hex):
            r = int(hex[0:2], 16)
            g = int(hex[2:4], 16)
            b = int(hex[4:6], 16)
            return [r, g, b]

        color_map1 = {
                0: Hex_to_RGB('a5c1ae'),
                1: Hex_to_RGB('90a955'),
                2: Hex_to_RGB('4f772d'),
                3: Hex_to_RGB('ffd166'),
                4: Hex_to_RGB('c75146'),
                5: Hex_to_RGB('ea8c55'),
                6: Hex_to_RGB('777ac9'),
                7: Hex_to_RGB('6096ba'),
                8: Hex_to_RGB('ffb9ac'),
                9: (0, 0, 0),  # 类别 9 对应黑色（默认）
                 
            }
        color_map2 = {
                10: Hex_to_RGB('a5c1ae'),
                1: Hex_to_RGB('90a955'),
                2: Hex_to_RGB('4f772d'),
                3: Hex_to_RGB('ffd166'),
                4: Hex_to_RGB('c75146'),
                5: Hex_to_RGB('ea8c55'),
                6: Hex_to_RGB('777ac9'),
                7: Hex_to_RGB('6096ba'),
                8: Hex_to_RGB('ffb9ac'),
                9: (0, 0, 0),  # 类别 9 对应黑色（默认）
                 
            }
        
        save_path=os.path.join(output_folder,'classmap')
        save_path1=os.path.join(save_path, img_name.replace(".tif", "_classmap.png"))
        save_path2 = os.path.join(save_path, img_name.replace(".tif",  "_classmap_cleaned.png"))
        os.makedirs(os.path.dirname(save_path1), exist_ok=True)
        os.makedirs(os.path.dirname(save_path2), exist_ok=True)
        masked_pred_stack = np.ma.masked_equal(pred_stack, 255)
        majority_vote = mode(masked_pred_stack, axis=0, nan_policy='omit', keepdims=False).mode.astype(np.uint8)

        # 移除每个类别中的小目标
        cleaned_majority_vote = majority_vote.copy()
        cleaned_majority_vote[cleaned_majority_vote == 0]=10
        cleaned_majority_vote = remove_small_objects2(cleaned_majority_vote, min_size=16)

        # 移除的小目标会被置为0，采用距离最近的类别进行填充
        cleaned_majority_vote = fill_zero_region_with_nearest_category(cleaned_majority_vote)

        rgb_image = np.zeros((majority_vote.shape[0], majority_vote.shape[1],3), dtype=np.uint8)
        cleaned_rgb_image = np.zeros((majority_vote.shape[0], majority_vote.shape[1],3), dtype=np.uint8)
        for i in range(majority_vote.shape[0]):
            for j in range(majority_vote.shape[1]):
                
                # 未清洗
                category1 = majority_vote[i, j]
                rgb_image[i, j] = color_map1.get(category1)

                # 清洗后的类别
                category2 = cleaned_majority_vote[i, j]
                cleaned_rgb_image[i, j] = color_map2.get(category2)
        # 保存未清洗
        majority_vote_im = Image.fromarray(rgb_image)
        majority_vote_im =  majority_vote_im.resize((img_W_size, img_H_size), Image.NEAREST)
        majority_vote_im.save(save_path1)

        # 保存清洗过的图  
        cleaned_majority_vote_im = Image.fromarray(cleaned_rgb_image)
        cleaned_majority_vote_im =  cleaned_majority_vote_im.resize((img_W_size, img_H_size), Image.NEAREST)
        cleaned_majority_vote_im.save(save_path2)
    

# ===== 主处理流程 =====
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]

for img_name in tqdm(image_files, desc="Processing images"):
    try:

        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = np.array(img)
            
        if image.ndim == 2:  # 灰度图 -> 添加通道维
            image = image[:, :, None]
        elif image.ndim == 3 and image.shape[0] <= 4:  # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))

        pred_stack = sliding_window_predict(image, model, patch_size, required_predictions, batch_size)
        save_npy_path = os.path.join(output_npy_folder, img_name.replace('.tif', '.npy'))
        np.save(save_npy_path,pred_stack)

        process_npy(pred_stack,image,img_name,output_image_folder,output)
    # 计算指标
    except:
        print(f"{img_name}错误！！！！！！！！！！")
        continue
    # print(f"完成{img_name}的预测")




    
