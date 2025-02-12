import os
import glob
import io
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image
import imageio


def pad_frame(frame, block_size=16):
    """
    如果图像尺寸不能被 block_size 整除，则在右侧和下方填充黑色像素，
    使图像尺寸调整为能被 block_size 整除。

    Args:
        frame (numpy.ndarray): 输入图像帧，形状为 (H, W, C)
        block_size (int): 要整除的块大小，默认值为 16

    Returns:
        numpy.ndarray: 填充后的图像帧
    """
    h, w = frame.shape[:2]
    new_w = ((w + block_size - 1) // block_size) * block_size
    new_h = ((h + block_size - 1) // block_size) * block_size
    if new_w == w and new_h == h:
        return frame
    padded_frame = np.pad(frame, ((0, new_h - h), (0, new_w - w), (0, 0)),
                          mode='constant', constant_values=0)
    return padded_frame

def load_baseline_data(data_dir, experiment_name):
    """
    加载特定实验的基准数据

    Args:
        data_dir (str): 数据根目录
        experiment_name (str): 实验目录名

    Returns:
        dict: 包含steps和baseline值的字典
    """
    exp_path = os.path.join(data_dir, experiment_name)
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"实验目录未找到: {exp_path}")
    
    # 获取事件文件
    event_files = glob.glob(os.path.join(exp_path, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"未找到事件文件: {exp_path}")
    
    # 加载事件数据
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()
    
    # 提取训练步数和基准值
    try:
        steps = [s.value for s in event_acc.Scalars('Train_EnvstepsSoFar')]
        baseline_values = [s.value for s in event_acc.Scalars('Baseline_Loss')]
        
        return {
            'steps': np.array(steps),
            'baseline': np.array(baseline_values)
        }
    except Exception as e:
        raise ValueError(f"数据提取错误: {str(e)}")

def plot_baseline_curve(data, output_path):
    """
    绘制基准学习曲线

    Args:
        data (dict): 包含steps和baseline值的字典
        output_path (str): 图表保存路径
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn')
    
    # 绘制主曲线
    plt.plot(data['steps'], 
             data['baseline'],
             color='#1f77b4',
             linewidth=2,
             marker='o',
             markersize=4,
             markevery=max(1, len(data['steps']) // 20),
             label='baseline loss')
    
    # 设置图表标题和标签
    plt.title('HalfCheetah-v4: baseline loss curve', fontsize=14, pad=20)
    plt.xlabel('env steps', fontsize=12, labelpad=10)
    plt.ylabel('baseline loss', fontsize=12, labelpad=10)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存至: {output_path}")

def analyze_baseline_metrics(data):
    """
    分析基准数据的关键指标

    Args:
        data (dict): 包含baseline值的字典
    """
    baseline_values = data['baseline']
    
    metrics = {
        'min loss': np.min(baseline_values),
        'max loss': np.max(baseline_values),
        'mean loss': np.mean(baseline_values),
        'std loss': np.std(baseline_values),
        'final loss': baseline_values[-1]
    }
    
    print("\n基准损失指标分析:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

def export_video_from_tensorboard(data_dir, experiment_name, output_video_path, video_tag=None, fps=10):
    """
    从 TensorBoard 事件文件中提取视频帧并导出为视频文件

    Args:
        data_dir (str): 数据根目录
        experiment_name (str): 实验目录名
        output_video_path (str): 视频保存路径（如 "output_video.mp4"）
        video_tag (str, optional): 指定要提取的 images 标签。若为 None，则自动选择第一个可用标签。
        fps (int): 视频的帧率
    """
    exp_path = os.path.join(data_dir, experiment_name)
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"实验目录未找到: {exp_path}")

    # 获取事件文件
    event_files = glob.glob(os.path.join(exp_path, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"未找到事件文件: {exp_path}")
    
    print(f"event_files is {event_files}")
    
    event_acc = EventAccumulator(event_files[0])
    event_acc.Reload()
    
    # 获取所有 images 标签
    available_image_tags = event_acc.Tags().get('images', [])
    print(f"可用 images 标签: {available_image_tags}")
    if not available_image_tags:
        raise ValueError("事件文件中没有找到任何 images 标签，无法导出视频。")
    
    # 如果没有指定视频标签，则自动使用第一个
    if video_tag is None:
        video_tag = available_image_tags[0]
        print(f"未指定视频标签，自动使用标签: {video_tag}")
    elif video_tag not in available_image_tags:
        raise ValueError(f"指定的视频标签 '{video_tag}' 不存在。可用标签：{available_image_tags}")
    
    print(f"开始提取标签 '{video_tag}' 下的视频帧...")
    image_events = event_acc.Images(video_tag)
    if not image_events:
        raise ValueError(f"在标签 '{video_tag}' 下未找到任何视频帧数据。")
    
    frames = []
    for event in image_events:
        # event.encoded_image_string 为图片的二进制数据
        try:
            img = Image.open(io.BytesIO(event.encoded_image_string))
            img = img.convert("RGB")
            frame = np.array(img)
            print(f"frame is {frame},and shape is {frame.shape}")
            # 填充帧，使其尺寸能被 16 整除，避免 ffmpeg 自动调整尺寸
            frame = pad_frame(frame, block_size=16)
            frames.append(frame)
        except Exception as e:
            print(f"警告：提取某帧时出错，跳过此帧。错误信息: {e}")
    
    if not frames:
        raise ValueError("未成功提取到任何视频帧。")
    
    print(f"the number of frames is {len(frames)}")
    
    # 保存视频（支持mp4格式）
    imageio.mimsave(output_video_path, frames, fps=fps)
    print(f"视频已保存至: {output_video_path}")

def main():
    # 设置参数
    data_dir = "./data"
    # 使用更新后的实验目录
    experiment_name = "q2_pg_cheetah_baseline_na_HalfCheetah-v4_2025-02-11_14-21-18"
    output_dir = "analysis_results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载数据并绘制学习曲线
        print("正在加载实验数据...")
        data = load_baseline_data(data_dir, experiment_name)
        curve_output_path = os.path.join(output_dir, "baseline_learning_curve.png")
        plot_baseline_curve(data, curve_output_path)
        analyze_baseline_metrics(data)
        
        # 导出视频
        # 若知道具体的视频标签，可在此处指定，如 video_tag="eval_video"
        video_output_path = os.path.join(output_dir, "halfcheetah_video.mp4")
        export_video_from_tensorboard(data_dir, experiment_name, video_output_path, video_tag='eval_rollouts', fps=10)
        
        print("\n所有分析结果已生成。")
        print("打开 TensorBoard 并进入 'Images' 标签页，也可查看导出的视频文件！")
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
