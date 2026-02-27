import os
import glob
import subprocess
import argparse

def main():
    # ===============================
    # 1. 设置命令行参数解析
    # ===============================
    parser = argparse.ArgumentParser(description="🚀 通用多实验批量运行脚本")
    parser.add_argument("--run_script", type=str, required=True, 
                        help="要执行的主训练脚本路径 (例如: cs336_alignment/sft_run.py)")
    parser.add_argument("--base_config", type=str, required=True, 
                        help="基础配置文件路径 (例如: config/sft/sft_base.yaml)")
    parser.add_argument("--exp_pattern", type=str, required=True, 
                        help="实验配置文件的通配符匹配规则 (例如: config/sft/qwen_2.5B_GSM8K*_*.yaml)")
    
    args = parser.parse_args()

    # ===============================
    # 2. 读取并校验传入的参数
    # ===============================
    run_script = args.run_script
    base_config = args.base_config
    search_pattern = args.exp_pattern

    # 校验：检查主脚本是否存在
    if not os.path.exists(run_script):
        print(f"❌ 找不到指定的执行脚本: {run_script}")
        return

    # 校验：检查基础配置是否存在
    if not os.path.exists(base_config):
        print(f"❌ 找不到基础配置文件: {base_config}")
        return

    # 获取所有匹配的实验配置
    exp_configs = sorted(glob.glob(search_pattern))
    
    if not exp_configs:
        print(f"❌ 没有找到匹配该规则的实验配置: {search_pattern}")
        return
        
    print(f"🔍 模式匹配成功，共发现 {len(exp_configs)} 个实验配置待运行。")
    print(f"📜 执行脚本: {run_script}")
    print(f"⚙️ 基础配置: {base_config}")
    print("=" * 50)
    
    # ===============================
    # 3. 循环拉起独立的训练进程
    # ===============================
    for i, exp_config in enumerate(exp_configs, 1):
        print(f"\n🚀 [ {i} / {len(exp_configs)} ] 正在启动实验: {exp_config}")
        
        # 动态组装命令
        cmd = [
            "python", run_script, 
            "--base_config", base_config, 
            "--exp_config", exp_config
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ 实验 {exp_config} 运行圆满结束！\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {exp_config} 运行失败，退出码: {e.returncode}")
            user_choice = input("⚠️ 是否忽略报错，继续运行下一个实验？(y/n): ")
            if user_choice.lower() != 'y':
                print("🛑 批处理已终止。")
                break

if __name__ == "__main__":
    main()