"""
实时监控训练进度
"""

import os
import time
from pathlib import Path

def watch_training():
    """监控训练日志"""
    log_file = Path("results/logs/rmha_seed42/training_log.txt")
    
    print("="*70)
    print("训练监控 - 按 Ctrl+C 退出")
    print("="*70)
    print()
    
    if not log_file.exists():
        print(f"等待日志文件: {log_file}")
        while not log_file.exists():
            time.sleep(1)
    
    last_size = 0
    last_lines = []
    
    try:
        while True:
            if log_file.exists():
                current_size = log_file.stat().st_size
                
                if current_size > last_size:
                    # 读取新内容
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        
                        # 显示最后5个episode
                        new_lines = lines[-5:]
                        
                        for line in new_lines:
                            if line.strip() and line not in last_lines:
                                # 解析日志行
                                if "Episode" in line:
                                    print(f"\n[{time.strftime('%H:%M:%S')}] {line.strip()}")
                                    
                                    # 提取关键信息
                                    if "success_rate" in line:
                                        import re
                                        sr_match = re.search(r"success_rate['\"]:\s*([\d.]+)", line)
                                        reward_match = re.search(r"reward['\"]:\s*([-\d.]+)", line)
                                        robots_match = re.search(r"robots_on_goal['\"]:\s*(\d+)", line)
                                        
                                        if sr_match:
                                            sr = float(sr_match.group(1))
                                            reward = float(reward_match.group(1)) if reward_match else 0
                                            robots = int(robots_match.group(1)) if robots_match else 0
                                            
                                            # 彩色输出
                                            if sr > 0:
                                                print(f"  ✓ 成功率: {sr:.1%} | 奖励: {reward:.1f} | 到达目标: {robots}")
                                            else:
                                                print(f"  - 成功率: {sr:.1%} | 奖励: {reward:.1f} | 到达目标: {robots}")
                    
                    last_size = current_size
                    last_lines = new_lines[-10:]  # 保留最后10行避免重复
                else:
                    # 显示等待状态
                    print(f"\r[{time.strftime('%H:%M:%S')}] 等待新日志...", end="", flush=True)
            
            time.sleep(2)  # 每2秒检查一次
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    watch_training()


