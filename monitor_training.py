"""
训练监控脚本
实时查看训练进度
"""

import os
import time
import glob
from pathlib import Path

def monitor_training():
    """监控训练进度"""
    log_dir = Path("results/logs")
    checkpoint_dir = Path("results/checkpoints")
    
    print("="*60)
    print("训练监控")
    print("="*60)
    print("\n监控训练日志和检查点...")
    print("按 Ctrl+C 退出\n")
    
    last_checkpoint = None
    last_log = None
    
    try:
        while True:
            # 检查检查点
            checkpoints = sorted(glob.glob(str(checkpoint_dir / "*.pth")), key=os.path.getmtime)
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                if latest_checkpoint != last_checkpoint:
                    print(f"[{time.strftime('%H:%M:%S')}] 新检查点: {os.path.basename(latest_checkpoint)}")
                    last_checkpoint = latest_checkpoint
            
            # 检查日志文件
            log_files = sorted(glob.glob(str(log_dir / "*.log")), key=os.path.getmtime)
            if log_files:
                latest_log = log_files[-1]
                if latest_log != last_log:
                    # 读取最后几行
                    try:
                        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            if lines:
                                print(f"\n最新日志 ({os.path.basename(latest_log)}):")
                                for line in lines[-5:]:
                                    print(f"  {line.strip()}")
                        last_log = latest_log
                    except:
                        pass
            
            time.sleep(5)  # 每5秒检查一次
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    monitor_training()


