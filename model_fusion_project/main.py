import logging
import argparse
import sys
from PyQt6.QtWidgets import QApplication
import qasync
import asyncio
from pathlib import Path

async def run_cli():
    """运行命令行界面"""
    from core.model_fusion_pipeline import ModelFusionPipeline
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并初始化pipeline
    pipeline = ModelFusionPipeline()
    await pipeline.initialize()
    
    # 启动交互式会话
    await pipeline.interactive_session()

def run_gui():
    """运行图形界面"""
    from ui.qt_app import MainWindow
    
    def close_future(future, loop):
        loop.call_later(10, future.cancel)
        future.cancel()

    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # 创建主窗口
    window = MainWindow(loop)
    window.show()

    future = asyncio.Future()
    app.aboutToQuit.connect(lambda: close_future(future, loop))

    with loop:
        loop.run_until_complete(window.init_models())  # 等待初始化完成
        loop.run_forever()

def main():
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='模型融合系统')
    parser.add_argument(
        '--ui',
        action='store_true',
        help='启动图形界面'
    )
    args = parser.parse_args()

    try:
        if args.ui:
            print("启动图形界面...")
            run_gui()  # 不再使用 qasync.run
        else:
            print("启动命令行界面...")
            asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\n程序已退出")
        sys.exit(0)

if __name__ == "__main__":
    main()