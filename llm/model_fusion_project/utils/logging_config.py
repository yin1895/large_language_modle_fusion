import logging
import sys
from pathlib import Path

def setup_logging(log_dir: str = "logs"):
    """配置日志系统"""
    # 创建日志目录
    Path(log_dir).mkdir(exist_ok=True)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/fusion.log")
        ]
    )

    # 为各个模块配置独立的日志记录器
    loggers = {
        'model_manager': logging.getLogger('core.model_manager'),
        'response_handler': logging.getLogger('core.response_handler'),
        'fusion_engine': logging.getLogger('core.fusion_engine')
    }

    for logger in loggers.values():
        logger.setLevel(logging.INFO) 