import asyncio
import logging
from pathlib import Path
import yaml
from typing import Dict, List, Any
import inquirer  # 用于交互式命令行

from core.model_manager import ModelManager
from core.response_handler import ResponseHandler
from core.fusion_engine import FusionEngine

class ModelFusionPipeline:
    def __init__(self):
        # 配置日志
        self.logger = logging.getLogger(__name__)
        
        # 获取项目根目录
        self.project_root = Path(__file__).parent
        
        # 构建配置文件的完整路径
        config_path = self.project_root / "config" / "models_config.yaml"
        
        # 加载配置
        self.config = self.load_config(config_path)
        
        # 初始化组件
        self.model_manager = ModelManager(
            config_path=str(config_path)
        )
        
        self.response_handler = ResponseHandler(
            config=self.config.get('response_handler_config', {})
        )
        
        self.fusion_engine = FusionEngine(
            config=self.config.get('fusion_engine_config', {})
        )
        
        self.logger.info("ModelFusionPipeline initialized")

    def load_config(self, config_path: Path) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            return {}

    async def initialize(self):
        """初始化pipeline"""
        await self.model_manager.initialize()

    async def process_query(
        self,
        query: str,
        model_names: List[str],
        fusion_method: str = 'weighted'
    ) -> Dict[str, Any]:
        """处理查询"""
        try:
            self.logger.info("Getting model responses...")
            # 获取模型响应
            responses = await self.model_manager.query_models(model_names, query)
            
            # 处理响应
            processed_responses = await self.response_handler.process_responses(responses)
            
            # 融合结果
            fusion_result = await self.fusion_engine.fuse_responses(
                processed_responses,
                method=fusion_method
            )
            
            return {
                'status': 'success',
                'final_response': fusion_result.final_response,
                'confidence_score': fusion_result.confidence_score,
                'source_responses': fusion_result.source_responses,
                'fusion_method': fusion_result.fusion_method
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def interactive_session(self):
        """交互式会话"""
        # 获取可用模型
        available_models = list(self.model_manager.models.keys())
        if not available_models:
            print("没有找到可用的模型！")
            return

        while True:
            # 用户选择模型
            questions = [
                inquirer.Checkbox(
                    'selected_models',
                    message="请选择要使用的模型（空格选择，回车确认）",
                    choices=available_models
                ),
                inquirer.Text(
                    'query',
                    message="请输入您的问题（输入 'q' 退出）"
                ),
                inquirer.Text(
                    'rounds',
                    message="请输入每个模型的思考轮数（1-5）",
                    validate=lambda _, x: x.isdigit() and 1 <= int(x) <= 5
                )
            ]
            
            answers = inquirer.prompt(questions)
            
            if not answers or answers['query'].lower() == 'q':
                break
                
            selected_models = answers['selected_models']
            query = answers['query']
            rounds = int(answers['rounds'])
            
            # 执行迭代式回答
            result = await self.process_iterative_query(
                query=query,
                model_names=selected_models,
                rounds=rounds
            )
            
            # 显示结果
            self._display_results(result)

    async def process_iterative_query(
        self,
        query: str,
        model_names: List[str],
        rounds: int
    ) -> Dict[str, Any]:
        """处理迭代式查询"""
        try:
            # 获取每个模型的迭代响应
            model_responses = await self.model_manager.query_models_iteratively(
                model_names=model_names,
                initial_prompt=query,
                rounds=rounds
            )
            
            # 处理响应
            processed_responses = await self.response_handler.process_responses(
                model_responses
            )
            
            # 融合结果
            fusion_result = await self.fusion_engine.fuse_responses(
                processed_responses,
                method='iterative'  # 使用专门的迭代融合方法
            )
            
            return {
                'status': 'success',
                'final_response': fusion_result.final_response,
                'confidence_score': fusion_result.confidence_score,
                'iteration_history': fusion_result.metadata.get('iteration_history', {}),
                'source_responses': fusion_result.source_responses
            }
            
        except Exception as e:
            self.logger.error(f"处理迭代查询时出错: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _display_results(self, result: Dict[str, Any]):
        """显示结果"""
        if result['status'] == 'success':
            print("\n=== 融合结果 ===")
            print(f"最终答案: {result['final_response']}")
            print(f"置信度: {result['confidence_score']:.2f}")
            
            print("\n=== 迭代历史 ===")
            for model, iterations in result['iteration_history'].items():
                print(f"\n模型: {model}")
                for i, response in enumerate(iterations, 1):
                    print(f"第 {i} 轮: {response[:200]}...")
        else:
            print(f"\n处理失败: {result['error']}")

async def main():
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

if __name__ == "__main__":
    asyncio.run(main()) 