import logging
from typing import Dict, List, Any
import inquirer
from .model_manager import ModelManager
from .response_handler import ResponseHandler
from .fusion_engine import FusionEngine

class ModelFusionPipeline:
    def __init__(self):
        self.model_manager = ModelManager()
        self.response_handler = ResponseHandler()
        self.fusion_engine = FusionEngine()
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """初始化pipeline"""
        await self.model_manager.initialize()
        self.logger.info("ModelFusionPipeline initialized")

    async def interactive_session(self):
        """交互式会话"""
        while True:
            # 获取可用模型
            available_models = list(self.model_manager.models.keys())
            if not available_models:
                print("没有找到可用的模型！")
                return

            # 用户选择
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
                ),
                inquirer.List(
                    'fusion_method',
                    message="请选择融合方法",
                    choices=['weighted', 'voting', 'best_confidence', 'iterative']
                )
            ]
            
            answers = inquirer.prompt(questions)
            
            if not answers or answers['query'].lower() == 'q':
                break
                
            selected_models = answers['selected_models']
            query = answers['query']
            rounds = int(answers['rounds'])
            fusion_method = answers['fusion_method']
            
            # 处理查询
            result = await self.process_query(
                query=query,
                model_names=selected_models,
                rounds=rounds,
                fusion_method=fusion_method
            )
            
            # 显示结果
            self._display_results(result)

    async def process_query(
        self,
        query: str,
        model_names: List[str],
        rounds: int,
        fusion_method: str
    ) -> Dict[str, Any]:
        """处理查询"""
        try:
            # 获取模型响应
            responses = await self.model_manager.query_models_iteratively(
                model_names=model_names,
                initial_prompt=query,
                rounds=rounds
            )
            
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
                'iteration_history': fusion_result.metadata.get('iteration_history', {}),
                'source_responses': fusion_result.source_responses
            }
            
        except Exception as e:
            self.logger.error(f"处理查询时出错: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _display_results(self, result: Dict[str, Any]):
        """显示处理结果"""
        if result['status'] == 'success':
            print("\n=== 融合结果 ===")
            print(result['final_response'])
            print(f"\n置信度: {result['confidence_score']:.2f}")
            
            print("\n=== 迭代历史 ===")
            for model, iterations in result['iteration_history'].items():
                print(f"\n模型: {model}")
                for i, response in enumerate(iterations, 1):
                    print(f"第 {i} 轮: {response[:200]}...")
        else:
            print(f"\n处理失败: {result['error']}") 