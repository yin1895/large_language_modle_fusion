from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import numpy as np

@dataclass
class FusionResult:
    """融合结果数据结构"""
    final_response: str
    confidence_score: float
    source_responses: List[Dict[str, Any]]
    fusion_method: str
    metadata: Dict[str, Any]

class FusionEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        logging.info("FusionEngine initialized")

    async def fuse_responses(
        self,
        responses: List[Dict[str, Any]],
        method: str = 'iterative'
    ) -> FusionResult:
        """
        根据指定方法融合响应
        
        Args:
            responses: 响应列表
            method: 融合方法 ('weighted', 'voting', 'best_confidence', 'iterative')
        """
        fusion_methods = {
            'weighted': self._weighted_fusion,
            'voting': self._voting_fusion,
            'best_confidence': self._best_confidence_fusion,
            'iterative': self._iterative_fusion
        }
        
        if method not in fusion_methods:
            raise ValueError(f"不支持的融合方法: {method}")
        
        fusion_func = fusion_methods[method]
        return await fusion_func(responses)

    async def _iterative_fusion(
        self,
        responses: List[Dict[str, Any]]
    ) -> FusionResult:
        """迭代式融合方法"""
        # 按模型分组
        model_responses = {}
        for response in responses:
            model_name = response['model_name']
            if model_name not in model_responses:
                model_responses[model_name] = []
            model_responses[model_name].append(response)
        
        # 获取每个模型的最终（最优）响应
        final_responses = []
        iteration_history = {}
        
        for model_name, model_iterations in model_responses.items():
            # 记录迭代历史
            iteration_history[model_name] = [
                r['response'] for r in model_iterations
            ]
            
            # 选择最后一轮或最高置信度的响应
            best_response = max(
                model_iterations,
                key=lambda x: x['confidence']
            )
            final_responses.append(best_response)
        
        # 从最终响应中选择最佳答案
        best_final = max(
            final_responses,
            key=lambda x: x['confidence']
        )
        
        return FusionResult(
            final_response=best_final['response'],
            confidence_score=best_final['confidence'],
            source_responses=responses,
            fusion_method='iterative',
            metadata={
                'iteration_history': iteration_history,
                'model_confidences': {
                    r['model_name']: r['confidence'] 
                    for r in final_responses
                }
            }
        )

    async def _weighted_fusion(
        self,
        responses: List[Dict[str, Any]]
    ) -> FusionResult:
        """加权融合方法"""
        if not responses:
            raise ValueError("没有响应可供融合")

        # 使用置信度作为权重
        weights = np.array([r['confidence'] for r in responses])
        weights = weights / np.sum(weights)
        
        # 选择权重最高的响应
        max_weight_idx = np.argmax(weights)
        selected_response = responses[max_weight_idx]
        
        return FusionResult(
            final_response=selected_response['response'],
            confidence_score=selected_response['confidence'],
            source_responses=responses,
            fusion_method='weighted',
            metadata={
                'weights': weights.tolist(),
                'selection_index': int(max_weight_idx)
            }
        )

    async def _voting_fusion(
        self,
        responses: List[Dict[str, Any]]
    ) -> FusionResult:
        """投票融合方法"""
        if not responses:
            raise ValueError("没有响应可供融合")

        # 选择置信度最高的响应
        best_response = max(responses, key=lambda x: x['confidence'])
        
        return FusionResult(
            final_response=best_response['response'],
            confidence_score=best_response['confidence'],
            source_responses=responses,
            fusion_method='voting',
            metadata={
                'total_responses': len(responses)
            }
        )

    async def _best_confidence_fusion(
        self,
        responses: List[Dict[str, Any]]
    ) -> FusionResult:
        """最佳置信度融合方法"""
        if not responses:
            raise ValueError("没有响应可供融合")

        # 选择置信度最高的响应
        best_response = max(responses, key=lambda x: x['confidence'])
        
        return FusionResult(
            final_response=best_response['response'],
            confidence_score=best_response['confidence'],
            source_responses=responses,
            fusion_method='best_confidence',
            metadata={
                'all_confidences': [r['confidence'] for r in responses]
            }
        )
