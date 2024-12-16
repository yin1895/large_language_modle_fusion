from typing import Dict, List, Any, Optional
import re
import json
from datetime import datetime
import logging
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

@dataclass
class ProcessedResponse:
    """处理后的响应数据结构"""
    model_name: str
    original_response: str
    cleaned_response: str
    response_length: int
    processing_time: float
    confidence_score: float
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

class ResponseHandler:
    """响应处理器：处理和标准化模型输出"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化响应处理器
        
        Args:
            config: 配置字典，包含处理参数
        """
        self.config = config or {}
        self.embedding_model = None
        self.tokenizer = None
        
        # 初始化embedding模型（按需加载）
        if self.config.get('use_embeddings', False):
            self._init_embedding_model()
            
        logging.info("ResponseHandler initialized")
    
    def _init_embedding_model(self) -> None:
        """初始化embedding模型"""
        try:
            model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)
            logging.info(f"Embedding model {model_name} loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {str(e)}")
            self.embedding_model = None
            self.tokenizer = None
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 输入文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
        
        # 统一标点符号
        text = text.replace('，', ',').replace('。', '.')
        
        return text
    
    def _calculate_confidence(self, response: str) -> float:
        """
        计算响应的置信度分数
        
        Args:
            response: 模型响应
            
        Returns:
            float: 置信度分数 (0-1)
        """
        # 这里实现一个简单的启发式计算方法
        # 可以根据具体需求扩展更复杂的计算方法
        
        # 基于响应长度的评分
        length_score = min(len(response.split()) / 100, 1.0)
        
        # 基于标点符号使用的评分
        punctuation_score = len(re.findall(r'[.,!?;:]', response)) / len(response.split())
        punctuation_score = min(punctuation_score, 1.0)
        
        # 基于是否包含常见停用词的评分
        common_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at'}
        words = set(response.lower().split())
        common_words_score = len(words.intersection(common_words)) / len(common_words)
        
        # 综合评分
        confidence = (length_score + punctuation_score + common_words_score) / 3
        return round(confidence, 3)
    
    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        生成文本的embedding向量
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: embedding向量
        """
        if not self.embedding_model:
            return None
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            
            # 使用[CLS]标记的输出作为句子表示
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            return embedding[0]  # 返回第一个样本的embedding
            
        except Exception as e:
            logging.error(f"Failed to generate embedding: {str(e)}")
            return None
    
    async def process_response(
        self,
        model_name: str,
        response: Dict[str, Any]
    ) -> ProcessedResponse:
        """
        处理单个模型响应
        
        Args:
            model_name: 模型名称
            response: 模型原始响应
            
        Returns:
            ProcessedResponse: 处理后的响应对象
        """
        start_time = datetime.now()
        
        try:
            # 提取原始响应文本
            original_text = response.get('response', '')
            
            # 清理文本
            cleaned_text = self._clean_text(original_text)
            
            # 计算置信度
            confidence = self._calculate_confidence(cleaned_text)
            
            # 生成embedding（如果启用）
            embedding = await self._generate_embedding(cleaned_text) if self.embedding_model else None
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 创建处理后的响应对象
            processed_response = ProcessedResponse(
                model_name=model_name,
                original_response=original_text,
                cleaned_response=cleaned_text,
                response_length=len(cleaned_text.split()),
                processing_time=processing_time,
                confidence_score=confidence,
                embedding=embedding,
                metadata={
                    'processed_at': datetime.now().isoformat(),
                    'original_metadata': response.get('metadata', {})
                }
            )
            
            return processed_response
            
        except Exception as e:
            logging.error(f"Error processing response from {model_name}: {str(e)}")
            raise
    
    async def process_responses(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理模型响应，增加更智能的置信度计算"""
        processed = []
        for response in responses:
            # 计算响应质量指标
            response_text = response['response']
            confidence = await self._calculate_confidence(response_text)
            
            processed.append({
                'model_name': response['model_name'],
                'response': response_text,
                'confidence': confidence,
                'metadata': {
                    **response.get('metadata', {}),
                    'length': len(response_text),
                    'quality_metrics': await self._get_quality_metrics(response_text)
                }
            })
        return processed
    
    def get_response_statistics(
        self,
        processed_responses: List[ProcessedResponse]
    ) -> Dict[str, Any]:
        """
        计算响应的统计信息
        
        Args:
            processed_responses: 处理后的响应列表
            
        Returns:
            Dict: 统计信息
        """
        if not processed_responses:
            return {}
            
        confidence_scores = [r.confidence_score for r in processed_responses]
        response_lengths = [r.response_length for r in processed_responses]
        
        return {
            'response_count': len(processed_responses),
            'average_confidence': np.mean(confidence_scores),
            'confidence_std': np.std(confidence_scores),
            'average_length': np.mean(response_lengths),
            'length_std': np.std(response_lengths),
            'models_used': [r.model_name for r in processed_responses]
        }

    async def _calculate_confidence(self, text: str) -> float:
        """计算响应的置信度"""
        try:
            # 基于多个因素计算置信度
            coherence = await self._assess_coherence(text)
            completeness = await self._assess_completeness(text)
            length_score = await self._assess_length(text)
            
            # 加权平均
            weights = self.config.get('confidence_weights', {
                'coherence': 0.4,
                'completeness': 0.4,
                'length': 0.2
            })
            
            confidence = (
                coherence * weights['coherence'] +
                completeness * weights['completeness'] +
                length_score * weights['length']
            )
            
            return min(max(confidence, 0.0), 1.0)  # 确保在 0-1 范围内
            
        except Exception as e:
            logging.error(f"计算置信度时出错: {str(e)}")
            return 0.5  # 出错时返回中等置信度

    async def _assess_coherence(self, text: str) -> float:
        """评估文本的连贯性"""
        try:
            # 简单的连贯性评估
            sentences = text.split('。')
            if len(sentences) < 2:
                return 0.5
            
            # 基于句子长度的简单评估
            lengths = [len(s.strip()) for s in sentences if s.strip()]
            avg_length = sum(lengths) / len(lengths)
            variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            
            # 长度适中且变化不大的句子可能更连贯
            coherence_score = 1.0 - min(variance / (avg_length * 2), 0.5)
            
            return coherence_score
            
        except Exception as e:
            logging.error(f"评估连贯性时出错: {str(e)}")
            return 0.5

    async def _assess_completeness(self, text: str) -> float:
        """评估响应的完整性"""
        try:
            # 简单的完整性评估
            # 检查是否有明显的结束标志
            has_conclusion = any(marker in text.lower() for marker in [
                '总之', '因此', '所以', '综上所述', '最后', '结论'
            ])
            
            # 检查最小长度要求
            min_length = self.config.get('quality_thresholds', {}).get('min_length', 50)
            meets_length = len(text) >= min_length
            
            # 综合评分
            completeness_score = (
                0.7 * meets_length +  # 长度权重
                0.3 * has_conclusion  # 结论标志权重
            )
            
            return completeness_score
            
        except Exception as e:
            logging.error(f"评估完整性时出错: {str(e)}")
            return 0.5

    async def _assess_length(self, text: str) -> float:
        """评估文本长度的适当性"""
        try:
            # 获取配置的长度阈值
            thresholds = self.config.get('quality_thresholds', {})
            min_length = thresholds.get('min_length', 50)
            max_length = thresholds.get('max_length', 2000)
            optimal_length = (min_length + max_length) / 2
            
            # 计算长度分数
            text_length = len(text)
            if text_length < min_length:
                return text_length / min_length
            elif text_length > max_length:
                return max_length / text_length
            else:
                # 在理想范围内，距离最佳长度越近分数越高
                distance_from_optimal = abs(text_length - optimal_length)
                max_distance = optimal_length - min_length
                return 1.0 - (distance_from_optimal / max_distance / 2)
                
        except Exception as e:
            logging.error(f"评估长度时出错: {str(e)}")
            return 0.5

    async def _get_quality_metrics(self, text: str) -> Dict[str, float]:
        """获取响应质量的详细指标"""
        try:
            return {
                'coherence': await self._assess_coherence(text),
                'completeness': await self._assess_completeness(text),
                'length_score': await self._assess_length(text)
            }
        except Exception as e:
            logging.error(f"获取质量指标时出错: {str(e)}")
            return {
                'coherence': 0.5,
                'completeness': 0.5,
                'length_score': 0.5
            }

# 使用示例
async def example_usage():
    # 创建响应处理器实例
    config = {
        'use_embeddings': True,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
    }
    handler = ResponseHandler(config)
    
    # 示例响应
    sample_responses = [
        {
            'model': 'llama2:7b',
            'response': 'The sky appears blue due to Rayleigh scattering.',
            'metadata': {'temperature': 0.7}
        },
        {
            'model': 'mistral:7b',
            'response': 'The blue color of the sky is caused by the scattering of sunlight.',
            'metadata': {'temperature': 0.7}
        }
    ]
    
    # 处理响应
    processed_responses = await handler.process_responses(sample_responses)
    
    # 打印处理结果
    for response in processed_responses:
        print(f"\n=== {response.model_name} 处理结果 ===")
        print(f"清理后的响应: {response.cleaned_response}")
        print(f"置信度分数: {response.confidence_score}")
        print(f"处理时间: {response.processing_time:.3f}秒")
        print("="*50)
    
    # 获取统计信息
    stats = handler.get_response_statistics(processed_responses)
    print("\n响应统计信息:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # 运行示例
    import asyncio
    asyncio.run(example_usage())
