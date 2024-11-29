import logging
import yaml
from typing import Dict, List, Any
import aiohttp
import json
import asyncio
from pathlib import Path

class ModelManager:
    def __init__(self, config_path: str = None):
        # 首先初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 获取项目根目录
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "models_config.yaml"
        
        # 加载配置
        self.config = self.load_config(str(config_path))
        self.base_url = self.config.get('api', {}).get('base_url', 'http://localhost:11434/api')
        self.timeout = self.config.get('api', {}).get('timeout', 30)
        self.models = {}
        self._response_cache = {}
        self._session = None

    def load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            return {}

    async def initialize(self):
        """初始化模型管理器"""
        try:
            # 创建持久化session
            self._session = aiohttp.ClientSession()
            
            # 获取可用模型
            available_models = await self.get_available_models()
            self.logger.info(f"发现可用模型: {available_models}")
            
            # 为每个可用模型创建配置
            self.models = {
                model_name: {
                    'name': model_name,
                    'type': 'ollama',
                    'parameters': {
                        'temperature': 0.7,
                        'max_tokens': 1000
                    },
                    'metadata': {
                        'description': f"{model_name} via Ollama API"
                    }
                }
                for model_name in available_models
            }
            
        except Exception as e:
            self.logger.error(f"初始化模型时出错: {str(e)}")
            raise

    async def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/tags",
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        raise Exception(f"获取模型列表失败: {response.status}")
                    data = await response.json()
                    return [model['name'] for model in data['models']]
        except Exception as e:
            self.logger.error(f"获取可用模型列表失败: {str(e)}")
            return []

    async def query_models_iteratively(
        self,
        model_names: List[str],
        initial_prompt: str,
        rounds: int
    ) -> List[Dict[str, Any]]:
        """迭代式查询多个模型"""
        all_responses = []
        
        for model_name in model_names:
            model_responses = await self._iterate_model_query(
                model_name=model_name,
                initial_prompt=initial_prompt,
                rounds=rounds
            )
            all_responses.extend(model_responses)
        
        return all_responses

    async def _iterate_model_query(
        self,
        model_name: str,
        initial_prompt: str,
        rounds: int
    ) -> List[Dict[str, Any]]:
        """对单个模型进行迭代查询"""
        responses = []
        current_prompt = initial_prompt
        
        for round_num in range(rounds):
            # 构建当前轮次的提示
            if round_num > 0:
                current_prompt = (
                    f"前一轮的回答是：\n{responses[-1]['response']}\n\n"
                    f"{self.config['iteration']['system_prompt']}"
                )
            
            # 获取响应
            response = await self._do_query(model_name, current_prompt)
            
            # 添加迭代信息
            response['iteration'] = {
                'round': round_num + 1,
                'total_rounds': rounds,
                'previous_response': responses[-1]['response'] if responses else None
            }
            
            responses.append(response)
            
        return responses

    async def _do_query(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """执行实际的模型查询"""
        if model_name not in self.models:
            raise ValueError(f"未知的模型: {model_name}")

        model_config = self.models[model_name]
        
        # 构建请求数据
        request_data = {
            "model": model_name,
            "prompt": prompt,
            **model_config['parameters']
        }

        try:
            async with self._session.post(
                f"{self.base_url}/generate",
                json=request_data,
                timeout=self.timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API请求失败: {error_text}")
                
                # 读取并处理 ndjson 响应
                full_response = ""
                async for line in response.content:
                    try:
                        if line:  # 确保行不为空
                            data = json.loads(line)
                            if 'response' in data:
                                full_response += data['response']
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"解析响应行时出错: {str(e)}")
                        continue
                
                return {
                    'model_name': model_name,
                    'response': full_response,
                    'metadata': {
                        'model_type': model_config['type'],
                        'parameters': model_config['parameters']
                    }
                }

        except asyncio.TimeoutError:
            self.logger.error(f"模型 {model_name} 请求超时")
            raise
        except Exception as e:
            self.logger.error(f"查询模型 {model_name} 时出错: {str(e)}")
            raise
