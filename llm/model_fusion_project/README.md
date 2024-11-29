# Model Fusion Project

这个项目实现了多模型迭代式问答融合，支持用户交互式选择模型、设置迭代轮数，并通过多种融合策略得到最终答案。

## 功能特点

- 支持多个 Ollama 模型同时使用
- 交互式选择模型和设置参数
- 每个模型支持多轮迭代思考
- 多种融合策略（加权、投票、最佳置信度等）
- 详细的迭代过程展示

## 安装步骤

1. 确保已安装 Python 3.8+ 和 Ollama

2. 下载并解压项目文件

3. 安装依赖：
```bash
cd model_fusion_project
pip install -r requirements.txt
```

## 使用方法

1. 确保 Ollama 服务正在运行

2. 运行主程序：
```bash
python main.py
```

3. 按照提示操作：
   - 选择要使用的模型（空格选择，回车确认）
   - 输入问题
   - 设置迭代轮数（1-5轮）
   - 查看融合结果

## 项目结构

```
model_fusion_project/
├── config/
│   └── models_config.yaml    # 模型配置文件
├── core/
│   ├── model_manager.py      # 模型管理
│   ├── response_handler.py   # 响应处理
│   └── fusion_engine.py      # 融合引擎
├── utils/
│   └── logging_config.py     # 日志配置
├── main.py                   # 主程序
└── requirements.txt          # 依赖列表
```

## 配置说明

在 `config/models_config.yaml` 中可以：
- 设置 API 参数（地址、超时等）
- 配置默认迭代轮数
- 自定义模型参数

## 常见问题

1. 如果遇到模型连接问题，请确保：
   - Ollama 服务正在运行
   - 配置文件中的 API 地址正确

2. 如果需要调整模型参数：
   - 修改 `config/models_config.yaml` 文件

## 依赖要求

- Python 3.8+
- Ollama
- 见 requirements.txt 的详细依赖列表

## 许可证

MIT License