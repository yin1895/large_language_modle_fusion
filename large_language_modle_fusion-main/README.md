# Model Fusion Project

这个项目实现了多模型迭代式问答融合，支持用户通过命令行或图形界面进行交互，可以选择模型、设置迭代轮数，并通过多种融合策略得到最终答案。

## 功能特点

- 支持多个 Ollama 模型同时使用
- 提供命令行和图形界面两种交互方式
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

2. 运行程序：

   命令行界面：
   ```bash
   python main.py
   ```

   图形界面：
   ```bash
   python main.py --ui
   ```

3. 使用说明：
   - 命令行界面：按照提示选择模型、输入问题、设置轮数
   - 图形界面：
     - 在左侧列表中选择要使用的模型（可多选）
     - 设置思考轮数（1-5轮）
     - 在输入框中输入问题
     - 点击"开始处理"按钮
     - 在输出区查看处理结果

## 项目结构

```
model_fusion_project/
├── config/
│   └── models_config.yaml    # 模型配置文件
├── core/
│   ├── model_manager.py      # 模型管理
│   ├── response_handler.py   # 响应处理
│   └── fusion_engine.py      # 融合引擎
├── ui/
│   └── qt_app.py            # 图形界面
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

3. 如果图形界面无法启动：
   - 确保已安装 PyQt6 和 qasync
   - 检查 Python 版本是否兼容

## 依赖要求

- Python 3.8+
- Ollama
- PyQt6（图形界面）
- 其他依赖见 requirements.txt

## 更新日志

- v1.1.0: 添加图形界面支持
- v1.0.0: 初始版本，命令行界面

## 许可证

MIT License