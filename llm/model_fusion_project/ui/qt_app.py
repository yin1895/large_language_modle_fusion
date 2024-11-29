from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QTextEdit, QPushButton, QComboBox,
    QLabel, QSpinBox, QListWidget, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Any
from qasync import QEventLoop, asyncSlot

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.model_manager import ModelManager
from core.response_handler import ResponseHandler
from core.fusion_engine import FusionEngine

class MainWindow(QMainWindow):
    def __init__(self, loop: QEventLoop):
        super().__init__()
        self.setWindowTitle("模型融合系统")
        self.setMinimumSize(800, 600)
        
        # 保存事件循环
        self.loop = loop
        
        # 初始化组件
        self.model_manager = ModelManager()
        self.response_handler = ResponseHandler()
        self.fusion_engine = FusionEngine()
        
        # 初始化UI
        self.init_ui()
    
    async def init_models(self):
        """异步初始化模型"""
        try:
            await self.model_manager.initialize()
            models = list(self.model_manager.models.keys())
            self.model_list.addItems(models)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化模型失败: {str(e)}")
    
    def init_ui(self):
        """初始化UI组件"""
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QHBoxLayout()
        main_widget.setLayout(layout)
        
        # 左侧面板（配置区）
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(300)
        
        # 模型选择
        left_layout.addWidget(QLabel("可用模型:"))
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        left_layout.addWidget(self.model_list)
        
        # 轮数设置
        rounds_layout = QHBoxLayout()
        rounds_layout.addWidget(QLabel("思考轮数:"))
        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(1, 5)
        self.rounds_spin.setValue(3)
        rounds_layout.addWidget(self.rounds_spin)
        left_layout.addLayout(rounds_layout)
        
        # 右侧面板（输入输出区）
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # 输入区
        right_layout.addWidget(QLabel("输入问题:"))
        self.input_text = QTextEdit()
        self.input_text.setMaximumHeight(100)
        right_layout.addWidget(self.input_text)
        
        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.on_process_clicked)
        right_layout.addWidget(self.process_btn)
        
        # 输出区
        right_layout.addWidget(QLabel("处理结果:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        right_layout.addWidget(self.output_text)
        
        # 添加到主布局
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
    
    @asyncSlot()
    async def on_process_clicked(self):
        """处理按钮点击事件"""
        # 获取选中的模型
        selected_models = [
            item.text() 
            for item in self.model_list.selectedItems()
        ]
        
        if not selected_models:
            QMessageBox.warning(self, "警告", "请至少选择一个模型")
            return
        
        # 获取输入
        query = self.input_text.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "警告", "请输入问题")
            return
        
        # 禁用处理按钮
        self.process_btn.setEnabled(False)
        self.process_btn.setText("处理中...")
        
        try:
            # 获取模型响应
            responses = await self.model_manager.query_models_iteratively(
                model_names=selected_models,
                initial_prompt=query,
                rounds=self.rounds_spin.value()
            )
            
            # 处理响应
            processed_responses = await self.response_handler.process_responses(responses)
            
            # 融合结果
            fusion_result = await self.fusion_engine.fuse_responses(
                processed_responses,
                method='iterative'
            )
            
            # 显示结果
            self.display_result({
                'status': 'success',
                'final_response': fusion_result.final_response,
                'confidence_score': fusion_result.confidence_score,
                'iteration_history': fusion_result.metadata.get('iteration_history', {}),
                'source_responses': fusion_result.source_responses
            })
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
        
        finally:
            # 恢复按钮状态
            self.process_btn.setEnabled(True)
            self.process_btn.setText("开始处理")
    
    def display_result(self, result: Dict[str, Any]):
        """显示处理结果"""
        if result['status'] == 'success':
            output = "=== 融合结果 ===\n"
            output += f"{result['final_response']}\n\n"
            output += f"置信度: {result['confidence_score']:.2f}\n\n"
            
            output += "=== 迭代历史 ===\n"
            for model, iterations in result['iteration_history'].items():
                output += f"\n模型: {model}\n"
                for i, response in enumerate(iterations, 1):
                    output += f"第 {i} 轮: {response[:200]}...\n"
            
            self.output_text.setText(output)

async def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # 创建事件循环
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # 创建主窗口
    window = MainWindow(loop)
    window.show()
    
    # 运行事件循环
    with loop:
        await loop.create_future()  # 保持程序运行

if __name__ == "__main__":
    asyncio.run(main())