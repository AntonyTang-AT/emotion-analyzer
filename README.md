# Emotion Analyzer – 多模态情感计算系统

> 基于六层流水线架构的 AI 情感分析系统，支持视频上传、实时摄像头分析、多模态矛盾检测、生成式报告与大五人格回归。

## 📌 项目简介

本项目是一个端到端的情感计算系统，能够从视频中提取文本、语音、宏表情、微表情四模态特征，预测效价（Valence）与唤醒度（Arousal），动态分段，检测模态间矛盾，生成自然语言报告，并推断用户的大五人格特质。

**主要特性：**

- 🎥 **视频分析**：上传视频文件，获得完整的 VA 曲线、矛盾热力图、文本报告。
- 🎙️ **实时摄像头分析**：通过 WebSocket 逐帧处理，实时绘制 VA 轨迹。
- 🧠 **长效记忆**：基于 Chroma 向量数据库存储历史片段，支持冷启动与个性化基线。
- 📊 **可视化输出**：VA 时序曲线、模态一致性热力图、人格雷达图。
- 🧩 **模块化设计**：六层架构完全解耦，支持配置式启用/跳过任一层次。

## 🏗️ 技术架构

```
视频输入 → L1 多模态特征提取 → L2 单模态 VA 预测 → L3 自适应分段 → L4 矛盾检测 → L5 生成式报告 → L6 人格回归
```

| 层级 | 名称 | 核心技术 |
|------|------|----------|
| L1 | 特征提取 | Whisper + BERT, Wav2Vec2+FoX, MediaPipe+VideoMAE, DGM+GCN |
| L2 | VA 预测 | 轻量 MLP + 多任务联合优化 (EmoLILMs) |
| L3 | 分段与个性化 | 动态分段控制器 + 基线校准 + 冷启动 + Chroma 记忆库 |
| L4 | 矛盾检测 | VA 空间距离 + QBTD 阈值 + 专家规则 |
| L5 | 报告生成 | DeepSeek/GPT 生成片段/整体报告 + 可视化 (Plotly/Matplotlib) |
| L6 | 人格回归 | LightGBM / 高斯过程，输出大五人格分数 |

后端：FastAPI + SQLAlchemy + SQLite/PostgreSQL  
前端：纯 HTML/CSS/JS + ECharts/Chart.js  
部署：Docker Compose (可选)

## 🚀 快速开始

### 环境要求

- Python 3.10
- Git
- （可选）CUDA 支持 GPU 加速

### 1. 克隆仓库

```bash
git clone https://github.com/AntonyTang-AT/emotion-analyzer.git
cd emotion-analyzer
```

### 2. 创建虚拟环境

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

或使用 conda：

```bash
conda create -n emotion python=3.10
conda activate emotion
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，并填写必要的密钥（如 OpenAI API Key，如不使用大模型可留空）。

```bash
cp .env.example .env
```

### 5. 初始化数据库

```bash
python scripts/init_db.py
```

### 6. 启动后端 API

```bash
cd server
uvicorn main:app --reload --port 8000
```

### 7. 启动前端静态服务（新开终端）

```bash
cd web
python -m http.server 3000
```

### 8. 访问应用

打开浏览器：http://localhost:3000

## 📂 目录结构

```
emotion-analyzer/
├── config/               # 配置文件（YAML）
├── src/                  # 核心算法库（六层模块化）
│   ├── layer1_feature/   # 多模态特征提取
│   ├── layer2_predict/   # VA 预测
│   ├── layer3_segment/   # 自适应分段 + 记忆
│   ├── layer4_contradiction/ # 矛盾检测
│   ├── layer5_report/    # 报告生成
│   ├── layer6_personality/   # 人格回归
│   ├── pipeline/         # 流水线控制器
│   └── utils/            # 通用工具
├── server/               # FastAPI 后端
│   ├── api/              # 路由层
│   ├── db/               # 数据库模型与操作
│   ├── vector_store/     # Chroma 封装
│   └── schemas/          # Pydantic 模型
├── web/                  # 前端静态文件
│   ├── index.html
│   ├── js/               # API 调用、摄像头、图表
│   └── css/
├── data/                 # 运行时数据（上传文件、数据库、向量库）
├── logs/                 # 日志
├── outputs/              # 生成报告、图表、人格结果
├── scripts/              # 辅助脚本（训练、初始化、测试）
├── tests/                # 单元测试
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 🧪 测试

运行所有单元测试：

```bash
pytest tests/
```

运行特定层测试：

```bash
pytest tests/test_layer1/
```

## 🐳 Docker 部署（可选）

```bash
docker-compose up -d
```

服务启动后：
- API: http://localhost:8000
- Web: http://localhost
- Chroma: http://localhost:8001

## 🤝 贡献指南

本项目采用 **GitHub Flow** 协作模式。

1. 从 `main` 分支创建功能分支：`git checkout -b feature/your-feature`
2. 提交更改，使用约定式提交格式：`feat:`, `fix:`, `docs:`, `test:` 等
3. 推送分支并创建 Pull Request，至少需要 1 人 Review
4. 通过 CI 检查后合并到 `main`

详细开发规范请参考 [Wiki](https://github.com/AntonyTang-AT/emotion-analyzer/wiki)（建设中）。

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。
