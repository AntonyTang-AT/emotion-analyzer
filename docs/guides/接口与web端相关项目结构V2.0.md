# 项目结构（在线接口服务 + Web端）

## 一、总体架构

系统采用前后端分离架构：
- **后端**：FastAPI 提供 REST API 和 WebSocket 实时流
- **前端**：纯静态 HTML/CSS/JS，通过 API 交互
- **核心算法库**（`core/`）独立，被后端调用

**新增能力**：
- API 返回结果中包含每个片段的**两分支VA**（`va_self` 和 `va_inter`）和 **L4输出的融合权重**（`suggested_fusion_weights`）
- 前端新增**模态信任权重时序图**，展示矛盾场景下系统对不同模态的信任变化
- 长效记忆存储两套 embedding（`va_self` 和 `va_inter`），支持按需检索

## 二、目录结构（更新）

```
project/
│
├── core/                     # 核心算法库（独立，无外部依赖）
│   ├── __init__.py
│   ├── layer1_feature/
│   ├── layer2_predict/        # 两分支MLP
│   ├── layer3_segment/
│   ├── layer4_contradiction/  # 输出权重
│   ├── layer5_report/
│   ├── layer6_personality/
│   ├── pipeline.py            # run_pipeline 返回结果含权重和两套VA
│   ├── models/                # 预训练模型权重
│   └── utils/
│
├── server/                   # 后端服务（FastAPI）
│   ├── __init__.py
│   ├── main.py               # 启动入口
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── endpoints/
│   │       │   ├── analyze.py     # 分析接口（结果含权重）
│   │       │   ├── user.py        # 用户注册/登录/基线管理
│   │       │   ├── memory.py      # 长效记忆检索（支持 embedding_type 参数）
│   │       │   └── personality.py # 人格查询
│   │       └── ws.py              # WebSocket 实时流（返回VA_inter）
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py
│   │   ├── models.py              # ORM 模型（Session 表存储含权重的 result_json）
│   │   └── crud.py
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── chroma_client.py
│   │   └── memory_store.py        # 新增 embedding_type 参数
│   ├── schemas/                   # Pydantic 模型
│   │   ├── user.py
│   │   ├── analyze.py             # 新增权重字段、两套VA字段
│   │   └── personality.py
│   ├── config.py
│   └── utils/
│
├── web/                      # Web前端（纯静态 + JS）
│   ├── index.html
│   ├── login.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── api.js
│   │   ├── upload.js              # 解析新字段，调用权重图表绘制
│   │   ├── camera.js              # WebSocket 实时接收 VA_inter
│   │   └── charts.js              # 新增 drawWeightChart() 绘制权重时序图
│   └── assets/
│
├── data/                     # 数据存储
│   ├── uploads/
│   ├── chroma/               # 存储两套 embedding
│   └── sqlite.db
│
├── scripts/
│   ├── init_db.py
│   ├── train_models.py
│   └── test_api.sh
│
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.web
├── .env.example
├── requirements.txt
└── README.md
```

## 三、各模块职责（更新部分）

### 3.1 core/ 核心算法库（不变，但输出结构扩展）

`run_pipeline()` 返回的字典新增字段：
- `va_self_predictions`: 每个模态的 (v, a, conf) 自建模结果
- `va_inter_predictions`: 每个模态的 (v, a, conf) 交互结果
- `segments[i].contradiction.suggested_fusion_weights`: 4维权重
- `segments[i].contradiction.routing_confidence`
- `raw_visual_features_summary`: 纯视觉旁路统计（如微表情平均强度）

### 3.2 server/schemas/analyze.py（新增/修改）

定义以下 Pydantic 模型：

```python
class VAPredictionSelf(BaseModel):
    valence: float
    arousal: float
    confidence: float

class VAPredictionInter(BaseModel):
    valence: float
    arousal: float
    confidence: float

class ContradictionResult(BaseModel):
    type: str
    intensity: float
    involved_modalities: List[str]
    suggested_fusion_weights: List[float]   # 4个权重
    routing_confidence: float

class SegmentResult(BaseModel):
    id: int
    start_time: float
    end_time: float
    va_self: Dict[str, VAPredictionSelf]   # 按模态名称
    va_inter: Dict[str, VAPredictionInter]
    contradiction: ContradictionResult
    report: str

class AnalysisResponse(BaseModel):
    session_id: str
    segments: List[SegmentResult]
    overall_report: str
    personality: Optional[Dict]
    raw_visual_summary: Optional[Dict]
```

### 3.3 server/api/v1/endpoints/analyze.py

- `POST /upload`：接收视频，调用 `core.run_pipeline`，将返回的完整结果（含权重）存入数据库的 `Session.result_json` 字段，返回 `session_id`。
- `GET /result/{session_id}`：从数据库读取 `result_json`，按上述 Schema 返回。
- `GET /report/{session_id}`：返回整体报告文本（从 result_json 中提取）。

### 3.4 server/api/v1/endpoints/memory.py（修改）

- `POST /similar_fragments` 请求体增加可选参数 `embedding_type`（`"self"` 或 `"inter"`，默认 `"inter"`）。
- 后端根据该参数调用 `memory_store.query_similar(embedding_type=...)`，返回相似片段。

### 3.5 server/vector_store/memory_store.py（修改）

- `add_fragment(user_id, embedding_self, embedding_inter, metadata)`：同时存储两个8维向量（可存两个不同的 collection 或同一 collection 带两个向量字段）。
- `query_similar(user_id, query_embedding, top_k=5, time_decay=True, embedding_type='inter')`：根据 `embedding_type` 选择使用哪个向量进行检索。

### 3.6 server/db/models.py（不变）

`Session` 表的 `result_json` 字段现在存储含权重的完整结果。

### 3.7 web/js/charts.js（新增函数）

```javascript
// 绘制模态信任权重时序图（堆叠面积图）
function drawWeightChart(segments) {
    // 输入 segments 数组，每个元素包含 contradiction.suggested_fusion_weights
    // 横轴：片段编号或时间中点，纵轴：权重（0~1）
    // 使用 ECharts 或 Chart.js 绘制堆叠面积图，显示四个模态的权重变化
}
```

### 3.8 web/js/upload.js（修改）

- 解析 API 返回的 `segments`，提取每个片段的 `contradiction.suggested_fusion_weights`。
- 调用 `drawWeightChart(segments)` 在结果页增加权重图卡片。
- 原有的 VA 曲线图改为基于加权平均后的 VA_inter 绘制（权重由 L4 提供）。

### 3.9 web/js/camera.js（不变）

- 实时模式下，WebSocket 每帧返回当前时刻的 VA_inter 值（单点），前端实时绘制曲线。权重图在会话结束后生成。

## 四、数据流说明（更新）

### 1. 用户上传视频

```
Web → POST /upload → API 保存临时文件 → 调用 core.run_pipeline
  → 返回结果（含 va_self, va_inter, 矛盾权重, 纯视觉摘要）
  → 存储到数据库 → 返回 session_id
Web → GET /result/{session_id} → 获取完整结果 → 前端绘制 VA 曲线（加权）、权重时序图、热力图、报告
```

### 2. 实时摄像头分析

```
Web → 打开 WebSocket → 每秒发送帧（base64）
API → 调用 core 单帧预测（L1+L2，只输出 VA_inter）→ 推送 VA_inter 点
Web → 实时绘制 VA_inter 曲线
结束时 → API 触发完整流水线（L3-L6）生成报告和权重 → 通过 WebSocket 推送最终结果
```

### 3. 冷启动（新用户）

- 新用户无基线，第一次上传分析时，服务器根据当前会话的前几个片段的 **VA_self** 向量，调用记忆检索接口（`embedding_type='self'`）获取相似用户，融合得到临时 ΔVA。
- 该临时基线仅用于本次会话的 **VA_self** 校准。

### 4. 数据库与向量库更新

- 每次分析完成后，将 `va_self_embedding` 和 `va_inter_embedding` 分别存入 Chroma（同一文档的不同字段或两个 collection）。
- `Session` 表存储完整的 `result_json`，包含权重信息。

## 五、部署方式（不变）

### 5.1 开发环境（单机）

- 使用 SQLite，无需 Docker。
- 启动 API：`cd server && uvicorn main:app --reload --port 8000`
- 启动前端静态服务：`cd web && python -m http.server 3000`
- 访问 `http://localhost:3000`

### 5.2 生产环境（Docker Compose）

服务组成：PostgreSQL + API + Nginx (Web) + Chroma

`docker-compose.yml` 需确保 Chroma 容器持久化存储两套 embedding。

## 六、优势总结（更新）

- **核心算法独立**：`core/` 可单独测试，返回丰富的结果结构。
- **REST API 清晰**：外部系统可轻松获取权重、两套VA、纯视觉摘要。
- **Web端增强**：新增权重时序图，提升可解释性；VA曲线使用动态融合权重更准确。
- **记忆检索灵活**：支持按 `va_self`（用户风格）或 `va_inter`（交互场景）检索，适配不同需求。
- **冷启动更准确**：使用自建模VA计算相似用户，反映真实表达倾向。

---

**文档版本**：V2.0（整合两分支VA、L4权重输出、纯视觉旁路）  
**最后更新**：2026-06-05