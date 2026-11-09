# 多模态模型部署选择 Agent

这是一个基于 LangGraph 的可运行 Demo。用户只需要描述任务，系统会自动检测当前电脑或服务器的硬件和运行环境，并为文本、图片、音频、视频或组合任务推荐模型及部署策略。

第一版只做检测与推荐，不会自动下载或启动大型模型。

## 当前能力

- 自动检测 CPU、内存、磁盘、NVIDIA GPU、显存和驱动。
- 识别个人电脑、单机服务器和 Kubernetes 环境。
- 检测 Docker、Ollama、kubectl、FFmpeg、llama.cpp 和 vLLM。
- 将自然语言任务拆成文本、图片、音频、视频或多阶段流水线。
- 根据任务能力、操作系统、内存、显存和运行时兼容性对模型评分。
- 每个任务阶段默认返回 3 个具体模型型号，也可以按需展示最多 10 个候选。
- 十类主要能力各内置至少 10 个离线候选：文本生成、摘要、翻译、代码、图片理解、视频理解、语音识别、语音合成、图片生成和视频生成。
- 每个候选解释参数规模、量化方式、显存、内存、运行时、硬件适配原因和主要取舍。
- 在本地资源不足时给出量化、串行加载、CPU Offload、服务器或远端服务建议。

## 项目结构

```text
backend/
├── agents/model_selector/
│   ├── state.py
│   ├── task_classifier.py
│   ├── hardware.py
│   ├── catalog.py
│   ├── matcher.py
│   ├── nodes.py
│   ├── graph.py
│   ├── models.json
│   └── models_extended.json
├── api/model_selector.py
├── static/
│   ├── index.html
│   ├── app.css
│   └── app.js
├── schemas.py
├── cli.py
└── main.py
```

LangGraph 流程：

```text
classify_task
    → detect_environment
    → personal_computer / single_server / cluster
    → select_models
    → build_plan
```

## 安装

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

如果 `ensurepip` 无法写入系统临时目录，可以临时把 `TEMP` 和 `TMP` 指向项目内可写目录后重试。

## 命令行运行

```powershell
.\.venv\Scripts\python.exe -m backend.cli "把会议视频转成文字，区分说话人，然后生成中文摘要"
```

## 启动 API

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

打开：

- 可视化推荐界面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

## 使用可视化界面

首页是一个和 API 同源运行的纯 HTML/CSS/JavaScript 页面，不需要另外安装
Node.js 或启动前端开发服务器。

使用流程：

1. 直接描述任务，或者从文本与代码、图像、音频、视频、多模态五类模板中快速填写。
2. 模板内容会进入任务输入框，仍可继续补充数据规模、语言和部署要求。
3. 选择速度优先、均衡或质量优先，并设置每个阶段显示的候选数量。
4. 点击“扫描硬件并推荐”。
5. 页面会调用现有 LangGraph Agent，展示检测到的 CPU、内存、GPU 和磁盘环境。
6. 查看任务流水线、每个阶段的具体模型候选、硬件适配原因、资源取舍和部署路线。

音频模板不限定会议场景，覆盖录音、播客、课程音频、一般声音理解、语音
合成和多说话人内容。多说话人任务会拆成语音识别、说话人分离和文本摘要
三个阶段。

如果 `8000` 端口已经被其他程序占用，可以改用其他端口：

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

然后打开 `http://127.0.0.1:8765/`。

请求示例：

```powershell
$body = @{
  task = "根据文字描述生成一段视频"
  preferences = @{
    quality = "balanced"
    max_candidates_per_stage = 3
  }
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/model-selector/recommend" `
  -ContentType "application/json" `
  -Body $body
```

## 如何阅读候选模型

每个任务阶段的 `candidates` 默认包含最多三个具体候选，可以通过
`preferences.max_candidates_per_stage` 设置为 1～10 个。模型目录提供 10 个
并不表示当前硬件一定能运行全部模型；接口仍会按系统、运行时、可用内存和
显存适配度排序，并明确标记 `unsupported` 候选。

候选示例：

```json
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "display_name": "Qwen2.5 7B Instruct",
  "score": 83,
  "fit_tier": "constrained",
  "recommended_runtime": "ollama",
  "parameter_size": "7B",
  "quantization": "Q4_K_M",
  "min_vram_gb": 6.0,
  "recommended_vram_gb": 8.0,
  "estimated_vram_gb": 6.5,
  "min_ram_gb": 12.0,
  "hardware_reasons": [
    "本机 GPU 总显存满足最低要求，但当前空闲显存不足。",
    "需要释放显存或启用 Offload。",
    "本机总内存满足最低要求。",
    "已检测到推荐运行时 ollama。"
  ],
  "tradeoffs": [
    "质量与资源消耗较均衡。",
    "速度通常慢于 3B 级轻量模型。",
    "量化可能带来少量精度损失。"
  ]
}
```

适配等级：

- `ideal`：当前空闲资源可以直接满足。
- `runnable`：总资源满足，运行余量一般。
- `constrained`：需要关闭其他 GPU 程序、量化或 CPU Offload。
- `unsupported`：当前节点不适合，建议更大服务器或远端推理。

### 推荐评分与偏好

排名采用“硬件门槛 + 100 分加权评分”：

- 只要存在 `ideal`、`runnable` 或 `constrained` 候选，`unsupported`
  模型就不会排在它们前面。
- 每个模型分别计算硬件适配、速度效率、预期质量、部署就绪度和估算可信度。
- 页面会显示最终分、五个分项、当前偏好权重和排名理由。

三种偏好的权重如下：

| 偏好 | 硬件 | 速度 | 质量 | 部署就绪 | 可信度 |
|---|---:|---:|---:|---:|---:|
| 速度优先 | 35% | 35% | 15% | 10% | 5% |
| 均衡 | 35% | 20% | 25% | 10% | 10% |
| 质量优先 | 30% | 10% | 45% | 10% | 5% |

目前速度和质量分包含目录数据与相对资源规模推断，因此属于可解释的启发式
排名，不应当作真实吞吐量或模型基准成绩。

当前目录覆盖的对比系列包括：

- 文本：Qwen2.5、Llama 3.2、Gemma 3、Phi-4 Mini、Mistral、DeepSeek R1 Distill。
- 代码：Qwen2.5 Coder、DeepSeek Coder、StarCoder2、CodeGemma。
- 视觉：Qwen2.5-VL、MiniCPM-V/O、SmolVLM2、InternVL3、Phi Vision、Llama Vision、LLaVA OneVision、Video-LLaVA、LongVA、InternVideo2.5。
- 语音识别：Faster Whisper、Whisper Large v3/Turbo、Distil-Whisper、Moonshine、SenseVoice。
- 语音合成：Kokoro、ChatTTS、CosyVoice2、F5-TTS、Fish Speech、Spark-TTS、MeloTTS、GPT-SoVITS、OpenVoice、Parler-TTS。
- 图片生成：Stable Diffusion、SDXL、FLUX、PixArt、HunyuanDiT、Kolors、Sana。
- 视频生成：Wan2.1/2.2、CogVideoX、HunyuanVideo、LTX-Video、Mochi、ModelScope Text-to-Video。

## 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## 模型目录

模型元数据位于 `backend/agents/model_selector/models.json` 和
`backend/agents/model_selector/models_extended.json`，加载时会合并并校验
`model_id` 唯一性。新增模型时配置：

- 支持能力和输入模态
- 支持的操作系统与推理框架
- 最低和推荐显存
- 最低系统内存
- 是否支持 CPU 或多 GPU
- 推荐量化和启动命令

## 第一版限制

- 任务分类采用离线关键词和规则，还没有接入 LLM 结构化分类。
- 资源需求是保守估算，不等同于真实基准测试。
- AMD、Intel GPU 和 Apple Silicon 暂时只保留扩展方向。
- Kubernetes 第一版只识别环境，不创建 Deployment。
- 当前模型目录用于演示，应在生产使用前补充模型卡、许可证和真实硬件基准数据。

## 后续优化方向

1. 接入 NVML、ROCm SMI、Intel Level Zero 和 Apple Metal 检测。
2. 将权重、KV Cache、上下文长度、批处理和并发纳入显存估算。
3. 增加真实模型启动与小规模基准测试，校准估算结果。
4. 接入 GPUStack 或 Kubernetes 节点资源，支持多节点调度。
5. 使用 LLM 结构化输出替代部分关键词分类，同时保留规则兜底。

