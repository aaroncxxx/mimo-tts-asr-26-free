# MiMo TTS & ASR v2.9

语音合成（TTS）与语音识别（ASR）一体化 Skill。三款 TTS 模型 + ASR 全部限时免费。

> **关于作者** — 十五年米粉，今天是用龙虾编程的第二天，撸起袖子就是干。

## 系统依赖

- **ffmpeg/ffprobe**：音频预处理、分片、格式转换必需
- **python3**：运行脚本

```bash
apt install ffmpeg  # Debian/Ubuntu
brew install ffmpeg  # macOS
```

## 配置

TTS 和 ASR 各需独立 API Key（可相同）：

```bash
export MIMO_API_KEY="your-tts-api-key"
export MIMO_ASR_KEY="your-asr-api-key"
```

或通过 OpenClaw 配置：
```bash
openclaw config set skills.entries.mimo-tts-asr.apiKey "your-key"
```

申请 Key：[platform.xiaomimimo.com](https://platform.xiaomimimo.com)（当前限时免费）

> 💡 不想配 Key？ASR 支持开源模型本地部署：[GitHub](https://github.com/XiaomiMiMo/MiMo-V2.5-ASR)

## TTS — 语音合成

### 基础用法

```bash
python3 "{baseDir}/scripts/tts.py" "要合成的文本" -o output.wav
```

### 三款模型

| 模型 | 用途 | 关键参数 |
|------|------|---------|
| `tts`（默认） | 内置音色 + 情感/语速控制 | `-v` 音色, `-s` 风格 |
| `voice-design` | 自然语言描述生成新音色 | `--voice-desc` |
| `voice-clone` | 参考音频克隆音色 | `--ref-audio` |

### 参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `-o` | output.wav | 输出路径 |
| `-m` | tts | 模型：tts / voice-design / voice-clone |
| `-v` | mimo_default | 音色（`--list-voices` 查全部） |
| `-s` | — | 风格标签，可组合如 `"开心 变快"` |
| `-f` | wav | 格式：wav / mp3 / ogg |
| `--voice-desc` | — | VoiceDesign 音色描述 |
| `--ref-audio` | — | VoiceClone 参考音频 |
| `--speed` | 1.0 | 语速（0.95~1.05） |
| `--pitch` | 1.0 | 音调（±5%） |
| `--preprocess` | 关 | 开启文本预处理（分句/清洗/数字替换） |
| `--denoise` | 关 | 后置降噪 |
| `--normalize` | 关 | 音量归一化 |
| `--cache` | 关 | 音频缓存（相同文本秒返） |
| `--optimize` | — | 推理优化：gpu / cpu / lite |
| `--list-voices` | — | 列出可用音色 |
| `--max-retries` | 3 | 最大重试次数 |

### 行内音频标签

在文本中插入精细控制：
`(停顿) (叹气) (笑声) (清嗓子) (耳语) (紧张) (小声) (语速加快) (深呼吸) (沉默片刻)`

### 核心示例

```bash
# 基础
python3 "{baseDir}/scripts/tts.py" "你好，今天天气真好" -o hello.wav

# 方言
python3 "{baseDir}/scripts/tts.py" "哎呀妈呀，这天儿也忒冷了吧" -s "东北话" -o dongbei.wav

# 情感 + 语速
python3 "{baseDir}/scripts/tts.py" "明天就是周五了！" -s "开心 变快" -o happy.wav

# 男声 / 童声 / 粤语
python3 "{baseDir}/scripts/tts.py" "大家好" -v mimo_male -o male.wav

# VoiceDesign — 生成新音色
python3 "{baseDir}/scripts/tts.py" "欢迎" -m voice-design \
  --voice-desc "元气少女，声线清脆，语尾上扬" -o genki.wav

# VoiceClone — 克隆音色
python3 "{baseDir}/scripts/tts.py" "克隆后的声音" -m voice-clone \
  --ref-audio reference.wav -o cloned.wav

# 推荐组合：预处理 + 归一化 + 微调
python3 "{baseDir}/scripts/tts.py" "长文本..." --preprocess --normalize --speed 0.98 --pitch 1.02 -o output.wav
```

## ASR — 语音识别

### 基础用法

```bash
python3 "{baseDir}/scripts/asr.py" audio.wav
```

### 参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `-o` | stdout | 输出路径 |
| `--lang` | auto | 语言：auto / zh / en / ja / ko |
| `--format` | text | 输出：text / json / srt |
| `--preprocess` | 关 | 音频预处理（采样率/静音裁剪） |
| `--no-diarization` | 关 | 关闭说话人分离（轻量版建议开启） |
| `--boost-cn` | 关 | 增强中文解码 |
| `--chunk` | 0 | 长音频分片秒数（0=不分片） |
| `--max-retries` | 3 | 最大重试次数 |

### 核心示例

```bash
# 基础转录
python3 "{baseDir}/scripts/asr.py" recording.wav

# 保存到文件
python3 "{baseDir}/scripts/asr.py" meeting.mp3 -o meeting.txt

# SRT 字幕
python3 "{baseDir}/scripts/asr.py" video_audio.wav --format srt -o subtitles.srt

# JSON（带时间戳）
python3 "{baseDir}/scripts/asr.py" audio.wav --format json

# 中文优化：预处理 + 关闭分离 + 增强中文
python3 "{baseDir}/scripts/asr.py" audio.wav --preprocess --no-diarization --boost-cn -o transcript.txt

# 长音频分片
python3 "{baseDir}/scripts/asr.py" long_audio.wav --chunk 30 -o transcript.txt
```

## TTS + ASR 联合工作流

```bash
# 1. 识别
python3 "{baseDir}/scripts/asr.py" input.wav -o transcript.txt

# 2. 修改后用不同音色重新合成
python3 "{baseDir}/scripts/tts.py" "$(cat transcript.txt)" -v mimo_male -o output.wav

# 3. 克隆重演绎
python3 "{baseDir}/scripts/tts.py" "$(cat transcript.txt)" \
  -m voice-clone --ref-audio original.wav -o cloned.wav
```

## 交付格式

- **TTS**：`MEDIA:output.wav`（或指定的 `-o` 路径）
- **ASR**：直接回复转录文本，或回复 `-o` 指定的文件路径

## 故障排查

| 错误 | 原因 | 解决 |
|------|------|------|
| 401 Invalid API Key | Key 未配置或格式错误 | 确认已配置 TTS/ASR Key |
| 429 Too Many Requests | 触发限流 | 等几秒重试（脚本自动重试） |
| 文件不存在 | 路径错误 | 检查音频文件路径 |
| ffmpeg 未找到 | 缺系统依赖 | `apt install ffmpeg` |

## 进阶优化

推理性能优化（GPU/CPU）、模型剪枝、TensorRT/OpenVINO 加速、FastAPI 封装等高级用法，详见 `docs/advanced.md`。

## 版本历史

### v2.9 (2026-04-24)
- 📐 SKILL.md 精简重构，聚焦用法，进阶内容拆至 docs/
- 🔧 修复 `--preprocess` 参数语义矛盾（改为显式开启）
- 📋 参数表统一格式，示例精简到核心场景

### v2.81 (2026-04-24)
- 📦 声明系统依赖 ffmpeg/ffprobe
- 🔑 明确 API Key 要求

### v2.8 (2026-04-24)
- 🚀 进阶优化：剪枝 / TensorRT / OpenVINO / FastAPI
- ⚠️ 避坑要点
