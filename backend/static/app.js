"use strict";

const API_PATH = "/api/v1/model-selector/recommend";

const form = document.querySelector("#task-form");
const taskInput = document.querySelector("#task-input");
const candidateCount = document.querySelector("#candidate-count");
const targetEnvironment = document.querySelector("#target-environment");
const allowOffload = document.querySelector("#allow-offload");
const useLlmfit = document.querySelector("#use-llmfit");
const submitButton = document.querySelector("#submit-button");
const formMessage = document.querySelector("#form-message");
const results = document.querySelector("#results");
const apiState = document.querySelector("#api-state");
const apiStateLabel = document.querySelector("#api-state-label");

const fitLabels = {
  ideal: "非常适合",
  runnable: "本机可运行",
  constrained: "受限运行",
  unsupported: "需要更强环境",
};

const environmentLabels = {
  personal_computer: "个人电脑",
  single_server: "单机服务器",
  cluster: "集群",
};

const modalityLabels = {
  text: "文本",
  image: "图像",
  audio: "音频",
  video: "视频",
};

const hardwarePresets = {
  current: null,
  "workstation-16": {
    name: "16GB 显存工作站",
    os_name: "Windows",
    ram_total_gb: 32,
    ram_available_gb: 26,
    gpu_count: 1,
    gpu_name: "Simulated 16GB GPU",
    gpu_total_vram_gb: 16,
    gpu_free_vram_gb: 14,
    environment_type: "personal_computer",
  },
  "server-24": {
    name: "24GB 显存服务器",
    os_name: "Linux",
    ram_total_gb: 64,
    ram_available_gb: 56,
    gpu_count: 1,
    gpu_name: "Simulated 24GB GPU",
    gpu_total_vram_gb: 24,
    gpu_free_vram_gb: 22,
    environment_type: "single_server",
    runtimes: {
      python: true,
      docker: true,
      ollama: true,
      ffmpeg: true,
      llama_cpp: true,
      vllm: true,
    },
  },
  "server-2x48": {
    name: "2×48GB 多 GPU 服务器",
    os_name: "Linux",
    ram_total_gb: 192,
    ram_available_gb: 168,
    gpu_count: 2,
    gpu_name: "Simulated 48GB GPU",
    gpu_total_vram_gb: 48,
    gpu_free_vram_gb: 44,
    environment_type: "single_server",
    runtimes: {
      python: true,
      docker: true,
      ollama: true,
      ffmpeg: true,
      llama_cpp: true,
      vllm: true,
    },
  },
};

const sourceLabels = {
  local_catalog: "LOCAL CATALOG",
  llmfit: "LLMFIT",
};

const executionModeLabels = {
  direct: "DIRECT",
  quantized: "QUANTIZED",
  cpu_offload: "CPU OFFLOAD",
  cpu_only: "CPU ONLY",
};

const confidenceLabels = {
  benchmarked: "实测",
  calibrated: "校准数据",
  catalog: "目录估算",
  heuristic: "启发式估算",
  unknown: "未知",
};

const preferenceLabels = {
  speed: "速度优先",
  balanced: "均衡",
  quality: "质量优先",
};

const scoreComponentLabels = {
  hardware: "硬件适配",
  speed: "速度效率",
  quality: "预期质量",
  readiness: "部署就绪",
  confidence: "估算可信",
};

function clearElement(element) {
  element.replaceChildren();
}

function appendTextList(container, items, emptyText = "暂无") {
  clearElement(container);
  const values = items && items.length ? items : [emptyText];
  values.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    container.append(li);
  });
}

function setApiState(mode, label) {
  apiState.classList.toggle("is-busy", mode === "busy");
  apiState.classList.toggle("is-error", mode === "error");
  apiStateLabel.textContent = label;
}

function setLoading(isLoading) {
  form.classList.toggle("is-loading", isLoading);
  submitButton.disabled = isLoading;
  taskInput.disabled = isLoading;
  candidateCount.disabled = isLoading;
  targetEnvironment.disabled = isLoading;
  allowOffload.disabled = isLoading;
  useLlmfit.disabled = isLoading;
  form
    .querySelectorAll(
      'input[name="quality"], .template-tab, .template-card',
    )
    .forEach((control) => {
      control.disabled = isLoading;
    });
  setApiState(isLoading ? "busy" : "ready", isLoading ? "SCANNING HARDWARE" : "API READY");
}

function showError(message) {
  formMessage.textContent = message;
  formMessage.hidden = false;
  setApiState("error", "RETRY AVAILABLE");
}

function clearError() {
  formMessage.hidden = true;
  formMessage.textContent = "";
}

function formatNumber(value, suffix = "") {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(numeric % 1 ? 1 : 0)}${suffix}` : `${value}${suffix}`;
}

function addDefinition(list, label, value) {
  const fragment = document.querySelector("#definition-template").content.cloneNode(true);
  fragment.querySelector("dt").textContent = label;
  fragment.querySelector("dd").textContent = value || "—";
  list.append(fragment);
}

function renderHardware(hardware) {
  const list = document.querySelector("#hardware-list");
  clearElement(list);

  const gpuSummary = hardware.gpus && hardware.gpus.length
    ? hardware.gpus
      .map((gpu) => `${gpu.name} · ${formatNumber(gpu.free_vram_gb, "GB")} / ${formatNumber(gpu.total_vram_gb, "GB")}`)
      .join("；")
    : "未检测到独立 GPU";

  addDefinition(list, "环境", environmentLabels[hardware.environment_type] || hardware.environment_type);
  addDefinition(
    list,
    "来源",
    hardware.metadata?.hardware_source === "simulated"
      ? `模拟：${hardware.metadata?.simulation_name || "自定义环境"}`
      : "当前设备自动检测",
  );
  addDefinition(list, "系统", `${hardware.os_name} ${hardware.os_version} · ${hardware.architecture}`);
  addDefinition(list, "处理器", `${hardware.cpu_model} · ${hardware.logical_cores} 线程`);
  addDefinition(
    list,
    "内存",
    `${formatNumber(hardware.ram_available_gb, "GB")} 可用 / ${formatNumber(hardware.ram_total_gb, "GB")} 总计`,
  );
  addDefinition(list, "GPU", gpuSummary);
  addDefinition(list, "磁盘", `${formatNumber(hardware.disk_free_gb, "GB")} 可用`);
}

function renderPipeline(analysis) {
  const list = document.querySelector("#pipeline-list");
  clearElement(list);
  document.querySelector("#pipeline-count").textContent = `${analysis.stages.length} 个阶段`;

  analysis.stages.forEach((stage, index) => {
    const fragment = document.querySelector("#pipeline-template").content.cloneNode(true);
    fragment.querySelector("li").dataset.capability = stage.capability;
    fragment.querySelector(".pipeline-number").textContent = String(index + 1).padStart(2, "0");
    fragment.querySelector("strong").textContent = stage.name;
    fragment.querySelector(".pipeline-detail").textContent =
      `${modalityLabels[stage.input_modality] || stage.input_modality} → ${modalityLabels[stage.output_modality] || stage.output_modality}`;
    list.append(fragment);
  });
}

function addSpec(list, label, value) {
  const wrapper = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value || "—";
  wrapper.append(term, description);
  list.append(wrapper);
}

function renderScoreBreakdown(fragment, candidate) {
  const container = fragment.querySelector(".score-breakdown");
  clearElement(container);
  const breakdown = candidate.score_breakdown || {};
  const weights = breakdown.weights || {};

  Object.entries(scoreComponentLabels).forEach(([key, label]) => {
    const row = document.querySelector("#score-row-template").content.cloneNode(true);
    const numericScore = Number(breakdown[key]);
    const score = Number.isFinite(numericScore)
      ? Math.max(0, Math.min(numericScore, 100))
      : 0;
    const weight = Number(weights[key]);
    row.querySelector(".score-row-label span").textContent =
      `${label} · 权重 ${Number.isFinite(weight) ? weight : 0}%`;
    row.querySelector(".score-row-label strong").textContent = `${Math.round(score)}`;
    row.querySelector(".score-fill").style.width = `${score}%`;
    container.append(row);
  });
}

function renderCandidate(candidate, index) {
  const fragment = document.querySelector("#candidate-template").content.cloneNode(true);
  const card = fragment.querySelector(".candidate-card");
  const fit = fragment.querySelector(".fit-badge");

  fragment.querySelector(".candidate-rank").textContent =
    index === 0 ? "PRIMARY CHOICE" : `OPTION ${String(index + 1).padStart(2, "0")}`;
  fragment.querySelector(".candidate-score-value").textContent =
    formatNumber(candidate.score);
  fit.textContent = fitLabels[candidate.fit_tier] || candidate.fit_tier;
  fit.classList.toggle("is-local", ["ideal", "runnable"].includes(candidate.fit_tier));
  fragment.querySelector("h5").textContent = candidate.display_name;
  fragment.querySelector(".candidate-id").textContent = candidate.model_id;
  fragment.querySelector(".candidate-source").textContent =
    sourceLabels[candidate.source] || candidate.source || "LOCAL CATALOG";
  fragment.querySelector(".candidate-mode").textContent =
    executionModeLabels[candidate.execution_mode]
    || candidate.execution_mode
    || "DIRECT";
  fragment.querySelector(".candidate-confidence").textContent =
    confidenceLabels[candidate.estimate_confidence]
    || candidate.estimate_confidence
    || "未知";
  fragment.querySelector(".candidate-preference").textContent =
    preferenceLabels[candidate.preference]
    || candidate.preference
    || "均衡";
  const peakRatio = candidate.utilization?.peak_ratio;
  fragment.querySelector(".candidate-utilization").textContent =
    typeof peakRatio === "number"
      ? `峰值资源占用约 ${(peakRatio * 100).toFixed(1)}% · 依据：${candidate.estimate_basis?.source || "未标记"}`
      : "当前资源不可用或缺少可比较的估算数据";
  renderScoreBreakdown(fragment, candidate);

  const specs = fragment.querySelector(".candidate-specs");
  addSpec(specs, "参数量", candidate.parameter_size);
  addSpec(specs, "量化策略", candidate.quantization || "无需额外量化");
  addSpec(specs, "预计显存", formatNumber(candidate.estimated_vram_gb, "GB"));
  addSpec(specs, "最低内存", formatNumber(candidate.min_ram_gb, "GB"));
  addSpec(specs, "环境依赖", candidate.recommended_runtime);

  appendTextList(
    fragment.querySelector(".ranking-reason-list"),
    candidate.ranking_reasons,
  );
  appendTextList(
    fragment.querySelector(".reason-list"),
    candidate.hardware_reasons && candidate.hardware_reasons.length
      ? candidate.hardware_reasons
      : candidate.reasons,
  );
  appendTextList(fragment.querySelector(".tradeoff-list"), candidate.tradeoffs);

  card.dataset.modelId = candidate.model_id;
  card.dataset.preference = candidate.preference || "balanced";
  return fragment;
}

function renderRecommendations(recommendations) {
  const list = document.querySelector("#recommendation-list");
  clearElement(list);

  recommendations.forEach((recommendation) => {
    const fragment = document.querySelector("#stage-template").content.cloneNode(true);
    fragment.querySelector(".stage-capability").textContent =
      recommendation.stage.capability.replaceAll("_", " ");
    fragment.querySelector("h4").textContent = recommendation.stage.name;
    fragment.querySelector(".stage-heading p").textContent = recommendation.local_supported
      ? "当前环境可执行该阶段，首选项已按适配度置顶。"
      : recommendation.fallback;

    const grid = fragment.querySelector(".candidate-grid");
    recommendation.candidates.forEach((candidate, index) => {
      grid.append(renderCandidate(candidate, index));
    });

    if (!recommendation.candidates.length) {
      const empty = document.createElement("p");
      empty.className = "empty-candidates";
      empty.textContent = recommendation.fallback || "当前模型目录没有匹配候选。";
      grid.append(empty);
    }
    list.append(fragment);
  });
}

function buildAgentSummary(data) {
  const chosenModels = data.recommendations
    .map((item) => item.candidates[0]?.display_name)
    .filter(Boolean);
  const supported = data.recommendations.filter((item) => item.local_supported).length;
  const total = data.recommendations.length;
  const modelText = chosenModels.length ? chosenModels.join("、") : "暂无可用候选";

  return `${data.deployment_plan.environment_summary}。任务被拆成 ${total} 个阶段，`
    + `其中 ${supported} 个阶段可在当前环境直接运行。建议优先组合 ${modelText}。`
    + `下方给出了候选差异、资源依据和部署顺序，便于按速度或质量继续取舍。`;
}

function createCommandRow(command, label) {
  const row = document.createElement("div");
  row.className = "command-row";
  const code = document.createElement("code");
  code.textContent = command;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy-button";
  button.textContent = `复制 ${label}`;
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(command);
      button.textContent = "已复制";
    } catch {
      button.textContent = "请手动复制";
    }
    window.setTimeout(() => {
      button.textContent = `复制 ${label}`;
    }, 1600);
  });
  row.append(code, button);
  return row;
}

function renderDeployment(data) {
  appendTextList(document.querySelector("#deployment-steps"), data.deployment_plan.steps);
  appendTextList(
    document.querySelector("#deployment-warnings"),
    [
      ...data.deployment_plan.warnings,
      ...data.deployment_plan.next_actions.map((item) => `下一步：${item}`),
    ],
    "当前没有额外警告，可按执行步骤开始验证。",
  );

  const commands = document.querySelector("#command-list");
  clearElement(commands);
  const seen = new Set();
  data.recommendations.forEach((item) => {
    const candidate = item.candidates[0];
    if (!candidate?.deployment_command || seen.has(candidate.deployment_command)) {
      return;
    }
    seen.add(candidate.deployment_command);
    commands.append(createCommandRow(candidate.deployment_command, candidate.display_name));
  });

  if (!seen.size) {
    commands.append(
      createCommandRow(
        "请先根据候选模型文档安装对应运行时，再回到此处生成具体命令。",
        "提示",
      ),
    );
  }
}

function renderResults(data) {
  document.querySelector("#agent-summary").textContent = buildAgentSummary(data);
  renderHardware(data.hardware);
  renderPipeline(data.task_analysis);
  renderRecommendations(data.recommendations);
  renderDeployment(data);
  results.hidden = false;
  window.requestAnimationFrame(() => {
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function extractErrorMessage(responseBody) {
  if (Array.isArray(responseBody?.detail)) {
    return responseBody.detail.map((item) => item.msg).join("；");
  }
  return responseBody?.detail || "推荐服务暂时不可用，请检查 FastAPI 是否仍在运行后重试。";
}

async function submitTask(event) {
  event.preventDefault();
  clearError();
  const task = taskInput.value.trim();
  if (!task) {
    showError("请先输入具体任务，例如处理什么内容、需要什么结果。");
    taskInput.focus();
    return;
  }

  const quality = form.querySelector('input[name="quality"]:checked').value;
  results.hidden = true;
  setLoading(true);
  try {
    const response = await fetch(API_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task,
        preferences: {
          quality,
          max_candidates_per_stage: Number(candidateCount.value),
          allow_offload: allowOffload.checked,
          use_llmfit: useLlmfit.checked,
        },
        hardware_override: hardwarePresets[targetEnvironment.value],
      }),
    });
    let body;
    try {
      body = await response.json();
    } catch {
      body = {
        detail: "推荐服务返回了无法解析的响应，请检查 FastAPI 日志后重试。",
      };
    }
    if (!response.ok) {
      throw new Error(extractErrorMessage(body));
    }
    renderResults(body);
  } catch (error) {
    setLoading(false);
    showError(error instanceof Error ? error.message : "请求失败，请重试。");
    return;
  }
  setLoading(false);
}

function setupTaskTemplates() {
  const tabs = [...document.querySelectorAll("[data-template-tab]")];
  const cards = [...document.querySelectorAll("[data-template-category]")];
  const panel = document.querySelector("#template-panel");

  function activateCategory(category, focusTab = false) {
    tabs.forEach((tab) => {
      const isActive = tab.dataset.templateTab === category;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
      tab.tabIndex = isActive ? 0 : -1;
      if (isActive) {
        panel.setAttribute("aria-labelledby", tab.id);
        if (focusTab) {
          tab.focus();
        }
      }
    });
    let visibleIndex = 0;
    cards.forEach((card) => {
      const isVisible = card.dataset.templateCategory === category;
      card.hidden = !isVisible;
      card.classList.toggle(
        "is-left-column",
        isVisible && visibleIndex % 2 === 0,
      );
      if (isVisible) {
        visibleIndex += 1;
      }
    });
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => {
      activateCategory(tab.dataset.templateTab);
    });
    tab.addEventListener("keydown", (event) => {
      let nextIndex = null;
      if (event.key === "ArrowRight") {
        nextIndex = (index + 1) % tabs.length;
      } else if (event.key === "ArrowLeft") {
        nextIndex = (index - 1 + tabs.length) % tabs.length;
      } else if (event.key === "Home") {
        nextIndex = 0;
      } else if (event.key === "End") {
        nextIndex = tabs.length - 1;
      }
      if (nextIndex === null) {
        return;
      }
      event.preventDefault();
      activateCategory(tabs[nextIndex].dataset.templateTab, true);
    });
  });

  cards.forEach((card) => {
    card.addEventListener("click", () => {
      taskInput.value = card.dataset.example;
      taskInput.focus();
      clearError();
    });
  });

  activateCategory("text");
}

function setupReset() {
  document.querySelector("#reset-button").addEventListener("click", () => {
    taskInput.focus();
    form.scrollIntoView({ behavior: "smooth", block: "center" });
  });
}

function ellipse(x, y, centerX, centerY, radiusX, radiusY) {
  return ((x - centerX) / radiusX) ** 2 + ((y - centerY) / radiusY) ** 2 <= 1;
}

function isLand(x, y) {
  const northAmerica =
    ellipse(x, y, 0.2, 0.32, 0.17, 0.15)
    || ellipse(x, y, 0.12, 0.4, 0.08, 0.12);
  const southAmerica =
    ellipse(x, y, 0.31, 0.64, 0.075, 0.2)
    || ellipse(x, y, 0.27, 0.52, 0.075, 0.1);
  const greenland = ellipse(x, y, 0.39, 0.18, 0.055, 0.08);
  const europe = ellipse(x, y, 0.51, 0.32, 0.09, 0.075);
  const africa =
    ellipse(x, y, 0.53, 0.54, 0.095, 0.18)
    || ellipse(x, y, 0.58, 0.46, 0.07, 0.1);
  const asia =
    ellipse(x, y, 0.68, 0.32, 0.2, 0.13)
    || ellipse(x, y, 0.78, 0.42, 0.13, 0.13);
  const australia = ellipse(x, y, 0.84, 0.68, 0.075, 0.075);
  const japan = ellipse(x, y, 0.91, 0.42, 0.012, 0.04);

  const centralAmericaCut = ellipse(x, y, 0.22, 0.47, 0.055, 0.04);
  const mediterraneanCut = ellipse(x, y, 0.55, 0.38, 0.09, 0.027);
  const asiaCut = ellipse(x, y, 0.71, 0.48, 0.13, 0.055);

  return (
    northAmerica
    || southAmerica
    || greenland
    || europe
    || africa
    || asia
    || australia
    || japan
  ) && !centralAmericaCut && !mediterraneanCut && !asiaCut;
}

function noise(column, row) {
  const value = Math.sin(column * 12.9898 + row * 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function drawWorldMap(phase = 0) {
  const canvas = document.querySelector("#world-map");
  const context = canvas.getContext("2d");
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const bounds = canvas.getBoundingClientRect();
  canvas.width = Math.round(bounds.width * ratio);
  canvas.height = Math.round(bounds.height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, bounds.width, bounds.height);

  const spacing = bounds.width < 680 ? 8 : 9;
  const size = bounds.width < 680 ? 3 : 4;
  const mapTop = bounds.width < 680 ? 60 : 76;
  const mapHeight = Math.min(bounds.height * 0.73, 620);
  const columns = Math.ceil(bounds.width / spacing);
  const rows = Math.ceil(mapHeight / spacing);

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const x = column / columns;
      const y = row / rows;
      if (!isLand(x, y) || noise(column, row) < 0.22) {
        continue;
      }
      const highlight = noise(column + 19, row + 41) > 0.9;
      const pulse = (column + row + phase) % 9 === 0;
      context.fillStyle = highlight
        ? `rgba(46, 234, 138, ${pulse ? 0.52 : 0.27})`
        : `rgba(160, 169, 163, ${0.14 + noise(column + 5, row + 9) * 0.18})`;
      context.fillRect(column * spacing, mapTop + row * spacing, size, size);
    }
  }
}

function setupMap() {
  let resizeFrame = 0;
  let phase = 0;
  drawWorldMap(phase);
  window.addEventListener("resize", () => {
    window.cancelAnimationFrame(resizeFrame);
    resizeFrame = window.requestAnimationFrame(() => drawWorldMap(phase));
  });

  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    window.setInterval(() => {
      phase = (phase + 1) % 9;
      drawWorldMap(phase);
    }, 1500);
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("unhealthy");
    }
    setApiState("ready", "API READY");
  } catch {
    setApiState("error", "API OFFLINE");
  }
}

form.addEventListener("submit", submitTask);
setupTaskTemplates();
setupReset();
setupMap();
checkHealth();
