<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import BoxOverlay from '../components/BoxOverlay.vue'
import TraceTimeline from '../components/TraceTimeline.vue'
import ResultDetail from '../components/ResultDetail.vue'
import { api, classLabel, classColor } from '../api'

const route = useRoute()

/* ------------------------------------------------------------ 状态 */
const samples = ref([])
const cfg = ref(null)
const file = ref(null)
const sampleName = ref('')
const previewUrl = ref('')
const result = ref(null)
const running = ref(false)
const error = ref('')
const errorHint = ref('')
const exporting = ref(false)

const hoverBox = ref(null)
const activeBox = ref(null)
const viewMode = ref('collab')          // collab | yolo
const showLabels = ref(true)
const showAdvanced = ref(false)
const hiddenClasses = ref([])

const params = ref({
  threshold: 0.6,
  infer_conf: 0.25,
  baseline_conf: 0.4,
  enable_vlm: true,
  enable_enhance: false,
  save_record: true
})

/* ------------------------------------------------------------ 初始化 */
onMounted(async () => {
  try {
    const [s, c] = await Promise.all([api.samples(), api.config()])
    samples.value = s.items || []
    cfg.value = c
    if (c.defaults) params.value = { ...params.value, ...c.defaults, save_record: true }
  } catch (e) {
    error.value = e.message
  }
})

/* ------------------------------------------------------------ 计算属性 */
const boxesToDraw = computed(() => {
  if (!result.value) return []
  const list = viewMode.value === 'yolo'
    ? (result.value.baseline_detections || [])
    : (result.value.final_boxes || [])
  if (!hiddenClasses.value.length) return list
  return list.filter(b => !hiddenClasses.value.includes(b.cls_name))
})

const overlayUrl = computed(() => {
  if (result.value?.image_url) return result.value.image_url
  return previewUrl.value
})

const hasImage = computed(() => !!previewUrl.value)

const thresholdExplain = computed(() => {
  const t = params.value.threshold
  if (t <= 0.45) return '低预算档：只有极少数最可疑的框会送大模型复核，成本最低。'
  if (t <= 0.55) return '均衡档：约 0.5 次复核/张，成本与效果折中。'
  if (t <= 0.65) return '推荐档（Pareto 膝点）：约 0.75 次复核/张，边际收益最高。'
  if (t <= 0.75) return '高精度档：约 1.3 次复核/张，F1 更高但调用成本接近翻倍。'
  return '极致档：绝大多数框都会被复核，成本接近"全复核"，请谨慎使用。'
})

/* ------------------------------------------------------------ 交互 */
function pickFile(f) {
  if (!f) return
  if (!/^image\//.test(f.type)) {
    error.value = '请选择图片文件（JPG / PNG / BMP / WEBP）'
    return
  }
  error.value = ''
  errorHint.value = ''
  file.value = f
  sampleName.value = ''
  result.value = null
  if (previewUrl.value && previewUrl.value.startsWith('blob:')) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(f)
}

function onDrop(e) {
  e.preventDefault()
  const f = e.dataTransfer?.files?.[0]
  pickFile(f)
}

function onFileInput(e) {
  pickFile(e.target.files?.[0])
  e.target.value = ''
}

function pickSample(s) {
  error.value = ''
  errorHint.value = ''
  sampleName.value = s.file
  file.value = null
  result.value = null
  if (previewUrl.value && previewUrl.value.startsWith('blob:')) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = s.url
}

function reset() {
  file.value = null
  sampleName.value = ''
  result.value = null
  error.value = ''
  if (previewUrl.value && previewUrl.value.startsWith('blob:')) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
}

function toggleClass(name) {
  const i = hiddenClasses.value.indexOf(name)
  if (i >= 0) hiddenClasses.value.splice(i, 1)
  else hiddenClasses.value.push(name)
}

function applyPreset(p) {
  params.value.threshold = p.threshold
}

async function detect() {
  if (!hasImage.value || running.value) return
  running.value = true
  error.value = ''
  errorHint.value = ''
  hoverBox.value = null
  activeBox.value = null
  const t0 = Date.now()
  try {
    const payload = { ...params.value }
    const res = sampleName.value
      ? await api.detectSample(sampleName.value, payload)
      : await api.detectUpload(file.value, payload)
    res.client_ms = Date.now() - t0
    result.value = res
    viewMode.value = 'collab'
    hiddenClasses.value = []
  } catch (e) {
    error.value = e.message
    errorHint.value = e.payload?.hint || ''
  } finally {
    running.value = false
  }
}

async function exportReport() {
  if (!result.value || exporting.value) return
  exporting.value = true
  try {
    await api.exportReport(result.value.record_id
      ? { record_id: result.value.record_id }
      : { result: result.value })
  } catch (e) {
    error.value = `导出报告失败：${e.message}`
  } finally {
    exporting.value = false
  }
}

/* 从检测记录页跳转过来时，直接载入该条记录 */
watch(() => route.query.record, async (id) => {
  if (!id) return
  try {
    const rec = await api.historyDetail(Number(id))
    result.value = rec.result_json
    previewUrl.value = rec.image_url
    sampleName.value = ''
    file.value = null
    if (rec.result_json?.params) {
      params.value = { ...params.value, ...rec.result_json.params, save_record: true }
    }
  } catch (e) {
    error.value = e.message
  }
}, { immediate: true })
</script>

<template>
  <div class="page">
    <!-- 未选图：上传区 + 样例库 -->
    <template v-if="!hasImage">
      <div class="welcome">
        <h2>开始一次协同推理检测</h2>
        <p class="muted">
          上传你自己的服装质检图片，或直接点选下方的内置样例——
          样例自带"参考答案"（真实标注），方便你对照系统的检测结果。
        </p>
      </div>

      <div class="entry-grid">
        <div
          class="dropzone"
          :class="{ dragging: false }"
          @dragover.prevent
          @drop="onDrop"
          @click="$refs.fileInput.click()"
        >
          <input ref="fileInput" type="file" accept="image/*" hidden @change="onFileInput" />
          <div class="dz-icon">📤</div>
          <div class="dz-title">把图片拖到这里，或点击选择文件</div>
          <div class="dz-sub">支持 JPG / PNG / BMP / WEBP，单张不超过 20 MB</div>
        </div>

        <div class="card">
          <div class="card-head">
            <h3>或是使用内置样例</h3>
            <span class="head-sub">来自重标 test 集，自带真实标注</span>
          </div>
          <div class="card-pad">
            <div v-if="!samples.length" class="skeleton" style="height:120px"></div>
            <div v-else class="sample-grid">
              <button v-for="s in samples" :key="s.file" class="sample-card" @click="pickSample(s)">
                <img :src="s.url" :alt="s.file" loading="lazy" />
                <div class="sample-meta">
                  <span class="xsmall mono dim">{{ s.file }}</span>
                  <span class="badge xsmall"
                        :class="s.primary_class === 'clean' ? 'badge-ok' : 'badge-' + s.primary_class">
                    {{ s.label_zh }}
                  </span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div v-if="error" class="alert alert-danger mt-3">
        <span class="ico">⚠️</span>
        <div>{{ error }}<div v-if="errorHint" class="xsmall mt-1">{{ errorHint }}</div></div>
      </div>
    </template>

    <!-- 已选图：工作台 -->
    <template v-else>
      <div class="detect-grid">
        <!-- ============ 左：图片与检测框 ============ -->
        <div class="card img-card">
          <div class="card-head">
            <h3>检测图像</h3>
            <span class="head-sub">
              {{ result ? `${result.image.width} × ${result.image.height} px` : '尚未检测' }}
            </span>
            <div class="head-right">
              <div v-if="result" class="seg">
                <button :class="{ on: viewMode === 'collab' }" @click="viewMode = 'collab'">
                  协同结果
                </button>
                <button :class="{ on: viewMode === 'yolo' }" @click="viewMode = 'yolo'">
                  仅 YOLO 基线
                </button>
              </div>
              <button class="btn btn-sm" @click="reset">换一张</button>
            </div>
          </div>

          <div class="card-pad">
            <div v-if="result" class="img-toolbar">
              <div class="row wrap gap-1">
                <span class="xsmall dim">按类别筛选：</span>
                <button
                  v-for="c in ['hole','stain','wrinkle']" :key="c"
                  class="chip" :class="{ off: hiddenClasses.includes(c) }"
                  :style="{ '--chip': classColor(c) }"
                  @click="toggleClass(c)">
                  <span class="chip-dot"></span>{{ classLabel(c) }}
                </button>
              </div>
              <div class="row gap-1">
                <button class="btn btn-sm btn-ghost" @click="showLabels = !showLabels">
                  {{ showLabels ? '隐藏标签' : '显示标签' }}
                </button>
              </div>
            </div>

            <div class="img-stage">
              <BoxOverlay
                :image-url="overlayUrl"
                :boxes="boxesToDraw"
                :hover-index="hoverBox"
                :active-index="activeBox"
                :show-labels="showLabels"
                @hover="v => hoverBox = v"
                @select="v => activeBox = activeBox === v ? null : v"
              />
              <div v-if="running" class="img-loading">
                <span class="spinner"></span>
                <span>正在执行协同推理…</span>
              </div>
            </div>

            <div v-if="result" class="img-legend">
              <span class="badge badge-gray">
                当前显示：{{ viewMode === 'collab' ? '协同推理结果' : '仅 YOLO 基线结果' }}
                · {{ boxesToDraw.length }} 个框
              </span>
              <span v-if="viewMode === 'yolo'" class="badge badge-warn">
                该模式下不进行大模型复核
              </span>
              <template v-if="viewMode === 'collab'">
                <span class="badge badge-ok">高置信直接接受</span>
                <span class="badge badge-warn">大模型复核后保留</span>
                <span class="badge badge-violet">类别被纠错</span>
              </template>
            </div>
            <div v-else class="img-legend">
              <span class="badge badge-gray">这是原始图片，点击右侧"开始检测"运行协同推理</span>
            </div>

            <!-- 样例参考答案 -->
            <div v-if="sampleName" class="ref-answer">
              <span class="xsmall dim">样例参考答案（真实标注）：</span>
              <span class="xsmall">
                {{ samples.find(s => s.file === sampleName)?.label_zh || '—' }}
              </span>
              <span class="xsmall dim">· 可用它检验系统是否漏检/误检</span>
            </div>
          </div>
        </div>

        <!-- ============ 右：参数与推理轨迹 ============ -->
        <div class="col-right">
          <div class="card">
            <div class="card-head">
              <h3>检测参数</h3>
              <div class="head-right">
                <span class="badge badge-blue xsmall">推荐 T = 0.6</span>
              </div>
            </div>
            <div class="card-pad">
              <div class="field">
                <div class="field-label">
                  路由阈值 T
                  <span class="hint">置信度 ≥ T 直接接受</span>
                </div>
                <div class="threshold-row">
                  <input type="range" min="0.30" max="0.90" step="0.05"
                         v-model.number="params.threshold" />
                  <span class="threshold-val">{{ params.threshold.toFixed(2) }}</span>
                </div>
                <div class="field-help">{{ thresholdExplain }}</div>
                <div class="preset-row">
                  <button v-for="p in (cfg?.recommended_points || [])" :key="p.threshold"
                          class="preset" :class="{ on: Math.abs(params.threshold - p.threshold) < 1e-6 }"
                          @click="applyPreset(p)">
                    <b>{{ p.threshold.toFixed(2) }}</b>
                    <span>{{ p.label }}</span>
                  </button>
                </div>
              </div>

              <div class="divider"></div>

              <label class="switch-row" @click="params.enable_vlm = !params.enable_vlm">
                <span class="switch" :class="{ on: params.enable_vlm }"></span>
                <span class="switch-label">启用多模态大模型复核</span>
                <span class="xsmall dim" style="margin-left:auto">关闭后等价于"仅 YOLO"</span>
              </label>
              <label class="switch-row" @click="params.enable_enhance = !params.enable_enhance">
                <span class="switch" :class="{ on: params.enable_enhance }"></span>
                <span class="switch-label">局部区域对比度增强</span>
                <span class="xsmall dim" style="margin-left:auto">论文可选步骤</span>
              </label>
              <label class="switch-row" @click="params.save_record = !params.save_record">
                <span class="switch" :class="{ on: params.save_record }"></span>
                <span class="switch-label">保存到检测记录</span>
              </label>

              <details class="adv" :open="showAdvanced" @toggle="e => showAdvanced = e.target.open">
                <summary>高级参数</summary>
                <div class="field mt-2">
                  <div class="field-label">
                    YOLO 推理阈值
                    <span class="hint">{{ params.infer_conf.toFixed(2) }}</span>
                  </div>
                  <input type="range" min="0.05" max="0.60" step="0.05"
                         v-model.number="params.infer_conf" />
                  <div class="field-help">
                    低于该置信度的候选框直接丢弃。论文采用 0.25 —— 放宽下限让更多疑难框
                    有机会进入复核，这正是协同能"找回漏检"的前提。
                  </div>
                </div>
                <div class="field">
                  <div class="field-label">
                    "仅 YOLO"对照阈值
                    <span class="hint">{{ params.baseline_conf.toFixed(2) }}</span>
                  </div>
                  <input type="range" min="0.10" max="0.90" step="0.05"
                         v-model.number="params.baseline_conf" />
                  <div class="field-help">
                    用于生成对比卡片里的"仅 YOLO 基线"。论文诚实基线为 0.40。
                  </div>
                </div>
              </details>

              <div class="row mt-3 gap-1">
                <button class="btn btn-primary grow" :disabled="running" @click="detect">
                  <span v-if="running" class="spinner"></span>
                  <span>{{ running ? '推理中…' : (result ? '重新检测' : '开始检测') }}</span>
                </button>
                <button class="btn" :disabled="!result || exporting" @click="exportReport">
                  {{ exporting ? '导出中…' : '导出报告' }}
                </button>
              </div>
            </div>
          </div>

          <div v-if="error" class="alert alert-danger mt-2">
            <span class="ico">⚠️</span>
            <div>{{ error }}<div v-if="errorHint" class="xsmall mt-1">{{ errorHint }}</div></div>
          </div>

          <div class="card mt-2">
            <div class="card-head">
              <h3>Agent 推理轨迹</h3>
              <span class="head-sub">每一步工具调用都被完整记录</span>
              <div class="head-right">
                <span v-if="result" class="badge badge-gray xsmall">
                  {{ result.trace.length }} 步
                </span>
              </div>
            </div>
            <div class="card-pad">
              <TraceTimeline :trace="result?.trace || []" :running="running" />
            </div>
          </div>
        </div>
      </div>

      <!-- ============ 下：结果详情 ============ -->
      <ResultDetail
        v-if="result"
        class="mt-3"
        :result="result"
        @hover-box="v => hoverBox = v"
        @active-box="v => activeBox = activeBox === v ? null : v"
      />
    </template>
  </div>
</template>

<style scoped>
.welcome { margin-bottom: 16px; }
.welcome h2 { font-size: 21px; margin-bottom: 5px; }
.welcome p { margin: 0; font-size: 13.5px; }

.entry-grid { display: grid; grid-template-columns: 1fr 1.5fr; gap: 16px; align-items: start; }

.dropzone {
  border: 2px dashed var(--border-strong); border-radius: var(--radius-lg);
  background: #fff; padding: 42px 24px; text-align: center; cursor: pointer;
  transition: all .18s; min-height: 200px;
  display: flex; flex-direction: column; justify-content: center;
}
.dropzone:hover { border-color: var(--primary); background: #fafcff; }
.dz-icon { font-size: 34px; margin-bottom: 10px; }
.dz-title { font-size: 14.5px; font-weight: 700; color: var(--text); }
.dz-sub { font-size: 12px; color: var(--text-4); margin-top: 6px; }

.sample-grid {
  display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fill, minmax(126px, 1fr));
  max-height: 380px; overflow-y: auto;
}
.sample-card {
  border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
  background: #fff; padding: 0; cursor: pointer; text-align: left;
  transition: all .15s; font-family: inherit;
}
.sample-card:hover { border-color: var(--primary); box-shadow: 0 3px 14px rgba(37,99,235,.16); transform: translateY(-1px); }
.sample-card img {
  width: 100%; height: 84px; object-fit: cover; display: block; background: var(--bg-soft);
}
.sample-meta { padding: 6px 8px 8px; display: flex; flex-direction: column; gap: 4px; }

.detect-grid { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(330px, 1fr); gap: 16px; align-items: start; }
.col-right { display: flex; flex-direction: column; }

.img-card .card-pad { padding: 14px 16px 16px; }
.img-toolbar {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; margin-bottom: 10px; flex-wrap: wrap;
}
.chip {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  border: 1px solid var(--border-strong); background: #fff; color: var(--text-2);
  cursor: pointer; font-family: inherit; transition: all .15s;
}
.chip:hover { border-color: var(--text-4); }
.chip.off { opacity: .42; text-decoration: line-through; }
.chip-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--chip); }

.img-stage { position: relative; }
.img-loading {
  position: absolute; inset: 0; border-radius: 10px;
  background: rgba(255,255,255,.82); backdrop-filter: blur(2px);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 12px; font-size: 13px; font-weight: 600; color: var(--primary);
  color: var(--primary-dark);
}
.img-loading .spinner { width: 26px; height: 26px; border-width: 3px; }

.img-legend { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 11px; }
.ref-answer {
  margin-top: 9px; padding: 8px 11px; border-radius: 8px;
  background: #f8fafc; border: 1px dashed var(--border-strong);
  display: flex; gap: 7px; align-items: center; flex-wrap: wrap;
}

.seg { display: inline-flex; background: var(--bg-soft); border-radius: 9px; padding: 2px; }
.seg button {
  border: none; background: transparent; padding: 5px 12px; border-radius: 7px;
  font-size: 12.5px; font-weight: 600; color: var(--text-3);
  cursor: pointer; font-family: inherit; transition: all .15s;
}
.seg button.on { background: #fff; color: var(--primary); box-shadow: var(--shadow-sm); }

.threshold-row { display: flex; align-items: center; gap: 12px; }
.threshold-row input { flex: 1; }
.threshold-val {
  font-family: var(--mono); font-size: 16px; font-weight: 700;
  color: var(--primary); min-width: 46px; text-align: right;
}
.preset-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin-top: 10px; }
.preset {
  border: 1px solid var(--border-strong); border-radius: 9px; background: #fff;
  padding: 7px 6px; cursor: pointer; font-family: inherit;
  display: flex; flex-direction: column; align-items: center; gap: 1px;
  transition: all .15s;
}
.preset:hover { border-color: var(--primary); }
.preset.on { border-color: var(--primary); background: var(--primary-soft); }
.preset b { font-size: 13.5px; color: var(--text); font-family: var(--mono); }
.preset span { font-size: 10.5px; color: var(--text-3); }

.adv summary {
  cursor: pointer; font-size: 12.5px; font-weight: 600; color: var(--text-3);
  padding: 6px 0; user-select: none;
}
.adv summary:hover { color: var(--primary); }

@media (max-width: 1150px) {
  .entry-grid, .detect-grid { grid-template-columns: 1fr; }
}
</style>
