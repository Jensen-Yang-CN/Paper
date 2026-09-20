<script setup>
import { ref, onMounted } from 'vue'
import { api } from '../api'

const tools = ref(null)
const model = ref(null)
const health = ref(null)
const err = ref('')

const modules = [
  { n: 1, name: '图片上传模块', desc: '支持拖拽上传、格式校验与内置样例图库（零上传体验）' },
  { n: 2, name: 'YOLOv11 缺陷检测模块', desc: '冻结权重推理，输出破洞 / 污渍 / 褶皱三类候选框与置信度' },
  { n: 3, name: '局部区域处理模块', desc: '按框外扩裁剪、过小自动放大、可选 CLAHE 对比度增强' },
  { n: 4, name: '多模态复核模块', desc: 'qwen3-vl-plus 对局部证据做语义判定，输出结构化 JSON' },
  { n: 5, name: 'Agent 协同决策模块', desc: '置信度驱动的动态路由 + 规则融合（含类别纠错）' },
  { n: 6, name: '结果展示模块', desc: '检测框可视化、逐框决策溯源、仅 YOLO 与协同对比、报告导出' },
  { n: 7, name: '检测记录模块', desc: 'SQLite 持久化、历史检索、统计总览' }
]

const stack = [
  { k: '后端', v: 'FastAPI + Uvicorn + Pydantic v2' },
  { k: '前端', v: 'Vue 3 + Vite + Vue Router' },
  { k: '视觉模型', v: 'YOLOv11（Ultralytics，PyTorch，权重冻结）' },
  { k: '多模态模型', v: 'qwen3-vl-plus（DashScope 兼容 OpenAI 协议）' },
  { k: '图像处理', v: 'OpenCV + Pillow + NumPy' },
  { k: '数据存储', v: 'SQLite（单文件，零运维）' }
]

onMounted(async () => {
  try {
    ;[tools.value, model.value, health.value] = await Promise.all([
      api.tools(), api.modelInfo(), api.health()
    ])
  } catch (e) {
    err.value = e.message
  }
})
</script>

<template>
  <div class="page page-narrow">
    <div class="card card-pad mb-3 about-hero">
      <div class="row-between wrap">
        <div>
          <h1 style="font-size:22px">ClothInspect-Agent 服装智能质检系统</h1>
          <div class="muted" style="margin-top:5px">
            版本 V1.0 · 视觉模型与多模态大模型协同推理
          </div>
        </div>
        <div class="col-right-badges">
          <span class="badge badge-blue">软件著作权候选成果</span>
          <span class="badge badge-gray">推理时协同</span>
        </div>
      </div>
      <p class="mt-2" style="font-size:13.5px;color:var(--text-2);margin-bottom:0">
        本系统是论文《面向服装表面微痕检测的视觉模型与多模态大模型协同推理方法研究》的
        配套软件实现，用于验证"由检测置信度驱动的动态路由 + 多模态大模型复核 +
        规则融合"这一协同推理机制在服装表面微痕检测中的有效性与效率优势。
      </p>
    </div>

    <div v-if="err" class="alert alert-danger mb-3">
      <span class="ico">⚠️</span><div>{{ err }}</div>
    </div>

    <!-- 运行环境 -->
    <div class="card mb-3">
      <div class="card-head"><h3>运行环境</h3></div>
      <div class="table-wrap">
        <table class="tbl">
          <tbody>
            <tr>
              <td style="width:170px">视觉检测模型</td>
              <td>
                <span v-if="model?.detector?.weights_found" class="badge badge-ok">已就绪</span>
                <span v-else class="badge badge-danger">权重未找到</span>
                <div class="xsmall dim mono mt-1">{{ model?.detector?.weights_path || '—' }}</div>
              </td>
            </tr>
            <tr>
              <td>多模态复核后端</td>
              <td>
                <span v-if="model?.vlm?.is_real" class="badge badge-ok">真实调用 {{ model.vlm.model }}</span>
                <span v-else class="badge badge-warn">离线启发式（未配置 API Key）</span>
                <div class="xsmall dim mt-1">{{ model?.vlm?.note }}</div>
              </td>
            </tr>
            <tr>
              <td>推理设备</td>
              <td class="mono">{{ model?.params?.device }} · 输入尺寸 {{ model?.params?.img_size }}</td>
            </tr>
            <tr>
              <td>算法超参</td>
              <td class="mono">
                默认阈值 T={{ health ? '0.60' : '—' }} ·
                框级 IoU 阈值 {{ model?.params?.iou_threshold }} ·
                裁剪外扩 {{ ((model?.params?.crop_margin ?? 0) * 100).toFixed(0) }}% ·
                最小裁剪边长 {{ model?.params?.min_crop }} px
              </td>
            </tr>
            <tr>
              <td>内置样例图库</td>
              <td>
                <span v-if="health?.samples?.available" class="badge badge-ok">
                  {{ health.samples.count }} 张可用
                </span>
                <span v-else class="badge badge-warn">未配置</span>
                <div class="xsmall dim mono mt-1">{{ health?.samples?.dir || '—' }}</div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 功能模块 -->
    <div class="card mb-3">
      <div class="card-head">
        <h3>软件功能模块</h3>
        <span class="head-sub">软著申报口径的七个模块</span>
      </div>
      <div class="card-pad">
        <div class="mod-grid">
          <div v-for="m in modules" :key="m.n" class="mod-item">
            <span class="mod-num">{{ m.n }}</span>
            <div>
              <div class="mod-name">{{ m.name }}</div>
              <div class="xsmall muted">{{ m.desc }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Agent 工具 -->
    <div class="card mb-3" v-if="tools">
      <div class="card-head">
        <h3>Agent 工具集</h3>
        <span class="head-sub">Function Calling 风格声明，共 {{ tools.count }} 个</span>
      </div>
      <div class="card-pad">
        <div class="tool-grid">
          <div v-for="t in tools.tools" :key="t.name" class="tool-card">
            <div class="row-between mb-1">
              <code class="tool-name">{{ t.name }}</code>
              <span class="badge badge-blue xsmall">{{ t.label }}</span>
            </div>
            <div class="xsmall muted">{{ t.description }}</div>
          </div>
        </div>

        <div class="divider"></div>
        <h4 style="font-size:13px" class="mb-2">执行顺序</h4>
        <div class="pipe">
          <template v-for="(p, i) in tools.pipeline" :key="p.step">
            <div class="pipe-step">
              <div class="pipe-num">{{ p.step }}</div>
              <div>
                <div class="pipe-title">{{ p.title }}</div>
                <div class="xsmall muted">{{ p.desc }}</div>
                <code class="xsmall dim">{{ p.tool }}</code>
              </div>
            </div>
            <div v-if="i < tools.pipeline.length - 1" class="pipe-arrow">↓</div>
          </template>
        </div>
      </div>
    </div>

    <!-- 技术栈 -->
    <div class="card mb-3">
      <div class="card-head"><h3>技术栈</h3></div>
      <div class="table-wrap">
        <table class="tbl">
          <tbody>
            <tr v-for="s in stack" :key="s.k">
              <td style="width:170px"><b>{{ s.k }}</b></td>
              <td>{{ s.v }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 使用与合规 -->
    <div class="card card-pad">
      <h3 style="font-size:14px" class="mb-2">使用与合规说明</h3>
      <ul class="notes">
        <li>
          <b>本系统展示的数据取自论文实验</b>（重标 test 集 300 张 / 模型 V6），
          系统界面中的单张图片结果仅为流程演示，论文结论以批量实验为准。
        </li>
        <li>
          <b>未配置 <code>DASHSCOPE_API_KEY</code> 时</b>，多模态复核会自动降级为
          离线启发式算法。它<b>不是大模型</b>，仅用于保证系统在完全离线时仍可完整演示，
          相关结果不代表论文结论。
        </li>
        <li>
          <b>关于基线诚实性：</b>论文基线在修正后的重标 test 上评估
          （mAP50 = 0.387、框级 F1 = 0.515）。早期 0.865 的口径因训练/验证集泄漏
          与旧标注漏标而虚高，不作为论文指标。
        </li>
        <li>
          接口文档：<a href="/docs" target="_blank" rel="noopener">/docs</a>
          （FastAPI 自动生成的 OpenAPI 文档，可直接在线调试全部接口）。
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.about-hero { border-color: #dbe7f6; background: linear-gradient(150deg,#fff,#f9fbff); }
.col-right-badges { display: flex; gap: 6px; }

.mod-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.mod-item { display: flex; gap: 11px; align-items: flex-start; }
.mod-num {
  flex: 0 0 24px; height: 24px; border-radius: 7px;
  background: var(--primary-soft); color: var(--primary-dark);
  display: grid; place-items: center; font-size: 12px; font-weight: 800;
}
.mod-name { font-size: 13px; font-weight: 700; margin-bottom: 1px; }

.tool-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 12px; }
.tool-card { border: 1px solid var(--border); border-radius: 11px; padding: 12px 14px; background: #fcfdff; }
.tool-name {
  font-family: var(--mono); font-size: 12.5px; font-weight: 700;
  color: var(--primary-dark); background: var(--primary-soft);
  padding: 2px 8px; border-radius: 6px;
}

.pipe { display: grid; gap: 4px; }
.pipe-step { display: flex; gap: 11px; align-items: flex-start; }
.pipe-num {
  flex: 0 0 22px; height: 22px; border-radius: 50%;
  background: var(--bg-soft); color: var(--text-3);
  display: grid; place-items: center; font-size: 11px; font-weight: 800;
  margin-top: 1px;
}
.pipe-title { font-size: 13px; font-weight: 700; }
.pipe-arrow { color: var(--border-strong); padding-left: 7px; font-size: 12px; line-height: 1; }

.notes { margin: 0; padding-left: 19px; font-size: 12.5px; color: var(--text-2); line-height: 1.75; }
.notes li { margin-bottom: 9px; }
.notes code {
  font-family: var(--mono); font-size: 11.5px; background: var(--bg-soft);
  padding: 1px 5px; border-radius: 5px;
}
</style>
