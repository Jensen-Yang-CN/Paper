<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api, fmtNum } from '../api'

const router = useRouter()
const exp = ref(null)
const tools = ref(null)
const err = ref('')

const steps = [
  { n: 1, icon: '🎯', title: '视觉检测', who: 'YOLOv11',
    desc: '对整图做一次推理，输出候选缺陷框（类别 / 坐标 / 置信度）。定位精确，但低置信度样本与褶皱容易漏检。' },
  { n: 2, icon: '🔀', title: '置信度路由', who: '协同策略',
    desc: '置信度 ≥ 阈值 T 的框直接接受；低于 T 的判为疑难样本。这是整套方法的"裁判"，决定什么时候值得花钱调大模型。' },
  { n: 3, icon: '✂️', title: '局部裁剪', who: 'crop_region',
    desc: '以框为中心外扩 30% 裁出局部区域，过小则放大，并告知大模型这块区域在原图中的尺度。' },
  { n: 4, icon: '🤖', title: '多模态复核', who: 'qwen3-vl-plus',
    desc: '大模型判断"这块区域到底是不是瑕疵、属于哪一类"，返回结构化 JSON。它擅长语义判断，但定位粗，不替代 YOLO。' },
  { n: 5, icon: '🧩', title: '结果融合', who: '规则融合',
    desc: '大模型判为瑕疵则保留该框且类别以它为准（类别纠错），否则剔除；框的精确定位始终沿用 YOLO。' }
]

const quickStart = [
  { icon: '①', title: '选一张图', desc: '进入检测工作台，直接使用内置样例，或拖入你自己的服装质检图。' },
  { icon: '②', title: '点"开始检测"', desc: '默认参数就是论文推荐的 T = 0.6（Pareto 膝点）。想对比不同预算，拖动阈值滑块即可。' },
  { icon: '③', title: '看协同过程', desc: '右侧时间线会逐步展示每个工具做了什么；下方对比卡片告诉你协同相对"仅 YOLO"改了什么。' }
]

onMounted(async () => {
  try {
    exp.value = await api.experiments()
    tools.value = await api.tools()
  } catch (e) {
    err.value = e.message
  }
})

const baseline = () => exp.value?.baseline?.prf_full?.ALL || {}
// 论文推荐操作点 T=0.60（Research MVP 定版）
const ours = () => (exp.value?.comparison || []).find(x => x.key === 'D_ours06') || {}
</script>

<template>
  <div class="page">
    <!-- ============ Hero ============ -->
    <section class="hero">
      <div class="hero-main">
        <span class="badge badge-blue mb-2">软件著作权候选成果 · V1.0</span>
        <h1 class="hero-title">
          让 YOLO 负责<span class="hl">快速发现</span>，<br />
          让多模态大模型负责<span class="hl2">疑难复核</span>
        </h1>
        <p class="hero-desc">
          本系统研究的是 <b>推理时协同</b>：模型权重完全冻结，在系统层面加一个"裁判"——
          由 YOLO 检测置信度驱动的动态路由，决定哪些疑难样本值得调用多模态大模型复核。
          目标是在<b>控制大模型调用成本</b>的同时，提升服装表面微痕检测的可靠性。
        </p>
        <div class="row wrap mt-3">
          <button class="btn btn-primary btn-lg" @click="router.push('/detect')">
            🔍 开始检测
          </button>
          <button class="btn btn-lg" @click="router.push('/experiments')">
            📊 查看实验数据
          </button>
        </div>
        <div class="row wrap mt-3 gap-1">
          <span class="badge badge-gray">YOLOv11 冻结权重</span>
          <span class="badge badge-gray">qwen3-vl-plus 复核</span>
          <span class="badge badge-gray">hole / stain / wrinkle 三类</span>
          <span class="badge badge-gray">FastAPI + Vue3</span>
        </div>
      </div>

      <div class="hero-side">
        <div class="hero-card">
          <div class="hero-card-title">论文核心结果（重标 test，300 张）</div>
          <div class="hero-metric">
            <div>
              <div class="hm-label">框级 F1（严格口径）</div>
              <div class="hm-flow">
                <span class="hm-from">{{ fmtNum(baseline().F1, 3) }}</span>
                <span class="hm-arrow">→</span>
                <span class="hm-to">{{ fmtNum(ours().F1, 3) }}</span>
              </div>
              <div class="xsmall muted mt-1">仅 YOLO 基线 → 协同推理（T=0.6）</div>
            </div>
          </div>
          <div class="hero-card-foot">
            <div class="hcf-item">
              <b>{{ fmtNum(ours().F1 - baseline().F1, 3) }}</b>
              <span>F1 绝对提升</span>
            </div>
            <div class="hcf-item">
              <b>0.75</b>
              <span>次调用 / 张</span>
            </div>
            <div class="hcf-item">
              <b>0.540</b>
              <span>低阈值控制组</span>
            </div>
          </div>
          <div class="alert alert-info mt-2" style="font-size:11.5px">
            <span class="ico">✓</span>
            <div>
              协同方法同时优于<b>基线 0.515</b> 与<b>低阈值控制组 0.540</b>，
              证明提升来自大模型语义复核，而非单纯降低 YOLO 阈值。
            </div>
          </div>
        </div>
      </div>
    </section>

    <div v-if="err" class="alert alert-danger mt-3">
      <span class="ico">⚠️</span>
      <div>{{ err }} —— 请确认后端服务已启动（默认 http://127.0.0.1:8000）。</div>
    </div>

    <!-- ============ 新用户三步上手 ============ -->
    <section class="mt-4">
      <div class="sec-head">
        <h2>三步上手</h2>
        <span class="muted small">不需要任何配置，打开就能跑</span>
      </div>
      <div class="quick-grid">
        <div v-for="q in quickStart" :key="q.title" class="quick-card">
          <div class="quick-badge">{{ q.icon }}</div>
          <div>
            <div class="quick-title">{{ q.title }}</div>
            <div class="quick-desc">{{ q.desc }}</div>
          </div>
        </div>
      </div>
    </section>

    <!-- ============ 方法流程 ============ -->
    <section class="mt-4">
      <div class="sec-head">
        <h2>系统是怎么工作的</h2>
        <span class="muted small">论文方法框架 · 五步协同推理</span>
      </div>
      <div class="flow">
        <template v-for="(s, i) in steps" :key="s.n">
          <div class="flow-card">
            <div class="flow-num">{{ s.n }}</div>
            <div class="flow-icon">{{ s.icon }}</div>
            <div class="flow-title">{{ s.title }}</div>
            <div class="flow-who">{{ s.who }}</div>
            <div class="flow-desc">{{ s.desc }}</div>
          </div>
          <div v-if="i < steps.length - 1" class="flow-arrow">→</div>
        </template>
      </div>
    </section>

    <!-- ============ 假设与边界 ============ -->
    <section class="mt-4 grid-2">
      <div class="card">
        <div class="card-head">
          <h3>三个研究假设及验证状态</h3>
          <div class="head-right"><span class="badge badge-ok">均有实验支撑</span></div>
        </div>
        <div class="card-pad">
          <div v-for="h in (exp?.hypotheses || [])" :key="h.id" class="hypo">
            <div class="hypo-head">
              <span class="badge badge-blue">{{ h.id }}</span>
              <span class="badge badge-ok">{{ h.status }}</span>
            </div>
            <div class="hypo-text">{{ h.text }}</div>
            <div class="xsmall muted">证据：{{ h.evidence }}</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h3>必须如实说明的边界</h3>
          <div class="head-right"><span class="badge badge-warn">诚实性</span></div>
        </div>
        <div class="card-pad">
          <div v-for="(b, i) in (exp?.boundary || [])" :key="i" class="boundary">
            <span class="boundary-dot">!</span>
            <div>{{ b }}</div>
          </div>
          <div v-if="tools" class="divider"></div>
          <div v-if="tools" class="xsmall muted">
            系统实现了论文要求的 {{ tools.count }} 个 Agent 工具：
            <code v-for="t in tools.tools" :key="t.name" class="code-chip">{{ t.name }}</code>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.hero {
  display: grid; grid-template-columns: 1.35fr 1fr; gap: 26px; align-items: start;
}
.hero-title { font-size: 30px; line-height: 1.3; letter-spacing: -.4px; }
.hl { color: var(--primary); }
.hl2 { color: var(--accent); }
.hero-desc { color: var(--text-2); font-size: 14.5px; margin-top: 12px; max-width: 620px; }

.hero-card {
  background: linear-gradient(160deg, #ffffff, #f8fbff);
  border: 1px solid #dbe7f6; border-radius: var(--radius-lg);
  padding: 20px; box-shadow: var(--shadow);
}
.hero-card-title { font-size: 12.5px; font-weight: 700; color: var(--text-3); margin-bottom: 12px; }
.hm-label { font-size: 12px; color: var(--text-3); font-weight: 600; }
.hm-flow { display: flex; align-items: baseline; gap: 10px; margin-top: 2px; }
.hm-from { font-size: 26px; font-weight: 700; color: var(--text-4); font-variant-numeric: tabular-nums; }
.hm-arrow { color: var(--text-4); font-size: 17px; }
.hm-to { font-size: 34px; font-weight: 800; color: var(--primary); font-variant-numeric: tabular-nums; }
.hero-card-foot {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
  margin-top: 16px; padding-top: 14px; border-top: 1px dashed var(--border);
}
.hcf-item { text-align: center; }
.hcf-item b { display: block; font-size: 16px; color: var(--text); font-variant-numeric: tabular-nums; }
.hcf-item span { font-size: 10.5px; color: var(--text-4); }

.sec-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px; }
.sec-head h2 { font-size: 18px; }

.quick-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
.quick-card {
  display: flex; gap: 13px; padding: 17px 18px;
  background: #fff; border: 1px solid var(--border); border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
}
.quick-badge {
  flex: 0 0 34px; height: 34px; border-radius: 10px;
  background: var(--primary-soft); color: var(--primary-dark);
  display: grid; place-items: center; font-size: 16px; font-weight: 800;
}
.quick-title { font-weight: 700; font-size: 14px; margin-bottom: 3px; }
.quick-desc { font-size: 12.5px; color: var(--text-3); line-height: 1.6; }

.flow { display: flex; align-items: stretch; gap: 6px; overflow-x: auto; padding-bottom: 6px; }
.flow-card {
  flex: 1 1 0; min-width: 186px;
  background: #fff; border: 1px solid var(--border); border-radius: var(--radius);
  padding: 15px 15px 16px; position: relative; box-shadow: var(--shadow-sm);
}
.flow-num {
  position: absolute; top: 11px; right: 13px;
  font-size: 22px; font-weight: 800; color: #eef2f9;
}
.flow-icon { font-size: 21px; margin-bottom: 7px; }
.flow-title { font-weight: 700; font-size: 14px; }
.flow-who {
  display: inline-block; font-size: 10.5px; font-family: var(--mono);
  color: var(--primary-dark); background: var(--primary-soft);
  padding: 1px 7px; border-radius: 5px; margin: 5px 0 8px;
}
.flow-desc { font-size: 11.5px; color: var(--text-3); line-height: 1.65; }
.flow-arrow { align-self: center; color: var(--border-strong); font-size: 17px; flex: 0 0 auto; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

.hypo { padding: 11px 0; border-bottom: 1px dashed var(--border); }
.hypo:last-child { border-bottom: none; padding-bottom: 0; }
.hypo-head { display: flex; gap: 6px; margin-bottom: 5px; }
.hypo-text { font-size: 13px; color: var(--text-2); line-height: 1.65; margin-bottom: 4px; }

.boundary {
  display: flex; gap: 9px; padding: 9px 0; font-size: 12.5px;
  color: var(--text-2); line-height: 1.65; border-bottom: 1px dashed var(--border);
}
.boundary:last-of-type { border-bottom: none; }
.boundary-dot {
  flex: 0 0 17px; height: 17px; border-radius: 50%;
  background: var(--warn-soft); color: var(--warn);
  display: grid; place-items: center; font-size: 11px; font-weight: 800;
  margin-top: 2px;
}
.code-chip {
  font-family: var(--mono); font-size: 11px; background: var(--bg-soft);
  padding: 1px 6px; border-radius: 5px; margin-left: 4px; color: var(--text-2);
}

@media (max-width: 1150px) {
  .hero { grid-template-columns: 1fr; }
  .grid-2 { grid-template-columns: 1fr; }
}
</style>
