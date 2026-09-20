<script setup>
import { ref, onMounted, computed } from 'vue'
import { api, fmtNum, classLabel } from '../api'

const exp = ref(null)
const err = ref('')
const tab = ref('compare')

const tabs = [
  { key: 'compare', label: '核心对比' },
  { key: 'pareto', label: '阈值与成本' },
  { key: 'ablation', label: '消融实验' },
  { key: 'hard', label: '难样本与假设' }
]

onMounted(async () => {
  try {
    exp.value = await api.experiments()
  } catch (e) {
    err.value = e.message
  }
})

const fig = (n) => `/api/experiments/figure/${n}`

const maxF1 = computed(() => {
  const rows = exp.value?.comparison || []
  return Math.max(0.01, ...rows.map(r => r.F1 || 0))
})

const h1Bins = computed(() => {
  const bins = exp.value?.h1_confidence_error || []
  const max = Math.max(0.01, ...bins.map(b => b.error_rate || 0))
  return bins.filter(b => b.n > 0).map(b => ({
    ...b,
    pct: (b.error_rate / max) * 100,
    label: `${b.lo.toFixed(1)}–${b.hi.toFixed(1)}`
  }))
})

const perClass = computed(() => {
  const h = exp.value?.hard_set || {}
  const cls = ['hole', 'stain', 'wrinkle']
  return cls.map(c => ({
    key: c,
    yolo40: h.yolo40?.per_class_recall?.[c] ?? 0,
    yolo25: h.yolo25?.per_class_recall?.[c] ?? 0,
    ours: h.ours?.per_class_recall?.[c] ?? 0,
    boxes: h.yolo40?.class_boxes?.[String(cls.indexOf(c))] ?? 0
  }))
})

function barW(v, max) { return `${Math.max(2, (v / max) * 100)}%` }
</script>

<template>
  <div class="page">
    <div v-if="err" class="alert alert-danger mb-3">
      <span class="ico">⚠️</span><div>{{ err }}</div>
    </div>

    <template v-if="exp">
      <!-- ============ 方法定义 ============ -->
      <div class="card card-pad mb-3 method-card">
        <div class="row-between wrap mb-2">
          <h2 style="font-size:17px">{{ exp.method.name }}</h2>
          <span class="badge badge-blue">重标 test 集 · 300 张 · 模型权重冻结</span>
        </div>
        <p class="method-line">「{{ exp.method.one_line }}」</p>
        <div class="framework">
          <div v-for="(s, i) in exp.method.framework_steps" :key="i" class="fw-step">
            <span class="fw-num">{{ i + 1 }}</span><span>{{ s }}</span>
          </div>
        </div>
        <div class="alert alert-info mt-2">
          <span class="ico">💡</span>
          <div><b>关键概念澄清：</b>{{ exp.method.key_point }}</div>
        </div>
      </div>

      <!-- ============ Tabs ============ -->
      <div class="tabs">
        <button v-for="t in tabs" :key="t.key"
                :class="{ on: tab === t.key }" @click="tab = t.key">{{ t.label }}</button>
      </div>

      <!-- ============ 核心对比 ============ -->
      <template v-if="tab === 'compare'">
        <div class="card mb-3">
          <div class="card-head">
            <h3>四组核心对比（主指标：类别严格的框级 F1，IoU = 0.5）</h3>
          </div>
          <div class="table-wrap">
            <table class="tbl">
              <thead>
                <tr>
                  <th>方法</th>
                  <th class="num">Precision</th>
                  <th class="num">Recall</th>
                  <th class="num">F1（严格）</th>
                  <th class="num">F1（类别无关）</th>
                  <th class="num">图像级 F1</th>
                  <th style="width:210px">F1 对比</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="r in exp.comparison" :key="r.key"
                    :class="{ ours: r.key.startsWith('D_') }">
                  <td>
                    <b>{{ r.label }}</b>
                    <div class="xsmall dim">{{ r.note }}</div>
                  </td>
                  <td class="num">{{ fmtNum(r.P) }}</td>
                  <td class="num">{{ fmtNum(r.R) }}</td>
                  <td class="num"><b>{{ fmtNum(r.F1) }}</b></td>
                  <td class="num">{{ fmtNum(r.F1_agnostic) }}</td>
                  <td class="num">{{ fmtNum(r.imgF1) }}</td>
                  <td>
                    <div class="bar-track">
                      <div class="bar-fill" :style="{ width: barW(r.F1 || 0, maxF1) }"></div>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="card-pad xsmall muted" style="border-top:1px solid var(--border)">
            口径说明：框级匹配 IoU = 0.5；<b>严格</b> = 检测框须命中同类别 GT（主指标）；
            <b>类别无关</b> = 命中任一真缺陷位置即算（辅助，支撑"找回漏检"的叙事）；
            <b>图像级</b> = 图像"有 / 无缺陷"二分类 F1（整体参考）。
          </div>
        </div>

        <div class="grid-2">
          <div class="card card-pad">
            <h3 style="font-size:14px" class="mb-2">方法框架图</h3>
            <img :src="fig('framework.png')" alt="方法框架图" class="fig-img" />
          </div>
          <div class="card card-pad">
            <h3 style="font-size:14px" class="mb-2">数据集与诚实基线</h3>
            <table class="tbl">
              <tbody>
                <tr><td>数据规模</td>
                    <td class="num">train {{ exp.dataset.train }} / val {{ exp.dataset.val }} / test {{ exp.dataset.test }}</td></tr>
                <tr><td>类别</td>
                    <td class="num">{{ exp.dataset.classes.map(classLabel).join(' / ') }}</td></tr>
                <tr><td>YOLOv11 mAP50</td>
                    <td class="num"><b>{{ fmtNum(exp.baseline?.mAP?.mAP50) }}</b></td></tr>
                <tr><td>YOLOv11 mAP50-95</td>
                    <td class="num">{{ fmtNum(exp.baseline?.mAP?.['mAP50-95']) }}</td></tr>
                <tr><td>框级 F1（严格）</td>
                    <td class="num"><b>{{ fmtNum(exp.baseline?.prf_full?.ALL?.F1) }}</b></td></tr>
                <tr><td>图像级 F1</td>
                    <td class="num">{{ fmtNum(exp.baseline?.prf_full ? 0.908 : null) }}</td></tr>
                <tr><td>YOLO 单图延迟</td>
                    <td class="num">{{ exp.collab_run?.t_yolo_per_img?.toFixed(4) }} s</td></tr>
                <tr><td>VLM 单次延迟</td>
                    <td class="num">{{ exp.collab_run?.t_vlm_avg?.toFixed(2) }} s</td></tr>
              </tbody>
            </table>
            <div class="alert alert-warn mt-2 xsmall">
              <span class="ico">⚠️</span>
              <div>{{ exp.dataset.note }}</div>
            </div>
          </div>
        </div>
      </template>

      <!-- ============ Pareto ============ -->
      <template v-if="tab === 'pareto'">
        <div class="card card-pad mb-3">
          <div class="row-between wrap mb-2">
            <h3 style="font-size:14px">阈值扫描与成本—性能权衡（Pareto 曲线）</h3>
            <span class="badge badge-blue">推荐操作点 T = 0.6</span>
          </div>
          <img :src="fig('pareto_curve.png')" alt="Pareto 曲线" class="fig-img" />
        </div>

        <div class="card mb-3">
          <div class="card-head"><h3>阈值扫描实测（严格口径）</h3></div>
          <div class="table-wrap">
            <table class="tbl">
              <thead>
                <tr>
                  <th>阈值 T</th>
                  <th class="num">VLM 调用 / 张</th>
                  <th class="num">F1（严格）</th>
                  <th class="num">边际收益 F1/次调用</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="p in exp.pareto" :key="p.T" :class="{ best: p.recommended }">
                  <td><b>{{ p.T.toFixed(2) }}</b></td>
                  <td class="num">{{ p.calls_per_img.toFixed(2) }}</td>
                  <td class="num"><b>{{ p.F1.toFixed(3) }}</b></td>
                  <td class="num">{{ p.marginal.toFixed(3) }}</td>
                  <td class="small">
                    <span v-if="p.recommended" class="badge badge-ok">Pareto 膝点 · 推荐</span>
                    <span v-else-if="p.T <= 0.5" class="muted">低预算场景</span>
                    <span v-else class="muted">高精度场景（调用成本接近翻倍）</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="grid-2">
          <div class="card card-pad">
            <h3 style="font-size:14px" class="mb-2">H3：成本—性能权衡（必须如实表述）</h3>
            <div class="tradeoff">
              <div class="to-item">
                <div class="to-num">28%</div>
                <div class="to-label">的调用量</div>
                <div class="to-sub">T = 0.6（0.75 次/张）</div>
              </div>
              <div class="to-item">
                <div class="to-num">35%</div>
                <div class="to-label">的 F1 提升</div>
                <div class="to-sub">相对全复核增益 0.240</div>
              </div>
            </div>
            <div class="tradeoff mt-2">
              <div class="to-item">
                <div class="to-num">48%</div>
                <div class="to-label">的调用量</div>
                <div class="to-sub">T = 0.7（1.30 次/张）</div>
              </div>
              <div class="to-item">
                <div class="to-num">45%</div>
                <div class="to-label">的 F1 提升</div>
                <div class="to-sub">相对全复核增益 0.240</div>
              </div>
            </div>
            <div class="alert alert-warn mt-2 xsmall">
              <span class="ico">⚠️</span>
              <div>全复核（F1 0.755）仍是性能上限。路由的价值是<b>在预算内逼近上限</b>，
                不能写成"超过全复核"。</div>
            </div>
          </div>

          <div class="card card-pad">
            <h3 style="font-size:14px" class="mb-2">H1：置信度—错误率曲线（方法的立论基础）</h3>
            <p class="xsmall muted mb-2">
              置信度驱动路由以"置信度"为唯一决策信号，因此必须先证明置信度确实反映检测质量。
            </p>
            <div class="h1-chart">
              <div v-for="b in h1Bins" :key="b.label" class="h1-col">
                <div class="h1-val">{{ (b.error_rate * 100).toFixed(0) }}%</div>
                <div class="h1-bar-track">
                  <div class="h1-bar" :style="{ height: b.pct + '%' }"
                       :class="{ low: b.hi <= 0.5 }"></div>
                </div>
                <div class="h1-lab">{{ b.label }}</div>
                <div class="h1-n">n={{ b.n }}</div>
              </div>
            </div>
            <div class="xsmall muted mt-2">
              置信度 &lt; 0.5 平均错误率约 25%，≥ 0.6 约 12%，0.6 → 0.9 单调下降 ——
              <b>低置信度样本确实是误检/漏检的高发区</b>。
            </div>
            <img :src="fig('h1_confidence_error.png')" alt="置信度-错误率曲线" class="fig-img mt-2" />
          </div>
        </div>
      </template>

      <!-- ============ 消融 ============ -->
      <template v-if="tab === 'ablation'">
        <div class="card mb-3">
          <div class="card-head">
            <h3>消融实验：每个组件各自贡献了多少</h3>
            <span class="head-sub">严格口径，重标 test</span>
          </div>
          <div class="table-wrap">
            <table class="tbl">
              <thead>
                <tr>
                  <th>方案</th>
                  <th class="center">YOLO</th>
                  <th class="center">局部 Crop</th>
                  <th class="center">VLM</th>
                  <th class="center">动态路由</th>
                  <th class="num">F1（严格）</th>
                  <th class="num">F1（无关）</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="r in exp.ablation" :key="r.key">
                  <td><b>{{ r.label }}</b><div class="xsmall dim">{{ r.note }}</div></td>
                  <td class="center">✓</td>
                  <td class="center">{{ r.crop ? '✓' : '—' }}</td>
                  <td class="center">{{ r.vlm ? '✓' : '—' }}</td>
                  <td class="center">{{ r.router ? '✓' : '—' }}</td>
                  <td class="num"><b>{{ fmtNum(r.F1) }}</b></td>
                  <td class="num">{{ fmtNum(r.F1_agnostic) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="grid-2">
          <div class="card card-pad">
            <h3 style="font-size:14px" class="mb-2">H2 直接证据：局部裁剪 vs 整图</h3>
            <p class="xsmall muted mb-2">
              同一批 {{ exp.h2_crop_vs_full?.n_det }} 个检测框，只改变送给大模型的输入形式。
            </p>
            <table class="tbl">
              <thead>
                <tr>
                  <th>VLM 输入</th>
                  <th class="num">准确率</th>
                  <th class="num">Precision</th>
                  <th class="num">Recall</th>
                  <th class="num">F1</th>
                </tr>
              </thead>
              <tbody>
                <tr class="best">
                  <td><b>裁剪图（+尺度提示）</b></td>
                  <td class="num">{{ fmtNum(exp.h2_crop_vs_full?.vlm_on_crop?.acc) }}</td>
                  <td class="num">{{ fmtNum(exp.h2_crop_vs_full?.vlm_on_crop?.P) }}</td>
                  <td class="num">{{ fmtNum(exp.h2_crop_vs_full?.vlm_on_crop?.R) }}</td>
                  <td class="num"><b>{{ fmtNum(exp.h2_crop_vs_full?.vlm_on_crop?.F1) }}</b></td>
                </tr>
                <tr>
                  <td>整图</td>
                  <td class="num">{{ fmtNum(exp.h2_crop_vs_full?.vlm_on_full_image?.acc) }}</td>
                  <td class="num">{{ fmtNum(exp.h2_crop_vs_full?.vlm_on_full_image?.P) }}</td>
                  <td class="num">{{ fmtNum(exp.h2_crop_vs_full?.vlm_on_full_image?.R) }}</td>
                  <td class="num">{{ fmtNum(exp.h2_crop_vs_full?.vlm_on_full_image?.F1) }}</td>
                </tr>
              </tbody>
            </table>
            <div class="alert alert-ok mt-2 xsmall">
              <span class="ico">✓</span>
              <div>裁剪图显著更好，提升主要在<b>拒绝假缺陷</b>
                （TN {{ exp.h2_crop_vs_full?.vlm_on_crop?.tn }} vs
                {{ exp.h2_crop_vs_full?.vlm_on_full_image?.tn }}），召回几乎不变。</div>
            </div>
          </div>

          <div class="card card-pad">
            <h3 style="font-size:14px" class="mb-2">消融结论</h3>
            <ol class="conc-list">
              <li><b>局部 Crop 有效：</b>B（逐框裁剪复核，{{ fmtNum(exp.h2_crop_vs_full?.vlm_on_crop?.F1) }} 的框级判定 F1）
                ≫ C（整图判定）——"聚焦局部区域"确实增强了大模型对细粒度微痕的复核能力。</li>
              <li><b>动态路由的价值是成本—性能权衡：</b>D 用 0.75 次调用/张拿到全复核 B 约 35% 的 F1 提升。</li>
              <li><b>全复核是性能上限：</b>B 的 F1 达 0.755、图像级 0.954，但需 2.7 次调用/张。</li>
              <li><b>协同 &gt; 基线且 &gt; 低阈值控制组：</b>证明提升来自大模型语义复核，而非单纯降低 YOLO 阈值。</li>
            </ol>
          </div>
        </div>
      </template>

      <!-- ============ 难样本与假设 ============ -->
      <template v-if="tab === 'hard'">
        <div class="card card-pad mb-3">
          <div class="row-between wrap mb-2">
            <h3 style="font-size:14px">真正的难度轴是「类别」，不是面积/对比度</h3>
            <span class="badge badge-warn">一次如实的负结果</span>
          </div>
          <div class="class-recall">
            <div v-for="c in perClass" :key="c.key" class="cr-item">
              <div class="cr-head">
                <span class="badge" :class="'badge-' + c.key">{{ classLabel(c.key) }}</span>
                <span class="xsmall dim">{{ c.boxes }} 个 GT 框</span>
              </div>
              <div class="cr-row">
                <span class="cr-lab">YOLO@0.4</span>
                <div class="bar-track"><div class="bar-fill gray" :style="{ width: (c.yolo40 * 100) + '%' }"></div></div>
                <span class="cr-val">{{ c.yolo40.toFixed(3) }}</span>
              </div>
              <div class="cr-row">
                <span class="cr-lab">YOLO@0.25</span>
                <div class="bar-track"><div class="bar-fill warn" :style="{ width: (c.yolo25 * 100) + '%' }"></div></div>
                <span class="cr-val">{{ c.yolo25.toFixed(3) }}</span>
              </div>
              <div class="cr-row">
                <span class="cr-lab">Ours(T=0.6)</span>
                <div class="bar-track"><div class="bar-fill blue" :style="{ width: (c.ours * 100) + '%' }"></div></div>
                <span class="cr-val strong">{{ c.ours.toFixed(3) }}</span>
              </div>
            </div>
          </div>
          <div class="alert alert-info mt-3">
            <span class="ico">📌</span>
            <div>
              <b>褶皱（wrinkle）是最难的类别</b>：基线召回仅
              {{ (exp.hard_set?.yolo40?.per_class_recall?.wrinkle ?? 0).toFixed(3) }}；
              协同方法在它上面提升最大
              （{{ (exp.hard_set?.yolo40?.per_class_recall?.wrinkle ?? 0).toFixed(3) }} →
              {{ (exp.hard_set?.ours?.per_class_recall?.wrinkle ?? 0).toFixed(3) }}，相对约 +50%），
              远高于破洞与污渍的增益 —— <b>"提升主要来自疑难样本"成立</b>。
            </div>
          </div>
          <div class="alert alert-warn mt-2">
            <span class="ico">⚠️</span>
            <div>
              面积/对比度定义的 Hard Set 与类别混淆（面积大的框多为褶皱），
              导致"简单集"召回反而低于"困难集"
              （{{ (exp.hard_set?.yolo40?.easy_recall ?? 0).toFixed(3) }} &lt;
              {{ (exp.hard_set?.yolo40?.hard_recall ?? 0).toFixed(3) }}）。
              因此<b>面积/对比度不是本数据集的有效难度轴</b>，不能据此宣称
              "提升来自小目标/低对比样本"。
            </div>
          </div>
        </div>

        <div class="card card-pad mb-3">
          <h3 style="font-size:14px" class="mb-2">研究假设与验证状态</h3>
          <div class="hypo-grid">
            <div v-for="h in exp.hypotheses" :key="h.id" class="hypo-card">
              <div class="row gap-1 mb-1">
                <span class="badge badge-blue">{{ h.id }}</span>
                <span class="badge badge-ok">{{ h.status }}</span>
              </div>
              <div class="small" style="color:var(--text-2)">{{ h.text }}</div>
              <div class="xsmall muted mt-1">证据：{{ h.evidence }}</div>
            </div>
          </div>
        </div>

        <div class="card card-pad">
          <h3 style="font-size:14px" class="mb-2">必须如实说明的边界</h3>
          <div v-for="(b, i) in exp.boundary" :key="i" class="boundary">
            <span class="boundary-dot">!</span><div>{{ b }}</div>
          </div>
        </div>
      </template>
    </template>

    <div v-else-if="!err" class="card card-pad">
      <div class="skeleton" style="height:130px;margin-bottom:12px"></div>
      <div class="skeleton" style="height:220px"></div>
    </div>
  </div>
</template>

<style scoped>
.method-card { border-color: #dbe7f6; background: linear-gradient(150deg,#fff,#f9fbff); }
.method-line { font-size: 14px; color: var(--primary-dark); font-weight: 600; margin: 0 0 12px; }
.framework { display: grid; gap: 7px; }
.fw-step {
  display: flex; gap: 10px; align-items: flex-start; font-size: 12.5px;
  color: var(--text-2); line-height: 1.6;
}
.fw-num {
  flex: 0 0 19px; height: 19px; border-radius: 50%; background: var(--primary-soft);
  color: var(--primary-dark); display: grid; place-items: center;
  font-size: 11px; font-weight: 800; margin-top: 1px;
}

.tabs { display: flex; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; }
.tabs button {
  padding: 8px 16px; border-radius: 9px; border: 1px solid var(--border);
  background: #fff; color: var(--text-3); font-size: 13px; font-weight: 600;
  cursor: pointer; font-family: inherit; transition: all .15s;
}
.tabs button:hover { border-color: var(--primary); color: var(--primary); }
.tabs button.on { background: var(--primary); border-color: var(--primary); color: #fff;
  box-shadow: 0 2px 10px rgba(37,99,235,.3); }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

.bar-track {
  height: 8px; background: var(--bg-soft); border-radius: 999px; overflow: hidden; min-width: 90px;
}
.bar-fill { height: 100%; background: linear-gradient(90deg,#60a5fa,#2563eb); border-radius: 999px; }
.bar-fill.gray { background: linear-gradient(90deg,#cbd5e1,#94a3b8); }
.bar-fill.warn { background: linear-gradient(90deg,#fcd34d,#f59e0b); }
.bar-fill.blue { background: linear-gradient(90deg,#60a5fa,#2563eb); }

.fig-img {
  width: 100%; border-radius: 10px; border: 1px solid var(--border); background: #fff;
}

.tradeoff { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.to-item {
  text-align: center; padding: 11px; border-radius: 10px;
  background: #f5f9ff; border: 1px solid #dbeafe;
}
.to-num { font-size: 22px; font-weight: 800; color: var(--primary); font-variant-numeric: tabular-nums; }
.to-label { font-size: 12px; font-weight: 600; color: var(--text-2); }
.to-sub { font-size: 10.5px; color: var(--text-4); margin-top: 2px; }

.h1-chart {
  display: flex; align-items: flex-end; gap: 5px; height: 150px;
  padding: 6px 2px 0; border-bottom: 1px solid var(--border);
}
.h1-col { flex: 1; display: flex; flex-direction: column; align-items: center; height: 100%; }
.h1-val { font-size: 10px; font-weight: 700; color: var(--text-2); font-variant-numeric: tabular-nums; }
.h1-bar-track { flex: 1; width: 100%; display: flex; align-items: flex-end; }
.h1-bar {
  width: 100%; background: linear-gradient(180deg,#93c5fd,#3b82f6);
  border-radius: 3px 3px 0 0; min-height: 2px;
}
.h1-bar.low { background: linear-gradient(180deg,#fca5a5,#ef4444); }
.h1-lab { font-size: 9px; color: var(--text-4); white-space: nowrap; margin-top: 3px; }
.h1-n { font-size: 8.5px; color: var(--text-4); }

.conc-list { margin: 0; padding-left: 18px; font-size: 12.5px; color: var(--text-2); line-height: 1.75; }
.conc-list li { margin-bottom: 8px; }

.class-recall { display: grid; grid-template-columns: repeat(auto-fit, minmax(255px, 1fr)); gap: 16px; }
.cr-item { border: 1px solid var(--border); border-radius: 11px; padding: 13px 15px; background: #fcfdff; }
.cr-head { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.cr-row { display: flex; align-items: center; gap: 9px; margin-bottom: 6px; }
.cr-lab { flex: 0 0 78px; font-size: 11px; color: var(--text-3); font-family: var(--mono); }
.cr-val { flex: 0 0 44px; text-align: right; font-size: 11.5px; font-family: var(--mono); color: var(--text-2); }

.hypo-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.hypo-card { border: 1px solid var(--border); border-radius: 11px; padding: 13px 15px; background: #fcfdff; }

.boundary {
  display: flex; gap: 9px; padding: 9px 0; font-size: 12.5px;
  color: var(--text-2); line-height: 1.65; border-bottom: 1px dashed var(--border);
}
.boundary:last-child { border-bottom: none; }
.boundary-dot {
  flex: 0 0 17px; height: 17px; border-radius: 50%; background: var(--warn-soft);
  color: var(--warn); display: grid; place-items: center;
  font-size: 11px; font-weight: 800; margin-top: 2px;
}

@media (max-width: 1150px) { .grid-2 { grid-template-columns: 1fr; } }
</style>
