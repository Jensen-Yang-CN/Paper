<script setup>
/**
 * 检测结果详情面板
 * 检测工作台与检测记录页共用，保证两处展示完全一致。
 */
import { computed } from 'vue'
import { classLabel, classColor, fmtMs, fmtNum } from '../api'

const props = defineProps({
  result: { type: Object, required: true },
  showCompare: { type: Boolean, default: true }
})
const emit = defineEmits(['hover-box', 'active-box'])

const r = computed(() => props.result || {})
const routing = computed(() => r.value.routing || {})
const fusion = computed(() => r.value.fusion || {})
const timing = computed(() => r.value.timing || {})
const vlm = computed(() => r.value.vlm || {})
const boxes = computed(() => r.value.final_boxes || [])
const decisions = computed(() => r.value.decisions || [])
const isRealVlm = computed(() => vlm.value.mode === 'real')

const classCounts = computed(() => {
  const m = {}
  boxes.value.forEach(b => { m[b.cls_name] = (m[b.cls_name] || 0) + 1 })
  return m
})

const boxRows = computed(() => boxes.value.map((b, i) => {
  const dec = decisions.value.find(d => Math.abs((d.conf ?? -1) - b.conf) < 1e-9) || {}
  return { ...b, i, routeReason: dec.reason || b.route_reason || '' }
}))

/** 协同相对"仅 YOLO 基线"的变化叙述 */
const compare = computed(() => {
  const c = r.value.comparison || {}
  const parts = []
  if (c.dropped_by_vlm) parts.push(`大模型剔除误检 ${c.dropped_by_vlm} 个`)
  if (c.class_corrected) parts.push(`类别纠错 ${c.class_corrected} 个`)
  if (c.kept_by_vlm) parts.push(`疑难框经复核后保留 ${c.kept_by_vlm} 个`)
  return parts
})

function confClass(conf) {
  if (conf >= 0.6) return 'badge-ok'
  if (conf >= 0.4) return 'badge-warn'
  return 'badge-danger'
}
</script>

<template>
  <div>
    <!-- 离线模式郑重提示 -->
    <div v-if="!isRealVlm && vlm.calls > 0" class="alert alert-warn mb-2">
      <span class="ico">⚠️</span>
      <div>
        <b>当前为离线启发式复核（未配置 DASHSCOPE_API_KEY）</b><br />
        系统用经典图像统计量（局部对比度 / 清晰度）近似"这块区域像不像瑕疵"，
        <b>它不是多模态大模型</b>。本页结果仅用于演示系统流程与界面，
        <b>不代表论文结论</b>。配置 API Key 后（或把 <code>VLM_MODE</code> 设为
        <code>real</code>）即走真实 qwen3-vl-plus 复核。
      </div>
    </div>

    <!-- 关键指标 -->
    <div class="stat-grid mb-3">
      <div class="stat">
        <div class="stat-label">最终缺陷框</div>
        <div class="stat-value stat-accent">
          {{ boxes.length }}<span class="stat-unit">个</span>
        </div>
        <div class="stat-sub">
          <template v-if="Object.keys(classCounts).length">
            <span v-for="(v, k) in classCounts" :key="k" class="badge xsmall"
                  :class="'badge-' + k" style="margin-right:4px">{{ classLabel(k) }} {{ v }}</span>
          </template>
          <template v-else>未检出缺陷</template>
        </div>
      </div>

      <div class="stat">
        <div class="stat-label">候选框 → 复核</div>
        <div class="stat-value sm">
          {{ routing.total || 0 }}<span class="stat-unit">→</span>{{ routing.verified || 0 }}
        </div>
        <div class="stat-sub">
          高置信直接接受 {{ routing.accepted || 0 }} 个
        </div>
      </div>

      <div class="stat">
        <div class="stat-label">大模型调用</div>
        <div class="stat-value sm">
          {{ vlm.calls || 0 }}<span class="stat-unit">次</span>
        </div>
        <div class="stat-sub">
          平均 {{ fmtMs(vlm.avg_latency_ms) }} / 次
        </div>
      </div>

      <div class="stat">
        <div class="stat-label">总耗时</div>
        <div class="stat-value sm">{{ fmtMs(timing.total_ms) }}</div>
        <div class="stat-sub">
          YOLO {{ timing.yolo_ms || 0 }} ms + 复核 {{ timing.vlm_ms || 0 }} ms
        </div>
      </div>

      <div class="stat">
        <div class="stat-label">协同增益</div>
        <div class="stat-value sm">
          <span style="color:var(--ok)">−{{ fusion.dropped_by_vlm || 0 }}</span>
          <span class="dim" style="font-size:15px"> / </span>
          <span style="color:var(--primary)">{{ fusion.class_corrected || 0 }}</span>
        </div>
        <div class="stat-sub">剔误检 / 类别纠错</div>
      </div>
    </div>

    <!-- 对比：仅 YOLO vs 协同 -->
    <div v-if="showCompare" class="card card-pad mb-3 compare-card">
      <div class="row-between mb-2">
        <h3 style="font-size:14px">📈 仅 YOLO 与协同结果对比</h3>
        <span class="xsmall dim">同一张图、同一模型，唯一差别是"疑难样本是否交大模型复核"</span>
      </div>
      <div class="compare-grid">
        <div class="cmp-col">
          <div class="cmp-head">
            <span class="badge badge-gray">仅 YOLO 基线</span>
            <span class="xsmall dim">conf ≥ {{ r.comparison?.yolo_only_conf }}</span>
          </div>
          <div class="cmp-num">{{ r.comparison?.yolo_only_count ?? 0 }}</div>
          <div class="xsmall muted">个缺陷框 · 0 次大模型调用</div>
        </div>
        <div class="cmp-arrow">
          <div class="cmp-arrow-line"></div>
          <div class="cmp-arrow-label">置信度路由 + 多模态复核</div>
        </div>
        <div class="cmp-col cmp-col-ours">
          <div class="cmp-head">
            <span class="badge badge-blue">协同推理（本系统）</span>
            <span class="xsmall dim">阈值 T = {{ r.params?.threshold }}</span>
          </div>
          <div class="cmp-num">{{ r.comparison?.collab_count ?? 0 }}</div>
          <div class="xsmall muted">
            个缺陷框 · {{ r.comparison?.vlm_calls ?? 0 }} 次大模型调用
          </div>
        </div>
      </div>
      <div v-if="compare.length" class="compare-notes mt-2">
        <span v-for="(t, i) in compare" :key="i" class="badge badge-ok">✓ {{ t }}</span>
      </div>
      <div v-else class="xsmall dim mt-2">
        本次未发生剔除或类别纠错——说明 YOLO 在这张图上的判定与大模型复核结论一致。
      </div>
    </div>

    <!-- 逐框明细 -->
    <div class="card mb-3">
      <div class="card-head">
        <h3>逐框明细</h3>
        <span class="head-sub">每个框的类别、置信度、路由去向与复核结论</span>
        <div class="head-right">
          <span class="badge badge-gray xsmall">共 {{ boxRows.length }} 个</span>
        </div>
      </div>
      <div class="table-wrap">
        <table class="tbl">
          <thead>
            <tr>
              <th style="width:44px">#</th>
              <th>最终类别</th>
              <th class="num">置信度</th>
              <th>路由</th>
              <th>复核结论</th>
              <th>说明</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="b in boxRows" :key="b.i"
                @mouseenter="emit('hover-box', b.i)"
                @mouseleave="emit('hover-box', null)"
                @click="emit('active-box', b.i)"
                style="cursor:pointer">
              <td class="mono">{{ b.i + 1 }}</td>
              <td>
                <span class="cls-dot" :style="{ background: classColor(b.cls_name) }"></span>
                <b>{{ classLabel(b.cls_name) }}</b>
                <span v-if="b.class_corrected" class="badge badge-violet xsmall" style="margin-left:5px">
                  纠错 · 原判{{ classLabel(b.yolo_cls_name) }}
                </span>
              </td>
              <td class="num">
                <span class="badge xsmall" :class="confClass(b.conf)">{{ fmtNum(b.conf, 2) }}</span>
              </td>
              <td>
                <span v-if="b.routed === 'accept'" class="badge badge-ok xsmall">直接接受</span>
                <span v-else class="badge badge-warn xsmall">大模型复核</span>
              </td>
              <td class="small">
                <template v-if="b.vlm">
                  <span v-if="b.vlm.is_defect === true" style="color:var(--ok)">✓ 判定为瑕疵</span>
                  <span v-else-if="b.vlm.is_defect === false" style="color:var(--danger)">✗ 判为非瑕疵</span>
                  <span v-else class="dim">调用异常</span>
                </template>
                <span v-else class="dim">未复核（高置信）</span>
              </td>
              <td class="xsmall muted truncate" style="max-width:320px"
                  :title="b.vlm?.reason || b.routeReason">
                {{ b.vlm?.reason || b.routeReason }}
              </td>
            </tr>
            <tr v-if="!boxRows.length">
              <td colspan="6" class="center muted" style="padding:26px">
                本张图片未检出任何缺陷框
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 发给大模型的局部证据 -->
    <div v-if="(r.crops || []).length" class="card card-pad">
      <div class="row-between mb-2">
        <h3 style="font-size:14px">✂️ 送给多模态大模型的局部视觉证据</h3>
        <span class="xsmall dim">
          论文 H2 已验证：看局部裁剪图比看整图准确率更高（0.643 vs 0.598）
        </span>
      </div>
      <div class="crop-grid">
        <div v-for="c in r.crops" :key="c.index" class="crop-item">
          <img v-if="c.url" :src="c.url" :alt="'局部裁剪 ' + c.index" />
          <div v-else class="crop-noimg">未保存</div>
          <div class="crop-meta">
            <span class="badge xsmall" :class="'badge-' + c.yolo_cls">
              {{ classLabel(c.yolo_cls) }} {{ fmtNum(c.yolo_conf, 2) }}
            </span>
            <span class="badge xsmall"
                  :class="c.verdict?.is_defect ? 'badge-ok' : 'badge-danger'">
              {{ c.verdict?.is_defect ? '✓ 瑕疵' : '✗ 非瑕疵' }}
            </span>
          </div>
          <div class="xsmall dim crop-reason" :title="c.verdict?.reason">
            {{ c.verdict?.reason }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cls-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 6px; vertical-align: middle;
}
.compare-grid {
  display: grid; grid-template-columns: 1fr auto 1fr; gap: 14px; align-items: center;
}
.cmp-col {
  border: 1px solid var(--border); border-radius: 11px;
  padding: 13px 16px; background: #fcfdff;
}
.cmp-col-ours { border-color: #bfdbfe; background: #f5f9ff; }
.cmp-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.cmp-num {
  font-size: 28px; font-weight: 700; line-height: 1.15;
  font-variant-numeric: tabular-nums; color: var(--text);
}
.cmp-col-ours .cmp-num { color: var(--primary); }
.cmp-arrow { text-align: center; min-width: 150px; }
.cmp-arrow-line {
  height: 2px; background: linear-gradient(90deg, var(--border-strong), var(--primary));
  position: relative; border-radius: 2px;
}
.cmp-arrow-line::after {
  content: '▶'; position: absolute; right: -7px; top: -9px;
  color: var(--primary); font-size: 11px;
}
.cmp-arrow-label { font-size: 11px; color: var(--text-3); margin-top: 6px; font-weight: 600; }
.compare-notes { display: flex; flex-wrap: wrap; gap: 6px; }

.crop-grid {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
}
.crop-item {
  border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden; background: #fff;
}
.crop-item img {
  width: 100%; height: 116px; object-fit: contain;
  background: repeating-conic-gradient(#f8fafc 0% 25%, #eef2f7 0% 50%) 50% / 14px 14px;
  display: block;
}
.crop-noimg {
  height: 116px; display: grid; place-items: center;
  color: var(--text-4); font-size: 12px; background: var(--bg-soft);
}
.crop-meta { display: flex; gap: 5px; padding: 7px 8px 3px; flex-wrap: wrap; }
.crop-reason {
  padding: 0 8px 8px; overflow: hidden; display: -webkit-box;
  -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.45;
}
</style>
