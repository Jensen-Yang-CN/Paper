<script setup>
/**
 * Agent 推理轨迹时间线
 * 展示流水线每一步的工具调用：步骤 / 工具 / 结论 / 耗时 / 展开细节。
 */
import { ref, computed } from 'vue'
import { fmtMs } from '../api'

const props = defineProps({
  trace: { type: Array, default: () => [] },
  running: { type: Boolean, default: false }
})

const expanded = ref(new Set())
function toggle(step) {
  const s = new Set(expanded.value)
  s.has(step) ? s.delete(step) : s.add(step)
  expanded.value = s
}

const TOOL_ICON = {
  detect_defect: '🎯',
  route_decision: '🔀',
  crop_region: '✂️',
  verify_with_vlm: '🤖',
  enhance_region: '✨',
  fuse_results: '🧩'
}

const TOOL_DESC = {
  detect_defect: 'YOLOv11 视觉检测',
  route_decision: '置信度驱动的动态路由',
  crop_region: '局部区域裁剪 + 尺度提示',
  verify_with_vlm: '多模态大模型语义复核',
  enhance_region: '局部区域对比度增强',
  fuse_results: '规则融合与最终判定'
}

// 同一批复核调用折叠展示，避免时间线过长
const displayTrace = computed(() => {
  const out = []
  let bucket = null
  for (const t of props.trace) {
    if (t.tool === 'crop_region' || t.tool === 'verify_with_vlm') {
      if (!bucket) {
        bucket = { group: true, tool: 'verify_group', label: '疑难样本复核',
                   steps: [], latency_ms: 0, status: 'ok' }
        out.push(bucket)
      }
      bucket.steps.push(t)
      bucket.latency_ms += t.latency_ms || 0
      if (t.status === 'error') bucket.status = 'error'
      continue
    }
    bucket = null
    out.push({ group: false, ...t })
  }
  return out
})

function groupSummary(g) {
  const n = g.steps.filter(s => s.tool === 'verify_with_vlm').length
  const err = g.steps.filter(s => s.status === 'error').length
  const kept = g.steps.filter(s => s.tool === 'verify_with_vlm' && s.detail?.verdict?.is_defect).length
  let txt = `对 ${n} 个疑难样本逐框裁剪并交由多模态大模型复核`
  if (n) txt += `，其中 ${kept} 个被判为真实瑕疵而保留`
  if (err) txt += `，${err} 次调用异常`
  return txt
}

function detailText(item) {
  if (item.group) {
    return item.steps.map(s =>
      `#${s.step} [${s.label}] ${s.summary}`
    ).join('\n')
  }
  const d = item.detail || {}
  const parts = []
  if (d.detections) {
    parts.push(`候选框 ${d.detections.length} 个`)
  }
  if (d.threshold !== undefined) {
    parts.push(`阈值 T=${d.threshold}，候选 ${d.total}，直接接受 ${d.accepted}，转复核 ${d.verified}`)
  }
  if (d.crop_size) {
    parts.push(`裁剪尺寸 ${d.crop_size[0]}x${d.crop_size[1]}${d.upscaled ? '（已放大）' : ''}`)
  }
  if (d.scale_prompt) parts.push(d.scale_prompt)
  if (d.verdict) {
    const v = d.verdict
    parts.push(`is_defect=${v.is_defect}  category=${v.category}  confidence=${v.confidence}`)
    if (v.reason) parts.push(`理由：${v.reason}`)
    if (v.mode) parts.push(`复核后端：${v.mode === 'real' ? '真实 VLM' : '离线启发式'}`)
  }
  if (d.rule) parts.push(`融合规则：${d.rule}`)
  if (item.args && Object.keys(item.args).length) {
    parts.push(`参数：${JSON.stringify(item.args, null, 1)}`)
  }
  return parts.join('\n')
}
</script>

<template>
  <div class="trace-wrap">
    <div v-if="!trace.length && !running" class="trace-empty">
      <div class="trace-empty-ico">🕐</div>
      <div>还没有检测记录</div>
      <div class="xsmall dim mt-1">上传或选择一张图片后，这里会实时显示协同推理的每一步</div>
    </div>

    <div v-else class="timeline">
      <div v-for="(item, idx) in displayTrace" :key="idx" class="tl-item">
        <div class="tl-dot" :class="{ err: item.status === 'error', skip: item.status === 'skipped' }">
          {{ item.group ? '🤖' : (TOOL_ICON[item.tool] || '•') }}
        </div>

        <div class="tl-head">
          <span class="tl-title">
            {{ item.group ? '多模态复核' : (item.label || item.tool) }}
          </span>
          <code class="tl-tool" :title="item.group ? TOOL_DESC.verify_with_vlm
                            : (TOOL_DESC[item.tool] || item.tool)">
            {{ item.group ? 'verify_with_vlm ×' + item.steps.filter(s => s.tool === 'verify_with_vlm').length
                          : item.tool }}
          </code>
          <span v-if="item.status === 'error'" class="badge badge-danger xsmall">异常</span>
          <span class="tl-time">{{ fmtMs(item.latency_ms) }}</span>
        </div>

        <div class="tl-body">
          {{ item.group ? groupSummary(item) : item.summary }}
        </div>

        <button class="tl-toggle" @click="toggle(item.group ? 'g' + idx : item.step)">
          {{ expanded.has(item.group ? 'g' + idx : item.step) ? '收起细节 ▲' : '查看细节 ▼' }}
        </button>

        <pre v-if="expanded.has(item.group ? 'g' + idx : item.step)" class="tl-detail">{{ detailText(item) }}</pre>
      </div>

      <div v-if="running" class="tl-item">
        <div class="tl-dot"><span class="spinner dark"></span></div>
        <div class="tl-head"><span class="tl-title">正在推理…</span></div>
        <div class="tl-body">YOLOv11 检测与多模态复核执行中</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.trace-wrap { max-height: 560px; overflow-y: auto; padding-right: 4px; }
.trace-empty { text-align: center; padding: 34px 10px; color: var(--text-3); font-size: 13px; }
.trace-empty-ico { font-size: 26px; margin-bottom: 8px; opacity: .5; }

.tl-toggle {
  margin-top: 4px; background: none; border: none; padding: 0;
  color: var(--primary); font-size: 11.5px; cursor: pointer; font-family: inherit;
}
.tl-toggle:hover { text-decoration: underline; }

.tl-tool {
  font-family: var(--mono); font-size: 11px; color: var(--primary-dark);
  background: var(--primary-soft); padding: 1px 7px; border-radius: 5px;
  white-space: nowrap; cursor: help;
}
</style>
