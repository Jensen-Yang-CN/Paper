<script setup>
import { ref, onMounted, computed } from 'vue'
import { RouterView, useRoute } from 'vue-router'
import { api } from './api'

const route = useRoute()
const health = ref(null)
const healthError = ref('')

const navGroups = [
  {
    title: '质检作业',
    items: [
      { to: '/', icon: '🏠', label: '首页引导' },
      { to: '/detect', icon: '🔍', label: '检测工作台' },
      { to: '/history', icon: '🗂️', label: '检测记录' }
    ]
  },
  {
    title: '研究与说明',
    items: [
      { to: '/experiments', icon: '📊', label: '实验结果' },
      { to: '/about', icon: 'ℹ️', label: '关于系统' }
    ]
  }
]

const detectorOk = computed(() => !!health.value?.detector?.loaded || !!health.value?.detector?.weights_found)
const vlmReal = computed(() => !!health.value?.vlm?.is_real)

const detectorText = computed(() => {
  if (healthError.value) return '无法连接后端'
  if (!health.value) return '检查中…'
  return health.value.detector?.weights_found ? 'YOLOv11 已就绪' : '权重未找到'
})

const vlmText = computed(() => {
  if (!health.value) return '检查中…'
  const v = health.value.vlm
  return v.is_real ? `${v.model} 真实复核` : '离线启发式复核'
})

onMounted(async () => {
  try {
    health.value = await api.health()
  } catch (e) {
    healthError.value = e.message || '后端未启动'
  }
})
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-title">
          <span class="brand-logo">🔍</span>
          <span>ClothInspect-Agent</span>
        </div>
        <div class="brand-sub">服装智能质检系统 V1.0</div>
      </div>

      <nav class="nav">
        <template v-for="g in navGroups" :key="g.title">
          <div class="nav-group-title">{{ g.title }}</div>
          <RouterLink
            v-for="it in g.items" :key="it.to" :to="it.to"
            class="nav-item" :class="{ active: route.path === it.to }">
            <span class="ico">{{ it.icon }}</span><span>{{ it.label }}</span>
          </RouterLink>
        </template>
      </nav>

      <div class="sidebar-foot">
        <div>
          <span class="status-dot" :class="detectorOk ? 'dot-ok' : (healthError ? 'dot-bad' : 'dot-warn')"></span>
          {{ detectorText }}
        </div>
        <div class="mt-1">
          <span class="status-dot" :class="vlmReal ? 'dot-ok' : 'dot-warn'"></span>
          {{ vlmText }}
        </div>
      </div>
    </aside>

    <div class="main">
      <header class="topbar">
        <div>
          <h2>{{ route.meta.title || 'ClothInspect-Agent' }}</h2>
          <div class="topbar-sub">
            视觉模型与多模态大模型协同推理 · 推理时协同（模型权重冻结）
          </div>
        </div>
        <div class="topbar-right">
          <span v-if="health && !vlmReal" class="badge badge-warn" title="未配置 DASHSCOPE_API_KEY，当前使用离线启发式复核，仅演示系统流程">
            ⚠ 离线启发式复核
          </span>
          <span v-else-if="vlmReal" class="badge badge-ok">✓ 真实 VLM 复核</span>
          <span v-if="healthError" class="badge badge-danger">后端未连接</span>
        </div>
      </header>

      <RouterView v-slot="{ Component }">
        <Transition name="fade" mode="out-in">
          <component :is="Component" :health="health" />
        </Transition>
      </RouterView>
    </div>
  </div>
</template>
