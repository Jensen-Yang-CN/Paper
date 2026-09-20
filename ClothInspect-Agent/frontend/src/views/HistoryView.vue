<script setup>
import { ref, onMounted, computed } from 'vue'
import BoxOverlay from '../components/BoxOverlay.vue'
import ResultDetail from '../components/ResultDetail.vue'
import { api, classLabel, fmtMs } from '../api'

const list = ref({ items: [], total: 0, page: 1, size: 12 })
const stats = ref(null)
const detail = ref(null)
const loading = ref(false)
const keyword = ref('')
const error = ref('')

async function load(page = 1) {
  loading.value = true
  error.value = ''
  try {
    list.value = await api.history(page, 12, keyword.value)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function loadStats() {
  try { stats.value = await api.historyStats() } catch { /* ignore */ }
}

onMounted(async () => {
  await Promise.all([load(1), loadStats()])
})

async function openDetail(id) {
  try {
    detail.value = await api.historyDetail(id)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  } catch (e) {
    error.value = e.message
  }
}

function closeDetail() { detail.value = null }

async function removeRecord(id, e) {
  e.stopPropagation()
  if (!confirm('确定删除这条检测记录吗？')) return
  try {
    await api.deleteHistory(id)
    if (detail.value?.id === id) detail.value = null
    await Promise.all([load(list.value.page), loadStats()])
  } catch (err) {
    error.value = err.message
  }
}

async function clearAll() {
  if (!confirm('确定清空全部检测记录吗？此操作不可撤销。')) return
  try {
    await api.clearHistory()
    detail.value = null
    await Promise.all([load(1), loadStats()])
  } catch (err) {
    error.value = err.message
  }
}

const totalPages = computed(() => Math.max(1, Math.ceil((list.value.total || 0) / (list.value.size || 12))))

const detailResult = computed(() => detail.value?.result_json || null)

function confBadge(c) {
  if (c >= 0.6) return 'badge-ok'
  if (c >= 0.4) return 'badge-warn'
  return 'badge-danger'
}
</script>

<template>
  <div class="page">
    <!-- 总览 -->
    <div class="stat-grid mb-3">
      <div class="stat">
        <div class="stat-label">📋 检测记录总数</div>
        <div class="stat-value">{{ stats?.total_records ?? 0 }}</div>
        <div class="stat-sub">累计检出缺陷框 {{ stats?.total_boxes ?? 0 }} 个</div>
      </div>
      <div class="stat">
        <div class="stat-label">🤖 大模型累计调用</div>
        <div class="stat-value sm">{{ stats?.total_vlm_calls ?? 0 }}<span class="stat-unit">次</span></div>
        <div class="stat-sub">剔除误检 {{ stats?.total_dropped ?? 0 }} · 类别纠错 {{ stats?.total_corrected ?? 0 }}</div>
      </div>
      <div class="stat">
        <div class="stat-label">⏱️ 平均单图耗时</div>
        <div class="stat-value sm">{{ fmtMs(stats?.avg_elapsed_ms) }}</div>
        <div class="stat-sub">含 YOLO 推理与大模型复核</div>
      </div>
      <div class="stat">
        <div class="stat-label">🏷️ 累计类别分布</div>
        <div class="stat-value sm" style="font-size:14px;line-height:1.9;font-weight:600">
          <template v-if="stats && Object.keys(stats.class_counts || {}).length">
            <span v-for="(v, k) in stats.class_counts" :key="k"
                  class="badge xsmall" :class="'badge-' + k" style="margin-right:4px">
              {{ classLabel(k) }} {{ v }}
            </span>
          </template>
          <span v-else class="dim">暂无数据</span>
        </div>
      </div>
    </div>

    <!-- 详情 -->
    <div v-if="detail" class="card mb-3 detail-card">
      <div class="card-head">
        <h3>记录详情 #{{ detail.id }}</h3>
        <span class="head-sub">
          {{ detail.source_name }} · {{ detail.created_at }} · 阈值 T = {{ detail.threshold }}
        </span>
        <div class="head-right">
          <span class="badge" :class="detail.vlm_mode === 'real' ? 'badge-ok' : 'badge-warn'">
            {{ detail.vlm_mode === 'real' ? '真实 VLM 复核' : '离线启发式复核' }}
          </span>
          <button class="btn btn-sm" @click="closeDetail">关闭</button>
        </div>
      </div>
      <div class="card-pad detail-grid">
        <div class="detail-img">
          <BoxOverlay :image-url="detail.image_url"
                      :boxes="detailResult?.final_boxes || []" />
          <div class="xsmall dim mt-1">
            原图 {{ detail.image_width }} × {{ detail.image_height }} px ·
            最终 {{ detail.n_final }} 个缺陷框
          </div>
        </div>
        <div>
          <ResultDetail :result="detailResult" :show-compare="true" />
        </div>
      </div>
    </div>

    <!-- 列表 -->
    <div class="card">
      <div class="card-head">
        <h3>检测记录</h3>
        <div class="head-right">
          <input v-model="keyword" type="text" placeholder="按文件名搜索…"
                 style="width:200px" @keyup.enter="load(1)" />
          <button class="btn btn-sm" @click="load(1)">搜索</button>
          <button class="btn btn-sm btn-danger" :disabled="!list.total" @click="clearAll">清空</button>
        </div>
      </div>

      <div v-if="error" class="card-pad">
        <div class="alert alert-danger"><span class="ico">⚠️</span><div>{{ error }}</div></div>
      </div>

      <div v-if="loading" class="card-pad">
        <div class="skeleton" style="height:80px;margin-bottom:10px"></div>
        <div class="skeleton" style="height:80px"></div>
      </div>

      <div v-else-if="!list.items.length" class="empty">
        <div class="empty-ico">🗂️</div>
        <div class="empty-title">还没有检测记录</div>
        <div class="muted small mt-1">到「检测工作台」跑一次检测，记录会自动出现在这里。</div>
      </div>

      <div v-else class="card-pad">
        <div class="rec-grid">
          <div v-for="rec in list.items" :key="rec.id"
               class="rec-card" @click="openDetail(rec.id)">
            <div class="rec-thumb">
              <img :src="rec.image_url" :alt="rec.source_name" loading="lazy" />
              <span class="rec-badge" :class="rec.n_final ? 'badge-danger' : 'badge-ok'">
                {{ rec.n_final ? rec.n_final + ' 缺陷' : '无缺陷' }}
              </span>
            </div>
            <div class="rec-body">
              <div class="rec-name truncate" :title="rec.source_name">{{ rec.source_name }}</div>
              <div class="xsmall dim">{{ rec.created_at }}</div>
              <div class="rec-tags">
                <span v-for="(v, k) in rec.class_counts" :key="k"
                      class="badge xsmall" :class="'badge-' + k">{{ classLabel(k) }} {{ v }}</span>
                <span class="badge xsmall badge-gray">T={{ rec.threshold }}</span>
              </div>
              <div class="rec-foot">
                <span class="xsmall muted">
                  🤖 {{ rec.n_vlm_calls }} 次 · ⏱ {{ fmtMs(rec.elapsed_ms) }}
                </span>
                <button class="rec-del" title="删除" @click="removeRecord(rec.id, $event)">✕</button>
              </div>
            </div>
          </div>
        </div>

        <div v-if="totalPages > 1" class="pager">
          <button class="btn btn-sm" :disabled="list.page <= 1" @click="load(list.page - 1)">上一页</button>
          <span class="small muted">第 {{ list.page }} / {{ totalPages }} 页（共 {{ list.total }} 条）</span>
          <button class="btn btn-sm" :disabled="list.page >= totalPages" @click="load(list.page + 1)">下一页</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.detail-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.5fr); gap: 18px; }
.detail-img { position: sticky; top: 74px; align-self: start; }
.detail-card { border-color: #bfdbfe; box-shadow: 0 6px 24px rgba(37,99,235,.10); }

.empty { text-align: center; padding: 56px 20px; }
.empty-ico { font-size: 34px; opacity: .45; margin-bottom: 10px; }
.empty-title { font-size: 15px; font-weight: 700; }

.rec-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); }
.rec-card {
  border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden;
  background: #fff; cursor: pointer; transition: all .15s;
  display: flex; flex-direction: column;
}
.rec-card:hover { border-color: var(--primary); box-shadow: 0 4px 18px rgba(37,99,235,.15); transform: translateY(-1px); }
.rec-thumb { position: relative; }
.rec-thumb img { width: 100%; height: 118px; object-fit: cover; display: block; background: var(--bg-soft); }
.rec-badge {
  position: absolute; top: 7px; right: 7px;
  padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 700;
}
.rec-body { padding: 9px 11px 10px; display: flex; flex-direction: column; gap: 4px; flex: 1; }
.rec-name { font-size: 12.5px; font-weight: 600; }
.rec-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 2px; }
.rec-foot {
  display: flex; align-items: center; justify-content: space-between;
  margin-top: auto; padding-top: 6px; border-top: 1px dashed var(--border);
}
.rec-del {
  border: none; background: none; color: var(--text-4); cursor: pointer;
  font-size: 13px; padding: 0 3px; border-radius: 4px; font-family: inherit;
}
.rec-del:hover { color: var(--danger); background: var(--danger-soft); }

.pager { display: flex; align-items: center; justify-content: center; gap: 14px; margin-top: 18px; }

@media (max-width: 1150px) { .detail-grid { grid-template-columns: 1fr; } .detail-img { position: static; } }
</style>
