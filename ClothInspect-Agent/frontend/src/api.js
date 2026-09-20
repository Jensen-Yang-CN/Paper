/**
 * 后端接口封装
 * 开发模式下由 vite 代理到 http://127.0.0.1:8000，
 * 生产模式由 FastAPI 同源托管，因此统一使用相对路径。
 */
const BASE = ''

async function request(path, options = {}) {
  const res = await fetch(BASE + path, options)
  const text = await res.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = { detail: text }
  }
  if (!res.ok) {
    const err = new Error((data && (data.detail || data.message)) || `请求失败 (${res.status})`)
    err.status = res.status
    err.payload = data
    throw err
  }
  return data
}

function form(obj) {
  const fd = new FormData()
  Object.entries(obj).forEach(([k, v]) => {
    if (v !== undefined && v !== null) fd.append(k, v)
  })
  return fd
}

export const api = {
  // ---------- 系统 ----------
  health: () => request('/api/health'),
  config: () => request('/api/config'),
  samples: () => request('/api/samples'),
  tools: () => request('/api/tools'),
  modelInfo: () => request('/api/model/info'),

  // ---------- 检测 ----------
  detectUpload: (file, params) => {
    const fd = form({ file, ...params })
    return request('/api/detect/upload', { method: 'POST', body: fd })
  },
  detectSample: (sample, params) => {
    const fd = form({ sample, ...params })
    return request('/api/detect/sample', { method: 'POST', body: fd })
  },
  exportReport: async (payload) => {
    const res = await fetch('/api/detect/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    if (!res.ok) {
      let msg = '导出失败'
      try { msg = (await res.json()).detail || msg } catch { /* ignore */ }
      throw new Error(msg)
    }
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ClothInspect检测报告_${Date.now()}.png`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  },

  // ---------- 记录 ----------
  history: (page = 1, size = 20, keyword = '') =>
    request(`/api/history?page=${page}&size=${size}&keyword=${encodeURIComponent(keyword)}`),
  historyStats: () => request('/api/history/statistics'),
  historyDetail: (id) => request(`/api/history/${id}`),
  deleteHistory: (id) => request(`/api/history/${id}`, { method: 'DELETE' }),
  clearHistory: () => request('/api/history', { method: 'DELETE' }),

  // ---------- 实验 ----------
  experiments: () => request('/api/experiments/summary'),
  mvpReport: () => request('/api/experiments/report')
}

/** 类别配色（与后端 report.py 保持一致） */
export const CLASS_COLORS = {
  hole: '#ef4444',
  stain: '#f59e0b',
  wrinkle: '#6366f1'
}

export const CLASS_LABELS = {
  hole: '破洞',
  stain: '污渍',
  wrinkle: '褶皱'
}

export function classColor(name) {
  return CLASS_COLORS[name] || '#3b82f6'
}

export function classLabel(name) {
  return CLASS_LABELS[name] || name || '未知'
}

export function fmtMs(ms) {
  if (ms === null || ms === undefined) return '-'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

export function fmtPct(v, digits = 1) {
  if (v === null || v === undefined) return '-'
  return `${(v * 100).toFixed(digits)}%`
}

export function fmtNum(v, digits = 3) {
  if (v === null || v === undefined) return '-'
  return Number(v).toFixed(digits)
}
