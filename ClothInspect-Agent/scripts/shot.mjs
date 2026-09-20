/**
 * 开发用截图工具（不属于交付功能，仅用于自检界面）
 * 用法： node scripts/shot.mjs
 * 依赖： 本机 Chrome + Node 18+（使用内置 WebSocket / fetch）
 */
import { spawn } from 'node:child_process'
import { writeFileSync, mkdirSync, existsSync } from 'node:fs'
import { setTimeout as sleep } from 'node:timers/promises'

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
const BASE = 'http://127.0.0.1:8000'
const PORT = 9333
const OUT = new URL('../docs/', import.meta.url).pathname.replace(/^\//, '')
const PROFILE = new URL('../.chrome-profile-cdp/', import.meta.url).pathname.replace(/^\//, '')

if (!existsSync(OUT)) mkdirSync(OUT, { recursive: true })

const chrome = spawn(CHROME, [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1680,1000', 'about:blank'
], { stdio: 'ignore' })

let ws, msgId = 0
const pending = new Map()

function send(method, params = {}) {
  const id = ++msgId
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    ws.send(JSON.stringify({ id, method, params }))
  })
}

async function connect() {
  for (let i = 0; i < 40; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
      const page = list.find(t => t.type === 'page')
      if (page) {
        ws = new WebSocket(page.webSocketDebuggerUrl)
        await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej })
        ws.onmessage = (e) => {
          const m = JSON.parse(e.data)
          if (m.id && pending.has(m.id)) {
            const { resolve, reject } = pending.get(m.id)
            pending.delete(m.id)
            m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result)
          }
        }
        return
      }
    } catch { /* retry */ }
    await sleep(250)
  }
  throw new Error('无法连接 Chrome 调试端口')
}

async function evaluate(expression) {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
  return r.result?.value
}

async function goto(url, waitMs = 2200) {
  await send('Page.navigate', { url })
  await sleep(waitMs)
}

async function shot(name, { full = true } = {}) {
  const r = await send('Page.captureScreenshot', {
    format: 'png', captureBeyondViewport: full
  })
  const f = `${OUT}_shot_${name}.png`
  writeFileSync(f, Buffer.from(r.data, 'base64'))
  console.log(`OK   ${name} -> ${f}`)
}

/** 等待某个条件为真 */
async function waitFor(expr, timeoutMs = 30000) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeoutMs) {
    if (await evaluate(expr)) return true
    await sleep(200)
  }
  return false
}

const main = async () => {
  await connect()
  await send('Page.enable')
  await send('Runtime.enable')

  // ---------------- 首页 ----------------
  await goto(`${BASE}/`)
  await shot('home')

  // ---------------- 检测工作台（空态） ----------------
  await goto(`${BASE}/detect`)
  await shot('detect_empty')

  // ---------------- 检测工作台：选样例 + 真实检测 ----------------
  await evaluate(`(() => {
    const cards = document.querySelectorAll('.sample-card');
    if (cards.length) cards[7].click();
    return cards.length;
  })()`)
  await sleep(600)
  await shot('detect_selected', { full: false })

  await evaluate(`(() => {
    const btns = [...document.querySelectorAll('button')];
    const b = btns.find(x => x.textContent.includes('开始检测'));
    if (b) b.click();
    return !!b;
  })()`)
  const ok = await waitFor(`!!document.querySelector('.timeline')`, 60000)
  await sleep(1500)
  console.log('检测完成:', ok)
  await shot('detect_single')

  // ---------------- 载入最丰富的一条记录（多框 + 多次复核 + 类别纠错）----------------
  const richId = await evaluate(`(async () => {
    const r = await (await fetch('/api/history?page=1&size=50')).json();
    const items = r.items || [];
    const best = items.slice().sort((a,b) =>
      ((b.n_dropped + b.n_corrected) * 10 + b.n_vlm_calls) -
      ((a.n_dropped + a.n_corrected) * 10 + a.n_vlm_calls))[0];
    return best ? best.id : null;
  })()`)
  if (richId) {
    await goto(`${BASE}/detect?record=${richId}`, 2600)
    await sleep(1500)
    await shot('detect_result')
    console.log('载入最丰富记录 id =', richId)
  } else {
    await shot('detect_result')
  }

  // ---------------- 切到"仅 YOLO 基线" ----------------
  await evaluate(`(() => {
    const b = [...document.querySelectorAll('.seg button')].find(x => x.textContent.includes('仅 YOLO'));
    if (b) b.click(); return !!b;
  })()`)
  await sleep(900)
  await shot('detect_yolo_mode', { full: false })

  // ---------------- 检查记录 ----------------
  await goto(`${BASE}/history`)
  await sleep(1200)
  await evaluate(`(() => {
    const c = document.querySelector('.rec-card');
    if (c) c.click(); return !!c;
  })()`)
  await sleep(2000)
  await shot('history')

  // ---------------- 实验数据 ----------------
  for (const [tab, name] of [['核心对比', 'experiments_compare'],
                             ['阈值与成本', 'experiments_pareto'],
                             ['消融实验', 'experiments_ablation'],
                             ['难样本与假设', 'experiments_hard']]) {
    if (name === 'experiments_compare') await goto(`${BASE}/experiments`, 2600)
    await evaluate(`(() => {
      const b = [...document.querySelectorAll('.tabs button')].find(x => x.textContent.includes('${tab}'));
      if (b) b.click(); return !!b;
    })()`)
    await sleep(1000)
    await shot(name)
  }

  // ---------------- 关于 ----------------
  await goto(`${BASE}/about`, 2200)
  await shot('about')

  // ---------------- 控制台错误检查 ----------------
  const errs = await evaluate(`JSON.stringify(window.__errs || [])`)
  console.log('页面运行时错误:', errs)

  await send('Browser.close').catch(() => {})
  chrome.kill()
  process.exit(0)
}

main().catch(e => { console.error('FAIL', e); chrome.kill(); process.exit(1) })
