import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'home', component: () => import('./views/HomeView.vue'), meta: { title: '首页' } },
  { path: '/detect', name: 'detect', component: () => import('./views/DetectView.vue'), meta: { title: '检测工作台' } },
  { path: '/history', name: 'history', component: () => import('./views/HistoryView.vue'), meta: { title: '检测记录' } },
  { path: '/experiments', name: 'experiments', component: () => import('./views/ExperimentView.vue'), meta: { title: '实验结果' } },
  { path: '/about', name: 'about', component: () => import('./views/AboutView.vue'), meta: { title: '关于系统' } },
  { path: '/:pathMatch(.*)*', redirect: '/' }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 })
})

router.afterEach((to) => {
  document.title = `${to.meta.title || ''} · ClothInspect-Agent 服装智能质检系统`
})

export default router
