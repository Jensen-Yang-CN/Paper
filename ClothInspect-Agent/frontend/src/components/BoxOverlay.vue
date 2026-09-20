<script setup>
/**
 * 检测框可视化覆盖层
 * 用绝对定位的 div 按百分比绘制（而不是 SVG），
 * 好处：文字不会被非等比缩放拉伸，中文标签始终清晰。
 */
import { computed } from 'vue'
import { classColor, classLabel } from '../api'

const props = defineProps({
  imageUrl: { type: String, default: '' },
  boxes: { type: Array, default: () => [] },       // [{box_norm, cls_name, conf, ...}]
  hoverIndex: { type: [Number, null], default: null },
  activeIndex: { type: [Number, null], default: null },
  dimOthers: { type: Boolean, default: false },
  showLabels: { type: Boolean, default: true }
})

const emit = defineEmits(['hover', 'select'])

const painted = computed(() =>
  props.boxes.map((b, i) => {
    const n = b.box_norm || [0, 0, 0, 0]
    const [x1, y1, x2, y2] = n
    const color = classColor(b.cls_name)
    const emphasized = props.hoverIndex === i || props.activeIndex === i
    const dimmed = props.dimOthers && props.activeIndex !== null && props.activeIndex !== i
    return {
      i, b, color,
      style: {
        left: `${x1 * 100}%`,
        top: `${y1 * 100}%`,
        width: `${Math.max(0.4, (x2 - x1) * 100)}%`,
        height: `${Math.max(0.4, (y2 - y1) * 100)}%`,
        borderColor: color,
        boxShadow: emphasized ? `0 0 0 2px #fff, 0 0 0 4px ${color}` : 'none',
        opacity: dimmed ? 0.25 : 1,
        zIndex: emphasized ? 30 : 10
      },
      tag: `${classLabel(b.cls_name)} ${Number(b.conf).toFixed(2)}`,
      corrected: !!b.class_corrected,
      correctedFrom: b.yolo_cls_name
    }
  })
)
</script>

<template>
  <div class="overlay-root">
    <img v-if="imageUrl" :src="imageUrl" alt="待检测图片" class="overlay-img" />
    <div v-else class="overlay-empty">暂无图片</div>

    <div
      v-for="p in painted" :key="p.i"
      class="bbox"
      :style="p.style"
      @mouseenter="emit('hover', p.i)"
      @mouseleave="emit('hover', null)"
      @click.stop="emit('select', p.i)"
    >
      <span v-if="showLabels" class="bbox-tag" :style="{ background: p.color }">
        {{ p.tag }}
        <em v-if="p.corrected" class="bbox-fix">纠错</em>
      </span>
    </div>
  </div>
</template>

<style scoped>
.overlay-root {
  position: relative;
  width: 100%;
  line-height: 0;
  background: repeating-conic-gradient(#f8fafc 0% 25%, #eef2f7 0% 50%) 50% / 18px 18px;
  border-radius: 10px;
  overflow: hidden;
}
.overlay-img { width: 100%; display: block; border-radius: 10px; }
.overlay-empty {
  padding: 60px 0; text-align: center; color: var(--text-4);
  font-size: 13px; line-height: 1.6;
}

.bbox {
  position: absolute;
  border: 2px solid #3b82f6;
  border-radius: 3px;
  box-sizing: border-box;
  cursor: pointer;
  transition: opacity .15s, box-shadow .15s;
}
.bbox-tag {
  position: absolute;
  left: -2px;
  top: -2px;
  transform: translateY(-100%);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  line-height: 1.55;
  padding: 1px 6px;
  border-radius: 4px 4px 0 0;
  white-space: nowrap;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: var(--mono);
}
.bbox-fix {
  font-style: normal;
  font-size: 10px;
  background: rgba(255, 255, 255, .3);
  padding: 0 3px;
  border-radius: 3px;
  font-family: inherit;
}
</style>
