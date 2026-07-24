const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ΨΩΦΔΛΞΣΘабвгдежзиклмнопрстуфхцчшщэюя';

function initMatrixRain() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const canvas = document.createElement('canvas');
  canvas.id = 'matrix-rain';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.prepend(canvas);
  const context = canvas.getContext('2d');
  const fontSize = 14;
  let width = 0;
  let height = 0;
  let columns = [];
  let frame = 0;

  const resize = () => {
    const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const count = Math.floor(width / fontSize);
    while (columns.length < count) columns.push(Math.random() * height / fontSize | 0);
    columns.length = count;
  };

  const draw = () => {
    if (!document.body.classList.contains('motion-paused')) {
      context.fillStyle = 'rgba(13, 11, 15, 0.075)';
      context.fillRect(0, 0, width, height);
      context.fillStyle = '#ee3344';
      context.font = `${fontSize}px ui-monospace, monospace`;
      for (let index = 0; index < columns.length; index += 1) {
        context.fillText(chars[Math.random() * chars.length | 0], index * fontSize, columns[index] * fontSize);
        if (columns[index] * fontSize > height && Math.random() > 0.975) columns[index] = 0;
        columns[index] += 1;
      }
    }
    frame = window.setTimeout(draw, 66);
  };

  resize();
  window.addEventListener('resize', resize, { passive: true });
  draw();
  window.addEventListener('pagehide', () => window.clearTimeout(frame), { once: true });
}

function initMotionToggle() {
  const button = document.querySelector('.motion-toggle');
  if (!button) return;
  button.addEventListener('click', () => {
    const paused = document.body.classList.toggle('motion-paused');
    button.setAttribute('aria-pressed', String(paused));
    button.setAttribute('aria-label', paused ? 'Resume motion' : 'Pause motion');
    button.textContent = paused ? '▶' : 'Ⅱ';
  });
}

initMatrixRain();
initMotionToggle();
