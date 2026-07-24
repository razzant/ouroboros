document.documentElement.classList.add("has-js");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const canvas = document.querySelector("#matrix-rain");
let matrixTimer = 0;

function initMatrixField() {
  if (!canvas || reducedMotion.matches) return;
  const context = canvas.getContext("2d");
  if (!context) return;

  const chars = "01アイウロボロスABCDEFGHIJKLMNOPQRSTUVWXYZΨΩΦΔΛΞΣΘ";
  const fontSize = 14;
  let width = 0;
  let height = 0;
  let columns = [];

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
    while (columns.length < count) columns.push(Math.floor(Math.random() * height / fontSize));
    columns.length = count;
  };

  const draw = () => {
    if (!document.body.classList.contains("motion-paused")) {
      context.fillStyle = "rgba(13, 11, 15, 0.08)";
      context.fillRect(0, 0, width, height);
      context.fillStyle = "#c93545";
      context.font = `${fontSize}px ui-monospace, monospace`;
      columns.forEach((drop, index) => {
        context.fillText(chars[Math.floor(Math.random() * chars.length)], index * fontSize, drop * fontSize);
        columns[index] = drop * fontSize > height && Math.random() > 0.978 ? 0 : drop + 1;
      });
    }
    matrixTimer = window.setTimeout(draw, 72);
  };

  resize();
  window.addEventListener("resize", resize, { passive: true });
  draw();
}

function initMotionToggle() {
  const button = document.querySelector(".motion-toggle");
  if (!button) return;
  button.addEventListener("click", () => {
    const paused = document.body.classList.toggle("motion-paused");
    button.setAttribute("aria-pressed", String(paused));
    button.setAttribute("aria-label", paused ? "Resume motion" : "Pause motion");
    const icon = button.querySelector("span[aria-hidden]");
    const label = button.querySelector(".motion-label");
    if (icon) icon.textContent = paused ? "▶" : "Ⅱ";
    if (label) label.textContent = paused ? "Resume motion" : "Pause motion";
  });
}

function initReveal() {
  const items = [...document.querySelectorAll(".reveal")];
  if (reducedMotion.matches || !("IntersectionObserver" in window)) {
    items.forEach((item) => item.classList.add("is-visible"));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    });
  }, { rootMargin: "0px 0px -8%", threshold: 0.08 });
  items.forEach((item) => observer.observe(item));
}

initMatrixField();
initMotionToggle();
initReveal();
window.addEventListener("pagehide", () => window.clearTimeout(matrixTimer), { once: true });
