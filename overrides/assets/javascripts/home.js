/* Homepage initialisation — re-runs on every instant navigation via document$ */
var __homeCleanup = [];

function __homeInit() {
  /* tear down previous instance (if navigating back) */
  __homeCleanup.forEach(function (fn) { fn(); });
  __homeCleanup = [];

  /* ── Scroll-reveal observer ── */
  var els = document.querySelectorAll('.reveal, .reveal-left, .reveal-right');
  if (els.length) {
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) e.target.classList.add('visible');
      });
    }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

    els.forEach(function (el) { obs.observe(el); });
    __homeCleanup.push(function () { obs.disconnect(); });
  }

  /* ── Code tabs ── */
  document.querySelectorAll('.mdx-code-tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      var block = tab.closest('.mdx-code-block');
      block.querySelectorAll('.mdx-code-tab').forEach(function (t) { t.classList.remove('active'); });
      tab.classList.add('active');
      var target = tab.dataset.tab;
      block.querySelectorAll('.mdx-code-pre').forEach(function (pre) {
        pre.style.display = pre.dataset.tab === target ? '' : 'none';
      });
    });
  });

  /* ── Animated waveform canvas ── */
  var canvas = document.getElementById('waveform');
  if (!canvas) return;

  var ctx = canvas.getContext('2d');
  var animId;

  function resize() {
    canvas.width  = canvas.offsetWidth  * devicePixelRatio;
    canvas.height = canvas.offsetHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
  }
  resize();
  window.addEventListener('resize', resize);
  __homeCleanup.push(function () { window.removeEventListener('resize', resize); });

  var layers = [
    { color: 'rgba(159,168,218,0.35)', amp: 35, freq: 0.007, speed: 0.0006, phase: 0 },
    { color: 'rgba(121,134,203,0.3)',  amp: 25, freq: 0.011, speed: 0.001,  phase: 2 },
    { color: 'rgba(197,202,233,0.15)', amp: 18, freq: 0.016, speed: 0.0005, phase: 4 },
  ];

  function drawWave(t) {
    var w = canvas.offsetWidth;
    var h = canvas.offsetHeight;
    ctx.clearRect(0, 0, w, h);

    layers.forEach(function (layer) {
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      for (var x = 0; x <= w; x += 2) {
        var y = h / 2
          + Math.sin(x * layer.freq + t * layer.speed + layer.phase) * layer.amp
          + Math.sin(x * layer.freq * 2.3 + t * layer.speed * 1.7) * (layer.amp * 0.4);
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = layer.color;
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    animId = requestAnimationFrame(function () { drawWave(t + 16); });
  }
  drawWave(0);
  __homeCleanup.push(function () { cancelAnimationFrame(animId); });
}

/* MkDocs Material exposes document$ (RxJS observable) that fires on every
   page load, including instant (XHR) navigations. */
if (typeof document$ !== 'undefined') {
  document$.subscribe(function () { __homeInit(); });
} else {
  /* fallback: no instant navigation */
  document.addEventListener('DOMContentLoaded', __homeInit);
}
