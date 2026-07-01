import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const threeSrc = fs.readFileSync(path.join(__dirname, 'vendor/three.module.js'), 'utf8');

const html = `<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0">
<canvas id="c" width="256" height="256"></canvas>
<script type="module">
import * as THREE from '/three.module.js';
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:false, preserveDrawingBuffer:true});
renderer.setClearColor(0x202030);
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 100); cam.position.z = 3;
const cube = new THREE.Mesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshNormalMaterial());
scene.add(cube);
let frame = 0;
window.__step = () => { cube.rotation.x += 0.2; cube.rotation.y += 0.3; renderer.render(scene, cam); frame++; return frame; };
window.__grab = () => canvas.toDataURL('image/png');
window.__ready = true;
</script>
</body></html>`;

const outDir = path.join(__dirname, 'smoke_out');
fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  executablePath: process.env.PW_CHROME || undefined,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--no-sandbox', '--disable-dev-shm-usage'],
});
const page = await browser.newPage({ viewport: { width: 256, height: 256 } });
page.on('console', m => console.log('  [page]', m.text()));
page.on('pageerror', e => console.log('  [pageerror]', e.message));
await page.route('**/three.module.js', r => r.fulfill({ contentType: 'application/javascript', body: threeSrc }));
await page.route('**/index.html', r => r.fulfill({ contentType: 'text/html', body: html }));
await page.goto('http://localhost/index.html', { waitUntil: 'load' });
await page.waitForFunction('window.__ready === true', null, { timeout: 15000 });

for (let i = 0; i < 5; i++) {
  const f = await page.evaluate('window.__step()');
  const dataUrl = await page.evaluate('window.__grab()');
  const b64 = dataUrl.split(',')[1];
  const buf = Buffer.from(b64, 'base64');
  fs.writeFileSync(path.join(outDir, `frame_${i}.png`), buf);
  console.log(`frame ${i}: step=${f}, bytes=${buf.length}`);
}
await browser.close();
console.log('DONE');
