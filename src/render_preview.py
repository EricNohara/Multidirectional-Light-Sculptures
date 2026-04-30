import trimesh
import base64
from pathlib import Path


def render_shadow_preview_threejs(stl_path, output_path):
    mesh = trimesh.load(stl_path)

    # ── Normalize mesh to fit in a unit cube centered at origin ──────────────
    # Scale so the longest axis = 1.0. The box interior will also be 1.0 wide,
    # so the sculpture fills it nicely with a small margin.
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / max(mesh.extents)   # fits in [-0.5, 0.5]^3
    mesh.apply_scale(scale)

    # Export to GLB (Three.js-friendly binary glTF)
    glb_path = str(Path(output_path).with_suffix(".glb"))
    mesh.export(glb_path)
    with open(glb_path, "rb") as f:
        glb_base64 = base64.b64encode(f.read()).decode()

    # ── All JS braces are doubled so Python f-string leaves them intact ───────
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Shadow Box Preview</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      background: #0d0b10;
      display: flex; align-items: center; justify-content: center;
      height: 100vh; overflow: hidden;
      font-family: 'Courier New', monospace;
    }}
    #info {{
      position: absolute; top: 18px; left: 50%; transform: translateX(-50%);
      color: #6677aa; font-size: 10px; letter-spacing: 0.18em;
      text-transform: uppercase; pointer-events: none;
    }}
    #status {{
      position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%);
      color: #ff6655; font-size: 10px; letter-spacing: 0.1em;
      pointer-events: none; opacity: 0; transition: opacity 0.3s;
    }}
  </style>
</head>
<body>
  <div id="info">drag to rotate &nbsp;·&nbsp; scroll to zoom</div>
  <div id="status" id="status"></div>

<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js';
import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/loaders/GLTFLoader.js';

// ─── Scene ────────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0b10);
// NO fog — fog was eating the geometry at short range

// ─── Camera ───────────────────────────────────────────────────────────────────
// Elevated 3/4 corner view matching the photo: front-right corner visible,
// looking into the box so back-wall and left-wall shadows are both visible.
const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.01, 50);
camera.position.set(1.6, 1.4, 1.8);   // units match the ~1.0 world scale
camera.lookAt(0, 0, 0);

// ─── Renderer ─────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.3;
document.body.appendChild(renderer.domElement);

// ─── OrbitControls ────────────────────────────────────────────────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.minDistance = 1.0;
controls.maxDistance = 6.0;
controls.maxPolarAngle = Math.PI * 0.54;  // don't go underground
controls.update();

// ─── Box interior dimensions ──────────────────────────────────────────────────
// Mesh is normalised to ~1.0 units.  Box is 1.4 units wide so there's margin.
const S = 0.7;   // half-size of one box side  →  full box = 1.4 units

// ─── Wall material — warm laser-cut MDF / kraft paper ─────────────────────────
const wallMat = new THREE.MeshStandardMaterial({{
  color: 0xc4a06a,
  roughness: 0.88,
  metalness: 0.0,
  side: THREE.FrontSide,
}});

function addWall(w, h, px, py, pz, rx, ry) {{
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), wallMat);
  mesh.position.set(px, py, pz);
  mesh.rotation.set(rx, ry, 0);
  mesh.receiveShadow = true;
  scene.add(mesh);
}}

// Floor  (y = -S, facing up)
addWall(S*2, S*2,   0,  -S,   0,  -Math.PI/2, 0);
// Back wall  (z = -S, facing +Z)
addWall(S*2, S*2,   0,   0,  -S,   0,          0);
// Left wall  (x = -S, facing +X)
addWall(S*2, S*2,  -S,   0,   0,   0,          Math.PI/2);

// ─── Lighting ─────────────────────────────────────────────────────────────────

// 1. Strong blue/violet point light near left wall — primary shadow caster
//    (matches the coloured LED strip on the left in the photo)
const blueLight = new THREE.PointLight(0x5566ff, 8.0, 4.0, 2);
blueLight.position.set(-S + 0.08, 0.05, 0.1);  // just inside left wall
blueLight.castShadow = true;
blueLight.shadow.mapSize.set(2048, 2048);
blueLight.shadow.camera.near = 0.02;
blueLight.shadow.camera.far  = 4.0;
blueLight.shadow.radius = 3;   // soft penumbra
scene.add(blueLight);

// 2. Warm narrow spotlight from top-centre — creates the circular gobo on floor
const topSpot = new THREE.SpotLight(0xffd580, 6.0);
topSpot.position.set(0.0, S - 0.04, 0.0);  // just under ceiling
topSpot.target.position.set(0, -S, 0);
topSpot.angle   = 0.36;
topSpot.penumbra = 0.5;
topSpot.decay   = 1.6;
topSpot.castShadow = true;
topSpot.shadow.mapSize.set(2048, 2048);
topSpot.shadow.camera.near = 0.05;
topSpot.shadow.camera.far  = 3.0;
scene.add(topSpot);
scene.add(topSpot.target);

// 3. Dim fill from slightly in front so we can see the sculpture face
const fillLight = new THREE.DirectionalLight(0x334466, 0.5);
fillLight.position.set(0.5, 0.3, 1.2);
scene.add(fillLight);

// 4. Ambient — neutral grey so nothing is pure black
const ambient = new THREE.AmbientLight(0x303040, 2.5);
scene.add(ambient);

// ─── Sculpture (GLB loaded from base64) ───────────────────────────────────────
const loader = new GLTFLoader();
const glbData = "data:model/gltf-binary;base64,{glb_base64}";

const statusEl = document.getElementById('status');

loader.load(
  glbData,
  (gltf) => {{
    const obj = gltf.scene;

    // Re-centre: bake trimesh normalisation in case GLB root has an offset
    const bbox = new THREE.Box3().setFromObject(obj);
    const ctr  = new THREE.Vector3();
    bbox.getCenter(ctr);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxExt = Math.max(size.x, size.y, size.z);

    // Rescale so the longest axis = 1.0 (belt-and-suspenders in case trimesh
    // exported a mesh that still has world-scale units in the GLB)
    if (maxExt > 0.001) {{
      obj.scale.setScalar(1.0 / maxExt);
    }}

    // Re-centre after scaling
    const bbox2 = new THREE.Box3().setFromObject(obj);
    const ctr2  = new THREE.Vector3();
    bbox2.getCenter(ctr2);
    obj.position.sub(ctr2);
    obj.position.y += 0.05;   // hang very slightly above the mid-point

    obj.traverse((child) => {{
      if (!child.isMesh) return;
      child.castShadow    = true;
      child.receiveShadow = false;
      // Blue-grey resin/plastic, matching the printed piece in the photo
      child.material = new THREE.MeshStandardMaterial({{
        color:     0x3a5faa,
        roughness: 0.4,
        metalness: 0.12,
      }});
    }});

    scene.add(obj);

    // ── Suspension wires ────────────────────────────────────────────────────
    // Estimate the top of the object after recentering
    const bbox3 = new THREE.Box3().setFromObject(obj);
    const objTopY = bbox3.max.y + obj.position.y;
    const ceilY   = S - 0.01;

    const wireMat = new THREE.LineBasicMaterial({{
      color: 0xdddddd, opacity: 0.4, transparent: true
    }});
    const wireOffsets = [[-0.18,-0.18],[0.18,-0.18],[-0.18,0.18],[0.18,0.18]];
    wireOffsets.forEach(([wx, wz]) => {{
      const pts = [
        new THREE.Vector3(wx * 0.7, objTopY,  wz * 0.7),
        new THREE.Vector3(wx * 1.0, ceilY,    wz * 1.0),
      ];
      scene.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts), wireMat
      ));
    }});
  }},

  // progress  (unused but required positional arg)
  undefined,

  // error callback — show a visible message instead of silent black screen
  (err) => {{
    console.error('GLTFLoader error:', err);
    statusEl.textContent = 'Model load error — check console';
    statusEl.style.opacity = '1';
  }}
);

// ─── Animation loop ───────────────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

// ─── Resize handler ───────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""

    html_path = output_path.replace(".png", ".html")
    with open(html_path, "w") as f:
        f.write(html)
    return html_path