import trimesh
import base64
from pathlib import Path


def render_shadow_preview_threejs(stl_path, output_path):
    mesh = trimesh.load(stl_path)

    # ── Normalize mesh to fit in a unit cube centered at origin ──────────────
    # Scale so the longest axis = 1.0. The box interior will also be 1.0 wide,
    # so the sculpture fills it nicely with a small margin.
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / max(mesh.extents)
    mesh.apply_scale(scale)

    # Export to GLB (Three.js-friendly binary glTF)
    glb_path = str(Path(output_path).with_suffix(".glb"))
    mesh.export(glb_path)
    with open(glb_path, "rb") as f:
        glb_base64 = base64.b64encode(f.read()).decode()

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Shadow Preview</title>
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: #0d0b10;
    }}
  </style>
</head>
<body>
<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js';
import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/loaders/GLTFLoader.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0b10);

const aspect = window.innerWidth / window.innerHeight;
const viewSize = 1.1;
const camera = new THREE.OrthographicCamera(
  -viewSize * aspect,
   viewSize * aspect,
   viewSize,
  -viewSize,
   0.01,
   50
);
camera.position.set(1.45, 1.1, 1.35);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enablePan = true;
controls.minDistance = 0.4;
controls.maxDistance = 6.0;
controls.maxPolarAngle = Math.PI * 0.75;
controls.update();

const blueLight = new THREE.PointLight(0x5a6eff, 7.5, 4.5, 2);
blueLight.position.set(-0.8, 0.9, 0.5);
blueLight.castShadow = true;
blueLight.shadow.mapSize.set(2048, 2048);
blueLight.shadow.camera.near = 0.02;
blueLight.shadow.camera.far = 10.0;
scene.add(blueLight);

const topSpot = new THREE.SpotLight(0xffd580, 5.5, 10.0, 0.45, 0.5, 1.8);
topSpot.position.set(0.0, 1.6, 0.1);
topSpot.target.position.set(0, 0, 0);
topSpot.castShadow = true;
topSpot.shadow.mapSize.set(2048, 2048);
topSpot.shadow.camera.near = 0.05;
topSpot.shadow.camera.far = 10.0;
scene.add(topSpot);
scene.add(topSpot.target);

const fillLight = new THREE.DirectionalLight(0xffffff, 0.55);
fillLight.position.set(0.85, 0.5, 1.05);
fillLight.castShadow = true;
fillLight.shadow.mapSize.set(2048, 2048);
fillLight.shadow.camera.left = -1.0;
fillLight.shadow.camera.right = 1.0;
fillLight.shadow.camera.top = 1.0;
fillLight.shadow.camera.bottom = -1.0;
fillLight.shadow.camera.near = 0.2;
fillLight.shadow.camera.far = 10.0;
scene.add(fillLight);

const ambient = new THREE.AmbientLight(0x404050, 1.5);
scene.add(ambient);

const loader = new GLTFLoader();
const glbData = "data:model/gltf-binary;base64,{glb_base64}";
loader.load(
  glbData,
  (gltf) => {{
    const obj = gltf.scene;
    const bbox = new THREE.Box3().setFromObject(obj);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxExt = Math.max(size.x, size.y, size.z);

    if (maxExt > 0.001) {{
      obj.scale.setScalar(1.0 / maxExt);
    }}

    const bbox2 = new THREE.Box3().setFromObject(obj);
    const ctr2 = new THREE.Vector3();
    bbox2.getCenter(ctr2);
    obj.position.sub(ctr2);
    obj.position.y += 0.01;

    obj.traverse((child) => {{
      if (!child.isMesh) return;
      child.castShadow = true;
      child.receiveShadow = false;
      child.material = new THREE.MeshStandardMaterial({{
        color: 0x3a5faa,
        roughness: 0.38,
        metalness: 0.08,
      }});
    }});

    scene.add(obj);
  }},
  undefined,
  (err) => {{
    console.error('GLTFLoader error:', err);
  }}
);

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
  const aspect = window.innerWidth / window.innerHeight;
  camera.left = -viewSize * aspect;
  camera.right = viewSize * aspect;
  camera.top = viewSize;
  camera.bottom = -viewSize;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""

    html_path = str(Path(output_path).with_suffix(".html"))
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path