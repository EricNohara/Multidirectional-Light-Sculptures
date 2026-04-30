import trimesh
import base64
from pathlib import Path


def render_shadow_preview_threejs(stl_path, output_path):
    mesh = trimesh.load(stl_path)

    # Normalize mesh
    mesh.apply_translation(-mesh.centroid)
    scale = 1.5 / max(mesh.extents)
    mesh.apply_scale(scale)

    # Export to GLB (Three.js-friendly)
    glb_path = str(Path(output_path).with_suffix(".glb"))
    mesh.export(glb_path)

    with open(glb_path, "rb") as f:
        glb_base64 = base64.b64encode(f.read()).decode()

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ margin: 0; overflow: hidden; background: #111; }}
canvas {{ display: block; }}
</style>
</head>
<body>

<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js';
import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/loaders/GLTFLoader.js';

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

// Camera
const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 100);
camera.position.set(4, 4, 4);

// Renderer
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();

// Lights
const light = new THREE.DirectionalLight(0xffffff, 1.2);
light.position.set(5, 8, 5);
light.castShadow = true;
light.shadow.mapSize.set(2048, 2048);
scene.add(light);

const ambient = new THREE.AmbientLight(0x404040, 0.6);
scene.add(ambient);

// Room (walls)
function makeWall(w, h, pos, rot) {{
    const geo = new THREE.PlaneGeometry(w, h);
    const mat = new THREE.MeshStandardMaterial({{
        color: 0xe0e0e0,
        side: THREE.DoubleSide
    }});
    const wall = new THREE.Mesh(geo, mat);
    wall.position.set(...pos);
    wall.rotation.set(...rot);
    wall.receiveShadow = true;
    scene.add(wall);
}}

// Floor + walls
makeWall(6, 6, [0, -1.2, 0], [-Math.PI/2, 0, 0]);
makeWall(6, 6, [0, 0, -3], [0, 0, 0]);
makeWall(6, 6, [-3, 0, 0], [0, Math.PI/2, 0]);

// Load STL (as GLB)
const loader = new GLTFLoader();

const data = "data:model/gltf-binary;base64,{glb_base64}";
loader.load(data, (gltf) => {{
    const obj = gltf.scene;

    obj.traverse((child) => {{
        if (child.isMesh) {{
            child.castShadow = true;
            child.receiveShadow = false;
            child.material = new THREE.MeshStandardMaterial({{
                color: 0x888888,
                roughness: 0.6,
                metalness: 0.1
            }});
        }}
    }});

    scene.add(obj);
}});

// Render loop
function animate() {{
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}}
animate();

// Resize
window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>

</body>
</html>
"""

    html_path = output_path.replace(".png", ".html")
    with open(html_path, "w") as f:
        f.write(html)

    return html_path