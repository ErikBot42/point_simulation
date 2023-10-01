
@group(0) @binding(0) 
var compute_texture: texture_2d<f32>;
@group(0) @binding(1) 
var<storage, read> points: array<GpuPoint>;

const NUM_POINTS: f32 = {NUM_POINTS}.0;
const NUM_KINDS: f32 = {NUM_KINDS}.0;

struct GpuPoint {
    x: f32,
    y: f32,
    k: u32,
    _unused: u32,
}


struct FillVertexInput {
    @location(0) position: vec2<f32>,
}
struct FillVertexOuput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pos: vec4<f32>,
}


@vertex
fn vs_fill(
    in: FillVertexInput
) -> FillVertexOuput {
    var out: FillVertexOuput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.pos = vec4<f32>(in.position, 0.0, 1.0);
    return out;
}
@fragment
fn fs_fill(
    in: FillVertexOuput,
) -> @location(0) vec4<f32> {
    let c: f32 = in.pos.x;
    return vec4<f32>(c, sin(c * (NUM_POINTS + 1.9008)), 0.0, 0.0);
}

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
}

struct VertexOutputMain {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutputMain {
    var out: VertexOutputMain;

    let group = in.vertex_index / 3u;
    let member = in.vertex_index % 3u;
    
    var offset: vec2<f32>;

    switch member {
        case 0u: {
            offset = vec2<f32>(sqrt(3.0) / 2.0, -0.5);
        }
        case 1u: {
            offset = vec2<f32>(0.0, 1.0);
        }
        case 2u, default: {
            offset = vec2<f32>(-sqrt(3.0) / 2.0, -0.5);
        }
    }
    let beta: f32 = 0.4;
    let range_index: f32 = 8.0;
    let hash_texture_size = 256.0;
    let radius_clip_space = (range_index) * 1.0 / hash_texture_size;
    let inner_radius_clip_space: f32 = beta * radius_clip_space;

    offset *= 0.8 * inner_radius_clip_space;
    
    let a = textureLoad(compute_texture, vec2<u32>(group, 0u), 0);
    let qqq = points[group];
    if (false) {
        offset += a.xy;

        let num_kinds: f32 = NUM_KINDS;

        let q = fract(f32(group) / num_kinds);
        let u = q * 1.0 * 3.141592;
        let w = (1.0 / 3.0) * 2.0 * 3.241592;
        let color = vec4<f32>(
            pow(sin(u + 0.0 * w), 4.0),
            pow(sin(u + 1.0 * w), 4.0),
            pow(sin(u + 2.0 * w), 4.0),
            1.0
        );
        out.color = color;

        out.clip_position = vec4<f32>(offset.x, offset.y, 0.0, 1.0);
    } else {
        offset += (vec2<f32>(qqq.x, qqq.y) * 2.0 - 1.0); // 0..1 -> -1..1

        let num_kinds: f32 = NUM_KINDS;

        let q = 0.0;//fract(f32(group) / num_kinds);
        let u = q * 1.0 * 3.141592;
        let w = (1.0 / 3.0) * 2.0 * 3.241592;
        let color = vec4<f32>(
            pow(sin(u + 0.0 * w), 4.0),
            pow(sin(u + 1.0 * w), 4.0),
            pow(sin(u + 2.0 * w), 4.0),
            1.0
        );
        out.color = color;

        out.clip_position = vec4<f32>(offset.x, offset.y, 0.0, 1.0);
    }
    return out;   
}
@fragment
fn fs_main(in: VertexOutputMain) -> @location(0) vec4<f32> {
    return in.color;
}
