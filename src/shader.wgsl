
//@group(0) @binding(0) 
//var<storage, read_write> points: array<GpuPoint>;
@group(0) @binding(1)
var<uniform> params: Params;


struct Params {
    zoom_x: f32,
    zoom_y: f32,
    offset_x: f32,
    offset_y: f32,
}


const NUM_POINTS: f32 = {NUM_POINTS}.0;
const NUM_KINDS: f32 = {NUM_KINDS}.0;

struct GpuPoint {
    x: f32,
    y: f32,
    k: u32,
    _unused: u32,
}

// @location(0) particle_pos: vec2<f32>,
// @location(1) particle_vel: vec2<f32>,
// @location(2) position: vec2<f32>,

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    @location(0) point_xy: vec2<f32>,
    @location(1) k: u32,
    @location(2) _unused: u32,
    @location(3) offset: vec2<f32>,
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
    

    //var remaining = in.vertex_index;
    //let tile = in.vertex_index / (3u * {NUM_POINTS}u);
    //remaining = remaining % (3u * {NUM_POINTS}u);
    //let group = remaining / 3u; // 0..NUM_POINTS
    //let member = remaining % 3u; // 0..3
    let group = in.instance_index;//in.vertex_index / 3u;
    let member = in.vertex_index;// % 3u;

    //let tile_x = tile % 3u; // 0..3
    //let tile_y = tile / 3u; // 0..3
    
    var offset: vec2<f32> = vec2<f32>(0.0);

    let beta: f32 = 0.4;
    let range_index: f32 = 8.0;
    let hash_texture_size = 256.0;
    let radius_clip_space = (range_index) * 1.0 / hash_texture_size;
    let inner_radius_clip_space: f32 = beta * radius_clip_space;

    
    //let qqq = points[group];
    var qqq: GpuPoint;
    qqq.x = in.point_xy.x;
    qqq.y = in.point_xy.y;
    qqq.k = in.k;
        

    //offset += vec2<f32>(f32(tile_x) - 1.0, f32(tile_y) - 1.0) * vec2<f32>({WIDTH_X}, {WIDTH_Y}) * 2.0;


    offset += (vec2<f32>(qqq.x, qqq.y) * 2.0 - 1.0); // 0..1 -> -1..1
    
    offset -= vec2<f32>(params.offset_x, params.offset_y);
    
    offset += in.offset;
    //let tri_size = 0.003;
    //switch member {
    //    case 0u: {
    //        offset += tri_size * vec2<f32>(sqrt(3.0) / 2.0, -0.5);
    //    }
    //    case 1u: {
    //        offset += tri_size * vec2<f32>(0.0, 1.0);
    //    }
    //    case 2u, default: {
    //        offset += tri_size * vec2<f32>(-sqrt(3.0) / 2.0, -0.5);
    //    }
    //}
    offset *= vec2<f32>(params.zoom_x, params.zoom_y);

    let num_kinds: f32 = NUM_KINDS;

    let q = fract(f32(qqq.k) / num_kinds);//fract(f32(group) / num_kinds);
    let u = q * 1.0 * 3.141592;
    let w = (1.0 / 3.0) * 2.0 * 3.241592;
    var color = vec4<f32>(
        pow(sin(u + 0.0 * w), 4.0),
        pow(sin(u + 1.0 * w), 4.0),
        pow(sin(u + 2.0 * w), 4.0),
        1.0
    );
    //if (tile_x != 1u || tile_y != 1u) {
    //    color *= .2;
    //}
    out.color = color;

    out.clip_position = vec4<f32>(offset.x, offset.y, 0.0, 1.0);
    return out;   
}
@fragment
fn fs_main(in: VertexOutputMain) -> @location(0) vec4<f32> {
    return in.color;
}
