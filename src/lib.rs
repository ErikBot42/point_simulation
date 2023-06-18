use wgpu::util::DeviceExt;
use wgpu::SurfaceError;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use prng::Prng;
use vertex::{Vertex, VERTICES};

mod prng;
mod vertex;

const NUM_POINTS: u32 = 2048;
const NUM_KINDS: usize = 7;
const NUM_KINDS2: usize = NUM_KINDS * NUM_KINDS;

const HASH_TEXTURE_SIZE: u32 = 256 * 2;
const DISTINCT_COLORS: [u32; 20] = [
    0xFFB30000, 0x803E7500, 0xFF680000, 0xA6BDD700, 0xC1002000, 0xCEA26200, 0x81706600, 0x007D3400,
    0xF6768E00, 0x00538A00, 0xFF7A5C00, 0x53377A00, 0xFF8E0000, 0xB3285100, 0xF4C80000, 0x7F180D00,
    0x93AA0000, 0x59331500, 0xF13A1300, 0x232C1600,
];

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Instant, SystemTime};

macro_rules! printlnc {
    ($aa:expr, $bb:expr) => {
        #[cfg(target_arch = "wasm32")]
        {
            web_sys::console::log_1(&format!($aa, $bb).into());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!($aa, $bb);
        }
    };
}

macro_rules! dbgc {
    ($aa:expr) => {{
        let a = $aa;
        printlnc!("{:#?}", &a);
        a
    }};
}

#[derive(Debug)]
struct SimParams {
    dt: f32,
    r_inner: f32,
    r_outer: f32,
    w: f32,
    kinds: u32,
    _0: u32,
    _1: u32,
    _2: u32,
    forces: Vec<Relation>,
}
#[derive(Debug)]
struct Relation {
    force: f32,
    r: f32,
    r_inner: f32,
    r_outer: f32,
}

impl SimParams {
    #[inline(never)]
    fn as_buffer(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer.extend(
            [self.dt, self.r_inner, self.r_outer, self.w]
                .into_iter()
                .flat_map(f32::to_le_bytes),
        );
        buffer.extend(
            [self.kinds, self._0, self._1, self._2]
                .into_iter()
                .flat_map(u32::to_le_bytes),
        );
        buffer.extend(
            self.forces
                .iter()
                .flat_map(|r| [r.force, r.r, r.r_inner, r.r_outer].into_iter())
                .flat_map(f32::to_le_bytes),
        );
        buffer
    }

    fn new() -> SimParams {
        let dt = 0.01;//0.01;
        let kinds = NUM_KINDS as _;
        let r_inner = 0.02;
        let r_outer = 0.06;
        let w = 16.0;

        let mut rng;
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                rng = Prng(13);
            } else {
                rng = Prng(SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64);
            }
        }

        printlnc!("seed: {}", rng.0);

        SimParams {
            dt,
            r_inner,
            r_outer,
            w,
            kinds,
            _0: 0,
            _1: 0,
            _2: 0,
            forces: (0..kinds * kinds)
                .map(|_| {
                    let force = rng.f32(-1.0..1.0);
                    let r = r_inner;
                    let r_inner = rng.f32(0.02..0.04);
                    let r_outer = rng.f32((r_inner + 0.01)..0.1);
                    //let r_inner = r_inner;
                    //let r_outer = r_outer;

                    Relation {
                        force,
                        r,
                        r_inner,
                        r_outer,
                    }
                })
                .collect(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: 960,
            height: 540,
        })
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("id-thingy")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut state = State::new(window).await;
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(physical_size) => state.resize(*physical_size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size)
            }
            _ => state.input(event),
        },
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                Err(SurfaceError::Lost) => state.resize(state.size),
                Err(SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            state.window().request_redraw();
        }
        _ => {}
    });
}
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,

    render_pipeline: wgpu::RenderPipeline,
    update_pipeline: wgpu::RenderPipeline,
    hash_pipeline: wgpu::RenderPipeline,
    debug_hash_pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,

    bind_groups: (wgpu::BindGroup, wgpu::BindGroup),

    point_texture_views: (wgpu::TextureView, wgpu::TextureView),
    hash_texture_views: (wgpu::TextureView, wgpu::TextureView),

    #[cfg(not(target_arch = "wasm32"))]
    last_print: Instant,
    frame: usize,

    sim_params: SimParams,
}

impl State {
    async fn new(window: Window) -> Self {
        // apply:     pos, vel -> pos, _
        // aggregate: pos+, _ -> _ , vel
        //
        // aggregate + apply: pos0+, vel0 -> pos1, vel1

        // Limits {
        //     max_texture_dimension_1d: 2048,
        //     max_texture_dimension_2d: 2048,
        //     max_texture_dimension_3d: 256,
        //     max_texture_array_layers: 256,
        //     max_bind_groups: 4,
        //     max_bindings_per_bind_group: 640,
        //     max_dynamic_uniform_buffers_per_pipeline_layout: 8,
        //     max_dynamic_storage_buffers_per_pipeline_layout: 0,
        //     max_sampled_textures_per_shader_stage: 16,
        //     max_samplers_per_shader_stage: 16,
        //     max_storage_buffers_per_shader_stage: 0,
        //     max_storage_textures_per_shader_stage: 0,
        //     max_uniform_buffers_per_shader_stage: 11,
        //     max_uniform_buffer_binding_size: 16384,
        //     max_storage_buffer_binding_size: 0,
        //     max_vertex_buffers: 8,
        //     max_buffer_size: 268435456,
        //     max_vertex_attributes: 16,
        //     max_vertex_buffer_array_stride: 255,
        //     min_uniform_buffer_offset_alignment: 256,
        //     min_storage_buffer_offset_alignment: 256,
        //     max_inter_stage_shader_components: 60,
        //     max_compute_workgroup_storage_size: 0,
        //     max_compute_invocations_per_workgroup: 0,
        //     max_compute_workgroup_size_x: 0,
        //     max_compute_workgroup_size_y: 0,
        //     max_compute_workgroup_size_z: 0,
        //     max_compute_workgroups_per_dimension: 0,
        //     max_push_constant_size: 0,
        // }

        let size: winit::dpi::PhysicalSize<u32> = window.inner_size();
        let instance: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let surface: wgpu::Surface = unsafe { instance.create_surface(&window) }.unwrap();
        dbgc!(instance
            .enumerate_adapters(wgpu::Backends::all())
            .collect::<Vec<_>>());
        let adapter: wgpu::Adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .find(|adapter| adapter.is_surface_supported(&surface))
            .unwrap();
        dbgc!(adapter.features());
        let limits = wgpu::Limits::downlevel_webgl2_defaults();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    //features: wgpu::Features::POLYGON_MODE_LINE
                    //    | wgpu::Features::POLYGON_MODE_POINT,
                    limits: dbgc!(limits),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format: wgpu::TextureFormat = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let num_vertices = VERTICES.len() as u32;

        let point_textures = {
            let make_point_texture = || {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Point Texture"),
                    size: wgpu::Extent3d {
                        width: NUM_POINTS,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[wgpu::TextureFormat::Rgba32Float],
                })
            };
            (make_point_texture(), make_point_texture())
        };
        let point_texture_views: (wgpu::TextureView, wgpu::TextureView) = (
            point_textures
                .0
                .create_view(&wgpu::TextureViewDescriptor::default()),
            point_textures
                .1
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );

        // x, y, kind, undef
        let hash_textures = {
            let make_hash_texture = || {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Hash Texture"),
                    size: wgpu::Extent3d {
                        width: HASH_TEXTURE_SIZE,
                        height: HASH_TEXTURE_SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[wgpu::TextureFormat::Rgba32Float],
                })
            };
            (make_hash_texture(), make_hash_texture())
        };
        let hash_texture_views: (wgpu::TextureView, wgpu::TextureView) = (
            hash_textures
                .0
                .create_view(&wgpu::TextureViewDescriptor::default()),
            hash_textures
                .1
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );

        let sim_params = SimParams::new();
        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim params buffer"),
            contents: &sim_params.as_buffer(),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout: wgpu::BindGroupLayout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let bind_groups: (wgpu::BindGroup, wgpu::BindGroup) = (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&point_texture_views.0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sim_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&hash_texture_views.0),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&point_texture_views.1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sim_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&hash_texture_views.1),
                    },
                ],
            }),
        );

        let shader_source: String = format!("
@group(0) @binding(0) 
var compute_texture: texture_2d<f32>;

@group(0) @binding(1)
var<uniform> params: SimParams;

@group(0) @binding(2) 
var hash_texture: texture_2d<f32>;

struct SimParams {{
    dt: f32,
    r_inner: f32,
    r_outer: f32,
    w: f32,
    kinds: u32,
    _0: u32,
    _1: u32,
    _2: u32,
    forces: array<Relation, {NUM_KINDS2}>,
}}

struct Relation {{
    force: f32,
    r: f32,
    r_inner: f32,
    r_outer: f32,
}}


struct FillVertexInput {{
    @location(0) position: vec2<f32>,
}}
struct FillVertexOuput {{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pos: vec4<f32>,
}}
@vertex
fn vs_hash(
    in: VertexInput
) -> VertexOutput {{
    var out: VertexOutput;
    let group = in.vertex_index;
    let pos = textureLoad(compute_texture, vec2<u32>(group, 0u), 0).xy;
    out.clip_position = vec4<f32>(pos.x, -pos.y, 0.0, 1.0);
    out.group = group;
    return out;   
}}
@fragment
fn fs_hash(in: VertexOutput) -> @location(0) vec4<f32> {{
    //let a = textureLoad(compute_texture, vec2<u32>(in.group, 0u), 0);
    //let q = fract(f32(in.group) / {NUM_KINDS}.0);
    //
    //let u = q * 2.0 * 3.141592;

    //let w = (1.0 / 3.0) * 2.0 * 3.241592;

    //return vec4<f32>(
    //pow(sin(u + 0.0 * w), 4.0),
    //pow(sin(u + 1.0 * w), 4.0),
    //pow(sin(u + 2.0 * w), 4.0),
    //        1.0
    //);
    let q = fract(f32(in.group) / {NUM_KINDS}.0);
    
    let u = q * 2.0 * 3.141592;

    let w = (1.0 / 3.0) * 2.0 * 3.241592;

    return vec4<f32>(
    0.2 * pow(sin(u + 0.0 * w), 4.0),
    0.2 * pow(sin(u + 1.0 * w), 4.0),
    0.2 * pow(sin(u + 2.0 * w), 4.0),
            1.0
    );



    //return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    ////return vec4<f32>(1.0-q, q, 0.0, 1.0);
}}


@vertex
fn vs_fill(
    in: FillVertexInput
) -> FillVertexOuput {{
    var out: FillVertexOuput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.pos = vec4<f32>(in.position, 0.0, 1.0);
    return out;
}}

@fragment
fn fs_fill(
    in: FillVertexOuput,
) -> @location(0) vec4<f32> {{
    let c: f32 = in.pos.x;
    return vec4<f32>(c, sin(c * ({NUM_POINTS}.0 + 1.9008)), cos(c*2338.23894), sin(c*2783.7893));
}}

@fragment
fn fs_debug_hash(
    in: FillVertexOuput,
) -> @location(0) vec4<f32> {{
    let pos = (vec2<f32>(in.pos.xy) + 1.0) * 0.5;

    let a = textureLoad(hash_texture, vec2<u32>(pos * {HASH_TEXTURE_SIZE}.0), 0);


    return a;
    //return vec4<f32>(1.0, pos.x, pos.y, 1.0);
    //return vec4<f32>(a.x, a.y, a.z, 1.0);
}}

@fragment
fn fs_update(
    in: FillVertexOuput,
) -> @location(0) vec4<f32> {{
    var a = textureLoad(compute_texture, vec2<u32>(u32(in.clip_position.x), 0u), 0);
    let dt = params.dt;
    //let dt = 0.004;
    var p = a.xy;
    var v = a.zw;


    //var a = textureLoad(compute_texture, vec2<u32>(u32(in.pos.x * {NUM_POINTS}.0 + 0.5), 0u), 0);
    
    var attract_p = vec2<f32>(0.0);
    var repell_p = vec2<f32>(0.0);
    for (var i: i32 = 0; i < {NUM_POINTS}; i++) {{
        let my_id = u32(in.clip_position.x);


        let relation = params.forces[my_id % {NUM_KINDS}u + (u32(i) % {NUM_KINDS}u) * {NUM_KINDS}u];
        var f = relation.force;

        var other = textureLoad(compute_texture, vec2<u32>(u32(i), 0u), 0);

        // TODO: let hardware do the wrapping?
        // TODO: rearrange to make sampling cheaper
        //
        if u32(i) != my_id {{
            let diff = (p.xy - other.xy);
            let l = length(diff);
            
            //{{
            //    let r_inner = relation.r_inner;
            //    let r_outer = relation.r_outer;
            //    let r_inner_c = 0.02;

            //    let a = 2.0 * f / (r_outer - r_inner);
            //    let q = (l - r_inner) * a;
            //    let u = (-l + r_outer) * a;

            //    let w = 64.0;
            //    var z = w - (l / r_inner_c) * w;
            //    f = max(
            //        max(
            //            0.0,
            //            min(q * sign(a), u * sign(a))
            //        ) * sign(a),
            //        z
            //    );
            //}}
            {{
                let x = l;
                let c = 4.0/{HASH_TEXTURE_SIZE}.0;
                let a = c * 2.0; 

                let d = a;
                let g = a * 4.0; // <- tweak me
                let h = (g + d) / 2.0;
                let k_1 = f / (h - d); // dyn (f)
                let k_2 = 1.0 / (a - c);

                
                if x < c {{
                    f = 100.0;
                }} else if x < a {{
                    f = min(100.0, 0.04 * (1.0 / (x - c) - k_2));
                }} else if x < h {{
                    f = (x - d) * k_1;
                }} else if x < g {{
                    f = ( - x + g) * k_1;
                }} else {{
                    f = 0.0;
                }}
            }}
            v += 0.1 * f * normalize(diff) * dt;
        }}
    }}
    v -= 0.01 * p * dt;
    if p.x > 1.0 {{ p.x = -1.0; }} else if p.x < -1.0 {{ p.x = 1.0; }}
    if p.y > 1.0 {{ p.y = -1.0; }} else if p.y < -1.0 {{ p.y = 1.0; }}

    v *= pow(0.3, dt);

    p += v * dt;
    return vec4<f32>(p, v);
}}

struct VertexInput {{
    //@location(0) position: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
}}

struct VertexOutput {{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) group: u32,
}};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {{
    var out: VertexOutput;

    let group = in.vertex_index / 3u;
    let member = in.vertex_index % 3u;
    
    var offset: vec2<f32>;

    switch member {{
        case 0u: {{
            offset = vec2<f32>(sqrt(3.0) / 2.0, -0.5);
        }}
        case 1u: {{
            offset = vec2<f32>(0.0, 1.0);
        }}
        case 2u, default: {{
            offset = vec2<f32>(-sqrt(3.0) / 2.0, -0.5);
        }}
    }}
    offset *= 0.01;
    
    let a = textureLoad(compute_texture, vec2<u32>(group, 0u), 0);

    offset *= max( - abs(a.zw) * 10.0 + vec2<f32>(1.0, 1.0), vec2<f32>(0.5, 0.5));

    offset += a.xy;

    
    //out.clip_position = vec4<f32>(offset.x + f32(group) * 0.01 - 1.0, offset.y + sin(f32(group)) * 0.5, 0.0, 1.0);
    out.clip_position = vec4<f32>(offset.x, offset.y, 0.0, 1.0);
    out.group = group;
    return out;   
}}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
    //let a = textureLoad(compute_texture, vec2<u32>(in.group, 0u), 0);
    let q = fract(f32(in.group) / {NUM_KINDS}.0);
    
    let u = q * 2.0 * 3.141592;

    let w = (1.0 / 3.0) * 2.0 * 3.241592;

    return vec4<f32>(
    pow(sin(u + 0.0 * w), 4.0),
    pow(sin(u + 1.0 * w), 4.0),
    pow(sin(u + 2.0 * w), 4.0),
            1.0
    );
    //return vec4<f32>(1.0-q, q, 0.0, 1.0);
}}

");
        let fragment_vertex_shader: wgpu::ShaderModule =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Fragment and vertex shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let render_pipeline;
        let update_pipeline;
        let fill_pipeline;
        let hash_pipeline;
        let debug_hash_pipeline;
        {
            let pipeline_layout: wgpu::PipelineLayout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
            let layout = Some(&pipeline_layout);
            let make_primitive = |topology| wgpu::PrimitiveState {
                topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            };
            let multisample: wgpu::MultisampleState = wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            };
            let vertex_fill = wgpu::VertexState {
                module: &fragment_vertex_shader,
                entry_point: "vs_fill",
                buffers: &[Vertex::desc()],
            };
            //: wgpu::TextureFormat::Rgba32Float,

            debug_hash_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Debug Hash Pipeline"),
                layout,
                vertex: vertex_fill.clone(),
                primitive: make_primitive(wgpu::PrimitiveTopology::TriangleList),
                depth_stencil: None,
                multisample,
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_vertex_shader,
                    entry_point: "fs_debug_hash",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });

            hash_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Hash Pipeline"),
                layout,
                vertex: wgpu::VertexState {
                    module: &fragment_vertex_shader,
                    entry_point: "vs_hash",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_vertex_shader,
                    entry_point: "fs_hash",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: make_primitive(wgpu::PrimitiveTopology::PointList),
                depth_stencil: None,
                multisample,
                multiview: None,
            });

            render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout,
                vertex: wgpu::VertexState {
                    module: &fragment_vertex_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_vertex_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: make_primitive(wgpu::PrimitiveTopology::TriangleList),
                depth_stencil: None,
                multisample,
                multiview: None,
            });
            fill_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Fill Pipeline"),
                layout,
                vertex: vertex_fill.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_vertex_shader,
                    entry_point: "fs_fill",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: make_primitive(wgpu::PrimitiveTopology::TriangleList),
                depth_stencil: None,
                multisample,
                multiview: None,
            });
            update_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Update Pipeline"),
                layout,
                vertex: vertex_fill.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_vertex_shader,
                    entry_point: "fs_update",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: make_primitive(wgpu::PrimitiveTopology::TriangleList),
                depth_stencil: None,
                multisample,
                multiview: None,
            });
        }

        // init state
        {
            let mut encoder: wgpu::CommandEncoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Fill Encoder"),
                });
            let mut fill_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fill Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &point_texture_views.0,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            fill_pass.set_pipeline(&fill_pipeline);
            fill_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            fill_pass.set_bind_group(0, &bind_groups.1, &[]);
            fill_pass.draw(0..num_vertices, 0..1);
            drop(fill_pass);
            queue.submit(Some(encoder.finish()));
        }

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            update_pipeline,
            hash_pipeline,
            debug_hash_pipeline,
            vertex_buffer,
            num_vertices,
            bind_groups,
            point_texture_views,
            hash_texture_views,
            #[cfg(not(target_arch = "wasm32"))]
            last_print: Instant::now(),
            frame: 0,
            sim_params,
        }
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        self.frame += 1;
        let max_frame = 240;
        #[cfg(not(target_arch = "wasm32"))]
        if self.frame == max_frame {
            self.frame = 0;
            let dt = std::mem::replace(&mut self.last_print, Instant::now()).elapsed();
            let fps = max_frame as f64 / dt.as_secs_f64();
            printlnc!("fps: {fps:?}, dt: {:50?}", dt);
        }

        let output: wgpu::SurfaceTexture = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder: wgpu::CommandEncoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Encoder"),
                });

        for (view, bind_group) in [
            (&self.hash_texture_views.1, &self.bind_groups.0),
            (&self.hash_texture_views.0, &self.bind_groups.1),
        ] {
            let mut hash_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Hash Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            hash_pass.set_pipeline(&self.hash_pipeline);
            //hash_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            hash_pass.set_bind_group(0, bind_group, &[]);
            hash_pass.draw(0..NUM_POINTS, 0..1);
        }

        for _ in 0..2 {
            for (view, bind_group) in [
                (&self.point_texture_views.1, &self.bind_groups.0),
                (&self.point_texture_views.0, &self.bind_groups.1),
            ] {
                let mut update_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Update Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });
                update_pass.set_pipeline(&self.update_pipeline);
                update_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                update_pass.set_bind_group(0, bind_group, &[]);
                update_pass.draw(0..self.num_vertices, 0..1);
                drop(update_pass);
            }
        }

        {
            let mut debug_hash_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Debug Hash Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            debug_hash_pass.set_pipeline(&self.debug_hash_pipeline);
            debug_hash_pass.set_bind_group(0, &self.bind_groups.1, &[]);
            debug_hash_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            debug_hash_pass.draw(0..self.num_vertices, 0..1);
            drop(debug_hash_pass);
        }

        if false {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        //load: wgpu::LoadOp::Clear(wgpu::Color {
                        //    r: 0.0,
                        //    g: 0.0,
                        //    b: 0.0,
                        //    a: 1.0,
                        //}),
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_groups.1, &[]);
            render_pass.draw(0..(3 * NUM_POINTS), 0..1);
            drop(render_pass);
        }
        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }

    fn update(&mut self) {}

    fn input(&mut self, _event: &WindowEvent<'_>) {}

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}
