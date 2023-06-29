use wgpu::util::DeviceExt;
use wgpu::SurfaceError;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use std::mem::{replace, size_of};

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Point {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}
unsafe impl bytemuck::Zeroable for Point {}
unsafe impl bytemuck::Pod for Point {}

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

//static MAX_NUM_ENTRIES_FOUND: AtomicUsize = AtomicUsize::new(0);

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::array::from_fn;

use prng::Prng;
use vertex::{Vertex, VERTICES};

mod prng;
mod vertex;

const NUM_POINTS: u32 = 4096; //2048; //128;
                              //const NUM_POINTS: u32 = 4 * 4096; //128;
const NUM_KINDS: usize = 8;
const NUM_KINDS2: usize = NUM_KINDS * NUM_KINDS;

const HASH_TEXTURE_SIZE: u32 = 256; //512;//128;

const RANGE_INDEX: u32 = 8;
const RADIUS_CLIP_SPACE: f64 = (RANGE_INDEX as f64) * 1.0 / (HASH_TEXTURE_SIZE as f64);

const BETA: f64 = 0.4;

const INNER_RADIUS_CLIP_SPACE: f64 = BETA * RADIUS_CLIP_SPACE;

const DISTINCT_COLORS: [u32; 20] = [
    0xFFB30000, 0x803E7500, 0xFF680000, 0xA6BDD700, 0xC1002000, 0xCEA26200, 0x81706600, 0x007D3400,
    0xF6768E00, 0x00538A00, 0xFF7A5C00, 0x53377A00, 0xFF8E0000, 0xB3285100, 0xF4C80000, 0x7F180D00,
    0x93AA0000, 0x59331500, 0xF13A1300, 0x232C1600,
];

macro_rules! debug_assert_ok_f32 {
    ($a:expr) => {{
        let a = $a;
        debug_assert!(check_ok_f32(a), "{}: {}", stringify!($a), a);
    }};
}
fn check_ok_f32(f: f32) -> bool {
    use std::num::FpCategory::*;
    match f.classify() {
        Nan | Infinite => false,
        Zero | Subnormal | Normal => true,
    }
}

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
        match $aa {
            tmp => {
                eprintln!(
                    "[{}:{}] {} = {:#?}",
                    file!(),
                    line!(),
                    stringify!($aa),
                    &tmp
                );
                tmp
            }
        }
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
        let dt = 0.01; // 0.01
        let kinds = NUM_KINDS as _;
        let r_inner = 0.02;
        let r_outer = 0.06;
        let w = 16.0;

        let mut rng;
        rng = Prng(13);
        //cfg_if::cfg_if! {
        //    if #[cfg(target_arch = "wasm32")] {
        //        rng = Prng(13);
        //    } else {
        //        rng = Prng(SystemTime::now()
        //            .duration_since(SystemTime::UNIX_EPOCH)
        //            .unwrap()
        //            .as_micros() as u64);
        //    }
        //}

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
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                use VirtualKeyCode::*;
                match keycode {
                    Space => state.render_tris = dbgc!(!state.render_tris),
                    W => state.use_render_bundles = dbgc!(!state.use_render_bundles),
                    C => state.use_cpu = dbgc!(!state.use_cpu),
                    S => state.start_sort = dbgc!(!state.start_sort),
                    _ => (),
                }
            }
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

    bind_groups: [wgpu::BindGroup; 2],
    hash_bind_groups: [wgpu::BindGroup; 2],

    point_texture_views: [wgpu::TextureView; 2],
    hash_texture_views: [wgpu::TextureView; 2],

    point_textures: [wgpu::Texture; 2],

    #[cfg(not(target_arch = "wasm32"))]
    last_print: Instant,
    frame: usize,

    sim_params: SimParams,

    render_tris: bool,
    use_render_bundles: bool,
    use_cpu: bool,
    start_sort: bool,

    update_bundles: [wgpu::RenderBundle; 2],
    hash_bundles: [wgpu::RenderBundle; 2],

    point_read_buffer: wgpu::Buffer,
    point_write_buffer: wgpu::Buffer,
}

impl State {
    async fn new(window: Window) -> Self {
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
        dbgc!(adapter.limits());
        let mut limits = wgpu::Limits::downlevel_webgl2_defaults();
        limits.max_texture_dimension_2d = 16384;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
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
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            //present_mode: wgpu::PresentMode::AutoVsync,
            present_mode: wgpu::PresentMode::Immediate,
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

        let point_textures: [wgpu::Texture; 2] = from_fn(|_| {
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
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[wgpu::TextureFormat::Rgba32Float],
            })
        });

        let point_read_buffer = {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Point Buffer"),
                size: NUM_POINTS as u64 * (4 * size_of::<f32>() as u64),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let point_write_buffer = {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Point Buffer"),
                size: NUM_POINTS as u64 * (4 * size_of::<f32>() as u64),
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let point_texture_views: [wgpu::TextureView; 2] = point_textures
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hash_textures: [wgpu::Texture; 2] = from_fn(|_| {
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
        });

        let hash_texture_views: [wgpu::TextureView; 2] = hash_textures
            .iter()
            .map(|texture| texture.create_view(&Default::default()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

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

        let bind_groups: [_; 2] = point_texture_views
            .iter()
            .zip(hash_texture_views.iter())
            .map(|(point_texture_view, hash_texture_view)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bind group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(point_texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: sim_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(hash_texture_view),
                        },
                    ],
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hash_bind_groups: [wgpu::BindGroup; 2] = point_texture_views
            .iter()
            .zip(hash_texture_views.iter().rev())
            .map(|(point_texture_view, hash_texture_view)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bind group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(point_texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: sim_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(hash_texture_view),
                        },
                    ],
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

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

    let group = in.group;
    return vec4<f32>(in.clip_position.xy / {HASH_TEXTURE_SIZE}.0 * 2.0 - 1.0, f32(group), 0.0);
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
    //return vec4<f32>(c, sin(c * ({NUM_POINTS}.0 + 1.9008)), cos(c*2338.23894), sin(c*2783.7893));
    return vec4<f32>(c, sin(c * ({NUM_POINTS}.0 + 1.9008)), 0.0, 0.0);
}}

@fragment
fn fs_debug_hash(
    in: FillVertexOuput,
) -> @location(0) vec4<f32> {{
    let pos = (vec2<f32>(in.pos.xy) + 1.0) * 0.5;
    let a = textureLoad(hash_texture, vec2<u32>(pos * {HASH_TEXTURE_SIZE}.0), 0);
    return a;
}}

// TODO: let hardware do the wrapping?
// TODO: rearrange to make sampling cheaper
@fragment
fn fs_update(
    in: FillVertexOuput,
) -> @location(0) vec4<f32> {{
    var a = textureLoad(compute_texture, vec2<u32>(u32(in.clip_position.x), 0u), 0);
    let dt = params.dt;
    var p = a.xy;
    var v = a.zw;

    let my_id = u32(in.clip_position.x);

    let my_origin = vec2<i32>(((p + 1.0) * 0.5) * {HASH_TEXTURE_SIZE}.0 + 0.5);

    let range: i32 = {RANGE_INDEX};
    
    var v_i = vec2<f32>(0.0, 0.0);
    for (var i: i32 = -range; i < range; i++) {{
    for (var j: i32 = -range; j < range; j++) {{
        
        var sample = textureLoad(hash_texture, vec2<i32>(my_origin.x + i, my_origin.y + j), 0);
        
        let other_id: u32 = u32(sample.z);//i;
        //var other = textureLoad(compute_texture, vec2<u32>(other_id, 0u), 0);
        //let other_pos = other.xy;
        let other_pos = vec2<f32>(sample.xy);

        if other_id != my_id  && sample.r != 0.0 && sample.g != 0.0 {{

            let relation = params.forces[my_id % {NUM_KINDS}u + (other_id % {NUM_KINDS}u) * {NUM_KINDS}u];
            var f = relation.force;

            let diff = (p - other_pos);
            let l = length(diff);

            let x = l;
            /* 
                let c = 4.0/{HASH_TEXTURE_SIZE}.0;
                //let c = 2.5/{HASH_TEXTURE_SIZE}.0;
                let a = c * 2.0; 

                let d = a;
                let g = a * 2.0; // <- tweak me
                let h = (g + d) / 2.0;
                let k_1 = f / (h - d); // dyn (f)
                let k_2 = 1.0 / (a - c);

                let i_s = 20.0;
                
                //if x < c {{
                //    f = 10.0;
                //}} else if x < a {{
                //    f = min(10.0, 0.04 * (1.0 / (x - c) - k_2));
                if x < a {{
                    f = max(- i_s * x / (a * 0.8) + i_s, 0.0);
                }} else if x < h {{
                    f = (x - d) * k_1;
                }} else if x < g {{
                    f = ( - x + g) * k_1;
                }} else {{
                    f = 0.0;
                }}
            */
            
            
                let beta = {BETA}; // const
                let alpha = 5.0; // const
                let g = {RADIUS_CLIP_SPACE}; // const

                let k1 = (2.0 / ((1.0 - beta) * alpha)); // const
                let k2 = - 1.0 / beta; // const
                let k3 = 1.0 / g; // const
                
                let x_g = x * k3;
                f = max(
                    fma(x_g, k2, 1.0), 
                    f * k1 * 
                        max(
                            0.0, 
                            min(
                                x_g - beta, 
                                1.0 - x_g
                            )
                        )
                ) * alpha;
            



            //v += 0.1 * f * normalize(diff) * dt;
            v_i += f * diff / l;
        }}
    }}
    }}
    v += v_i * 0.1 * dt;
    //v -= 0.01 * p * dt;
    if p.x > 1.0 {{ p.x = -1.0; }} else if p.x < -1.0 {{ p.x = 1.0; }}
    if p.y > 1.0 {{ p.y = -1.0; }} else if p.y < -1.0 {{ p.y = 1.0; }}

    v *= pow(0.1, dt);

    p += v * dt;// / {HASH_TEXTURE_SIZE}.0 * 512.0;
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

struct VertexOutputMain {{
    @builtin(position) clip_position: vec4<f32>,
    @location(1) color: vec4<f32>,
}};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutputMain {{
    var out: VertexOutputMain;

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
    offset *= 0.8 * {INNER_RADIUS_CLIP_SPACE};//3.0/{HASH_TEXTURE_SIZE}.0;

    let a = textureLoad(compute_texture, vec2<u32>(group, 0u), 0);

    offset += a.xy;

    let q = fract(f32(group) / {NUM_KINDS}.0);
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
    return out;   
}}

@fragment
fn fs_main(in: VertexOutputMain) -> @location(0) vec4<f32> {{
    return in.color;
}}

");
        dbgc!(RADIUS_CLIP_SPACE);

        let render_pipeline;
        let update_pipeline;
        let fill_pipeline;
        let hash_pipeline;
        let debug_hash_pipeline;
        {
            let fragment_vertex_shader: wgpu::ShaderModule =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Fragment and vertex shader"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });
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

        let hash_bundles: [_; 2];
        let update_bundles: [_; 2];
        {
            hash_bundles = hash_bind_groups
                .iter()
                .map(|hash_bind_group| {
                    let mut hash_pass =
                        device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                            label: Some("Hash Bundle Encoder"),
                            color_formats: &[Some(wgpu::TextureFormat::Rgba32Float)],
                            depth_stencil: None,
                            sample_count: 1,
                            multiview: None,
                        });
                    hash_pass.set_pipeline(&hash_pipeline);
                    hash_pass.set_bind_group(0, hash_bind_group, &[]);
                    hash_pass.draw(0..NUM_POINTS, 0..1);
                    hash_pass.finish(&wgpu::RenderBundleDescriptor {
                        label: Some("Hash Pass Bundle"),
                    })
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            update_bundles = bind_groups
                .iter()
                .map(|bind_group| {
                    let mut update_pass =
                        device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                            label: Some("Update Bundle Encoder"),
                            color_formats: &[Some(wgpu::TextureFormat::Rgba32Float)],
                            depth_stencil: None,
                            sample_count: 1,
                            multiview: None,
                        });

                    update_pass.set_pipeline(&update_pipeline);
                    update_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    update_pass.set_bind_group(0, bind_group, &[]);
                    update_pass.draw(0..num_vertices, 0..1);
                    update_pass.finish(&wgpu::RenderBundleDescriptor {
                        label: Some("Update Pass Bundle"),
                    })
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        }

        // init state
        {
            let mut fill_encoder: wgpu::CommandEncoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Fill Encoder"),
                });
            let mut fill_pass = fill_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fill Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &point_texture_views[0],
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
            fill_pass.set_bind_group(0, &bind_groups[1], &[]);
            fill_pass.draw(0..num_vertices, 0..1);
            drop(fill_pass);
            queue.submit(Some(fill_encoder.finish()));
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
            hash_bind_groups,
            point_texture_views,
            hash_texture_views,
            #[cfg(not(target_arch = "wasm32"))]
            last_print: Instant::now(),
            frame: 0,
            sim_params,
            update_bundles,
            hash_bundles,
            render_tris: true,
            use_render_bundles: false,
            use_cpu: true,
            point_read_buffer,
            point_write_buffer,
            point_textures,
            start_sort: false,
        }
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        self.frame += 1;
        let max_frame = 240;
        #[cfg(not(target_arch = "wasm32"))]
        if self.frame == max_frame {
            self.frame = 0;
            let dt = replace(&mut self.last_print, Instant::now()).elapsed();
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

        if !self.use_cpu {
            for _ in 0..5 {
                if self.use_render_bundles {
                    for (((hash_view, view), hash_bundle), update_bundle) in self
                        .hash_texture_views
                        .iter()
                        .zip(self.point_texture_views.iter().rev())
                        .zip(self.hash_bundles.iter())
                        .zip(self.update_bundles.iter())
                    {
                        let mut hash_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Hash Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: hash_view,
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
                        hash_pass.execute_bundles(Some(hash_bundle));
                        drop(hash_pass);

                        let mut update_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                        update_pass.execute_bundles(Some(update_bundle));
                        drop(update_pass);
                    }
                } else {
                    for (((hash_view, hash_bind_group), view), bind_group) in self
                        .hash_texture_views
                        .iter()
                        .zip(self.hash_bind_groups.iter())
                        .zip(self.point_texture_views.iter().rev())
                        .zip(self.bind_groups.iter())
                    {
                        let mut hash_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Hash Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: hash_view,
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
                        hash_pass.set_bind_group(0, hash_bind_group, &[]);
                        hash_pass.draw(0..NUM_POINTS, 0..1);
                        drop(hash_pass);

                        let mut update_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
            }
        } else {
            encoder.copy_texture_to_buffer(
                self.point_textures[0].as_image_copy(),
                wgpu::ImageCopyBuffer {
                    buffer: &self.point_read_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(size_of::<[f32; 4]>() as u32 * NUM_POINTS),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: NUM_POINTS,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );

            self.queue.submit(Some(encoder.finish()));
            encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Encoder"),
                });

            let read_buffer_slice = self.point_read_buffer.slice(..);
            read_buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
            self.device.poll(wgpu::Maintain::Wait);
            let mut data: Vec<Point> =
                bytemuck::cast_slice(&read_buffer_slice.get_mapped_range()).to_vec();

            update_points(
                &mut data,
                &self.sim_params,
                replace(&mut self.start_sort, false),
            );

            let write_buffer_slice = self.point_write_buffer.slice(..);
            write_buffer_slice.map_async(wgpu::MapMode::Write, |_| ());
            self.device.poll(wgpu::Maintain::Wait);
            bytemuck::cast_slice_mut(&mut write_buffer_slice.get_mapped_range_mut())
                .copy_from_slice(&data);

            self.point_read_buffer.unmap();
            self.point_write_buffer.unmap();

            encoder.copy_buffer_to_texture(
                wgpu::ImageCopyBuffer {
                    buffer: &self.point_write_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(size_of::<[f32; 4]>() as u32 * NUM_POINTS),
                        rows_per_image: Some(1),
                    },
                },
                self.point_textures[0].as_image_copy(),
                wgpu::Extent3d {
                    width: NUM_POINTS,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        if self.render_tris {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_groups[0], &[]);
            render_pass.draw(0..(3 * NUM_POINTS), 0..1);
            drop(render_pass);
        } else {
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
            debug_hash_pass.set_bind_group(0, &self.bind_groups[0], &[]);
            debug_hash_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            debug_hash_pass.draw(0..self.num_vertices, 0..1);
            drop(debug_hash_pass);
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

const GRID_SIZE: usize = 24;
const GRID_LEN: f32 = 2.0 / GRID_SIZE as f32;
const GRID_RADIUS: f32 = GRID_LEN / 2.0;

#[inline(never)]
fn update_points(points: &mut [Point], params: &SimParams, sort_points: bool) {
    #[inline(always)]
    fn hash_point_pos(px: f32, py: f32) -> (isize, isize) {
        let px = ((px + 1.0) / 2.0).rem_euclid(1.0);
        let py = ((py + 1.0) / 2.0).rem_euclid(1.0);
        ((px * GRID_SIZE as f32) as _, (py * GRID_SIZE as f32) as _)
    }
    #[inline(always)]
    fn join_coords((px, py): (isize, isize)) -> usize {
        ((px.rem_euclid(GRID_SIZE as isize))
            + (py.rem_euclid(GRID_SIZE as isize)) * GRID_SIZE as isize) as usize
    }

    let relation_lookup: [[f32; NUM_KINDS]; NUM_KINDS] =
        from_fn(|i| from_fn(|j| params.forces[i + NUM_KINDS * j].force * -0.1));

    let mut spatial_hash: Vec<Vec<u32>> =
        (0..(GRID_SIZE * GRID_SIZE)).map(|_| Vec::new()).collect();

    if sort_points {
        let mut vecs: Vec<_> = (0..NUM_KINDS).map(|_| Vec::new()).collect();
        for chunk in points.chunks(NUM_KINDS) {
            for (i, &point) in chunk.into_iter().enumerate() {
                vecs[i].push(point);
            }
        }
        for v in &mut vecs {
            v.sort_by_cached_key(|p| join_coords(hash_point_pos(p.x, p.y)));
        }
        let mut iters: Vec<_> = vecs.iter().map(|v| v.iter()).collect();
        let mut reconstructed = Vec::new();
        loop {
            let mut found_some = false;
            for iter in &mut iters {
                if let Some(&point) = iter.next() {
                    found_some = true;
                    reconstructed.push(point);
                }
            }
            if !found_some {
                break;
            }
        }
        points.copy_from_slice(&reconstructed);
    }

    let mut outer_group: Vec<u32> = Vec::with_capacity(NUM_POINTS as _);

    for _ in 0..10 {
        for e in spatial_hash.iter_mut() {
            e.clear();
        }

        for i in 0..NUM_POINTS as usize {
            let p = &points[i];
            let key = join_coords(hash_point_pos(p.x, p.y));
            //spatial_hash.entry(key).or_default().push(i);
            spatial_hash[key].push(i as u32);
        }

        let dt = 0.01;

        let grid_size = GRID_SIZE as isize;
        for (hash_index, inner_group) in spatial_hash.iter().enumerate()
        //.flat_map(|(i, v)| std::iter::repeat(i).zip(v.iter()))
        {
            outer_group.clear();

            for jj in [
                -grid_size - 1,
                -grid_size,
                -grid_size + 1,
                -1,
                0,
                1,
                grid_size - 1,
                grid_size,
                grid_size + 1,
            ]
            .into_iter()
            .map(|d| {
                &spatial_hash[(d + hash_index as isize).rem_euclid(grid_size * grid_size) as usize]
            }) {
                outer_group.extend_from_slice(jj);
            }

            evaluate_groups(inner_group, &outer_group, points, relation_lookup, dt);
        }
    }
}

fn evaluate_groups(
    inner_group: &Vec<u32>,
    outer_group: &Vec<u32>,
    points: &mut [Point],
    relation_lookup: [[f32; NUM_KINDS]; NUM_KINDS],
    dt: f32,
) {
    fn as_rchunks<T, const N: usize>(this: &[T]) -> (&[T], &[[T; N]]) {
        assert!(N != 0, "chunk size must be non-zero");
        let len = this.len() / N;
        let (remainder, multiple_of_n) = this.split_at(this.len() - len * N);
        // SAFETY: We already panicked for zero, and ensured by construction
        // that the length of the subslice is a multiple of N.
        let array_slice = unsafe { as_chunks_unchecked(multiple_of_n) };
        (remainder, array_slice)
    }

    unsafe fn as_chunks_unchecked<T, const N: usize>(s: &[T]) -> &[[T; N]] {
        let new_len = s.len() / N;
        unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), new_len) }
    }

    let (outer_remaining, outer_chunks) = as_rchunks::<_, 8>(outer_group);
    assert_eq!(
        outer_remaining.len() + outer_chunks.len() * 8,
        outer_group.len()
    );
    //let outer_chunks: &[u32] = &[];
    //let outer_remaining = outer_group;
    for &i in inner_group {
        unsafe {
            /*use core::arch::x86_64::*;
            #[inline(always)]
            unsafe fn mul(a: __m256, b: __m256) -> __m256 {
                _mm256_mul_ps(a, b)
            }
            #[inline(always)]
            unsafe fn addi(a: __m256i, b: __m256i) -> __m256i {
                _mm256_add_epi32(a, b)
            }
            #[inline(always)]
            unsafe fn add(a: __m256, b: __m256) -> __m256 {
                _mm256_add_ps(a, b)
            }
            #[inline(always)]
            unsafe fn andi(a: __m256i, b: __m256i) -> __m256i {
                _mm256_and_si256(a, b)
            }
            #[inline(always)]
            unsafe fn and(a: __m256, b: __m256) -> __m256 {
                _mm256_and_ps(a, b)
            }
            #[inline(always)]
            unsafe fn sub(a: __m256, b: __m256) -> __m256 {
                _mm256_sub_ps(a, b)
            }
            #[inline(always)]
            unsafe fn fmadd(a: __m256, b: __m256, c: __m256) -> __m256 {
                _mm256_fmadd_ps(a, b, c)
            }
            #[inline(always)]
            unsafe fn fmsub(a: __m256, b: __m256, c: __m256) -> __m256 {
                _mm256_fmsub_ps(a, b, c)
            }
            #[inline(always)]
            unsafe fn dot(a: __m256, b: __m256) -> __m256 {
                fmadd(a, a, mul(b, b))
            }
            #[inline(always)]
            unsafe fn gather<const SCALE: i32>(p: *const f32, indexes: __m256i) -> __m256 {
                _mm256_i32gather_ps::<SCALE>(p, indexes)
            }
            #[inline(always)]
            unsafe fn rsqrt(a: __m256) -> __m256 {
                _mm256_rsqrt_ps(a)
            }
            #[inline(always)]
            unsafe fn splat(a: f32) -> __m256 {
                _mm256_set1_ps(a)
            }
            #[inline(always)]
            unsafe fn splati(a: i32) -> __m256i {
                _mm256_set1_epi32(a)
            }
            #[inline(always)]
            unsafe fn mod8(a: __m256i) -> __m256i {
                assert_eq!(NUM_KINDS, 8);
                let avoid_zero = addi(splati(GRID_SIZE as i32 * 2), a);
                andi(avoid_zero, splati(0b111))
            }
            #[inline(always)]
            unsafe fn max(a: __m256, b: __m256) -> __m256 {
                _mm256_max_ps(a, b)
            }
            #[inline(always)]
            unsafe fn min(a: __m256, b: __m256) -> __m256 {
                _mm256_min_ps(a, b)
            }



            let x1 = splat(points[i as usize].x);
            let y1 = splat(points[i as usize].y);
            //let vx1 = splat(points[i as usize].vx);
            //let vy1 = splat(points[i as usize].vy);

            let kind_i = i.rem_euclid(NUM_KINDS as _);

            let mut simd_sum_dx = splat(0.0);
            let mut simd_sum_dy = splat(0.0);

            for outer_chunk in outer_chunks {
                let outer_chunk: __m256i = _mm256_loadu_si256(outer_chunk.as_ptr() as _);
                let points_pointer: *const f32 = points.as_ptr() as _;

                let x2: __m256 = gather::<4>(points_pointer.offset(0), outer_chunk);
                let y2: __m256 = gather::<4>(points_pointer.offset(1), outer_chunk);
                //let vx2: __m256 = gather::<4>(points_pointer.offset(2), outer_chunk);
                //let vy2: __m256 = gather::<4>(points_pointer.offset(3), outer_chunk);

                let dx: __m256 = sub(x1, x2);
                let dy: __m256 = sub(y1, y2);

                let r2: __m256 = dot(dx, dy);
                let r_inv: __m256 = rsqrt(r2);
                let r: __m256 = mul(r2, r_inv);

                let kind_j: __m256i = mod8(outer_chunk);

                let forces: __m256 = gather::<{ NUM_KINDS as i32 }>(
                    (relation_lookup.as_ptr() as *const f32).offset(kind_i as _),
                    _mm256_srli_epi32(kind_j, 3),
                );

                // TODO: more fusing

                let beta = 0.3;

                let rx = mul(r, splat(1.0 / GRID_RADIUS as f32));

                let fac = max(
                    mul(
                        max(min(sub(rx, splat(beta)), sub(splat(1.0), rx)), splat(0.0)),
                        forces,
                    ),
                    sub(splat(1.0), mul(rx, splat(1.0 / beta))),
                );

                let t = mul(mul(splat(dt), r_inv), fac); // t >= 0.0

                // NaN mask
                // 12: not equal ordered, non signaling

                // _CMP_EQ_OQ    0x00 /* Equal (ordered, non-signaling)  */
                // _CMP_LT_OS    0x01 /* Less-than (ordered, signaling)  */
                // _CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
                // _CMP_UNORD_Q  0x03 /* Unordered (non-signaling)  */
                // _CMP_NEQ_UQ   0x04 /* Not-equal (unordered, non-signaling)  */
                // _CMP_NLT_US   0x05 /* Not-less-than (unordered, signaling)  */
                // _CMP_NLE_US   0x06 /* Not-less-than-or-equal (unordered, signaling)  */
                // _CMP_ORD_Q    0x07 /* Ordered (nonsignaling)   */
                // _CMP_EQ_UQ    0x08 /* Equal (unordered, non-signaling)  */
                // _CMP_NGE_US   0x09 /* Not-greater-than-or-equal (unord, signaling)  */
                // _CMP_NGT_US   0x0a /* Not-greater-than (unordered, signaling)  */
                // _CMP_FALSE_OQ 0x0b /* False (ordered, non-signaling)  */
                // _CMP_NEQ_OQ   0x0c /* Not-equal (ordered, non-signaling)  */
                // _CMP_GE_OS    0x0d /* Greater-than-or-equal (ordered, signaling)  */
                // _CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */
                // _CMP_TRUE_UQ  0x0f /* True (unordered, non-signaling)  */
                // _CMP_EQ_OS    0x10 /* Equal (ordered, signaling)  */
                // _CMP_LT_OQ    0x11 /* Less-than (ordered, non-signaling)  */
                // _CMP_LE_OQ    0x12 /* Less-than-or-equal (ordered, non-signaling)  */
                // _CMP_UNORD_S  0x13 /* Unordered (signaling)  */
                // _CMP_NEQ_US   0x14 /* Not-equal (unordered, signaling)  */
                // _CMP_NLT_UQ   0x15 /* Not-less-than (unordered, non-signaling)  */
                // _CMP_NLE_UQ   0x16 /* Not-less-than-or-equal (unord, non-signaling)  */
                // _CMP_ORD_S    0x17 /* Ordered (signaling)  */
                // _CMP_EQ_US    0x18 /* Equal (unordered, signaling)  */
                // _CMP_NGE_UQ   0x19 /* Not-greater-than-or-equal (unord, non-sign)  */
                // _CMP_NGT_UQ   0x1a /* Not-greater-than (unordered, non-signaling)  */
                // _CMP_FALSE_OS 0x1b /* False (ordered, signaling)  */
                // _CMP_NEQ_OS   0x1c /* Not-equal (ordered, signaling)  */
                // _CMP_GE_OQ    0x1d /* Greater-than-or-equal (ordered, non-signaling)  */
                // _CMP_GT_OQ    0x1e /* Greater-than (ordered, non-signaling)  */
                // _CMP_TRUE_US  0x1f /* True (unordered, signaling)  */
                //

                // unordered eq zero?
                //let ok_mask = _mm256_cmp_ps::<12>(r_inv, splat(0.0));
                let ok_mask = _mm256_cmp_ps::<0x8>(r_inv, splat(0.0));
                let t_masked = and(ok_mask, t);

                //dbg!(t, ok_mask, t_masked);
                //dbg!();

                {
                    let mut t_buffer: [f32; 8] = [0.0; 8];
                    //_mm256_storeu_ps(t_buffer.as_mut_ptr() as _, t_masked);
                    _mm256_storeu_ps(t_buffer.as_mut_ptr() as _, t_masked);

                    let mut sum_t: f32 = t_buffer.into_iter().sum();
                    assert!(!sum_t.is_nan(), "{sum_t}, {t:?}, {ok_mask:?}");
                }
                simd_sum_dx = fmadd(t_masked, dx, simd_sum_dx);
                simd_sum_dy = fmadd(t_masked, dy, simd_sum_dy);
            }*/

            let mut p_vx = points[i as usize].vx;
            let mut p_vy = points[i as usize].vy;

            for outer_chunk in outer_chunks {
                for &j in outer_chunk {
                    let p2 = points[j as usize];
                    let p = points[i as usize];

                    let (dix, diy) = (p.x - p2.x, p.y - p2.y);
                    let r2 = dix * dix + diy * diy;
                    let r = r2.sqrt();
                    let (dixn, diyn) = (dix / r, diy / r);
                    let fac = {
                        let x = r / GRID_RADIUS;
                        let beta = 0.3;
                        let relation_force =
                            relation_lookup[i as usize % NUM_KINDS][j as usize % NUM_KINDS];
                        let f = relation_force;

                        f32::max(
                            f32::max(f32::min(x - beta, -x + 1.0), 0.0) * f,
                            1.0 - x / beta,
                        )
                    };

                    let dx = fac * dixn * dt;
                    let dy = fac * diyn * dt;
                    p_vx += if dx.is_nan() {
                        //(i as f32 / NUM_POINTS as f32 * 0.5) * 0.0001
                        0.0
                    } else {
                        dx
                    };
                    p_vy += if dy.is_nan() {
                        //(i as f32 / NUM_POINTS as f32 * 0.5) * 0.0001
                        0.0
                    } else {
                        dy
                    };
                }
            }

            for &j in outer_remaining {
                let p2 = points[j as usize];
                let p = points[i as usize];

                let (dix, diy) = (p.x - p2.x, p.y - p2.y);
                let r2 = dix * dix + diy * diy;
                let r = r2.sqrt();
                let (dixn, diyn) = (dix / r, diy / r);
                let fac = {
                    let x = r / GRID_RADIUS;
                    let beta = 0.3;
                    let relation_force =
                        relation_lookup[i as usize % NUM_KINDS][j as usize % NUM_KINDS];
                    let f = relation_force;

                    f32::max(
                        f32::max(f32::min(x - beta, -x + 1.0), 0.0) * f,
                        1.0 - x / beta,
                    )
                };

                let dx = fac * dixn * dt;
                let dy = fac * diyn * dt;
                p_vx += if dx.is_nan() {
                    //(i as f32 / NUM_POINTS as f32 * 0.5) * 0.0001
                    0.0
                } else {
                    dx
                };
                p_vy += if dy.is_nan() {
                    //(i as f32 / NUM_POINTS as f32 * 0.5) * 0.0001
                    0.0
                } else {
                    dy
                };
            }
            //{
            //    let mut dx_buffer: [f32; 8] = [0.0; 8];
            //    let mut dy_buffer: [f32; 8] = [0.0; 8];
            //    _mm256_storeu_ps(dx_buffer.as_mut_ptr() as _, simd_sum_dx);
            //    _mm256_storeu_ps(dy_buffer.as_mut_ptr() as _, simd_sum_dy);
            //    let mut sum_dx: f32 = dx_buffer.into_iter().sum();
            //    let mut sum_dy: f32 = dy_buffer.into_iter().sum();
            //    if sum_dx != 0.0 || sum_dy != 0.0 {
            //        assert!(!sum_dx.is_nan(), "{sum_dx}");
            //        assert!(!sum_dy.is_nan(), "{sum_dy}");
            //        dbg!(sum_dx, sum_dy);
            //    }
            //    p_vx += sum_dx;
            //    p_vx += sum_dx;
            //}
            points[i as usize].vx = p_vx;
            points[i as usize].vy = p_vy;
        }
        /*for &j in outer_group {
            // !0: we know that i != j here and that positions are distinct
            // 0: i may equal j, r may equal zero

            let p2 = points[j as usize];
            let p = &mut points[i as usize];

            let (dix, diy) = (p.x - p2.x, p.y - p2.y);
            let r2 = dix * dix + diy * diy;
            let r = r2.sqrt();
            let (dixn, diyn) = (dix / r, diy / r);
            let fac = {
                let x = r / GRID_RADIUS;
                let beta = 0.3;
                let relation_force =
                    relation_lookup[i as usize % NUM_KINDS][j as usize % NUM_KINDS];
                let f = relation_force;

                f32::max(
                    f32::max(f32::min(x - beta, -x + 1.0), 0.0) * f,
                    1.0 - x / beta,
                )
            };

            let dx = fac * dixn * dt;
            let dy = fac * diyn * dt;
            p.vx += if dx.is_nan() {
                //(i as f32 / NUM_POINTS as f32 * 0.5) * 0.0001
                0.0
            } else {
                dx
            };
            p.vy += if dy.is_nan() {
                //(i as f32 / NUM_POINTS as f32 * 0.5) * 0.0001
                0.0
            } else {
                dy
            };
        }*/

        let p = &mut points[i as usize];

        if p.x > 1.0 {
            p.x = 1.0;
            p.vx = -p.vx.abs() - dt;
        } else if p.x < -1.0 {
            p.x = -1.0;
            p.vx = p.vx.abs() - dt;
        }
        if p.y > 1.0 {
            p.y = 1.0;
            p.vy = -p.vy.abs() - dt;
        } else if p.y < -1.0 {
            p.y = -1.0;
            p.vy = p.vy.abs() + dt;
        }

        p.vx *= 0.05_f32.powf(dt);
        p.vy *= 0.05_f32.powf(dt);

        p.vx -= p.x * (dt * 0.01);
        p.vy -= p.y * (dt * 0.01);

        p.x += p.vx * dt;
        p.y += p.vy * dt;
    }
}
